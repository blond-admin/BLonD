# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Wrap a BLonD trackable so it can be tracked inside an xsuite ``Line``."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from blond.core.backends.backend import backend
from blond.core.base import DynamicParameter
from blond.core.beam.beams import Beam
from blond.core.beam.flags import BeamFlags
from blond.core.beam.particle_types import ParticleType
from blond.core.scheduling import ScheduledInterpolation
from blond.interfaces.xsuite.elements.helpers import (
    ReferenceFrame,
    blond_beam_to_xsuite_particles,
    xsuite_particles_to_blond_beam,
)

if TYPE_CHECKING:  # pragma: no cover
    from xtrack import Particles as XsuiteParticles

    from blond.core.base import BeamPhysicsRelevant


def _scalar(x: npt.ArrayLike) -> float:
    """
    Extract a Python float from an xsuite scalar / 0-d / 1-d quantity.

    Parameters
    ----------
    x
        An xsuite scalar, 0-d or 1-d array-like quantity.

    Returns
    -------
    float
        The first (or only) element as a Python float.
    """
    # ``float()`` handles scalars and size-1 arrays of any library (numpy,
    # cupy); per-particle arrays raise and fall back to their first element
    # — without copying the whole (possibly device) array to host.
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(x[0])


class WrapBlond4Xsuite:
    """
    Run a BLonD element inside an xsuite ``Line``.

    Wrap any BLonD trackable (an RF station, a drift, a wake field, ...) so it
    exposes the ``track(particles)`` interface an xsuite ``Line`` expects, then
    drop it into your line like any other xsuite element::

        line = xt.Line(elements=[..., WrapBlond4Xsuite(rf_station), ...])

    xsuite stays in charge of the reference energy: the wrapper reads the live
    reference (``beta0``, ``energy0``) from the particles each turn and feeds it
    to the BLonD element, so an attached ``energy_program`` or an explicit
    ``ReferenceEnergyIncrease`` ramps the BLonD physics too. Because of that, a
    wrapped element must **not** advance the reference itself: construct RF
    stations with ``magnetic_cycle=None`` so xsuite remains the sole driver.

    Particle losses cross the boundary in both directions: slots xsuite has
    already killed are skipped, and particles the BLonD element loses during its
    track are marked lost in xsuite's state so downstream elements skip them too.

    Schedules: turn-keyed schedules (``ScheduledArray``, e.g.
    ``element.schedule(attribute="voltage", value=per_turn_array)``) work out
    of the box — the wrapper feeds the element's turn counter from xsuite's
    ``particles.at_turn`` each track call. Time-keyed schedules
    (``ScheduledInterpolation``) are refused: nothing advances
    ``beam.reference.time`` inside an xsuite-owned loop.

    GPU: the particles' xobjects context decides. Particles on a
    ``ContextCupy`` switch the (process-global) BLonD backend to cuda via
    ``setup_backend`` on first track (with a warning — call
    ``setup_backend("cuda")`` yourself at script start to silence it), and
    all per-turn data stays on the device. CPU particles combined with an
    explicitly chosen GPU BLonD backend are refused.

    Parameters
    ----------
    element
        A BLonD trackable, typically constructed via the element's
        ``headless(...)`` factory. RF stations (and any element that would
        otherwise advance the reference itself) must be constructed with
        ``magnetic_cycle=None`` so xsuite remains the sole driver of the
        reference energy.

    See Also
    --------
    blond.interfaces.xsuite.elements.wrap_xsuite_element.WrapXsuite4Blond :
        The mirrored direction — an xsuite ``Line`` tracked inside a BLonD
        ring (BLonD owns the loop and the reference energy).

    Examples
    --------
    PS triple splitting with xsuite owning the loop and the wrapped BLonD
    station providing the :math:`h=7/14/21` splitting cavity
    (``blond/examples/scripts/EX_29_BLonD_in_Xsuite.py``):

    .. literalinclude:: /../blond/examples/scripts/EX_29_BLonD_in_Xsuite.py
       :language: python
       :start-at: import os
    """

    def __init__(self, element: BeamPhysicsRelevant) -> None:
        # xsuite owns the reference energy inside its line: a wrapped element
        # that drives the reference itself would relabel ``dE`` every turn
        # while xsuite's ``p0c`` never moves — silent double counting.
        # Mirror of ``WrapXsuite4Blond``'s ``energy_program`` guard.
        if getattr(element, "magnetic_cycle", None) is not None:
            raise ValueError(
                "WrapBlond4Xsuite was given an element with an attached "
                "magnetic_cycle. xsuite owns the reference energy inside an "
                "xt.Line, and a wrapped element advancing it too would "
                "double-count the acceleration. Construct the element with "
                "magnetic_cycle=None (e.g. via its headless(...) factory) "
                "and drive the ramp from xsuite's energy_program, or let "
                "BLonD own the loop and use WrapXsuite4Blond instead."
            )
        # xsuite owns the geometry: this wrapper is a thin element (no
        # ``length``), so a wrapped element whose track applies drift
        # kinematics over an orbit length would change ``dt`` while xsuite's
        # ``s``/survey bookkeeping never moves. Keep drift *physics* on the
        # xsuite side. RF-manipulation elements are exempt even when they
        # carry a nonzero ``orbit_length``: real cavities occupy metres of
        # the ring, but their track is a thin kick — the length is pure
        # ring-layout bookkeeping and never moves ``dt`` or the reference
        # clock.
        # Imports not at top to prevent cyclic imports.
        from blond.core.base import HasOrbitLength
        from blond.physics.rf_station import RFManipulationBaseClass

        if (
            isinstance(element, HasOrbitLength)
            and element.orbit_length != 0.0
            and not isinstance(element, RFManipulationBaseClass)
        ):
            raise ValueError(
                "WrapBlond4Xsuite wraps thin-tracking elements only, but "
                f"the given element occupies orbit_length="
                f"{element.orbit_length} m. xsuite owns the geometry inside "
                "an xt.Line — model the longitudinal drift with an xsuite "
                "element (xt.Drift, LineSegmentMap) and wrap only the "
                "thin-kick BLonD physics (cavities, wakefields), or let "
                "BLonD own the loop and use WrapXsuite4Blond instead."
            )
        # Turn-keyed schedules are driven from xsuite's own bookkeeping: the
        # wrapper owns a turn counter and feeds it from ``particles.at_turn``
        # each track call. Time-keyed schedules would need a reference clock
        # nobody advances in an xsuite-owned loop, so they stay refused.
        self._turn_counter: DynamicParameter | None = None
        if getattr(element, "schedule_active", False):
            time_keyed = [
                attribute
                for attribute, scheduled in element.schedules.items()
                if isinstance(scheduled, ScheduledInterpolation)
            ]
            if time_keyed:
                raise NotImplementedError(
                    "WrapBlond4Xsuite cannot drive time-keyed schedules "
                    "(ScheduledInterpolation): beam.reference.time is not "
                    "advanced inside an xsuite-owned loop. Use turn-indexed "
                    "arrays (ScheduledArray) instead. Got time-keyed "
                    f"schedules for {time_keyed}."
                )
            self._turn_counter = DynamicParameter(0)
            # ``Schedulable`` owns the counter; the element's track applies
            # the active schedules from it, the wrapper keeps it current.
            element.turn_counter = self._turn_counter

        self._element = element
        self._beam: Beam | None = None

    def track(self, particles: XsuiteParticles) -> None:
        """
        Track xsuite ``particles`` through the wrapped BLonD element.

        Parameters
        ----------
        particles
            An xsuite ``Particles`` whose ``zeta`` / ``ptau`` are updated in
            place.
        """
        self._sync_backend_with_context(particles)

        # Read the reference from the live ``particles`` state, not from
        # ``line.particle_ref`` (design state): xsuite advances beta0/energy0/
        # p0c on the particles between turns when an ``energy_program`` is
        # attached, or in-line when a ``ReferenceEnergyIncrease`` fires, so by
        # the time we run they already carry the current reference.
        beta0 = _scalar(particles.beta0)
        energy0 = _scalar(particles.energy0)
        frame = ReferenceFrame(
            beta0=beta0,
            energy0=energy0,
        )

        # Slot count, not active-particle count: BLonD's beam mirrors every
        # xsuite slot so lost slots keep their index. ``xt.Particles`` offers
        # no public equivalent — it has no ``__len__`` and no
        # ``num_particles``; ``_capacity`` is private and
        # ``_num_active_particles`` counts only the live slots.
        n = int(particles.zeta.shape[0])
        if self._beam is None or len(self._beam.read_partial_dt()) != n:
            self._build_beam(
                particles=particles,
                energy0=energy0,
                n=n,
            )

        # Push the live reference into the BLonD beam so any element reading
        # ``beam.reference.beta``/``gamma`` (e.g. an RF station computing
        # ``omega_rev``) follows the xsuite ramp. Then convert active particle
        # coordinates into the reusable beam, marking xsuite-lost slots LOST.
        self._beam.reference.total_energy = energy0
        # Feed the wrapped element's schedule machinery with xsuite's own
        # turn bookkeeping: any *active* particle's ``at_turn`` is the
        # current turn index (lost particles freeze at their loss turn).
        if self._turn_counter is not None:
            active = particles.state > 0
            if bool(active.any()):
                self._turn_counter.value = int(
                    particles.at_turn[int(active.argmax())]
                )
        active_at_input = xsuite_particles_to_blond_beam(
            particles=particles,
            beam=self._beam,
            frame=frame,
        )
        self._element.track(self._beam)

        # The BLonD element may have flagged additional particles LOST
        # during its track (loss boxes, energy cuts, ...). Propagate those
        # losses back into xsuite's state so subsequent xsuite elements
        # also skip them — and only write coordinates back for slots that
        # survived BLonD's track.
        blond_active_now = (
            self._beam.read_partial_flags() == BeamFlags.ACTIVE.value
        )
        newly_lost = active_at_input & ~blond_active_now
        if bool(newly_lost.any()):
            particles.state[newly_lost] = BeamFlags.LOST.value
        blond_beam_to_xsuite_particles(
            beam=self._beam,
            particles=particles,
            frame=frame,
            active=active_at_input & blond_active_now,
        )

    def _sync_backend_with_context(self, particles: XsuiteParticles) -> None:
        """
        Make BLonD's backend follow the particles' xobjects context.

        xsuite is the master here: GPU particles (``ContextCupy``) switch
        BLonD to the cuda backend via
        :func:`blond.core.backends.helpers.setup_backend` (with a warning,
        since the backend is process-global). The reverse mismatch — CPU
        particles with an explicitly chosen GPU BLonD backend — is refused
        rather than silently downgraded.

        Parameters
        ----------
        particles
            The xsuite ``Particles`` about to be tracked.
        """
        # Import not at top to prevent import crash on computers without Xsuite.
        from xobjects import ContextCpu, ContextCupy

        context = particles._context
        if isinstance(context, ContextCupy):
            if not backend.is_gpu:
                warnings.warn(
                    "WrapBlond4Xsuite received particles on a ContextCupy "
                    "while BLonD runs a CPU backend; switching the "
                    "(process-global) BLonD backend to cuda to match "
                    "xsuite. Call setup_backend('cuda') yourself at script "
                    "start to silence this warning.",
                    UserWarning,
                    stacklevel=3,
                )
                from blond.core.backends.helpers import setup_backend

                setup_backend("cuda")
                # Any internal beam built on the CPU backend is stale.
                self._beam = None
        elif isinstance(context, ContextCpu):
            if backend.is_gpu:
                warnings.warn(
                    "WrapBlond4Xsuite received particles on a ContextCpu "
                    "while BLonD runs a GPU backend; switching the "
                    "(process-global) BLonD backend to cpp or numba to match "
                    "xsuite. Call setup_backend('xxx') yourself at script "
                    "start to silence this warning.",
                    UserWarning,
                    stacklevel=3,
                )
                from blond.core.backends.helpers import setup_backend

                try:
                    setup_backend("cpp")
                except Exception:
                    setup_backend("numba")
                # Any internal beam built on the GPU backend is stale.
                self._beam = None
        else:
            raise NotImplementedError(
                f"WrapBlond4Xsuite supports ContextCpu and ContextCupy, got "
                f"{type(context).__name__}."
            )

    def _build_beam(
        self, particles: XsuiteParticles, energy0: float, n: int
    ) -> None:
        """
        Construct a BLonD beam from xsuite particles.

        Validates that particles have uniform charge ratio and weight (required
        for BLonD's uniform-charge model), then builds a beam with the particle
        type and total intensity from ``particles``.

        Parameters
        ----------
        particles
            The xsuite ``Particles`` to extract properties from.
        energy0
            Reference total energy [eV].
        n
            Number of macroparticles to allocate in the beam.

        Raises
        ------
        NotImplementedError
            If particles have non-uniform charge_ratio or weight.
        """
        # BLonD kicks every macroparticle with the same charge; xsuite's
        # per-particle charge_ratio (and the chi = charge/mass ratio used for
        # partially stripped ions) has no BLonD counterpart. Refuse instead
        # of silently applying wrong kicks.
        charge_ratio = particles.charge_ratio
        if bool((charge_ratio != 1.0).any()):
            if hasattr(charge_ratio, "get"):  # device array, for the message
                charge_ratio = charge_ratio.get()
            raise NotImplementedError(
                "WrapBlond4Xsuite does not support per-particle charge "
                "ratios: BLonD applies a uniform charge to all "
                "macroparticles, so particles with charge_ratio != 1 would "
                "receive wrong kicks. Got charge_ratio values "
                f"{np.unique(np.asarray(charge_ratio))}."
            )
        particle_type = ParticleType(
            mass=_scalar(particles.mass0),
            charge=_scalar(particles.q0),
        )
        # xsuite carries the bunch population as a per-macroparticle ``weight``
        # (the number of real particles each macroparticle represents); the
        # total intensity is their sum. Summing over all slots — not just the
        # active ones — keeps BLonD's per-macroparticle
        # ``ratio = intensity / n_macroparticles`` equal to xsuite's weight,
        # which is what BLonD's intensity-effect elements consume. That
        # equality only holds for uniform weights — refuse spreads instead of
        # silently distorting wrapped intensity effects.
        weight = particles.weight
        if bool((weight != weight[0]).any()):
            raise NotImplementedError(
                "WrapBlond4Xsuite does not support per-particle weight "
                "spreads: BLonD models the bunch as a uniform "
                "intensity / n_macroparticles per slot, so non-uniform "
                "xsuite weights would distort any wrapped intensity effect."
            )
        intensity = float(weight.sum())
        self._beam = Beam(
            intensity=intensity,
            particle_type=particle_type,
        )
        self._beam.setup_beam(
            dt=np.zeros(n),
            dE=np.zeros(n),
            reference_total_energy=energy0,
        )
