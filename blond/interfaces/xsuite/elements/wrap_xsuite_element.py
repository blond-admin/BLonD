# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Wrap an xsuite ``Line`` as a single black-box BLonD ring element.

The wrapper accepts an :class:`xtrack.Line` (wrap a single element in a
one-element line if that is all you have). It transports particles
between BLonD's ``(dt, dE)`` and xsuite's ``(zeta, ptau)`` coordinates,
runs the line's ``track``, and tells the BLonD ring how long it is and
how much reference time it consumes — *nothing else*. It deliberately
does not introspect what the line contains. From BLonD's analytics'
point of view (``phi_s``, slip-factor averaging, voltage sums) the xsuite
block is invisible.

This is the black-box treatment we want: xsuite owns its physics, BLonD
owns its loop, the wrapper is purely for reference transport. If a user needs BLonD
to compute things that depend on the lattice (``phi_s``, eta_0
averaging), they add parallel BLonD elements with matching parameters
alongside the wrapper.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from blond.acc_math.analytic.simple_math import calc_total_energy
from blond.core.backends.backend import backend
from blond.core.base import (
    AltersReference,
    BeamPhysicsRelevant,
    HasOrbitLength,
)
from blond.core.beam.flags import BeamFlags
from blond.interfaces.xsuite.elements.helpers import (
    ReferenceFrame,
    dE_to_ptau,
    dt_to_zeta,
    ptau_to_dE,
    zeta_to_dt,
)

if TYPE_CHECKING:  # pragma: no cover
    from xtrack import Line
    from xtrack import Particles as XsuiteParticles

    from blond.core.beam.base import BeamBaseClass
    from blond.core.reference_clock.reference_clock import ReferenceCoordinates


class WrapXsuite4Blond(BeamPhysicsRelevant, AltersReference, HasOrbitLength):
    """
    Run an xsuite ``Line`` inside a BLonD ring, as a black-box element.

    Wrap any :class:`xtrack.Line` so it behaves like a BLonD ring element, then
    add it to your ring like any other element::

        ring.add_elements([WrapXsuite4Blond(my_line), rf, ...])

    If you only have a single xsuite element (a drift, a cavity, a magnet),
    wrap it in a one-element line first::

        WrapXsuite4Blond(xt.Line(elements=[element], element_names=["e"]))

    The wrapper is a **black box**: it transports particles between BLonD's
    ``(dt, dE)`` and xsuite's ``(zeta, ptau)``, runs the line's ``track``, and
    reports its length (``line.get_length()``) and reference-time consumption to
    the ring — nothing else. It does not introspect what the line contains, so
    the xsuite block is invisible to BLonD's longitudinal analytics (``phi_s``,
    slip-factor averaging, voltage sums). If you need those to include the
    xsuite block, add parallel BLonD elements (a ``SingleHarmonicRFStation``, a
    ``DriftSimple``) carrying the matching parameters alongside the wrapper.

    Reference energy: BLonD's ring owns the reference. If the wrapped line
    advances it (it contains an ``xt.ReferenceEnergyIncrease``), the wrapper
    picks the new energy up from the tracked particles and writes it back
    into ``beam.reference``. Reference scans (``track_reference`` — the
    per-turn forecast, ``get_t_rev_init``, ``n_turns_max``) measure the
    same advance by tracking a single probe particle through a throwaway
    copy of the line, so they follow a guest-driven ramp without ever
    touching the (possibly stateful) live line; passive guests are probed
    once and skipped thereafter. A line carrying its own
    ``energy_program`` is rejected — drive the ramp from BLonD's
    ``MagneticCycle`` instead, or use ``WrapBlond4Xsuite`` and let xsuite
    own the loop.

    GPU: BLonD's backend decides where the guest runs. With a GPU backend
    (``setup_backend("cuda")``) the wrapper builds the guest tracker and its
    per-pass ``Particles`` on a ``ContextCupy``, so all per-turn data stays
    on the device; a guest tracker prebuilt on a mismatched context is
    refused.

    By type the wrapper is *only* a length-bearing, reference-altering ring
    element (``BeamPhysicsRelevant`` + ``AltersReference`` +
    ``HasOrbitLength``). It is deliberately **not** a ``DriftBaseClass`` or
    ``RFStationBaseClass``: drift analytics (``eta_0`` averaging, momentum
    compaction, synchrotron radiation) and RF accounting (``n_rf_stations``,
    separatrix and feedback iteration, the ``MagneticCyclePerTurn`` energy
    split) never see it.

    Parameters
    ----------
    line
        The :class:`xtrack.Line` to wrap. Wrap a single element in a
        one-element line (``xt.Line(elements=[el], element_names=[...])``)
        if that is all you have.

    See Also
    --------
    blond.interfaces.xsuite.elements.wrap_blond_element.WrapBlond4Xsuite :
        The mirrored direction — a BLonD element tracked inside an xsuite
        ``Line`` (xsuite owns the loop and the reference energy).

    Examples
    --------
    PS triple splitting with BLonD owning the loop and the wrapped xsuite
    segment map providing the drift
    (``blond/examples/scripts/EX_30_Xsuite_in_BLonD.py``):

    .. literalinclude:: /../blond/examples/scripts/EX_30_Xsuite_in_BLonD.py
       :language: python
       :start-at: import os
    """

    skip_find_instances_attributes = ["_line"]

    def __init__(
        self,
        line: Line,
    ) -> None:
        # Import not at top to prevent import crash on computers without Xsuite.
        from xtrack import Line

        # The wrapper's contract is a full ``xt.Line``: one ``track`` entry
        # point, one ``get_length``, and the place xsuite expects collective
        # effects / energy programs to live. A bare ``BeamElement`` (Drift,
        # Cavity, LineSegmentMap, …) is rejected here — wrap it in a
        # one-element line first.
        if not isinstance(line, Line):
            raise TypeError(
                f"WrapXsuite4Blond wraps an xt.Line, got "
                f"{type(line).__name__}. Wrap a single element in a "
                f"one-element line first: "
                f"xt.Line(elements=[element], element_names=['e'])."
            )

        self._line = line
        # The most recent pass's ``Particles``, kept only for inspection and
        # debugging. It is never reused: every pass builds a fresh one (see
        # ``_track``), so no xsuite bookkeeping survives from pass to pass.
        self._particles: XsuiteParticles | None = None

        # Refuse a wrapped ``xt.Line`` that carries its own ``EnergyProgram``:
        # BLonD owns the reference here and the wrapper's per-turn refresh
        # would overwrite the program's advance.
        if getattr(line, "energy_program", None) is not None:
            raise ValueError(
                "WrapXsuite4Blond was given an xt.Line with an attached "
                "energy_program. BLonD owns the reference inside its Ring, "
                "and the wrapper stamps the reference onto the particles "
                "each pass, which would overwrite any program-driven "
                "advance. Either drive the ramp from BLonD's "
                "MagneticCycle (and remove the line's energy_program), or "
                "let xsuite own the loop and use WrapBlond4Xsuite instead."
            )

        super().__init__(
            section_index=0,
            orbit_length=line.get_length(),
        )

        # Does the guest advance the reference energy (p0c)? ``None`` =
        # unknown; the first reference probe classifies. ``False`` lets
        # ``track_reference`` skip the probe (a line copy + one-particle
        # track) for the common passive guest — the flag is only ever
        # promoted to ``True``, never demoted, so a ramp with flat
        # segments keeps being probed.
        self._guest_advances_reference: bool | None = None

    def track_reference(
        self,
        reference: ReferenceCoordinates,
        is_counter_rotating: bool = False,
        **kwargs,
    ) -> float:
        """
        Advance the reference clock by one pass through the wrapped line.

        Advances ``reference.time`` by the time of flight
        ``orbit_length / (beta c)`` and ``reference.total_energy`` by the
        advance one pass through the guest line would produce, measured by
        tracking a single probe particle through a **throwaway copy** of
        the line — the (possibly stateful) guest itself is never tracked
        here, so ``track_reference`` shares no state with ``track``. The
        forecast is a pure function of the incoming reference state:
        repeated calls simulate successive passes, and scans from any
        state (the per-turn forecast,
        :meth:`~blond.core.simulation.simulation.Simulation.get_t_rev_init`,
        ``n_turns_max``) get the advance a pass from *that* state would
        produce.

        Cost: the probe (line copy + tracker build + one-particle track)
        takes ~10 ms for small lines and grows with line size. A guest
        whose first probe shows no ``p0c`` advance is classified passive
        and never probed again; only guest-driven ramps (a line containing
        an ``xt.ReferenceEnergyIncrease``) pay the probe per call.

        Counter-rotating beams are not supported and raise
        :class:`NotImplementedError`.

        Parameters
        ----------
        reference
            Reference coordinates whose ``time`` and ``total_energy`` are
            advanced in place.
        is_counter_rotating
            Must be ``False``; counter-rotating beams are not supported.
        **kwargs
            Ignored; accepted for interface compatibility.

        Returns
        -------
        float
            The reference total-energy delta [eV] applied this call.
        """
        # Counter-rotating beams would need the guest tracked in the opposite
        # orbit direction, which the wrapper does not do — raise rather than
        # silently return a wrong reference advance.
        if is_counter_rotating:
            raise NotImplementedError(
                "WrapXsuite4Blond does not support counter-rotating beams; "
                "the wrapped xsuite element is tracked in a single orbit "
                "direction only."
            )
        reference_time_change = self.orbit_length / reference.velocity
        reference.time += reference_time_change

        if self._guest_advances_reference is False:
            return 0.0
        delta_e = self._probe_reference_energy_advance(reference)
        if self._guest_advances_reference is None:
            self._guest_advances_reference = delta_e != 0.0
        reference.total_energy += delta_e
        return delta_e

    def _probe_reference_energy_advance(
        self, reference: ReferenceCoordinates
    ) -> float:
        """
        Measure the guest's reference-energy advance on a throwaway copy.

        Tracks a single on-reference probe particle (``zeta = ptau = 0``)
        through a copy of the guest line and reads the resulting ``p0c``.
        The live line, its tracker and its state are untouched.

        Parameters
        ----------
        reference
            The reference state to probe from (energy and particle type).

        Returns
        -------
        float
            Total-energy advance one pass would produce, in [eV]
            (``0.0`` when the probe is lost or ``p0c`` did not move).
        """
        # Import not at top to prevent import crash on computers without Xsuite.
        from xtrack import Particles as XsuiteParticles

        mass = float(reference.particle_type.mass)
        energy0 = float(reference.total_energy)
        p0c = float(math.sqrt(energy0**2 - mass**2))

        probe_line = self._copy_line_for_probe()
        probe = XsuiteParticles(
            p0c=p0c,
            mass0=mass,
            q0=float(reference.particle_type.charge),
            zeta=np.zeros(1),
            ptau=np.zeros(1),
        )
        probe_line.build_tracker()
        probe_line.track(probe)

        if int(probe.state[0]) <= 0:
            # The probe got lost inside the guest — no reliable readout.
            return 0.0
        new_p0c = float(probe.p0c[0])
        if new_p0c == p0c:
            return 0.0
        return float(calc_total_energy(mass, new_p0c)) - energy0

    def _copy_line_for_probe(self) -> Line:
        """
        Return a CPU-side throwaway copy of the guest line.

        The copy lives on a ``ContextCpu`` regardless of where the live
        line runs: the reference advance is context-independent, and a
        one-particle CPU track avoids touching the (possibly GPU) live
        tracker. ``Line.copy`` reconstructs via ``self.__class__`` and can
        fail for user subclasses of ``xt.Line`` with custom constructors —
        fall back to ``copy.deepcopy`` there.

        Returns
        -------
        Line
            A tracker-less deep copy of the wrapped line.
        """
        # Import not at top to prevent import crash on computers without Xsuite.
        from xobjects import ContextCpu

        try:
            return self._line.copy(_context=ContextCpu())
        except Exception:
            import copy as copy_module

            probe_line = copy_module.deepcopy(self._line)
            probe_line.discard_tracker()
            return probe_line

    def _resolve_context(self):
        """
        Return the xobjects context matching BLonD's backend.

        BLonD is the master here, so the BLonD backend decides where the
        guest runs: ``backend.is_gpu`` requires a ``ContextCupy``, otherwise
        a ``ContextCpu``. If the guest line has no tracker yet, one is built
        on the matching context; a tracker prebuilt on a mismatched context
        is refused rather than silently copied between devices.

        Returns
        -------
        context
            The xobjects context shared by the guest tracker and the
            wrapper's per-pass ``Particles``.
        """
        # Import not at top to prevent import crash on computers without Xsuite.
        if backend.is_gpu:
            from xobjects import ContextCupy as context_cls
        else:
            from xobjects import ContextCpu as context_cls

        tracker = self._line.tracker
        if tracker is not None:
            if not isinstance(tracker._context, context_cls):
                raise ValueError(
                    f"The wrapped xt.Line's tracker runs on a "
                    f"{type(tracker._context).__name__}, but BLonD's backend "
                    f"requires a {context_cls.__name__} "
                    f"(backend.is_gpu={backend.is_gpu}). BLonD owns the loop "
                    f"here: either rebuild the line's tracker on the "
                    f"matching context, or switch the BLonD backend with "
                    f"blond.core.backends.helpers.setup_backend(...)."
                )
            return tracker._context

        context = context_cls()
        self._line.build_tracker(_context=context)
        return context

    def _build_particles(
        self, beam: BeamBaseClass, p0c: float, n: int, context
    ) -> XsuiteParticles:
        """
        Build a fresh xsuite ``Particles`` for one pass through the guest.

        Parameters
        ----------
        beam
            The BLonD beam supplying the particle type.
        p0c
            Reference momentum [eV/c] to stamp on every slot.
        n
            Number of slots to allocate.
        context
            The xobjects context the guest tracker runs on.

        Returns
        -------
        XsuiteParticles
            A zero-initialised ``Particles`` on ``context``.
        """
        # Import not at top to prevent import crash on computers without Xsuite.
        from xtrack import Particles as XsuiteParticles

        # Input fields must be host arrays — ``Particles.__init__`` runs them
        # through ``np.array`` before moving them onto ``_context``.
        # ``weight`` is left at 1: the guest line sees one real particle per
        # macroparticle, so intensity effects inside it (space charge,
        # wakefields) do NOT see BLonD's bunch population. Model intensity
        # effects on the BLonD side of the ring.
        return XsuiteParticles(
            _context=context,
            p0c=p0c,
            mass0=float(beam.particle_type.mass),
            q0=float(beam.particle_type.charge),
            zeta=np.zeros(n),
            ptau=np.zeros(n),
            weight=np.ones(n),
        )

    def _track(self, beam: BeamBaseClass) -> None:
        # Round-trip one beam through the line: build a ReferenceFrame from
        # beam.reference, build the pass's xsuite Particles around it,
        # propagate BLonD's LOST flags via the particle_id mapping (xsuite keeps
        # active slots contiguous, BLonD keeps slots stable), convert (dt, dE) ->
        # (zeta, ptau), track, pick up any reference advance the line performed,
        # then write surviving particles back and flag lost xsuite slots LOST.
        beta0 = float(beam.reference.beta)
        energy0 = float(beam.reference.total_energy)
        frame = ReferenceFrame(beta0=beta0, energy0=energy0)

        # The wrapper owns the live reference clock for its share of the
        # orbit, exactly like ``DriftBaseClass._track``: time-based magnetic
        # cycles and schedules read ``beam.reference.time`` each turn.
        beam.reference.time += self.orbit_length / beam.reference.velocity

        n = len(beam.dt.array_local)

        # The guest line, the fresh Particles and BLonD's beam arrays all
        # live on the context dictated by BLonD's backend (CPU numpy or GPU
        # cupy), so every array operation below is same-library and no
        # host/device copies happen in the loop.
        context = self._resolve_context()
        # Build a fresh ``Particles`` for every pass instead of reviving a
        # cached one. Writing ``state`` on a reused ``Particles`` leaves
        # xsuite's own ``_num_active_particles`` / ``_num_lost_particles``
        # counters stale, so a slot lost on one pass stays outside the
        # tracked range on every later pass — silently never tracked again,
        # even once BLonD revives it. ``at_turn`` and the slot permutation
        # left by xsuite's compaction leak the same way. A fresh object is
        # the only state-free starting point.
        mass = float(beam.particle_type.mass)
        p0c = float(math.sqrt(energy0**2 - mass**2))
        particles = self._build_particles(beam, p0c, n, context)
        self._particles = particles

        pid_in = particles.particle_id
        blond_dt = beam.dt.array_local
        blond_dE = beam.dE.array_local
        blond_active = beam.flags.array_local == BeamFlags.ACTIVE.value

        # Mark xsuite slots as lost when their mapped BLonD slot is lost.
        particles.state[~blond_active[pid_in]] = BeamFlags.LOST.value

        particles.zeta[:] = dt_to_zeta(blond_dt[pid_in], frame)
        particles.ptau[:] = dE_to_ptau(blond_dE[pid_in], frame)

        self._line.track(particles)

        pid_out = particles.particle_id
        state = particles.state

        # Pick up any reference advance the guest performed and apply it to
        # the live reference. The advance is detected on ``p0c`` (a stored
        # field — one scalar device-to-host transfer) instead of the
        # ``energy0`` property, which would compute a whole derived array
        # just to read one entry.
        i = int(state.argmax())  # first active slot; any slot if none active
        new_p0c = float(particles.p0c[i])
        if new_p0c != p0c and int(state[i]) > 0:
            new_energy0 = float(calc_total_energy(mass, new_p0c))
            beam.reference.total_energy = new_energy0
            frame = ReferenceFrame(
                beta0=float(particles.beta0[i]),
                energy0=new_energy0,
            )
            # Performance hint only (no physics flows back into the
            # forecast): a live pass proved the guest ramps, so make sure
            # ``track_reference`` keeps probing even if an earlier probe
            # classified the line as passive (e.g. a time-dependent ramp
            # that was still flat when first probed).
            self._guest_advances_reference = True

        active_xs = state > 0
        active_slots = pid_out[active_xs]
        beam.dt.array_local[active_slots] = zeta_to_dt(
            particles.zeta[active_xs], frame
        )
        beam.dE.array_local[active_slots] = ptau_to_dE(
            particles.ptau[active_xs], frame
        )
        # Flag the slots whose particle the guest killed; an empty index
        # write is a no-op, so no host-synchronizing ``.all()`` guard is
        # needed.
        beam.flags.array_local[pid_out[~active_xs]] = BeamFlags.LOST.value
