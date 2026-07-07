# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Multibunch and multiparticle simulation code (MuSiC) algorithm.

Time-domain induced voltage of a single resonator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import elementary_charge as e

from blond.core.backends.backend import backend
from blond.core.base import BeamPhysicsRelevant
from blond.generals.distributed.helpers import mpi_is_distributed

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.physics.impedances.sources import Resonators


class Music(BeamPhysicsRelevant):
    r"""
    MuSiC time-domain induced voltage from a single resonator.

    Alternative to :class:`~blond.physics.impedances.base.WakeField` that
    computes the *exact* induced voltage of one resonant mode directly from
    the macro-particles in time domain, without slicing the beam into a
    profile. The cost is :math:`O(n \log n)` in the number of macro-particles
    (dominated by sorting; the recurrence itself is :math:`O(n)` via
    Migliorati & Palumbo).

    Unlike :class:`~blond.physics.impedances.base.WakeField`, this element has
    **no profile**: it sorts the beam by ``dt`` every turn and updates each
    particle's energy ``dE`` in place. Sorting permutes *all* per-particle
    arrays (``dt``, ``dE``, ``ids``, ``flags``) consistently, so particle
    identity is preserved.

    Parameters
    ----------
    source
        A :class:`~blond.physics.impedances.sources.Resonators` holding
        exactly **one** resonance.
    section_index
        Section index to group elements into sections.
    name
        Optional human-readable name for the element.

    Attributes
    ----------
    induced_voltage
        Induced voltage of the most recent turn [V] (one entry per
        macro-particle, in the sorted order).

    See Also
    --------
    blond.physics.impedances.base.WakeField : Profile-based induced voltage.

    Notes
    -----
    Only the ``python`` and ``cpp`` backends are supported (those that
    BLonD2 shipped); ``numba``, ``cuda`` and MPI raise
    :class:`NotImplementedError`. Like BLonD2, only singly-charged
    particles (``charge == 1``) are supported.

    References
    ----------
    M. Migliorati, L. Palumbo, "Multibunch and multiparticle simulation
    code with an alternative approach to wakefield effects", Phys. Rev. ST
    Accel. Beams 18, 031001 (2015).
    https://journals.aps.org/prab/abstract/10.1103/PhysRevSTAB.18.031001

    Examples
    --------
    >>> from blond import Music, Resonators
    >>>
    >>> music = Music(
    ...     source=Resonators(
    ...         shunt_impedances=1e6,
    ...         center_frequencies=1e9,
    ...         quality_factors=1.0,
    ...     )
    ... )
    >>> ring.add_elements([music])
    """

    # TODO 20260629.0 : Fix Notes when implementing CUDA/NUMBA backend

    def __init__(
        self,
        source: Resonators,
        section_index: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(section_index=section_index, name=name)
        from blond.physics.impedances.sources import (
            Resonators,  # prevent cyclic import
        )

        if not isinstance(source, Resonators):
            raise TypeError(
                f"`source` must be a `Resonators`, got {type(source).__name__}."
            )
        if source._n_resonators != 1:
            raise ValueError(
                "MuSiC supports exactly one resonance, but `source` has "
                f"{source._n_resonators}."
                f" Contact the developers if you need more."
            )
        self.source = source
        # Resonator parameters of the single resonance (scalars on host).
        self._R_S = float(source._shunt_impedances[0])
        self._omega_R = (  # angular resonant frequency [rad/s] = 2*pi*f
            2 * np.pi * float(source._center_frequencies[0])
        )
        self._Q = float(source._quality_factors[0])

        # Quantities derived once from the resonator; they parametrise the
        # O(n) recurrence (Migliorati & Palumbo). alpha is the damping rate
        # and omega_bar the damped angular frequency.
        self._alpha = self._omega_R / (2 * self._Q)
        self._omega_bar = float(np.sqrt(self._omega_R**2 - self._alpha**2))
        self._coeff1 = -self._alpha / self._omega_bar
        self._coeff2 = -self._R_S * self._omega_R / (self._Q * self._omega_bar)
        self._coeff3 = self._omega_R * self._Q / (self._R_S * self._omega_bar)
        self._coeff4 = self._alpha / self._omega_bar

        # MuSiC prefactor [V]; depends on the beam, so computed in
        # `configure_run` once the beam is known.
        self._const: float | None = None
        # Turn 1 starts the recurrence fresh; later turns bridge the wake.
        self._first_turn = True
        # Reference clock time [s] at the previous track; the difference to
        # the current reference time is the exact elapsed time between the
        # two passages (used to bridge the inter-turn gap).
        self._prev_reference_time: float | None = None
        # Running state carried across turns, layout
        # [input_first, input_second, delta_t, last_dt]:
        #   - input_first/second: 2-component oscillator state of the
        #     recurrence after the last processed particle,
        #   - delta_t: reference-time elapsed since the previous turn,
        #     refreshed each turn before bridging,
        #   - last_dt: dt of the last (largest-dt) particle of the previous
        #     turn, used to span the gap to this turn's first particle.
        self._parameter_array: NumpyArray | None = None
        # Induced voltage [V] of the most recent turn (one entry/particle).
        self.induced_voltage: NumpyArray | None = None

    def _check_supported(self) -> None:
        """
        Raise if the active backend / parallelization is unsupported.

        Raises
        ------
        NotImplementedError
            If MPI is in use or the ``numba`` or ``cuda`` backend is selected.
            The per-turn sort (:meth:`~blond.core.beam.base.BeamBaseClass.sort_by_dt`)
            also rejects distributed beams as a low-level safeguard.
        """
        if mpi_is_distributed():
            raise NotImplementedError(
                "MuSiC does not support MPI: the per-turn sort cannot order "
                "a beam split across ranks."
            )
        if backend.specials_mode == "numba":
            raise NotImplementedError(
                "MuSiC does not support the `numba` backend; only `python` "
                "and `cpp` are available (as in BLonD2)."
            )
        if backend.specials_mode == "cuda":
            raise NotImplementedError(
                "MuSiC does not support the `cuda` backend."
            )

    def configure_run(
        self, beam: BeamBaseClass, n_turns: int, **kwargs
    ) -> None:
        """
        Reset per-run state and compute the MuSiC prefactor.

        Parameters
        ----------
        beam
            The beam being simulated.
        n_turns
            Number of turns for this run.
        **kwargs
            Simulation-extracted values passed down the MRO chain.
        """
        super().configure_run(beam=beam, n_turns=n_turns, **kwargs)
        self._check_supported()
        assert self.each_turn_i == 1, (
            "MuSiC requires `each_turn_i == 1`: the multi-turn bridging "
            "assumes the element runs on consecutive turns, but got "
            f"{self.each_turn_i}."
        )
        n_macroparticles = beam.n_macroparticles_partial()
        charge = beam.particle_type.charge
        assert charge == 1, (
            "MuSiC currently only supports singly-charged particles "
            f"(charge == 1), but got charge={charge}."
        )
        self._const = (
            -e
            # TODO investigate: the energy kick arguably scales with
            # charge**2 (one factor from the beam current, one from the
            # kick onto a test particle), mirroring `WakeField`. Disabled
            # until verified; `charge == 1` is asserted above so it would
            # have no effect anyway.
            # * charge**2
            * self._R_S
            * self._omega_R
            * beam.intensity
            / (n_macroparticles * self._Q)
        )
        self._first_turn = True
        self._prev_reference_time = None
        self._parameter_array = backend.array(
            [1.0, 0.0, 0.0, 0.0], dtype=backend.float
        )
        self.induced_voltage = None

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Sort the beam, compute the induced voltage and kick ``dE``.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        # Re-checked here (not only in `configure_run`) because the active
        # backend can be switched between run setup and tracking.
        self._check_supported()

        # Sort by dt (permutes dt/dE/ids/flags consistently) so the
        # recurrence below sees the particles in time order.
        beam.sort_by_dt()
        dt = beam.write_partial_dt()
        dE = beam.write_partial_dE()

        n = len(dt)
        if self.induced_voltage is None or len(self.induced_voltage) != n:
            self.induced_voltage = backend.zeros(n, dtype=backend.float)
        else:
            self.induced_voltage[:] = 0.0

        # On turn 1 there is no previous-turn wake to bridge; afterwards the
        # bridge needs the exact time elapsed since the previous track, read
        # from the reference clock (accurate under ramps, unlike a nominal
        # t_rev). MuSiC runs at a fixed ring position, so consecutive
        # samples differ by exactly one revolution.
        reference_time = float(beam.reference.time)
        multiturn = not self._first_turn
        if multiturn:
            self._parameter_array[2] = (
                reference_time - self._prev_reference_time
            )
        backend.specials.music_track(
            dt,
            dE,
            self.induced_voltage,
            self._parameter_array,
            self._alpha,
            self._omega_bar,
            self._const,
            self._coeff1,
            self._coeff2,
            self._coeff3,
            self._coeff4,
            multiturn,
        )
        self._prev_reference_time = reference_time
        self._first_turn = False

    @staticmethod
    def headless(
        beam: BeamBaseClass,
        source: Resonators,
        section_index: int = 0,
    ) -> Music:
        """
        Build a tracking-ready `Music` without a full `Simulation`.

        Runs the configure hooks directly. For multi-turn tracking the
        caller must advance ``beam.reference.time`` between ``track`` calls
        (a real simulation does this via drift/RF elements); the elapsed
        reference time is what bridges the inter-turn gap.

        Parameters
        ----------
        beam
            The `Beam` object whose state will be updated by this element.
        source
            A `Resonators` source holding exactly one resonance.
        section_index
            Section index to group elements into sections.

        Returns
        -------
        music
            Instance ready to be tracked.
        """
        music = Music(source=source, section_index=section_index)
        music.configure()
        music.configure_run(beam=beam, n_turns=None)
        return music
