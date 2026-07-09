# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Synchrotron radiation base classes.

Author:
L. Valle
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray as NumpyArray

from blond.acc_math.analytic.synchrotron_radiation.utilities import (
    gather_longitudinal_synchrotron_radiation_parameters,
)
from blond.core.backends.backend import backend
from blond.core.base import BeamPhysicsRelevant, DynamicParameter, Schedulable

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


class SynchrotronRadiationBaseClass(BeamPhysicsRelevant, Schedulable):
    """
    Base class for radiating ring elements.

    Parameters
    ----------
    name
        Human-readable name for the element. If not provided, a unique name is
        automatically generated.
    section_index
        Section index to group elements into sections.
    share_of_radiation_integrals
        Share of synchrotron radiation integrals.
    disable_quantum_excitation
       Disables the quantum excitation kick.
    seed
        Currently unsupported and must be left as ``None``. The
        quantum-excitation noise is generated inside
        ``backend.specials.apply_synchrotron_radiation_and_quantum_excitation_energy_kick``
        and each backend uses its own RNG (NumPy global state on the Python
        backend, Numba's per-thread parallel PRNG, ``std::mt19937_64`` on the
        C++ backend, cuRAND on the CUDA backend), so a single user-supplied
        seed cannot be threaded through uniformly. Passing a value here will
        raise ``NotImplementedError`` rather than silently being ignored.
    """

    def __init__(
        self,
        name: str | None = None,
        section_index: int | None = None,
        share_of_radiation_integrals: NumpyArray | None = None,
        disable_quantum_excitation: bool = False,
        seed: int | None = None,
    ):
        if seed is not None:  # pragma: no cover
            raise NotImplementedError(
                "`seed` is not supported: the quantum-excitation noise is "
                "drawn inside the active backend's `specials` implementation "
                "and the four backends (Python/Numba/C++/CUDA) each use a "
                "different RNG (NumPy global state, Numba's per-thread "
                "parallel PRNG, `std::mt19937_64`, cuRAND). A single seed "
                "cannot be plumbed through uniformly today, so we refuse it "
                "instead of silently ignoring it. Pass `seed=None` and (for "
                "the Python backend only) call `np.random.seed(...)` before "
                "tracking if you need reproducibility."
            )

        super().__init__(name=name, section_index=section_index)

        self._add_intended_schedule(
            "share_of_radiation_integrals",
        )

        self._simulation: Simulation | None = None
        self.share_of_radiation_integrals = share_of_radiation_integrals

        self._disable_quantum_excitation = disable_quantum_excitation

        self._energy_lost_due_to_synchrotron_radiation: float | None = None
        self._damping_time: float | None = None
        self._natural_energy_spread: float | None = None

    def _apply_kick(
        self,
        beam: BeamBaseClass,
    ) -> None:
        """
        Apply synchrotron radiation and quantum excitation energy kicks.

        Mutates ``beam._dE`` in place via
        ``backend.specials.apply_synchrotron_radiation_and_quantum_excitation_energy_kick``.

        Parameters
        ----------
        beam
             BeamBaseClass object.
        """
        total_energy = beam.reference.total_energy
        (
            estimated_energy_lost,
            estimated_damping_time,
            estimated_natural_energy_spread,
        ) = gather_longitudinal_synchrotron_radiation_parameters(
            particle_type=beam.particle_type,
            energy=total_energy,
            radiation_integrals=self.share_of_radiation_integrals,
        )
        self._energy_lost_due_to_synchrotron_radiation = estimated_energy_lost
        self._damping_time = estimated_damping_time
        self._natural_energy_spread = estimated_natural_energy_spread

        beam_dE = beam.write_partial_dE()
        backend.specials.apply_synchrotron_radiation_and_quantum_excitation_energy_kick(
            energy_lost=estimated_energy_lost,
            beam_dE=beam_dE,
            natural_energy_spread=estimated_natural_energy_spread,
            longitudinal_damping_time=estimated_damping_time,
            total_energy=total_energy,
            disable_quantum_excitation=self._disable_quantum_excitation,
        )

    def on_init_simulation(self, simulation: Simulation, **kwargs) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        **kwargs
            Configure parameters collected by the MRO chain.
        """
        super().on_init_simulation(
            simulation,
            turn_counter=simulation.turn_counter,
            **kwargs,
        )

    def configure(
        self,
        *,
        turn_counter: DynamicParameter | None = None,
        **kwargs,
    ) -> None:
        """
        Store the runtime references needed during tracking.

        Parameters
        ----------
        turn_counter
            Live turn counter; accessed as ``turn_counter.value`` each track call.
        **kwargs
            Passed to the next level in the MRO chain.
        """
        super().configure(**kwargs)
        self._turn_counter = turn_counter

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        if self.schedule_active:
            assert self._turn_counter is not None, (
                "Turn counter must be set with active scheduling."
            )
            self.apply_schedules(
                turn_i=self._turn_counter.value,
                reference_time=float(beam.reference.time),
            )
        self._apply_kick(beam)
