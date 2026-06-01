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

from abc import ABC
from typing import TYPE_CHECKING

from numpy.typing import NDArray as NumpyArray

from blond.acc_math.analytic.synchrotron_radiation.utilities import (
    gather_longitudinal_synchrotron_radiation_parameters,
)
from blond.core.backends.backend import backend
from blond.core.base import BeamPhysicsRelevant, DynamicParameter

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


class SynchrotronRadiationBaseClass(BeamPhysicsRelevant, ABC):
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
        Expert user only. Disables the quantum excitation kick.
    seed
        Currently unsupported and must be left as ``None``. The
        quantum-excitation noise is generated inside
        ``backend.specials.apply_synchrotron_radiation_and_quantum_excitation_energy_kick``
        and each backend uses its own RNG (NumPy global state on the Python
        backend, Numba's per-thread parallel PRNG, xoshiro256+ seeded from
        wall-clock on the C++/CUDA backends), so a single user-supplied seed
        cannot be threaded through uniformly. Passing a value here will raise
        ``NotImplementedError`` rather than silently being ignored.
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
                "parallel PRNG, xoshiro256+ from wall-clock). A single seed "
                "cannot be plumbed through uniformly today, so we refuse it "
                "instead of silently ignoring it. Pass `seed=None` and (for "
                "the Python backend only) call `np.random.seed(...)` before "
                "tracking if you need reproducibility."
            )

        super().__init__(name=name, section_index=section_index)

        self._simulation: Simulation | None = None
        self._turn_i: DynamicParameter | int = 0
        self._share_of_radiation_integrals = share_of_radiation_integrals

        self._disable_quantum_excitation = disable_quantum_excitation

        self._energy_lost_due_to_synchrotron_radiation: float | None = None
        self._damping_time: float | None = None
        self._natural_energy_spread: float | None = None

    @property
    def share_of_radiation_integrals(self) -> NumpyArray | None:
        """
        Synchrotron radiation integrals of the drift.

        Returns
        -------
        synchrotron_radiation_integrals_drift
            Synchrotron radiation integrals of the drift.
        """
        return self._share_of_radiation_integrals

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
            radiation_integrals=self._share_of_radiation_integrals,
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

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        """
        super().on_init_simulation(simulation=simulation)
        self._simulation = simulation
        self._turn_i = simulation.turn_i

    def on_run_simulation(
        self,
        simulation: Simulation,
        **kwargs,
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        **kwargs
            Additional keyword arguments for simulation setup.
        """
        pass

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        self._apply_kick(beam)
