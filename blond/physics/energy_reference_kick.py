# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Gives a kick to the beam to update its reference energy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from blond.core.backends.backend import backend
from blond.core.base import (
    AltersReference,
    BeamPhysicsRelevant,
    DynamicParameter,
)
from blond.core.beam.base import BeamBaseClass
from blond.core.simulation.simulation import Simulation
from blond.cycles.magnetic_cycle import MagneticCycleBase, MagneticCycleByTime

if TYPE_CHECKING:  # pragma: no cover
    from blond import Ring
    from blond.core.reference_clock.reference_clock import ReferenceCoordinates


class ReferenceEnergyChange(BeamPhysicsRelevant, AltersReference):
    """
    Update beam's `reference.total_energy` and `dE` array, but constant in absolute terms.

    Can be used in simulations where RF ramping is asynchronous with respect to the
    beam's energy.

    Parameters
    ----------
    section_index
        Index of the ring section where this element is placed.
    name
        An optional name for the element.
    **kwargs
        Additional keyword arguments for compatibility.

    Attributes
    ----------
    _turn_i
        Current simulation turn number (initialized during simulation).
    _magnetic_cycle
        Reference to the simulation's magnetic cycle.
    _ring
        Reference to the ring being simulated.

    Examples
    --------
    >>> elem = ReferenceEnergyChange(section_index=1, name="energy_reference_kick")
    >>> # Add to element map before simulation
    """

    def __init__(
        self,
        section_index: int = 0,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
            **kwargs,
        )

        self._turn_i: DynamicParameter | None = None
        self._magnetic_cycle: MagneticCycleBase | None = None
        self._ring: Ring | None = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        """
        super().on_init_simulation(simulation=simulation)
        self._turn_i = simulation.turn_i
        self._magnetic_cycle = simulation.magnetic_cycle
        if not isinstance(self._magnetic_cycle, MagneticCycleByTime):
            raise TypeError(
                f"Expected MagneticCycleByTime, got {type(self._magnetic_cycle).__name__}"
            )
        self._ring = simulation.ring

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs,
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation beam object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        pass

    def track_reference(
        self, reference: ReferenceCoordinates, **kwargs
    ) -> float:
        """
        Update the coordinates of the reference coordinate system.

        Parameters
        ----------
        reference
            The object that holds the reference time [s] and total energy [eV].
        **kwargs
            Allows more arguments in the method definition outside the
            abstract class.

        Returns
        -------
        change
            Change of reference time or energy.
        """
        target_total_energy = self._magnetic_cycle.get_target_total_energy(
            turn_i=self._turn_i.value,
            section_i=self.section_index,
            reference_time=reference.time,
            particle_type=reference.particle_type,
        )

        reference_energy_change = backend.float(
            target_total_energy - reference.total_energy
        )
        reference.total_energy = target_total_energy
        return reference_energy_change

    def _track(self, beam: BeamBaseClass):
        """
        Update reference energy of the beam.

        Parameters
        ----------
        beam
            Simulation beam object.
        """
        super()._track(beam=beam)

        reference_energy_change = self.track_reference(
            reference=beam.reference,
        )
        dE = beam.write_partial_dE()
        dE -= reference_energy_change
