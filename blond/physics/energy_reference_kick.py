# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Gives a kick to the beam to update its reference energy."""

from typing import TYPE_CHECKING

from blond.core.backends.backend import backend
from blond.core.base import (
    BeamPhysicsRelevant,
    DynamicParameter,
    SchedulableMixIn,
)
from blond.core.beam.base import BeamBaseClass
from blond.core.simulation.simulation import Simulation
from blond.cycles.magnetic_cycle import MagneticCycleBase, MagneticCycleByTime

if TYPE_CHECKING:  # pragma: no cover
    from blond import Ring


class ReferenceEnergyChange(BeamPhysicsRelevant, SchedulableMixIn):
    """Updates beam's `reference_total_energy` and `dE` array, but constant in absolute terms.

    Can be used in simulations where RF ramping is asynchronous with respect to the
    beam’s energy.

    Parameters
    ----------
    section_index:
        Index of the ring section where this element is placed.
    name:
        An optional name for the element.
    **kwargs:
        Additional keyword arguments for compatibility.

    Attributes
    ----------
    _turn_i:
        Current simulation turn number (initialized during simulation).
    _magnetic_cycle:
        Reference to the simulation’s magnetic cycle.
    _ring:
         Reference to the ring being simulated.

    Example:
        >>> elem = ReferenceEnergyChange(section_index=1, name="energy_reference_kick")
        >>> # Add to element map before simulation
    """

    def __init__(
        self,
        section_index: int,
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
        """Lateinit method when `simulation.__init__` is called.

        simulation
            Simulation context manager
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
        turn_i_init: int,
        **kwargs,
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called.

        simulation
            Simulation context manager
        beam
            Simulation beam object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        pass

    def track(self, beam: BeamBaseClass):
        """Updates reference energy of the beam.

        beam
            Simulation beam object
        """
        super().track(beam=beam)
        if self.schedule_active:
            self.apply_schedules(
                turn_i=self._turn_i.value,
                reference_time=beam.reference_time,
            )

        target_total_energy = self._magnetic_cycle.get_target_total_energy(
            turn_i=self._turn_i.value,
            section_i=self.section_index,
            reference_time=beam.reference_time,
            particle_type=beam.particle_type,
        )

        reference_energy_change = backend.float(
            target_total_energy - beam.reference_total_energy
        )

        beam._dE -= reference_energy_change
        beam.reference_total_energy += reference_energy_change
