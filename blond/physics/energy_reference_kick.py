"""Gives a kick to the beam to update its reference energy."""

from typing import TYPE_CHECKING

from .._core.backends.backend import backend
from .._core.base import BeamPhysicsRelevant, DynamicParameter, Schedulable
from .._core.beam.base import BeamBaseClass
from .._core.simulation.simulation import Simulation
from ..cycles.magnetic_cycle import MagneticCycleByTime

if TYPE_CHECKING:  # pragma: no cover
    from .. import Ring
    from ..cycles.magnetic_cycle import MagneticCycleBase


class ReferenceEnergyChange(BeamPhysicsRelevant, Schedulable):
    """Updates beam's reference energy, for example asynchronous ramping.

    Can be used in simulations where RF ramping is asynchronous with respect to the
    beam’s energy. The resulting offset affects the beam's `dE` (energy deviation) and simulates the physics of an energy
    mismatch relative to the reference trajectory.

    Parameters
    ----------
        section_index (int): Index of the ring section where this element is placed.
        name (str, optional): An optional name for the element.
        **kwargs (dict): Additional keyword arguments for compatibility with fused or composite elements.

    Attributes
    ----------
        _turn_i (DynamicParameter | None): Current simulation turn number (initialized during simulation).
        _magnetic_cycle (MagneticCycleBase | None): Reference to the simulation’s magnetic cycle.
        _ring (Ring | None): Reference to the ring being simulated.

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
        if not isinstance(self._magnetic_cycle, MagneticCycleByTime):
            raise TypeError(
                f"Expected MagneticCycleByTime, got {type(self._magnetic_cycle).__name__}"
            )

        super().on_init_simulation(simulation=simulation)
        self._turn_i = simulation.turn_i
        self._magnetic_cycle = simulation.magnetic_cycle
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
        """Updates reference energy of the beam."""
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
