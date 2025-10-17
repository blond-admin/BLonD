from .._core.base import Schedulable, BeamPhysicsRelevant, DynamicParameter
from .._core.backends.backend import backend
from .._core.beam.base import BeamBaseClass
from .._core.simulation.simulation import Simulation

from typing import TYPE_CHECKING, Any
#test
if TYPE_CHECKING:  # pragma: no cover

    from .. import Ring, Simulation
    from ..cycles.magnetic_cycle import MagneticCycleBase

class EnergyReferenceKick(BeamPhysicsRelevant, Schedulable):

    def __init__(self,
                 n_rf: int,
                 section_index: int,
                 name: str | None = None,
                 **kwargs: dict[str, Any],):  # for MRO of fused elements

        super().__init__(
            section_index=section_index,
            name=name,
            **kwargs,  # for MRO of fused elements
        )

        self._n_rf = n_rf
        self._turn_i: DynamicParameter | None = None
        self._magnetic_cycle: MagneticCycleBase | None = None
        self._ring: Ring | None = None

    def on_init_simulation(self, simulation: Simulation) -> None:
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
        **kwargs: dict[str, Any],
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

        # alternative 1
        beam._dE -= reference_energy_change
        beam.reference_total_energy += reference_energy_change

        # #alternative 2
        # backend.specials.kick_single_harmonic(
        #     dt=beam.read_partial_dt(),
        #     dE=beam.write_partial_dE(),
        #     voltage=backend.float(0),
        #     phi_rf=backend.float(0),
        #     omega_rf=backend.float(0),
        #     charge=backend.float(beam.particle_type.charge),
        #     acceleration_kick=+reference_energy_change,
        # )





























