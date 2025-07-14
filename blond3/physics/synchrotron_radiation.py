from blond3 import Simulation, DriftSimple
from blond3._core.base import BeamPhysicsRelevant
from blond3._core.beam.base import BeamBaseClass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Optional as LateInit


class SynchrotronRadiation(BeamPhysicsRelevant):
    def __init__(self, section_index: int = 0, name: Optional[str] = None):
        super().__init__(
            section_index=section_index,
            name=name,
        )
        raise NotImplementedError("For Lina")
        self._simulation: LateInit[DriftSimple] = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        self._simulation = simulation

    def on_run_simulation(
        self,
        simulation: Simulation,
        n_turns: int,
        turn_i_init: int,
    ) -> None:
        pass

    def track(self, beam: BeamBaseClass) -> None:
        pass
