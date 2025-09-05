from typing import TYPE_CHECKING

from blond import DriftSimple, Simulation
from blond._core.base import BeamPhysicsRelevant
from blond._core.beam.base import BeamBaseClass

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any, Dict
    from typing import Optional
    from typing import Optional as LateInit


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
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called

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

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        pass
