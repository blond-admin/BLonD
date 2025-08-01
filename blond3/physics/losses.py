from __future__ import annotations

from typing import (
    Optional,
)
from typing import Optional as LateInit

from .._core.backends.backend import backend
from .._core.base import BeamPhysicsRelevant
from .._core.beam.base import BeamBaseClass
from .._core.simulation.simulation import Simulation


class LossesBaseClass(BeamPhysicsRelevant):
    def __init__(self):
        super().__init__()


class BoxLosses(LossesBaseClass):
    def __init__(
        self,
        t_min: Optional[backend.float] = None,
        t_max: Optional[backend.float] = None,
        e_min: Optional[backend.float] = None,
        e_max: Optional[backend.float] = None,
    ):
        super().__init__()

        self.t_min = backend.float(t_min)
        self.t_max = backend.float(t_max)
        self.e_min = backend.float(e_min)
        self.e_max = backend.float(e_max)

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        pass

    def on_run_simulation(self, simulation: Simulation, beam: BeamBaseClass,
                          n_turns: int, turn_i_init: int, **kwargs) -> None:
        pass

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        backend.loss_box(
            beam.write_partial_flags(), self.t_min, self.t_max, self.e_min, self.e_max
        )


class SeparatrixLosses(LossesBaseClass):
    def __init__(self):
        super().__init__()
        self._simulation: LateInit[Simulation] = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        self._simulation = simulation

    def on_run_simulation(self, simulation: Simulation, beam: BeamBaseClass,
                          n_turns: int, turn_i_init: int, **kwargs) -> None:
        pass

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        self._simulation.get_separatrix()  # TODO
