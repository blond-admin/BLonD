from __future__ import annotations

from typing import (
    Optional,
)
from typing import Optional as LateInit

from blond3.core.backend import backend
from ..core.base import BeamPhysicsRelevant
from ..core.beam.base import BeamBaseClass
from ..core.simulation.simulation import Simulation


class Losses(BeamPhysicsRelevant):
    def __init__(self):
        super().__init__()


class BoxLosses(Losses):
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

    def track(self, beam: BeamBaseClass):
        backend.loss_box(
            beam.write_partial_flags(), self.t_min, self.t_max, self.e_min, self.e_max
        )

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass


class SeparatrixLosses(Losses):
    def __init__(self):
        super().__init__()
        self._simulation: LateInit[Simulation] = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        self._simulation = simulation

    def track(self, beam: BeamBaseClass):
        self._simulation.get_separatrix()  # TODO
