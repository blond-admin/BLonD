from __future__ import annotations

from abc import ABC
from typing import Optional as LateInit, TYPE_CHECKING

from .._core.base import Preparable

if TYPE_CHECKING:
    from .._core.simulation.simulation import Simulation
    from ..physics.cavities import (
        CavityBaseClass,
        MultiHarmonicCavity,
        SingleHarmonicCavity,
    )


class ProgrammedCycle(Preparable, ABC):
    def __init__(self):
        super().__init__()


class RfParameterCycle(ProgrammedCycle, ABC):
    def __init__(self):
        super().__init__()
        self._simulation: LateInit[Simulation] = None
        self._owner: SingleHarmonicCavity | MultiHarmonicCavity | None = None

    def set_owner(self, cavity: CavityBaseClass):
        assert self._owner is None
        self._owner = cavity

    def on_init_simulation(self, simulation: Simulation) -> None:
        assert self._owner is not None
        self._simulation = simulation

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        pass
