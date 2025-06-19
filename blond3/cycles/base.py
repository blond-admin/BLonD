from __future__ import annotations

from abc import abstractmethod, ABC

from numpy.typing import NDArray as NumpyArray

from blond3.core.backend import backend
from ..core.base import Preparable
from ..core.simulation.simulation import Simulation


class ProgrammedCycle(Preparable, ABC):
    def __init__(self):
        super().__init__()



class EnergyCycle(ProgrammedCycle):
    def __init__(self, beam_energy_by_turn: NumpyArray):
        super().__init__()
        self._beam_energy_by_turn = beam_energy_by_turn.astype(backend.float)

    @property
    def beam_energy_by_turn(self):
        return self._beam_energy_by_turn

    @staticmethod
    def from_linspace(start, stop, turns, endpoint: bool = True):
        return EnergyCycle(
            beam_energy_by_turn=backend.linspace(
                start, stop, turns, endpoint=endpoint, dtype=backend.float
            )
        )

    def late_init(self, simulation: Simulation, **kwargs) -> None:
        pass


class RfParameterCycle(ProgrammedCycle):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_phase(self, turn_i: int) -> backend.float:
        pass

    @abstractmethod
    def get_effective_voltage(self, turn_i: int) -> backend.float:
        pass
