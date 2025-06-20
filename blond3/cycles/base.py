from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Optional as LateInit
from numpy.typing import NDArray as NumpyArray

from blond3.core.backend import backend
from .. import SingleHarmonicCavity
from ..core.base import Preparable
from ..core.simulation.simulation import Simulation
from ..physics.cavities import CavityBaseClass, MultiHarmonicCavity
from ..physics.impedances.sovlers import MutliTurnResonatorSolver


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

    def on_run_simulation(self, simulation: Simulation, n_turns: int, turn_i_init: int) -> None:
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass


class RfParameterCycle(ProgrammedCycle, ABC):
    def __init__(self):
        super().__init__()
        self._simulation = LateInit[Simulation]
        self.owner: SingleHarmonicCavity | MultiHarmonicCavity | None = None

    def set_owner(self, cavity: CavityBaseClass):
        assert self.owner is None
        self.owner = cavity


    def on_init_simulation(self, simulation: Simulation) -> None:
        assert self.owner is not None
        self._simulation = simulation

    def on_run_simulation(self, simulation: Simulation, n_turns: int, turn_i_init: int) -> None:
        pass

    def get_frequency(self, turn_i: int) -> backend.float:
        if isinstance(self.owner, SingleHarmonicCavity):
            shc = self.owner
            shc.harmonic self._simulation.

        elif isinstance(self.owner, MultiHarmonicCavity):
            mhc = self.owner
            shc.harmonics

            pass # TODO
        else:
            raise TypeError(type(self.owner))

    @abstractmethod
    def get_phase(self, turn_i: int) -> backend.float:
        pass

    @abstractmethod
    def get_effective_voltage(self, turn_i: int) -> backend.float:
        pass
