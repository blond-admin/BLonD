from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Optional as LateInit, TYPE_CHECKING

from ..core.backends.backend import backend
from ..core.base import Preparable
from ..core.simulation.simulation import Simulation
from ..physics.cavities import (
    CavityBaseClass,
    MultiHarmonicCavity,
    SingleHarmonicCavity,
)

if TYPE_CHECKING:
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray


class ProgrammedCycle(Preparable, ABC):
    def __init__(self):
        super().__init__()


class EnergyCycle(ProgrammedCycle):
    def __init__(self, beam_energy_by_turn: NumpyArray):
        super().__init__()
        self._beam_energy_by_turn = beam_energy_by_turn.astype(backend.float)
        self._beam_energy_by_turn.flags.writeable = False

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

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass


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


class RfProgramSingleHarmonic(RfParameterCycle):
    _owner: SingleHarmonicCavity

    def get_frequency(self, turn_i: int) -> backend.float:
        freq: backend.float = (
            self._owner.harmonic
            * self._simulation.revolution_frequency.by_turn(turn_i=turn_i)
        )
        return freq

    def get_omega(self, turn_i: int) -> backend.float:
        return backend.twopi * self.get_frequency(turn_i=turn_i)

    @abstractmethod
    def get_phase(self, turn_i: int) -> backend.float:
        pass

    @abstractmethod
    def get_effective_voltage(self, turn_i: int) -> backend.float:
        pass


class RfProgramMultiHarmonic(RfParameterCycle):
    _owner = MultiHarmonicCavity

    def get_frequencies(self, turn_i: int) -> NumpyArray | CupyArray:
        freqs = self._owner.harmonics * self._simulation.revolution_frequency.by_turn(
            turn_i=turn_i
        )
        return freqs

    def get_omegas(self, turn_i: int) -> NumpyArray | CupyArray:
        return backend.twopi * self.get_frequencies(turn_i=turn_i)

    @abstractmethod
    def get_phases(self, turn_i: int) -> NumpyArray | CupyArray:
        pass

    @abstractmethod
    def get_effective_voltages(self, turn_i: int) -> NumpyArray | CupyArray:
        pass
