from __future__ import annotations

import warnings
from abc import abstractmethod, ABC
from functools import cached_property
from typing import Optional as LateInit, TYPE_CHECKING

import numpy as np

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
    def __init__(self, synchronous_data: NumpyArray, synchronous_data_type="momentum"):
        super().__init__()
        self._synchronous_data = synchronous_data
        self._synchronous_data_type = synchronous_data_type
        from blond.input_parameters.ring import Ring as Blond2Ring

        self._ring: LateInit[Blond2Ring] = None

    @cached_property
    def n_turns(self):
        return len(self._synchronous_data) - 1

    @property
    def beta(self) -> NumpyArray:
        return self._ring.beta  # TODO correct dtype

    @property
    def gamma(self) -> NumpyArray:
        return self._ring.gamma  # TODO correct dtype

    @property
    def energy(self) -> NumpyArray:
        return self._ring.energy  # TODO correct dtype

    @property
    def kin_energy(self) -> NumpyArray:
        return self._ring.kin_energy  # TODO correct dtype

    @property
    def delta_E(self) -> NumpyArray:
        return self._ring.delta_E  # TODO correct dtype

    @property
    def t_rev(self) -> NumpyArray:
        return self._ring.t_rev  # TODO correct dtype

    @property
    def cycle_time(self) -> NumpyArray:
        return self._ring.cycle_time  # TODO correct dtype

    @property
    def f_rev(self) -> NumpyArray:
        return self._ring.f_rev  # TODO correct dtype

    @property
    def omega_rev(self) -> NumpyArray:
        return self._ring.omega_rev  # TODO correct dtype

    @staticmethod
    def from_linspace(start, stop, turns, endpoint: bool = True):
        return EnergyCycle(
            synchronous_data=backend.linspace(
                start, stop, turns, endpoint=endpoint, dtype=backend.float
            )
        )

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        from blond.input_parameters.ring import Ring as Blond2Ring

        self._ring = Blond2Ring(
            ring_length=simulation.ring.circumference,
            alpha_0=np.nan,
            particle=simulation.beams[0].particle_type,
            n_turns=len(self._synchronous_data),
            synchronous_data_type=self._synchronous_data_type,
            bending_radius=simulation.ring.bending_radius,
            n_sections=len(simulation.ring.elements.get_elements(CavityBaseClass)),
        )


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
