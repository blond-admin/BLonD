from __future__ import annotations

from typing import Optional as LateInit

from numpy.typing import NDArray as NumpyArray

from .base import RfParameterCycle
from .noise_generators.base import NoiseGenerator
from blond3.core.backend import backend
from ..core.simulation.simulation import Simulation


class ConstantProgram(RfParameterCycle):
    def __init__(self, phase: float, effective_voltage: float):
        super().__init__()
        self._phase = backend.float(phase)
        self._effective_voltage = backend.float(effective_voltage)

    def get_phase(self, turn_i: int):
        return self._phase

    def get_effective_voltage(self, turn_i: int):
        return self._effective_voltage

    def late_init(self, simulation: Simulation, **kwargs) -> None:
        pass


class RFNoiseProgram(RfParameterCycle):
    def __init__(
        self,
        phase: float,
        effective_voltage: float,
        phase_noise_generator: NoiseGenerator,
    ):
        super().__init__()
        self._phase = backend.float(phase)
        self._effective_voltage = backend.float(effective_voltage)
        self._phase_noise_generator = phase_noise_generator

        self._phase_noise: LateInit[NumpyArray] = None

    def on_run_simulation(self, simulation: Simulation, n_turns: int, turn_i_init: int) -> None:
        self._phase_noise = self._phase_noise_generator.get_noise(
            n_turns=n_turns
        ).astype(backend.float)

    def get_phase(self, turn_i: int):
        return self._phase + self._phase_noise[turn_i]

    def get_effective_voltage(self, turn_i: int):
        return self._effective_voltage
