from __future__ import annotations

from typing import Optional as LateInit, Iterable

from numpy.typing import NDArray as NumpyArray

from .base import RfProgramSingleHarmonic, RfProgramMultiHarmonic
from .noise_generators.base import NoiseGenerator
from ..core.backend import backend
from ..core.simulation.simulation import Simulation


class ConstantProgramSingleHarmonic(RfProgramSingleHarmonic):
    def __init__(self, phase: float, effective_voltage: float):
        super().__init__()
        self._phase = backend.float(phase)
        self._effective_voltage = backend.float(effective_voltage)

    def get_phase(self, turn_i: int):
        return self._phase

    def get_effective_voltage(self, turn_i: int):
        return self._effective_voltage

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass



class ConstantProgramMultiHarmonic(RfProgramMultiHarmonic):
    def __init__(self, phases: Iterable[float], effective_voltages: Iterable[float]):
        super().__init__()
        self._phases: NumpyArray | CupyArray = backend.array(phases, dtype=backend.float)
        self._effective_voltages: NumpyArray | CupyArray = backend.array(effective_voltages, dtype=backend.float)

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def get_phases(self, turn_i: int) -> NumpyArray | CupyArray:
        return self._phases

    def get_effective_voltages(self, turn_i: int)-> NumpyArray | CupyArray:
        return self._effective_voltages


class RFNoiseProgram(ConstantProgramSingleHarmonic):
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

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        self._phase_noise = self._phase_noise_generator.get_noise(
            n_turns=n_turns
        ).astype(backend.float)

    def get_phase(self, turn_i: int) -> backend.float:
        return self._phase + self._phase_noise[turn_i]

    def get_effective_voltage(self, turn_i: int) -> backend.float:
        return self._effective_voltage
