from __future__ import annotations

from typing import Optional as LateInit, Iterable, TYPE_CHECKING

from .base import RfProgramSingleHarmonic, RfProgramMultiHarmonic
from .noise_generators.base import NoiseGenerator
from ..core.backends.backend import backend
from ..core.simulation.simulation import Simulation

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray
    from cupy.typing import NDArray as CupyArray


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
    def __init__(
        self,
        phase_per_harmonic: Iterable[float],
        effective_voltage_per_harmonic: Iterable[float],
    ):
        super().__init__()
        self._phase_per_harmonic: NumpyArray | CupyArray = backend.array(
            phase_per_harmonic, dtype=backend.float
        )
        self._effective_voltage_per_harmonic: NumpyArray | CupyArray = backend.array(
            effective_voltage_per_harmonic, dtype=backend.float
        )

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def get_phases(self, turn_i: int) -> NumpyArray | CupyArray:
        return self._phase_per_harmonic

    def get_effective_voltages(self, turn_i: int) -> NumpyArray | CupyArray:
        return self._effective_voltage_per_harmonic


class RFNoiseProgram(ConstantProgramSingleHarmonic):
    def __init__(
        self,
        phase: float,
        effective_voltage: float,
        phase_noise_generator: NoiseGenerator,
    ):
        super().__init__(phase=phase, effective_voltage=effective_voltage)
        self._phase = backend.float(phase)
        self._effective_voltage = backend.float(effective_voltage)
        self._phase_noise_generator = phase_noise_generator

        self._phase_noise: LateInit[NumpyArray] = None
        self._offset: LateInit[int] = None

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        self._offset = turn_i_init
        self._phase_noise = self._phase_noise_generator.get_noise(
            n_turns=n_turns
        ).astype(backend.float)

    def get_phase(self, turn_i: int) -> backend.float:
        return self._phase + self._phase_noise[turn_i - self._offset]

    def get_effective_voltage(self, turn_i: int) -> backend.float:
        return self._effective_voltage
