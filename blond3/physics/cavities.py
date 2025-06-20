from __future__ import annotations

from abc import ABC
from typing import (
    Optional, Iterable, TYPE_CHECKING,
)
from typing import Optional as LateInit

import numpy as np

from .impedances.base import WakeField
from ..core.backend import backend
from ..core.base import BeamPhysicsRelevant
from ..core.beam.base import BeamBaseClass
from ..core.simulation.simulation import Simulation
from ..cycles.base import RfParameterCycle, RfProgramSingleHarmonic, RfProgramMultiHarmonic

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray
    from cupy.typing import NDArray as CupyArray

class CavityBaseClass(BeamPhysicsRelevant, ABC):
    def __init__(
        self,
        rf_program: Optional[RfParameterCycle] = None,
        group: int = 0,
        local_wakefield: Optional[WakeField] = None,
    ):
        super().__init__(group=group)
        rf_program.set_owner(cavity=self)
        self._rf_program: RfParameterCycle = rf_program
        self._local_wakefield = local_wakefield
        self._turn_i_dynamic: LateInit[None]

    @property
    def rf_program(self):
        return self._rf_program

    def on_init_simulation(self, simulation: Simulation) -> None:
        self._turn_i_dynamic = simulation.turn_i
        assert self._rf_program is not None

    def track(self, beam: BeamBaseClass):
        if self._local_wakefield is not None:
            self._local_wakefield.track(beam=beam)


class SingleHarmonicCavity(CavityBaseClass):
    def __init__(
        self,
        harmonic: int | float,
        rf_program: Optional[RfProgramSingleHarmonic] = None,
        group: int = 0,
        local_wakefield: Optional[WakeField] = None,
    ):
        super().__init__(
            rf_program=rf_program, group=group, local_wakefield=local_wakefield
        )
        self._harmonic = harmonic

    @property
    def harmonic(self):
        return self._harmonic

    def track(self, beam: BeamBaseClass):
        super().track(beam=beam)
        backend.kick_single_harmonic(
            beam.read_partial_dt(),
            beam.write_partial_dE(),
            self._rf_program.get_phase(turn_i=self._turn_i_dynamic.value),
            self._rf_program.get_effective_voltage(turn_i=self._turn_i_dynamic.value),
        )

    def on_init_simulation(self, simulation: Simulation) -> None:
        super().on_init_simulation(simulation=simulation)
        pass


class MultiHarmonicCavity(CavityBaseClass):
    def __init__(
        self,
        harmonics: Iterable[float],
        rf_program: Optional[RfProgramMultiHarmonic] = None,
        group: int = 0,
        local_wakefield: Optional[WakeField] = None,
    ):
        super().__init__(
            rf_program=rf_program, group=group, local_wakefield=local_wakefield
        )
        self._harmonics: NumpyArray | CupyArray = backend.array(harmonics, dtype=backend.float)

    @property
    def harmonics(self):
        return self._harmonics

    def track(self, beam: BeamBaseClass):
        super().track(beam=beam)
        backend.kick_multi_harmonic(
            beam.read_partial_dt(),
            beam.write_partial_dE(),
            self._rf_program.get_phases(turn_i=self._turn_i_dynamic.value),
            self._rf_program.get_effective_voltages(turn_i=self._turn_i_dynamic.value),
        )

    def on_init_simulation(self, simulation: Simulation) -> None:
        super().on_init_simulation(simulation=simulation)
        pass
