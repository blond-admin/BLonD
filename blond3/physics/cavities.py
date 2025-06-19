from __future__ import annotations

from abc import ABC
from typing import (
    Optional,
)
from typing import Optional as LateInit

from .impedances.base import WakeField
from blond3.core.backend import backend
from ..core.base import BeamPhysicsRelevant
from ..core.beam.base import BeamBaseClass
from ..core.simulation.simulation import Simulation
from ..cycles.base import RfParameterCycle


class CavityBaseClass(BeamPhysicsRelevant, ABC):
    def __init__(
        self,
        rf_program: Optional[RfParameterCycle] = None,
        group: int = 0,
        local_wakefield: Optional[WakeField] = None,
    ):
        super().__init__(group=group)
        self._rf_program: RfParameterCycle = rf_program
        self._local_wakefield = local_wakefield
        self._turn_i_dynamic: LateInit[None]

    def late_init(self, simulation: Simulation, **kwargs):
        self._turn_i_dynamic = simulation.turn_i
        assert self._rf_program is not None
        self._rf_program.late_init(simulation=simulation)

    def track(self, beam: BeamBaseClass):
        if self._local_wakefield is not None:
            self._local_wakefield.track(beam=beam)


class SingleHarmonicCavity(CavityBaseClass):
    def __init__(
        self,
        harmonic: int | float,
        rf_program: Optional[RfParameterCycle] = None,
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

    def derive_rf_program(self, simulation: Simulation):
        pass

    def track(self, beam: BeamBaseClass):
        super().track(beam=beam)
        backend.kick_single_harmonic(
            beam.read_partial_dt(),
            beam.write_partial_dE(),
            self._rf_program.get_phase(turn_i=self._turn_i_dynamic.value),
            self._rf_program.get_effective_voltage(turn_i=self._turn_i_dynamic.value),
        )

    def late_init(self, simulation: Simulation, **kwargs):
        super().late_init(simulation=simulation)
        pass


class MultiHarmonicCavity(CavityBaseClass):
    def track(self, beam: BeamBaseClass):
        pass

    def derive_rf_program(self, simulation: Simulation):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        pass
