from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from .._core.backends.backend import backend
from .._core.base import BeamPhysicsRelevant, DynamicParameter

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional as LateInit
    from typing import (
        Optional,
    )
    from .impedances.base import WakeField
    from .. import EnergyCycle
    from .._core.beam.base import BeamBaseClass
    from .._core.simulation.simulation import Simulation
    from ..cycles.rf_parameter_cycle import RfStationParams


class CavityBaseClass(BeamPhysicsRelevant, ABC):
    def __init__(
        self,
        n_rf: int,
        rf_program: RfStationParams,
        section_index: int = 0,
        local_wakefield: Optional[WakeField] = None,
    ):
        super().__init__(section_index=section_index)
        rf_program.set_owner(cavity=self)
        self._rf_program: RfStationParams = rf_program
        self._local_wakefield = local_wakefield
        self._turn_i: LateInit[DynamicParameter] = None
        self._n_rf = n_rf
        self._energy_cycle: LateInit[EnergyCycle] = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        assert self._rf_program is not None
        self._turn_i = simulation.turn_i
        self._energy_cycle = simulation.energy_cycle

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        pass

    @property  # as readonly attributes
    def n_rf(self):
        return self._n_rf

    @property  # as readonly attributes
    def rf_program(self) -> RfStationParams:
        return self._rf_program

    def track(self, beam: BeamBaseClass):
        self.rf_program.track()
        if self._local_wakefield is not None:
            self._local_wakefield.track(beam=beam)


class SingleHarmonicCavity(CavityBaseClass):
    _rf_program: RfStationParams  # make type hint more specific

    def __init__(
        self,
        rf_program: RfStationParams = None,
        section_index: int = 0,
        local_wakefield: Optional[WakeField] = None,
    ):
        super().__init__(
            n_rf=1,
            rf_program=rf_program,
            section_index=section_index,
            local_wakefield=local_wakefield,
        )

    def track(self, beam: BeamBaseClass):
        super().track(beam=beam)
        backend.specials.kick_single_harmonic(
            dt=beam.read_partial_dt(),
            dE=beam.write_partial_dE(),
            voltage=self._rf_program.voltage[0, self._turn_i.value],
            omega_rf=self._rf_program.omega_rf[0, self._turn_i.value],
            phi_rf=self._rf_program.phi_rf[0, self._turn_i.value],
            charge=backend.float(beam.particle_type.charge),  #  FIXME
            acceleration_kick=-self._energy_cycle.delta_E[
                self.section_index, self._turn_i.value
            ],
        )


class MultiHarmonicCavity(CavityBaseClass):
    _rf_program: RfStationParams  # make type hint more specific

    def __init__(
        self,
        n_harmonics: int,
        rf_program: RfStationParams,
        section_index: int = 0,
        local_wakefield: Optional[WakeField] = None,
    ):
        super().__init__(
            n_rf=n_harmonics,
            rf_program=rf_program,
            section_index=section_index,
            local_wakefield=local_wakefield,
        )

    def track(self, beam: BeamBaseClass):
        super().track(beam=beam)
        backend.kick_multi_harmonic(
            beam.read_partial_dt(),
            beam.write_partial_dE(),
            self._rf_program.phi_rf[:, self._turn_i.value],
            self._rf_program.voltage[:, self._turn_i.value],
            self._rf_program.omega_rf[:, self._turn_i.value],
        )
