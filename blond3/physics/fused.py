from typing import Optional, Iterable

from blond3._core.backends.backend import backend
from blond3._core.beam.base import BeamBaseClass
from blond3.cycles.rf_parameter_cycle import RfStationParams
from blond3.physics.cavities import MultiHarmonicCavity, SingleHarmonicCavity
from blond3.physics.drifts import DriftSimple
from blond3.physics.impedances.base import WakeField


class FusedDriftMultiHarmonicCavity(
    MultiHarmonicCavity, DriftSimple, SingleHarmonicCavity
):
    def __init__(
        self,
        n_harmonics: int,
        share_of_circumference: float,
        transition_gamma: float | Iterable,
        rf_program: RfStationParams,
        local_wakefield: Optional[WakeField] = None,
        section_index: int = 0,
    ):
        super().__init__(
            n_harmonics=n_harmonics,
            share_of_circumference=share_of_circumference,
            transition_gamma=transition_gamma,
            rf_program=rf_program,
            local_wakefield=local_wakefield,
            section_index=section_index,
        )

    def track(self, beam: BeamBaseClass) -> None:
        super().track(beam=beam)
        current_turn_i = self._turn_i.value
        backend.specials.kick_drift_multi_harmonic(
            dt=beam.read_partial_dt(),
            dE=beam.write_partial_dE(),
            voltage=self._rf_program.voltage[0, current_turn_i],
            omega_rf=self._rf_program.omega_rf[0, current_turn_i],
            phi_rf=self._rf_program.phi_rf[0, current_turn_i],
            charge=backend.float(beam.particle_type.charge),  # FIXME
            acceleration_kick=-self._energy_cycle.delta_E[
                self.section_index, current_turn_i
            ],
            t_rev=self._simulation.energy_cycle.t_rev[current_turn_i],
            length_ratio=self._share_of_circumference,
            eta_0=self._eta_0[current_turn_i],
            beta=self._simulation.energy_cycle.beta[self.section_index, current_turn_i],
            energy=self._simulation.energy_cycle.energy[
                self.section_index, current_turn_i
            ],
        )


FusedDriftMultiHarmonicCavity(
    n_harmonics=10,
    share_of_circumference=10,
    transition_gamma=10,
    rf_program=RfStationParams(harmonic=35640, voltage=6e6, phi_rf=0),
)
