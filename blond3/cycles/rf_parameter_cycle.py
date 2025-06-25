from __future__ import annotations

from typing import Optional as LateInit
from typing import TYPE_CHECKING, Optional

import numpy as np

from .base import RfParameterCycle
from .._core.backends.backend import backend
from .._core.ring.helpers import requires
from .._core.simulation.simulation import Simulation

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray


class _Blon2LikeInit(object):
    def __init__(
        self,
        harmonic: int | float | NumpyArray,
        voltage: float | NumpyArray,
        phi_rf: float | NumpyArray,
        omega_rf: Optional[float | NumpyArray],
        phi_noise: Optional[NumpyArray],
        phi_modulation: Optional[object],
    ):
        self.harmonic = harmonic
        self.voltage = voltage
        self.phi_rf = phi_rf
        self.omega_rf = omega_rf
        self.phi_noise = phi_noise
        self.phi_modulation = phi_modulation


class RfStationParams(RfParameterCycle):
    def __init__(
        self,
        harmonic: int | float | NumpyArray,
        voltage: float | NumpyArray,
        phi_rf: float | NumpyArray,
        omega_rf: Optional[float | NumpyArray] = None,
        phi_noise: Optional[NumpyArray] = None,
        phi_modulation: Optional[object] = None,  # TODO required??
    ):
        super().__init__()

        self._init_params = _Blon2LikeInit(
            harmonic=harmonic,
            voltage=voltage,
            phi_rf=phi_rf,
            omega_rf=omega_rf,
            phi_noise=phi_noise,
            phi_modulation=phi_modulation,
        )

        self.harmonic: LateInit[NumpyArray[backend.float]] = None

        self.voltage: LateInit[NumpyArray[backend.float]] = None

        self.omega_rf: LateInit[NumpyArray[backend.float]] = None
        self.omega_rf_design: LateInit[NumpyArray[backend.float]] = None

        self.phi_rf: LateInit[NumpyArray[backend.float]] = None
        self.phi_rf_design: LateInit[NumpyArray[backend.float]] = None
        self.dphi_rf: LateInit[NumpyArray[backend.float]] = None

        self.phi_noise: Optional[NumpyArray[backend.float]] = None
        self.phi_modulation: Optional[NumpyArray] = None

        self.noise_feedback = None  # TODO it is not clear if they should be
        # here
        self.beam_feedback = None  # TODO it is not clear if they should be here
        self.cavity_feedback = None  # TODO it is not clear if they should be
        # here

    @requires(["EnergyCycleBase"])
    def on_init_simulation(self, simulation: Simulation) -> None:
        super().on_init_simulation(simulation=simulation)
        from blond.input_parameters.rf_parameters import RFStationOptions

        rf_station_options = RFStationOptions()
        cycle_time = self._simulation.energy_cycle.cycle_time[
            self._owner.section_index, :]
        n_turns = self._simulation.energy_cycle.n_turns
        n_rf = self._owner.n_rf
        t_start = 0.0  # TODO expose parameter if required, else legacy

        # Reshape input rf programs
        # Reshape design harmonic
        harmonic = rf_station_options.reshape_data(
            self._init_params.harmonic,
            n_turns,
            n_rf,
            cycle_time,
            t_start,
        )
        self.harmonic = harmonic.astype(backend.float, order="C", copy=False)

        # Reshape design voltage
        voltage = rf_station_options.reshape_data(
            self._init_params.voltage, n_turns, n_rf, cycle_time, t_start
        )
        self.voltage = voltage.astype(
            backend.float,
            order="F",
            # F contigous, so voltage[:,turn_i] is contigous
            copy=False,
        )

        # Reshape design phase
        self.phi_rf_design = rf_station_options.reshape_data(
            self._init_params.phi_rf,
            n_turns,
            n_rf,
            cycle_time,
            t_start,
        )

        # Calculating design rf angular frequency
        if self._init_params.omega_rf is None:
            omega_rf_d = (
                2.0 * np.pi * self._simulation.energy_cycle.f_rev * self.harmonic
            )
        else:
            omega_rf_d = rf_station_options.reshape_data(
                self._init_params.omega_rf,
                n_turns,
                n_rf,
                cycle_time,
                t_start,
            )
        self.omega_rf_design = omega_rf_d.astype(backend.float, order="C", copy=False)

        # Reshape phase noise
        if self._init_params.phi_noise is not None:
            phi_noise = rf_station_options.reshape_data(
                self._init_params.phi_noise,
                n_turns,
                n_rf,
                cycle_time,
                t_start,
            )
            self.phi_noise = phi_noise.astype(backend.float, order="C", copy=False)

        else:
            self.phi_noise = None

        if self._init_params.phi_modulation is not None:
            try:
                iter(self._init_params.phi_modulation)
            except TypeError:
                phi_modulation = [self._init_params.phi_modulation]

            dPhi = np.zeros([n_rf, self.n_turns + 1], dtype=backend.float)
            dOmega = np.zeros([n_rf, self.n_turns + 1], dtype=backend.float)
            for pMod in phi_modulation:
                system = np.where(self.harmonic[:, 0] == pMod.harmonic)[0]
                if len(system) == 0:
                    raise ValueError("No matching harmonic in phi_modulation")
                elif len(system) > 1:
                    raise RuntimeError("""Phase modulation not yet 
                                               implemented with multiple systems 
                                               at the same harmonic.""")
                else:
                    system = system[0]

                pMod.calc_modulation()
                pMod.calc_delta_omega((ring.cycle_time[system], self.omega_rf_design[system]))
                dPhiInput, dOmegaInput = pMod.extend_to_n_rf(self.harmonic[:, 0])
                dPhi += rf_station_options.reshape_data(
                    dPhiInput,
                    n_turns,
                    n_rf,
                    cycle_time,
                    t_start,
                )
                dOmega += rf_station_options.reshape_data(
                    dOmegaInput,
                    n_turns,
                    n_rf,
                    cycle_time,
                    t_start,
                )

            self.phi_modulation = (dPhi, dOmega)
        else:
            self.phi_modulation = None

        # Copy of the design rf programs in the one used for tracking
        # and that can be changed by feedbacks
        self.phi_rf = np.array(self.phi_rf_design).astype(backend.float, order="F")
        # F contigous, so phi_rf[:,turn_i] is contigous

        self.dphi_rf = np.zeros(self._owner.n_rf).astype(backend.float, order="F")
        # F contigous, so dphi_rf[:,turn_i] is contigous

        self.omega_rf = np.array(self.omega_rf_design).astype(backend.float, order="F")
        # F contigous, so omega_rf[:,turn_i] is contigous

        # self.phi_s = calculate_phi_s(self, self.particle)
        # self.Q_s = calculate_Q_s(self, self.particle)
        # self.omega_s0 = self.Q_s * ring.omega_rev

    def track(self):
        """Tracking method for the section. Applies first the kick, then the
        drift. Calls also RF/beam feedbacks if applicable. Updates the counter
        of the corresponding RFStation class and the energy-related variables
        of the Beam class.

        """
        turn = self._simulation.turn_i.value
        # Add phase noise directly to the cavity RF phase
        if self.phi_noise is not None:
            if self.noise_feedback is not None:
                self.phi_rf[:, turn] += self.noise_feedback.x * self.phi_noise[:, turn]
            else:
                self.phi_rf[:, turn] += self.phi_noise[:, turn]

        # Add phase modulation directly to the cavity RF phase
        if self.phi_modulation is not None:
            self.phi_rf[:, turn] += self.phi_modulation[0][:, turn]
            self.omega_rf[:, turn] += self.phi_modulation[1][:, turn]

        # Determine phase loop correction on RF phase and frequency
        if self.beam_feedback is not None and turn >= self.beam_feedback.delay:
            self.beam_feedback.track()

        # Update the RF phase of all systems for the next turn
        # Accumulated phase offset due to beam phase loop or frequency offset
        self.dphi_rf += (
            2.0
            * np.pi
            * self.harmonic[:, turn + 1]
            * (self.omega_rf[:, turn + 1] - self.omega_rf_design[:, turn + 1])
            / self.omega_rf_design[:, turn + 1]
        )

        # Total phase offset
        self.phi_rf[:, turn + 1] += self.dphi_rf

        # Correction from cavity loop
        if self.cavity_feedback is not None:
            for feedback in self.cavity_feedback:
                if feedback is not None:
                    feedback.track()
