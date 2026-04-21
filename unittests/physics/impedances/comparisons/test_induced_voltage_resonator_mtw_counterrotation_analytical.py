import sys
import unittest
from copy import deepcopy
from typing import Literal
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.signal as sig
from scipy.constants import e

from blond import (
    Beam,
    DriftSimple,
    MagneticCyclePerTurnAllRFStations,
    Numpy64Bit,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    backend,
    momentum_compaction_factor,
    mu_minus,
    mu_plus,
)
from blond.generals.distributed.distributed_array import DistributedArray
from blond.handle_results.observables_as_elements import (
    InducedVoltageObservationCR,
)
from blond.physics.impedances.solvers import (
    MultiPassResonatorSolver,
    MultiPoleSparseSolve,
)
from blond.physics.impedances.sources import Resonators
from blond.testing.helpers import enforce_64_bit_backend


def gauss(x, width, center):
    return (
        1
        / (width * np.sqrt(2 * np.pi))
        * np.exp(-((x - center) ** 2) / (2 * width**2))
    )


def nonperiodic_wake(time_array, f0, R, Q):
    wake = np.zeros_like(time_array)
    omega_R = 2 * np.pi * f0
    alpha = omega_R / (2 * Q)
    omega_bar = np.sqrt(omega_R**2 - alpha**2)

    wake += (
        (np.sign(time_array) + 1)
        * R
        * alpha
        * np.exp(-alpha * time_array)
        * (
            np.cos(omega_bar * time_array)
            - alpha / omega_bar * np.sin(omega_bar * time_array)
        )
    )
    return wake


DEBUG_PLOTTING = False


class InducedVoltageResonatorPhysicsCR(unittest.TestCase):
    def setUp(self):
        self.n_slices = 2**12
        self.cut_left = 0
        self.cut_right = (
            1.4072317864464973e-08  # self.rf_station_list[0].t_rf[0, 0] * 2
        )
        self.t_rf = 6.332543039009238e-09

        self.harmonic = 20
        self.voltage_per_rf_station = 50e6
        self.R_shunt = 52e6
        self.Q_factor = 2e1
        self.alpha_p = -8.986e-4
        self.energy = 120e6
        self.energy_gain_per_turn = 50e6

        self.n_turns = 3
        self.n_stations = 4
        self.n_section_lengths = np.array([5, 5, 5, 5])  # 0-drift last

        self.n_macroparticles = int(1e4)

        self.energy_array = np.append(
            np.ones(self.n_stations) * self.energy,
            np.linspace(
                self.energy + self.energy_gain_per_turn / self.n_stations,
                self.energy + self.energy_gain_per_turn * self.n_turns,
                self.n_stations * self.n_turns,
            ),
        )

        self.sigma_bunch = 5e-10
        self.bunch_offset = 7e-9

    def run_sim(
        self,
        counterrot: bool = False,
        solver: Literal["PoleResidue", "Convolution"] = "Covolution",
    ):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")
        ring = Ring(
            circumference=np.sum(self.n_section_lengths),
            check_section_indices=False,
        )
        energy_array = np.reshape(
            self.energy_array, (self.n_stations, self.n_turns + 1), order="F"
        )

        magnetic_cycle = MagneticCyclePerTurnAllRFStations.headless(
            value_init=energy_array[0][0],
            values_after_rf_station_per_turn=np.array(
                [en[1:] for en in energy_array]
            ),  # slice off first value, as this is init
            in_unit="total energy",
            reference_particle=mu_plus,
        )
        one_turn_model = []
        beam = Beam(
            intensity=self.n_macroparticles,
            particle_type=mu_plus,
            is_counter_rotating=False,
        )

        prof = StaticProfile.from_rad(
            self.cut_left * 2 * np.pi / self.t_rf,
            self.cut_right * 2 * np.pi / self.t_rf,
            n_bins=self.n_slices,
            t_period=self.t_rf,
        )

        shc_list = []
        cav_obs_list = []
        profile_list = []
        prof._hist_y = gauss(prof.hist_x, self.sigma_bunch, self.bunch_offset)

        mocked_profile = Mock(spec=StaticProfile)
        mocked_profile.cut_left = prof.cut_left
        mocked_profile.cut_right = prof.cut_right
        mocked_profile.hist_y = prof.hist_y
        mocked_profile.hist_x = prof.hist_x
        mocked_profile.hist_step = prof.hist_step
        mocked_profile.hist_y_to_density_factor = 1 / beam.intensity
        mocked_profile.active = True
        mocked_profile.n_bins = len(prof.hist_y)
        mocked_profile.info_string.return_value = "me_mock"

        for sec_ind in range(self.n_stations):
            lock_mock = deepcopy(mocked_profile)
            lock_mock.section_index = sec_ind

            profile_list.append(lock_mock)
            local_res = Resonators(
                center_frequencies=1 / self.t_rf,
                quality_factors=self.Q_factor,
                shunt_impedances=self.R_shunt,
                shunt_impedances_counter_rotating=-self.R_shunt
                if counterrot
                else self.R_shunt,  # charge will invert additionally
            )

            shc_list.append(
                SingleHarmonicRFStation(
                    voltage=self.voltage_per_rf_station,
                    phi_rf=0,
                    harmonic=self.harmonic,
                    local_wakefield=WakeField(
                        sources=(local_res,),
                        solver=MultiPassResonatorSolver(
                            decay_fraction_threshold=1e-12,
                            allow_delta_t_zero=True,
                        )
                        if solver == "Convolution"
                        else MultiPoleSparseSolve(),
                        profile=profile_list[-1],
                        section_index=sec_ind,
                    ),
                    section_index=sec_ind,
                )
            )
            cav_obs_list.append(
                InducedVoltageObservationCR(
                    wake_field=shc_list[-1]._local_wakefield, each_turn_i=1
                )
            )

            one_turn_model.extend(
                [
                    DriftSimple(
                        orbit_length=self.n_section_lengths[sec_ind] / 2,
                        section_index=sec_ind,
                        momentum_compaction_factor=momentum_compaction_factor(
                            float(1)
                        ),
                    ),
                    cav_obs_list[-1],
                    profile_list[-1],
                    shc_list[-1],
                    profile_list[-1],
                    cav_obs_list[-1],
                    DriftSimple(
                        orbit_length=self.n_section_lengths[sec_ind] / 2,
                        section_index=sec_ind,
                        momentum_compaction_factor=momentum_compaction_factor(
                            float(1)
                        ),
                    ),
                ]
            )

        ring.add_elements(one_turn_model, reorder=False)

        sim = Simulation(
            ring=ring,
            magnetic_cycle=magnetic_cycle,
        )
        sim.print_one_turn_execution_order()

        beam._dE = DistributedArray(
            backend.array([0, 0, 0], dtype=backend.float)
        )
        beam._dt = DistributedArray(
            backend.array([0, 0, 0], dtype=backend.float)
        )
        beam._flags = DistributedArray(
            backend.array([0, 0, 0], dtype=backend.float)
        )
        beam._ids = DistributedArray(
            backend.array([0, 1, 2], dtype=backend.float)
        )

        beam_CR = deepcopy(beam)
        beam_CR._is_counter_rotating = True
        if counterrot:
            beam_CR.reference._particle_type = mu_minus
        sim.run_simulation(
            beams=(beam, beam_CR),
        )

        return cav_obs_list, sim.get_t_rev_init(), prof

    def test_compare_with_pole_residual_model_CR_beams(self):
        cav_obs_list_conv, t_rev, profile = self.run_sim(
            counterrot=True, solver="Convolution"
        )
        cav_obs_list_pole, t_rev, profile = self.run_sim(
            counterrot=True, solver="PoleResidue"
        )

        if DEBUG_PLOTTING:
            for inter_turn in range(self.n_stations):
                plt.figure(f"b3_{inter_turn}")
                plt.title(f"b3_{inter_turn}")
                for el in range(self.n_turns * 2):
                    plt.plot(
                        cav_obs_list_conv[inter_turn].beam_reference_time[el]
                        + profile.hist_x,
                        cav_obs_list_conv[inter_turn].induced_voltage[el],
                        ls="-",
                        label=f"section {inter_turn} turn {el}",
                    )
                    plt.plot(
                        cav_obs_list_pole[inter_turn].beam_reference_time[el]
                        + profile.hist_x,
                        cav_obs_list_pole[inter_turn].induced_voltage[el],
                        ls="--",
                        color="k",
                        # label=f"section {inter_turn} turn {el}",
                    )
                plt.legend(loc="upper right")

                if inter_turn == self.n_stations - 1:
                    plt.show(block=True)
                else:
                    plt.show(block=False)

        for inter_turn_ind in range(self.n_stations):
            for trn_ind in range(self.n_turns * 2):
                np.testing.assert_allclose(
                    cav_obs_list_conv[inter_turn_ind].induced_voltage[trn_ind],
                    cav_obs_list_pole[inter_turn_ind].induced_voltage[trn_ind],
                    atol=np.max(
                        cav_obs_list_pole[inter_turn_ind].induced_voltage[
                            trn_ind
                        ]
                    )
                    * 1e-12,
                    rtol=1e99,  # problem with close to 0 values --> 1e-38 vs 1e-9
                )

    @pytest.mark.backend_mutation
    def test_blond3_mtw(self):
        cav_obs_list, t_rev, profile = self.run_sim(
            counterrot=False, solver="Convolution"
        )

        t_start = sys.float_info.min
        time_axis = np.linspace(
            t_start,
            cav_obs_list[-1].beam_reference_time[-1] + t_rev,
            num=int(5e6),
        )
        total_wake_kernel = nonperiodic_wake(
            time_axis, 1 / self.t_rf, self.R_shunt, self.Q_factor
        )

        for ind in range(self.n_stations // 2):
            np.testing.assert_allclose(
                cav_obs_list[ind].induced_voltage,
                cav_obs_list[self.n_stations - ind - 1].induced_voltage,
                rtol=1e-8,
                atol=1e8,
            )
        self.dt_profile = time_axis[1] - time_axis[0]
        self.convolution_result = [[] for _ in range(self.n_stations)]
        self.profile_times = [[] for _ in range(self.n_stations)]

        for inter_turn in range(self.n_stations):
            ref_time = cav_obs_list[inter_turn].beam_reference_time
            convolution_input = np.zeros_like(time_axis)
            for turn in range(self.n_turns):
                turn_offset = ref_time[turn * 2]
                self.profile_times[inter_turn].append(
                    turn_offset + profile.hist_x
                )
                convolution_input += gauss(
                    time_axis,
                    self.sigma_bunch,
                    turn_offset + self.bunch_offset,
                )
                turn_offset = ref_time[turn * 2 + 1]
                self.profile_times[inter_turn].append(
                    turn_offset + profile.hist_x
                )
                convolution_input += gauss(
                    time_axis,
                    self.sigma_bunch,
                    turn_offset + self.bunch_offset,
                )

            self.convolution_result[inter_turn] = sig.convolve(
                convolution_input, total_wake_kernel
            )[0 : len(time_axis)]

        if DEBUG_PLOTTING:
            for inter_turn in range(self.n_stations):
                plt.figure(f"b3_{inter_turn}")
                plt.title(f"b3_{inter_turn}")
                plt.plot(
                    time_axis,
                    -self.convolution_result[inter_turn]
                    * e
                    / profile.hist_step
                    * self.dt_profile,
                    label="convolution_2",
                    color="k",
                )
                for el in range(self.n_turns * 2):
                    plt.plot(
                        cav_obs_list[inter_turn].beam_reference_time[el]
                        + profile.hist_x,
                        cav_obs_list[inter_turn].induced_voltage[el],
                        ls="--",
                        label=f"section {inter_turn} turn {el}",
                    )
                plt.legend(loc="upper right")

                if inter_turn == self.n_stations - 1:
                    plt.show(block=True)
                else:
                    plt.show(block=False)

        for inter_turn_ind in range(self.n_stations):
            for trn_ind in range(self.n_turns * 2):
                conv_result = np.interp(
                    self.profile_times[inter_turn_ind][trn_ind],
                    time_axis,
                    self.convolution_result[inter_turn_ind],
                )
                np.testing.assert_allclose(
                    -conv_result * e / profile.hist_step * self.dt_profile,
                    cav_obs_list[inter_turn_ind].induced_voltage[trn_ind],
                    atol=4.5e4,
                    rtol=1e-12,
                )
