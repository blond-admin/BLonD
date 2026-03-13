import sys
from copy import deepcopy
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from scipy.constants import e

from blond import Beam as beam_b3
from blond import (
    DriftSimple,
    MagneticCyclePerTurnAllRFStations,
)
from blond import Ring as ring_b3
from blond import (
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    momentum_compaction_factor,
    mu_plus,
)
from blond.experimental.beam_preparation.semi_empiric_matcher import (
    SemiEmpiricMatcher,
)
from blond.generals.distributed.distributed_array import DistributedArray
from blond.handle_results.observables import RFStationInducedVoltageObservation
from blond.handle_results.observables_as_elements import (
    InducedVoltageObservationCR,
)
from blond.legacy.blond2.beam.beam import Beam, MuPlus
from blond.legacy.blond2.beam.profile import CutOptions, Profile
from blond.legacy.blond2.impedances.impedance import (
    InducedVoltageResonator,
)
from blond.legacy.blond2.impedances.impedance_sources import Resonators
from blond.legacy.blond2.input_parameters.rf_parameters import RFStation
from blond.legacy.blond2.input_parameters.ring import Ring
from blond.physics.impedances.solvers import MultiPassResonatorSolver
from blond.physics.impedances.sources import Resonators as res_b3


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


DEBUG_PLOTTING = True


class InducedVoltageResonatorPhysicsCR:
    def __init__(self):
        self.n_slices = 2**12
        self.cut_left = 0
        self.cut_right = (
            1.4072317864464973e-08  # self.rf_station_list[0].t_rf[0, 0] * 2
        )

        self.harmonic = 10
        self.voltage_per_rf_station = 50e6
        self.R_shunt = 52e6
        self.Q_factor = 2e1
        self.alpha_p = -8.986e-4
        self.energy = 120e6
        self.energy_gain_per_turn = 50e6

        self.n_turns = 5
        self.n_stations = 3
        self.n_section_lengths = np.array([3, 3, 3, 0])  # 0-drift last

        self.n_macroparticles = int(1e4)

        self.energy_array = np.ones(self.n_stations + 1) * self.energy
        for _ in range(self.n_turns):
            init_en = (
                self.energy_array[-1]
                + self.energy_gain_per_turn / self.n_stations / 2
            )
            self.energy_array = np.append(
                self.energy_array,
                np.concatenate(
                    (
                        [init_en],
                        [
                            init_en
                            + _ * self.energy_gain_per_turn / self.n_stations
                            for _ in range(1, self.n_stations)
                        ],
                        [self.energy_array[-1] + self.energy_gain_per_turn],
                    )
                ),
            )
        self.n_stations += 1  # splitting of first station

        self.sigma_bunch = 5e-10
        self.bunch_offset = 3e-9

    def setUpB2(self, old_impl: bool = True):  # just for time arrays
        ring = Ring(
            self.n_section_lengths,
            self.alpha_p,
            self.energy_array.reshape(
                self.n_stations, self.n_turns + 1, order="F"
            ),
            MuPlus(),
            synchronous_data_type="total energy",
            n_turns=self.n_turns,
            n_sections=self.n_stations,
        )

        rf_station_list = []
        for sec_ind in range(self.n_stations):
            rf_station_list.append(
                RFStation(
                    ring,
                    self.harmonic,
                    self.voltage_per_rf_station
                    if sec_ind not in [0, self.n_stations]
                    else self.voltage_per_rf_station / 2,
                    0,
                    n_rf=1,
                    section_index=sec_ind + 1,
                )
            )  # amazing indexing

        self.beam = Beam(
            ring,
            n_macroparticles=self.n_macroparticles,
            intensity=self.n_macroparticles,
        )

        cut_options = CutOptions(
            cut_left=self.cut_left,
            cut_right=self.cut_right,
            n_slices=self.n_slices,
        )

        self.profile = Profile(self.beam, cut_options=cut_options)
        self.profile.track()
        self.profile.fwhm()
        self.hist_x = self.profile.bin_centers
        self.hist_y = self.profile.n_macroparticles
        self.hist_step = self.profile.bin_size

        self.t_rf = 1 / (rf_station_list[0].omega_rf[0, 0] / 2 / np.pi)

        resonator = Resonators(
            self.R_shunt,
            1 / self.t_rf,
            self.Q_factor,
        )  # low Q for fast decay in small machine, although phasing will dominate

        ind_volt_list = []
        for _ in range(self.n_stations):
            ind_volt_list.append(
                InducedVoltageResonator(
                    self.beam,
                    self.profile,
                    resonator,
                    rf_station=rf_station_list[_],
                    rf_station_list=rf_station_list,
                    mtw_mode="time",
                    time_decay_factor=1e-12,
                    multi_turn_wake=True,
                    old_time_array_impl=old_impl,
                )
            )  # never release

        # setup analytical solution
        t_start = sys.float_info.min
        t_end = np.sum(ring.t_rev)
        self.time_axis = np.linspace(t_start, t_end, num=int(5e6))
        wake_kernel = nonperiodic_wake(
            self.time_axis,
            resonator.frequency_R[0],
            resonator.R_S[0],
            resonator.Q[0],
        )

        beta_arrays = []
        section_length_arrays = []
        for _ in range(self.n_stations):
            beta_arrays.append(rf_station_list[_].beta.tolist())
            section_length_arrays.append(rf_station_list[_].section_length)
        beta_array = np.array(
            [
                result
                for combination in zip(*beta_arrays)
                for result in combination
            ]
        )
        section_length_array_extended = []
        [
            section_length_array_extended.append(section_length_arrays)
            for _ in range(self.n_turns)
        ]
        section_length_array_extended = np.array(
            section_length_array_extended
        ).flatten()
        from scipy.constants import c

        self.section_time = (
            1
            / (beta_array[self.n_stations :] * c)
            * section_length_array_extended
        )

        profiles = np.zeros(
            (self.n_stations, len(self.time_axis)), dtype=float
        )
        profiles[0] += gauss(
            self.time_axis, self.sigma_bunch, self.bunch_offset
        )

        profile_time_corot = [[] for _ in range(self.n_stations)]
        profile_time_corot[0].append(0)
        for prof_ind in range(0, self.n_turns):
            for inter_turn_ind in range(self.n_stations):
                if prof_ind == 0 and inter_turn_ind == 0:
                    continue
                profiles[inter_turn_ind] += gauss(
                    self.time_axis,
                    self.sigma_bunch,
                    np.sum(
                        self.section_time[
                            0 : prof_ind * self.n_stations + inter_turn_ind
                        ]
                    )
                    + self.bunch_offset,
                )
                profile_time_corot[inter_turn_ind].append(np.sum(
                    self.section_time[
                        0 : prof_ind * self.n_stations + inter_turn_ind
                    ]
                ) + self.bunch_offset)

        # profiles_CR = np.zeros_like(profiles)
        profile_time_counterrot = [[] for _ in range(self.n_stations)]
        for prof_ind in range(0, self.n_turns):
            for inter_turn_ind in range(self.n_stations):
                if prof_ind == 0 and inter_turn_ind < self.n_stations // 2:
                    profiles[inter_turn_ind] += gauss(
                        self.time_axis,
                        self.sigma_bunch,
                        np.sum(
                            self.section_time[
                                0: self.n_stations - inter_turn_ind - 1
                            ]
                        ) + self.bunch_offset,
                    )
                    profile_time_counterrot[inter_turn_ind].append(np.sum(
                                                                            self.section_time[
                                                                                0: self.n_stations - inter_turn_ind - 1
                                                                            ]
                                                                        ) + self.bunch_offset)
                else:
                    profiles[inter_turn_ind] += gauss(
                        self.time_axis,
                        self.sigma_bunch,
                        np.sum(
                            self.section_time[
                                inter_turn_ind : prof_ind * self.n_stations
                                + self.n_stations
                                - inter_turn_ind
                            ]
                        )
                        + self.bunch_offset,
                    )
                    profile_time_counterrot[inter_turn_ind].append(np.sum(
                            self.section_time[
                                inter_turn_ind : prof_ind * self.n_stations
                                + self.n_stations
                                - inter_turn_ind
                            ]
                        )
                        + self.bunch_offset)

        profile_time_combined = [[] for _ in range(self.n_turns)]
        for turn in range(0, self.n_turns):
            for inter_turn_ind in range(self.n_stations):
                if profile_time_corot[turn][inter_turn_ind] < profile_time_counterrot[turn][inter_turn_ind]:
                    profile_time_combined[turn].append(profile_time_corot[turn][inter_turn_ind]) # + self.profile.bin_centers)
                    profile_time_combined[turn].append(profile_time_counterrot[turn][inter_turn_ind]) # + self.profile.bin_centers)
                else:
                    profile_time_combined[turn].append(profile_time_corot[turn][inter_turn_ind]) # + self.profile.bin_centers)
                    profile_time_combined[turn].append(profile_time_counterrot[turn][inter_turn_ind]) # + self.profile.bin_centers)

        self.convolution_result = np.zeros_like(profiles)
        DEBUG_PLOTTING = False
        for inter_turn_ind in range(self.n_stations):
            self.convolution_result[inter_turn_ind] = sig.convolve(
                profiles[inter_turn_ind], wake_kernel
            )[0 : len(self.time_axis)]

            if DEBUG_PLOTTING:
                fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
                fig.suptitle(f"beam {inter_turn_ind}")
                ax[0].plot(self.time_axis, profiles[inter_turn_ind])
                ax[1].plot(
                    self.time_axis,
                    self.convolution_result[inter_turn_ind][
                        0 : len(self.time_axis)
                    ],
                )
                plt.show(block=False)

        self.profile.n_macroparticles = gauss(
            self.profile.bin_centers, self.sigma_bunch, self.bunch_offset
        )
        self.hist_y = self.profile.n_macroparticles

        self.time_array_profile = [[] for _ in range(self.n_stations)]

        for trn_ind in range(self.n_turns):
            for inter_turn_ind in range(self.n_stations):
                turn_time_corot = (np.sum(
                            self.section_time[
                                0 : trn_ind * self.n_stations + inter_turn_ind
                            ]
                        )
                        if trn_ind > 0
                        else 0
                        if inter_turn_ind == 0
                        else np.sum(self.section_time[:inter_turn_ind])
                    )
                turn_time_CR = (np.sum(
                                    self.section_time[
                                        0: trn_ind * self.n_stations + (inter_turn_ind - self.n_stations) % self.n_stations
                                    ]
                                )
                                if trn_ind > 0
                                else 0
                                if inter_turn_ind == 3  #
                                else np.sum(self.section_time[:inter_turn_ind])
                                )
                self.time_array_profile[inter_turn_ind].append(
                    # self.profile.bin_centers
                    # + (
                    turn_time_corot
                )
                self.time_array_profile[inter_turn_ind].append()

        self.dt_profile = self.time_axis[1] - self.time_axis[0]
        return
        if DEBUG_PLOTTING:
            for inter_turn_ind in range(self.n_stations):
                plt.figure()
                plt.title(f"blond2 old_impl = {old_impl}")
                plt.plot(
                    self.time_axis,
                    -self.convolution_result[inter_turn_ind][
                        0 : len(self.time_axis)
                    ]
                    * e
                    / self.profile.bin_size
                    * self.dt_profile,
                    label="convolution_2",
                )
                for el in range(self.n_turns):
                    plt.plot(
                        self.time_array_profile[inter_turn_ind][el],
                        save_voltage_array[inter_turn_ind][el],
                        ls="--",
                        label=f"resonator turn {el}",
                    )
                plt.legend(loc="upper right")
                plt.show(block=False)
        plt.show(block=True)
        if not old_impl:
            for inter_turn_ind in range(self.n_stations):
                for trn_ind in range(self.n_turns):
                    conv_result = np.interp(
                        self.time_array_profile[inter_turn_ind][trn_ind],
                        self.time_axis,
                        self.convolution_result[inter_turn_ind],
                    )
                    np.testing.assert_allclose(
                        -conv_result
                        * e
                        / self.profile.bin_size
                        * self.dt_profile,
                        save_voltage_array[inter_turn_ind][trn_ind],
                        atol=1e8,
                        rtol=1e-8,
                    )

    def setUpB3(self):
        ring = ring_b3(
            circumference=np.sum(self.n_section_lengths),
            check_section_indices=False,
        )
        energy_array = np.reshape(
            self.energy_array, (self.n_stations, self.n_turns + 1), order="F"
        )  # TODO: this seems to be very complicated

        magnetic_cycle = MagneticCyclePerTurnAllRFStations.headless(
            value_init=energy_array[0][0],
            values_after_rf_station_per_turn=np.array(
                [en[1:] for en in energy_array]
            ),  # slice off first value, as this is init
            in_unit="total energy",
            reference_particle=mu_plus,
        )
        one_turn_model = []
        beam = beam_b3(
            intensity=self.n_macroparticles,
            particle_type=mu_plus,
            is_counter_rotating=False,
        )

        shc_list = []
        cav_obs_list = []
        profile_list = []
        for sec_ind in range(self.n_stations):
            mocked_profile = Mock(spec=StaticProfile)
            # prof = StaticProfile.from_rad(
            #         self.cut_left * 2 * np.pi / self.t_rf,
            #         self.cut_right * 2 * np.pi / self.t_rf,
            #         n_bins=self.n_slices,
            #         t_period=self.t_rf,
            #         section_index=sec_ind,
            #     )
            mocked_profile.cut_left = self.cut_left
            mocked_profile.cut_right = self.cut_right
            mocked_profile.hist_y = self.hist_y
            mocked_profile.hist_x = self.hist_x
            mocked_profile.hist_step = self.hist_step
            mocked_profile.hist_y_to_density_factor = 1 / self.beam.intensity
            mocked_profile.active = True
            mocked_profile.n_bins = len(self.hist_y)
            mocked_profile.section_index = sec_ind
            mocked_profile.info_string.return_value = "me_mock"
            profile_list.append(
                mocked_profile
            )
            local_res = res_b3(
                center_frequencies=1 / self.t_rf,
                quality_factors=self.Q_factor,
                shunt_impedances=self.R_shunt,
                shunt_impedances_counter_rotating=-self.R_shunt,
            )
            voltage = (
                self.voltage_per_rf_station
                if sec_ind not in [0, self.n_stations]
                else self.voltage_per_rf_station / 2
            )
            shc_list.append(
                SingleHarmonicRFStation(
                    voltage=voltage,
                    phi_rf=0,
                    harmonic=self.harmonic,
                    local_wakefield=WakeField(
                        sources=(local_res,),
                        solver=MultiPassResonatorSolver(
                            decay_fraction_threshold=1e-12,
                            allow_delta_t_zero=True,
                        ),
                        profile=profile_list[-1],
                        section_index=sec_ind,
                    ),
                    section_index=sec_ind,
                )
            )
            cav_obs_list.append(
                InducedVoltageObservationCR(
                    rf_station=shc_list[-1], each_turn_i=1
                )
            )
            one_turn_model.extend(
                [
                    cav_obs_list[-1],
                    profile_list[
                        -1
                    ],  # TODO: is this necessary --> should not be, wakefield should have profile tracking implemented.
                    shc_list[-1],
                    profile_list[-1],
                    cav_obs_list[-1],
                ]
            )
            if sec_ind is not self.n_stations:
                one_turn_model.extend(
                    [
                        DriftSimple(
                            orbit_length=self.n_section_lengths[sec_ind],
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

        ind_volt_obs_list = []
        for sec_ind in range(self.n_stations):
            ind_volt_obs_list.append(
                RFStationInducedVoltageObservation(
                    rf_station=shc_list[sec_ind], each_turn_i=1
                )
            )
        beam._dE = DistributedArray(np.array([0, 0, 0], dtype=np.float64))
        beam._dt = DistributedArray(np.array([0, 0, 0], dtype=np.float64))
        beam._flags = DistributedArray(np.array([0, 0, 0], dtype=np.float64))
        beam._ids = DistributedArray(np.array([0, 1, 2], dtype=np.float64))
        # sim.prepare_beam(
        #     beam=beam,
        #     preparation_routine=SemiEmpiricMatcher(
        #         time_limit=[self.cut_left, self.cut_right],
        #         n_macroparticles=int(1e6),
        #         animate=True,
        #         hamilton_to_density_kwargs={
        #             "density_modifier": 0.25,
        #             "hamilton_max": 5000,
        #         },
        #     ),
        #     turn_i=0,
        # )
        plt.show()
        beam_CR = deepcopy(beam)
        beam_CR._is_counter_rotating = True
        sim.run_simulation(
            beams=(beam, beam_CR),
        )

        for ind in range(self.n_stations // 2):
            np.testing.assert_allclose(
                cav_obs_list[ind].induced_voltage,
                cav_obs_list[self.n_stations - ind - 1].induced_voltage,
                rtol=1e-8,
                atol=1e8,
            )
        self.dt_profile = self.time_axis[1] - self.time_axis[0]
        if DEBUG_PLOTTING:
            for inter_turn in range(self.n_stations):
                plt.figure(f"b3_{inter_turn}")
                plt.title(f"b3_{inter_turn}")
                plt.plot(
                    self.time_axis,
                    -self.convolution_result[inter_turn][
                        0 : len(self.time_axis)
                    ]
                    * e
                    / self.profile.bin_size
                    * self.dt_profile,
                    label="convolution_2",
                )
                for el in range(self.n_turns):
                    plt.plot(
                        self.time_array_profile[inter_turn][el],
                        cav_obs_list[inter_turn].induced_voltage[el],
                        ls="--",
                        label=f"section {inter_turn} turn {el}",
                    )
                plt.legend(loc="upper right")

                if inter_turn == self.n_stations - 1:
                    plt.show(block=True)
                else:
                    plt.show(block=False)


if __name__ == "__main__":
    indi = InducedVoltageResonatorPhysicsCR()
    indi.setUpB2(old_impl=False)
    # indi.setUpB2(old_impl=True)
    indi.setUpB3()
