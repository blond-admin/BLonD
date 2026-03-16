import sys
import unittest
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from scipy.constants import c, e

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
from blond.generals.distributed.distributed_array import DistributedArray
from blond.handle_results.observables import RFStationInducedVoltageObservation
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


DEBUG_PLOTTING = False
SAVE_PLOTS = False

from cycler import cycler

plt.rcParams["axes.prop_cycle"] = cycler(
    color=["#0033a0", "#e15e32", "#2f2f2f", "#708238", "#6a4c93", "#c9a227"]
)
plt.rcParams["font.size"] = 12
plt.rcParams["lines.linewidth"] = 2.0


class InducedVoltageResonatorComparisonTest(unittest.TestCase):
    def setUp(self):
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
        self.n_stations = 2
        self.n_section_lengths = np.array([3, 5])

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
        self.bunch_offset = 3e-9

    def setUpB2(self, old_impl: bool = True):
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
        for station_ind in range(self.n_stations):
            rf_station_list.append(
                RFStation(
                    ring,
                    self.harmonic,
                    self.voltage_per_rf_station,
                    0,
                    n_rf=1,
                    section_index=station_ind + 1,
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
        for station_ind in range(self.n_stations):
            ind_volt_list.append(
                InducedVoltageResonator(
                    self.beam,
                    self.profile,
                    resonator,
                    rf_station=rf_station_list[station_ind],
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
        for station_ind in range(self.n_stations):
            beta_arrays.append(rf_station_list[station_ind].beta.tolist())
            section_length_arrays.append(
                rf_station_list[station_ind].section_length
            )
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

        section_time = (
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

        for prof_ind in range(0, self.n_turns):
            for inter_turn_ind in range(self.n_stations):
                if prof_ind == 0 and inter_turn_ind == 0:
                    continue
                profiles[inter_turn_ind] += gauss(
                    self.time_axis,
                    self.sigma_bunch,
                    np.sum(
                        section_time[
                            0 : prof_ind * self.n_stations + inter_turn_ind
                        ]
                    )
                    + self.bunch_offset,
                )
        self.dt_profile = self.time_axis[1] - self.time_axis[0]
        self.convolution_result = np.zeros_like(profiles)
        self.plot_normalisation_const = max(profiles[0])
        for inter_turn_ind in range(self.n_stations):
            self.convolution_result[inter_turn_ind] = sig.convolve(
                profiles[inter_turn_ind], wake_kernel
            )[0 : len(self.time_axis)]

            if DEBUG_PLOTTING:
                fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
                fig.suptitle(f"Section {inter_turn_ind}")
                ax[0].plot(
                    self.time_axis * 1e9,
                    profiles[inter_turn_ind] / self.plot_normalisation_const,
                )
                ax[0].set_ylabel("Beam Profiles [arb.]")
                ax[1].plot(
                    self.time_axis * 1e9,
                    -self.convolution_result[inter_turn_ind][
                        0 : len(self.time_axis)
                    ]
                    * e
                    / self.profile.bin_size
                    * self.dt_profile
                    / self.plot_normalisation_const,
                )
                ax[1].set_xlabel("time [ns]")
                ax[1].set_ylabel("Induced Voltage [arb.]")
                ax[1].set_xlim([-10, 180])
                if SAVE_PLOTS:
                    plt.savefig(
                        f"profiles_convolution_results_sec_{inter_turn_ind}.png",
                        dpi=400,
                        bbox_inches="tight",
                    )
                plt.show(block=False)

        self.profile.n_macroparticles = gauss(
            self.profile.bin_centers, self.sigma_bunch, self.bunch_offset
        )
        self.hist_y = self.profile.n_macroparticles

        self.time_array_profile = [[] for _ in range(self.n_stations)]
        save_voltage_array = [[] for _ in range(self.n_stations)]
        for trn_ind in range(self.n_turns):
            for inter_turn_ind in range(self.n_stations):
                ind_volt_list[inter_turn_ind].induced_voltage_generation()
                ind_volt_list[inter_turn_ind].rf_params.counter[0] += 1
                save_voltage_array[inter_turn_ind].append(
                    ind_volt_list[inter_turn_ind].induced_voltage[
                        : self.profile.n_slices
                    ]
                )
                self.time_array_profile[inter_turn_ind].append(
                    self.profile.bin_centers
                    + (
                        np.sum(
                            section_time[
                                0 : trn_ind * self.n_stations + inter_turn_ind
                            ]
                        )
                        if trn_ind > 0
                        else 0
                        if inter_turn_ind == 0
                        else np.sum(section_time[:inter_turn_ind])
                    )
                )

        if DEBUG_PLOTTING:
            for inter_turn_ind in range(self.n_stations):
                plt.figure()
                plt.plot(
                    self.time_axis * 1e9,
                    -self.convolution_result[inter_turn_ind][
                        0 : len(self.time_axis)
                    ]
                    * e
                    / self.profile.bin_size
                    * self.dt_profile
                    / self.plot_normalisation_const,
                    label="Analytical result",
                    alpha=0.6,
                )
                for el in range(self.n_turns):
                    plt.plot(
                        self.time_array_profile[inter_turn_ind][el] * 1e9,
                        save_voltage_array[inter_turn_ind][el]
                        / self.plot_normalisation_const,
                        ls="--",
                        label=f"turn {el}",
                    )
                plt.ylabel("Induced Voltage [arb.]")
                plt.xlabel("Time [ns]")
                plt.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.13),
                    ncol=3,
                    frameon=False,
                )
                plt.xlim([-10, 180])
                if SAVE_PLOTS:
                    plt.savefig(
                        f"blond_2_old_impl_{old_impl}_section_{inter_turn_ind}.png",
                        dpi=400,
                        bbox_inches="tight",
                        transparent=True,
                    )
                    if not old_impl:
                        plt.gca().tick_params(axis="y", labelleft=False)
                        plt.ylabel("")
                        plt.savefig(
                            f"blond_2_old_impl_{old_impl}_section_{inter_turn_ind}_no_leftlabel.png",
                            dpi=400,
                            bbox_inches="tight",
                            transparent=True,
                        )

                plt.show(block=False)
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
        beam = beam_b3(
            intensity=self.n_macroparticles,
            particle_type=mu_plus,
            is_counter_rotating=False,
        )
        beam._dt = DistributedArray(np.zeros(self.n_macroparticles))
        beam._dE = DistributedArray(np.zeros(self.n_macroparticles))
        beam._flags = DistributedArray(np.ones(self.n_macroparticles))
        beam._ids = DistributedArray(np.arange(self.n_macroparticles))
        profile = Mock(StaticProfile)
        profile.cut_left = self.cut_left
        profile.cut_right = self.cut_right
        profile.hist_x = self.hist_x
        profile.hist_y = self.hist_y
        profile.hist_step = self.hist_step
        profile.active = True
        profile.hist_y_to_density_factor = 1 / self.beam.intensity
        profile.n_bins = len(profile.hist_y)
        shc_list = []
        for sec_ind in range(self.n_stations):
            local_res = res_b3(
                center_frequencies=1 / self.t_rf,
                quality_factors=self.Q_factor,
                shunt_impedances=self.R_shunt,
            )
            shc_list.append(
                SingleHarmonicRFStation(
                    voltage=self.voltage_per_rf_station,
                    phi_rf=0,
                    harmonic=self.harmonic,
                    local_wakefield=WakeField(
                        sources=(local_res,),
                        solver=MultiPassResonatorSolver(
                            decay_fraction_threshold=1e-12
                        ),
                        profile=profile,
                        section_index=sec_ind,
                    ),
                    section_index=sec_ind,
                )
            )
            one_turn_model.extend(
                [
                    shc_list[-1],
                    DriftSimple(
                        orbit_length=self.n_section_lengths[sec_ind],
                        section_index=sec_ind,
                        momentum_compaction_factor=momentum_compaction_factor(
                            1
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

        sim.run_simulation(
            beams=(beam,),
            observe=tuple(ind_volt_obs_list),
        )
        if DEBUG_PLOTTING:
            for inter_turn in range(self.n_stations):
                plt.figure(f"b3_{inter_turn}")
                # plt.title(f"b3_{inter_turn}")
                plt.plot(
                    self.time_axis * 1e9,
                    -self.convolution_result[inter_turn][
                        0 : len(self.time_axis)
                    ]
                    * e
                    / self.profile.bin_size
                    * self.dt_profile
                    / self.plot_normalisation_const,
                    label="Analytical result",
                    alpha=0.6,
                )
                for el in range(self.n_turns):
                    plt.plot(
                        self.time_array_profile[inter_turn][el] * 1e9,
                        ind_volt_obs_list[inter_turn].induced_voltage[el]
                        / self.plot_normalisation_const,
                        ls="--",
                        label=f"turn {el}",
                    )
                # plt.ylabel("Induced Voltage [arb.]")
                plt.gca().tick_params(axis="y", labelleft=False)
                plt.xlabel("Time [ns]")
                plt.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.13),
                    ncol=3,
                    frameon=False,
                )
                plt.xlim([-10, 180])
                if SAVE_PLOTS:
                    plt.savefig(
                        f"blond_3_section_{inter_turn}.png",
                        dpi=400,
                        bbox_inches="tight",
                        transparent=True,
                    )

                if inter_turn == self.n_stations - 1:
                    plt.show(block=True)
                else:
                    plt.show(block=False)

        for inter_turn_ind in range(self.n_stations):
            for trn_ind in range(self.n_turns):
                conv_result = np.interp(
                    self.time_array_profile[inter_turn_ind][trn_ind],
                    self.time_axis,
                    self.convolution_result[inter_turn_ind],
                )
                np.testing.assert_allclose(
                    -conv_result * e / self.profile.bin_size * self.dt_profile,
                    ind_volt_obs_list[inter_turn_ind].induced_voltage[trn_ind],
                    atol=1e8,
                    rtol=1e-8,
                )

    def test_blond2_3(self):
        self.setUpB2(old_impl=False)
        # self.setUpB2(old_impl=True)  # This will give wrong results, leaving in as comparison
        self.setUpB3()  # some of the code relies on the matcher in b2, therefore blond2 has to run first
