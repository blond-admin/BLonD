import numpy as np
from _pytest import unittest

from blond import BiGaussian
from blond.legacy.blond2.beam.beam import Beam, MuPlus
from blond.legacy.blond2.beam.distributions import bigaussian
from blond.legacy.blond2.beam.profile import CutOptions, Profile
from blond.legacy.blond2.impedances.impedance import (
    InducedVoltageFreq,
    InducedVoltageResonator,
    InducedVoltageTime,
    TotalInducedVoltage,
)
from blond.legacy.blond2.impedances.impedance_sources import Resonators
from blond.legacy.blond2.input_parameters.rf_parameters import RFStation
from blond.legacy.blond2.input_parameters.ring import Ring


class InducdedVoltageResonator:
    def setUp(self):
        sigma_bunch = 5e-10
        bunch_offset = 3e-9
        self.harmonic = 10
        self.voltage_per_rf_station = 50e6
        self.R_shunt = 52e6
        self.alpha_p = -8.986e-4
        self.energy = 120e6
        self.energy_gain_per_turn = 50.68e6

        self.n_turns = 5
        self.n_stations = 1
        self.n_section_lengths = np.array([10])

        self.n_macroparticles = int(1e4)

        self.energy_array = np.linspace(
            self.energy,
            self.energy + self.energy_gain_per_turn * self.n_turns,
            self.n_stations * (self.n_turns + 1),
        )
        self.energy_array = np.reshape(
            self.energy_array, (self.n_stations, self.n_turns + 1)
        )

        ring = Ring(
            self.n_section_lengths,
            self.alpha_p,
            self.energy_array,
            MuPlus(),
            synchronous_data_type="total energy",
            n_turns=self.n_turns,
            n_sections=self.n_stations,
        )

        self.rf_station_list = []
        self.rf_station_list.append(
            RFStation(
                ring,
                self.harmonic,
                self.voltage_per_rf_station,
                0,
                n_rf=1,
                section_index=1,
            )
        )  # amazing indexing
        # self.rf_station_list.append(
        #     RFStation(
        #         ring,
        #         self.harmonic,
        #         self.voltage_per_rf_station,
        #         0,
        #         n_rf=1,
        #         section_index=2,
        #     )
        # )

        self.beam = Beam(
            ring,
            n_macroparticles=self.n_macroparticles,
            intensity=self.n_macroparticles,
        )

        cut_options = CutOptions(
            cut_left=0,
            cut_right=self.rf_station_list[0].t_rf[0, 0] * 2,
            n_slices=2**12,
        )

        self.profile = Profile(self.beam, cut_options=cut_options)
        self.profile.track()
        self.profile.fwhm()

        self.resonator = Resonators(
            self.R_shunt,
            self.rf_station_list[0].omega_rf[0, 0] / 2 / np.pi,
            2e1,
        )  # low Q for fast decay in small machine, although phasing will dominate

        self.ind_volt_res_1 = InducedVoltageResonator(
            self.beam,
            self.profile,
            self.resonator,
            rf_station=self.rf_station_list[0],
            mtw_mode="time",
            time_decay_factor=1e-12,
            multi_turn_wake=True,
        )  # never release

        assert len(self.ind_volt_res_1.time_array) == self.n_turns
        assert (
            len(self.ind_volt_res_1.time_array[0])
            == (self.n_turns + 1) * self.profile.n_slices
        )
        assert (
            len(self.ind_volt_res_1.time_array[1])
            == self.n_turns * self.profile.n_slices
        )
        assert (
            len(self.ind_volt_res_1.time_array[-1])
            == 2 * self.profile.n_slices
        )  # only this and next turn

        self.induced_voltage_time = InducedVoltageFreq(
            self.beam,
            self.profile,
            [self.resonator],
            multi_turn_wake=True,
            rf_station=self.rf_station_list[0],
            frequency_resolution=0.5 * ring.f_rev[0] / 10,
        )

        def gauss(x, widt, center):
            return (
                1
                / (widt * np.sqrt(2 * np.pi))
                * np.exp(-((x - center) ** 2) / (2 * widt**2))
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

        # setup analytical solution
        import sys

        t_start = sys.float_info.min
        t_end = np.sum(ring.t_rev)
        time_axis = np.linspace(t_start, t_end, num=int(5e6))
        wake_kernel = nonperiodic_wake(
            time_axis,
            self.resonator.frequency_R[0],
            self.resonator.R_S[0],
            self.resonator.Q[0],
        )
        profiles = np.zeros_like(time_axis)
        profiles += gauss(time_axis, sigma_bunch, bunch_offset)
        for prof_ind in range(1, self.n_turns):
            profiles += gauss(
                time_axis,
                sigma_bunch,
                np.sum(self.rf_station_list[0].t_rev[0:prof_ind])
                + bunch_offset,
            )

        import matplotlib.pyplot as plt

        # plt.clf()
        # plt.plot(profiles)
        # plt.show()
        import scipy.signal as sig

        convolution_result = sig.convolve(profiles, wake_kernel)
        convol_res_time_axis = np.arange(len(convolution_result)) * (
            time_axis[1] - time_axis[0]
        )

        plt.clf()
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

        ax[0].plot(time_axis, profiles)
        ax[1].plot(time_axis, convolution_result[0 : len(time_axis)])
        plt.show(block=False)

        # self.ind_volt_res_2 = InducedVoltageResonator(
        #     self.beam,
        #     self.profile,
        #     self.resonator,
        #     rf_station=self.rf_station_list[1],
        #     mtw_mode="time",
        #     time_decay_factor=1e-12,
        #     multi_turn_wake=True,
        # )  # never release

        self.profile.n_macroparticles = gauss(
            self.profile.bin_centers, sigma_bunch, bunch_offset
        )

        tot_ind_volt = TotalInducedVoltage(
            self.beam, self.profile, [self.ind_volt_res_1]
        )
        time_array_profile = []
        save_voltage_array = []
        for trn_ind in range(self.n_turns):
            tot_ind_volt.induced_voltage_sum()
            self.ind_volt_res_1.rf_params.counter[0] += 1
            save_voltage_array.append(tot_ind_volt.induced_voltage)
            time_array_profile.append(
                self.profile.bin_centers
                + (
                    np.sum(self.ind_volt_res_1.rf_params.t_rev[0:trn_ind])
                    if trn_ind > 0
                    else 0
                )
            )

        from scipy.constants import e

        dt = time_axis[1] - time_axis[0]
        plt.clf()
        plt.plot(
            time_axis,
            -convolution_result[0 : len(time_axis)]
            * e
            / self.profile.bin_size
            * dt,
            label="convolution_2",
        )
        for el in range(len(save_voltage_array)):
            plt.plot(
                time_array_profile[el],
                save_voltage_array[el],
                ls="--",
                label=f"resonator turn {el}",
            )
        # plt.plot(time_axis, -convolution_result[0 : len(time_axis)] * e, label="convolution")
        plt.legend()
        plt.show()

        pass


if __name__ == "__main__":
    indi = InducdedVoltageResonator()
    indi.setUp()
