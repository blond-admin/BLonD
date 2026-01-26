from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from _pytest import unittest
from scipy.constants import e

from blond import Beam as beam_b3
from blond import (
    BiGaussian,
    DriftSimple,
    MagneticCyclePerTurn,
)
from blond import Ring as ring_b3
from blond import (
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    mu_plus,
)
from blond.handle_results.observables import RFStationInducedVoltageObservation
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


class InducdedVoltageResonator:
    def __init__(self, new_impl: bool = True):
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

        self.sigma_bunch = 5e-10
        self.bunch_offset = 3e-9

    def setUpB2(self):
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

        self.resonator = Resonators(
            self.R_shunt,
            self.rf_station_list[0].omega_rf[0, 0] / 2 / np.pi,
            self.Q_factor,
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

        # setup analytical solution
        import sys

        t_start = sys.float_info.min
        t_end = np.sum(ring.t_rev)
        self.time_axis = np.linspace(t_start, t_end, num=int(5e6))
        wake_kernel = nonperiodic_wake(
            self.time_axis,
            self.resonator.frequency_R[0],
            self.resonator.R_S[0],
            self.resonator.Q[0],
        )
        profiles = np.zeros_like(self.time_axis)
        profiles += gauss(self.time_axis, self.sigma_bunch, self.bunch_offset)
        for prof_ind in range(1, self.n_turns):
            profiles += gauss(
                self.time_axis,
                self.sigma_bunch,
                np.sum(self.rf_station_list[0].t_rev[0:prof_ind])
                + self.bunch_offset,
            )

        self.convolution_result = sig.convolve(profiles, wake_kernel)
        convol_res_time_axis = np.arange(len(self.convolution_result)) * (
            self.time_axis[1] - self.time_axis[0]
        )

        plt.clf()
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

        ax[0].plot(self.time_axis, profiles)
        ax[1].plot(
            self.time_axis, self.convolution_result[0 : len(self.time_axis)]
        )
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
            self.profile.bin_centers, self.sigma_bunch, self.bunch_offset
        )
        self.hist_y = self.profile.n_macroparticles

        tot_ind_volt = TotalInducedVoltage(
            self.beam, self.profile, [self.ind_volt_res_1]
        )
        self.time_array_profile = []
        save_voltage_array = []
        for trn_ind in range(self.n_turns):
            tot_ind_volt.induced_voltage_sum()
            self.ind_volt_res_1.rf_params.counter[0] += 1
            save_voltage_array.append(tot_ind_volt.induced_voltage)
            self.time_array_profile.append(
                self.profile.bin_centers
                + (
                    np.sum(self.ind_volt_res_1.rf_params.t_rev[0:trn_ind])
                    if trn_ind > 0
                    else 0
                )
            )

        self.dt = self.time_axis[1] - self.time_axis[0]
        plt.figure()
        plt.title("blond3")
        plt.plot(
            self.time_axis,
            -self.convolution_result[0 : len(self.time_axis)]
            * e
            / self.profile.bin_size
            * self.dt,
            label="convolution_2",
        )
        for el in range(len(save_voltage_array)):
            plt.plot(
                self.time_array_profile[el],
                save_voltage_array[el],
                ls="--",
                label=f"resonator turn {el}",
            )
        plt.legend()
        plt.show(block=False)

    def setUpB3(self):
        ring = ring_b3(
            circumference=np.sum(self.n_section_lengths),
            check_section_indices=False,
        )
        magnetic_cycle = MagneticCyclePerTurn(
            value_init=self.energy_array[0][0],
            values_after_turn=self.energy_array[0],
            in_unit="total energy",
            reference_particle=mu_plus,
        )
        one_turn_model = []
        profile_list = []
        t_rf = (
            magnetic_cycle.get_t_rev_init(
                ring.circumference,
                particle_type=mu_plus,
            )
            / self.harmonic
        )
        beam = beam_b3(
            intensity=self.n_macroparticles,
            particle_type=mu_plus,
            is_counter_rotating=False,
        )
        beam._dt = np.zeros(self.n_macroparticles)
        beam._dE = np.zeros(self.n_macroparticles)
        beam._flags = np.ones(self.n_macroparticles)
        beam._ids = np.arange(self.n_macroparticles)
        profile = Mock(StaticProfile)
        profile.cut_left = self.cut_left
        profile.cut_right = self.cut_right
        profile.hist_x = self.hist_x
        profile.hist_y = self.hist_y
        profile.hist_step = self.hist_step
        profile.active = True
        profile.hist_y_to_density_factor = 1 / self.beam.intensity
        profile.n_bins = len(profile.hist_y)

        local_res = res_b3(
            center_frequencies=1 / t_rf,
            quality_factors=self.Q_factor,
            shunt_impedances=self.R_shunt,
        )
        shc = SingleHarmonicRFStation(
            voltage=self.voltage_per_rf_station,
            phi_rf=0,
            harmonic=self.harmonic,
            local_wakefield=WakeField(
                sources=(local_res,),
                solver=MultiPassResonatorSolver(
                    decay_fraction_threshold=1e-12
                ),
                profile=profile,
            ),
            section_index=0,
        )
        one_turn_model.extend(
            [
                shc,
                DriftSimple(
                    orbit_length=np.sum(self.n_section_lengths),
                    section_index=0,
                    transition_gamma=1,
                ),
            ]
        )
        ring.add_elements(one_turn_model, reorder=False)

        sim = Simulation(
            ring=ring,
            magnetic_cycle=magnetic_cycle,
        )
        sim.print_one_turn_execution_order()

        ind_volt_obs = RFStationInducedVoltageObservation(
            rf_station=shc, each_turn_i=1
        )

        sim.run_simulation(
            beams=(beam,),
            observe=(ind_volt_obs,),
        )

        indu_voll = ind_volt_obs.induced_voltage
        plt.figure()
        plt.title("blond3")
        plt.plot(
            self.time_axis,
            -self.convolution_result[0 : len(self.time_axis)]
            * e
            / self.profile.bin_size
            * self.dt,
            label="convolution_2",
        )
        for el in range(5):
            plt.plot(
                self.time_array_profile[el],
                indu_voll[el],
                ls="--",
                label=f"resonator turn {el}",
            )
        plt.legend()
        plt.show()


if __name__ == "__main__":
    indi = InducdedVoltageResonator(new_impl=True)
    indi.setUpB2()
    indi.setUpB3()
