import numpy as np
from _pytest import unittest

from blond import BiGaussian
from blond.legacy.blond2.beam.beam import Beam, MuPlus
from blond.legacy.blond2.beam.distributions import bigaussian
from blond.legacy.blond2.beam.profile import CutOptions, Profile
from blond.legacy.blond2.impedances.impedance import (
    InducedVoltageResonator,
    InducedVoltageTime,
)
from blond.legacy.blond2.impedances.impedance_sources import Resonators
from blond.legacy.blond2.input_parameters.rf_parameters import RFStation
from blond.legacy.blond2.input_parameters.ring import Ring


class InducdedVoltageResonator:
    def setUp(self):
        self.harmonic = 130  # TODO: double check
        self.voltage_per_rf_station = 5e6
        self.R_shunt = 52e6
        self.alpha_p = 8.986e-4
        self.energy = 31.68e7
        self.energy_gain_per_turn = 31.68e8

        self.n_turns = 5
        self.n_stations = 2
        self.n_section_lengths = np.array([5, 5])

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
        self.rf_station_list.append(
            RFStation(
                ring,
                self.harmonic,
                self.voltage_per_rf_station,
                0,
                n_rf=1,
                section_index=2,
            )
        )

        self.beam = Beam(
            ring, n_macroparticles=self.n_macroparticles, intensity=2e10
        )

        cut_options = CutOptions(
            cut_left=0,
            cut_right=self.rf_station_list[0].t_rf[0, 0],
            n_slices=5,
        )

        self.profile = Profile(self.beam, cut_options=cut_options)
        self.profile.track()
        self.profile.fwhm()

        self.resonator = Resonators(
            self.R_shunt,
            self.rf_station_list[0].omega_rf[0, 0] / 2 / np.pi,
            1e3,
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

        self.ind_volt_res_2 = InducedVoltageResonator(
            self.beam,
            self.profile,
            self.resonator,
            rf_station=self.rf_station_list[1],
            mtw_mode="time",
            time_decay_factor=1e-12,
            multi_turn_wake=True,
        )  # never release

        pass


if __name__ == "__main__":
    indi = InducdedVoltageResonator()
    indi.setUp()
