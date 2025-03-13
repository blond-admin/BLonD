# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for impedances.impedance

:Authors: **Markus Schwarz**
"""

import unittest

import numpy as np
from blond.beam.profile import CutOptions, Profile
from blond.impedances.impedance import InducedVoltageFreq, InducedVoltageTime, InducedVoltageResonator
from blond.impedances.impedance_sources import Resonators
from blond.impedances.impedance import TotalInducedVoltage


class TestInducedVoltageFreq(unittest.TestCase):

    def setUp(self):
        # Nyquist frequency 1.6 GHz; frequency spacing 50 MHz
        self.profile = Profile(None,
                               cut_options=CutOptions(cut_left=0, cut_right=5e-9, n_slices=16))
        self.impedance_source = Resonators([4.5e6], [200.222e6], [200])

    def test_default_wake_length(self):
        test_object = InducedVoltageFreq(
            None, self.profile, [self.impedance_source])

        np.testing.assert_allclose(
            test_object.wake_length,
            self.profile.bin_size * self.profile.n_slices)

    def test_default_frequency_resolution(self):
        test_object = InducedVoltageFreq(
            None, self.profile, [self.impedance_source])

        np.testing.assert_allclose(test_object.frequency_resolution,
                                   1 / (self.profile.bin_size * self.profile.n_slices))

    def test_default_n_rfft(self):
        test_object = InducedVoltageFreq(
            None, self.profile, [self.impedance_source])

        np.testing.assert_equal(test_object.n_fft, 16)

    def test_frequency_resolution_input(self):
        test_object = InducedVoltageFreq(
            None, self.profile, [self.impedance_source],
            frequency_resolution=3e6)

        np.testing.assert_allclose(test_object.frequency_resolution_input, 3e6)

    def test_frequency_resolution_n_induced_voltage(self):
        test_object = InducedVoltageFreq(
            None, self.profile, [self.impedance_source],
            frequency_resolution=3e6)

        np.testing.assert_equal(test_object.n_induced_voltage, 1067)

    def test_frequency_resolution_n_fft(self):
        test_object = InducedVoltageFreq(
            None, self.profile, [self.impedance_source],
            frequency_resolution=3e6)

        np.testing.assert_equal(test_object.n_fft, 1080)

    def test_frequency_resolution_frequency_resolution(self):
        test_object = InducedVoltageFreq(
            None, self.profile, [self.impedance_source],
            frequency_resolution=3e6)

        np.testing.assert_allclose(
            test_object.frequency_resolution,
            test_object.freq[1] - test_object.freq[0])


class TestInducedVoltageTime(unittest.TestCase):

    def setUp(self):
        # Nyquist frequency 1.6 GHz; frequency spacing 50 MHz
        self.profile = Profile(None,
                               cut_options=CutOptions(cut_left=0, cut_right=5e-9, n_slices=16))
        self.impedance_source = Resonators([4.5e6], [200.222e6], [200])

    def test_default_wake_length(self):
        test_object = InducedVoltageTime(
            None, self.profile, [self.impedance_source])

        np.testing.assert_allclose(
            test_object.wake_length,
            self.profile.bin_size * self.profile.n_slices)

    def test_default_frequency_resolution(self):
        test_object = InducedVoltageTime(
            None, self.profile, [self.impedance_source])

        np.testing.assert_allclose(test_object.frequency_resolution,
                                   1 / (self.profile.bin_size * 2 * test_object.n_induced_voltage))

    def test_default_n_rfft(self):
        test_object = InducedVoltageTime(
            None, self.profile, [self.impedance_source])

        np.testing.assert_equal(test_object.n_fft, 2 * 16)

    def test_wake_length_input(self):
        test_object = InducedVoltageTime(
            None, self.profile, [self.impedance_source],
            wake_length=11e-9)

        np.testing.assert_allclose(test_object.wake_length_input, 11e-9)


class TestInducedVoltageResonator_multi_turn_wake(unittest.TestCase):

    def setUp(self):
        import os
        import numpy as np
        from blond.beam.beam import Beam, Proton
        from blond.beam.profile import CutOptions, FitOptions, Profile
        from blond.impedances.impedance_sources import Resonators
        from blond.input_parameters.rf_parameters import RFStation
        from blond.input_parameters.ring import Ring
        from blond.impedances.impedance import TotalInducedVoltage, InducedVoltageResonator
        from blond.trackers.tracker import RingAndRFTracker

        gamma_transition = 1 / np.sqrt(0.00192)
        mom_compaction = 1 / gamma_transition ** 2
        circ = 6911.56
        n_rf_stations = 2
        l_per_section = circ / n_rf_stations
        self.n_turns = 2
        sectionlengths = np.full(n_rf_stations, l_per_section)
        alpha_c = np.full(n_rf_stations, mom_compaction)
        energy_program = np.array([[1 * 25e9, 1 * 25e9, 1.04 * 25e9],
                                   [1 * 25e9, 1.01 * 25e9, 1.05 * 25e9]])

        ring = Ring(ring_length=sectionlengths, alpha_0=alpha_c,
                    particle=Proton(), n_turns=self.n_turns,
                    n_sections=n_rf_stations, synchronous_data=energy_program)
        self.rf_station = RFStation(ring=ring, harmonic=[4620], voltage=[300e6], phi_rf_d=[0.0], n_rf=1)

        self.beam = Beam(ring=ring,
                         n_macroparticles=1001,
                         intensity=int(1e10))
        self.n_slices = 2 ** 9
        cut_options = CutOptions(cut_left=0, cut_right=2 * np.pi, n_slices=self.n_slices,
                                 cuts_unit='rad', rf_station=self.rf_station)
        self.profile = Profile(self.beam, cut_options,
                               FitOptions(fit_option='gaussian'))

        self.R_shunt = 11897424000
        self.Q_factor = 696000.0
        fdet = -1320
        self.f_res = 1297263703.3482404
        self.resonator = Resonators(self.R_shunt, self.f_res + fdet, self.Q_factor)

    def test_total_induced_voltage(self):
        from blond.trackers.tracker import RingAndRFTracker
        buckets = 1
        potential_min_cav = self.rf_station.phi_s[0] / self.rf_station.omega_rf[0, 0]
        min_index = np.abs(self.profile.bin_centers - potential_min_cav).argmin()
        self.timeArray = []
        for turn_ind in range(self.n_turns):
            self.timeArray = np.append(self.timeArray,
                                       self.rf_station.t_rev[turn_ind] * turn_ind +
                                       np.linspace(self.profile.bin_centers[0],
                                                   self.profile.bin_centers[-1] + 2 * (
                                                           self.profile.bin_centers[min_index] -
                                                           self.profile.bin_centers[0]),
                                                   self.n_slices * buckets + 2 * min_index))

        ivr = InducedVoltageResonator(beam=self.beam, profile=self.profile, time_array=self.timeArray,
                                        multi_turn_wake=True,
                                        resonators=self.resonator,
                                        rf_station=self.rf_station,
                                        wake_length=len(self.timeArray) * self.profile.bin_size)
                                        #array_length=int(len(self.timeArray) / self.n_turns))
        ivr.induced_voltage_mtw()

if __name__ == '__main__':
    unittest.main()
