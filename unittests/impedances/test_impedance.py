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
import os
from blond.beam.profile import CutOptions, Profile, FitOptions
from blond.impedances.impedance import InducedVoltageFreq, InducedVoltageTime, InducedVoltageResonator, \
    TotalInducedVoltage
from blond.impedances.impedance_sources import Resonators
from blond.beam.beam import Beam, Proton
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.trackers.tracker import RingAndRFTracker


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


class TestInducedVoltageResonatorMultiTurnWake(unittest.TestCase):

    def setUp(self):

        mom_compaction = 4.68e-4
        circ = 5989.95
        self.n_stations = 1
        l_per_section = circ / self.n_stations
        self.n_turns = 3
        sectionlengths = np.full(self.n_stations, l_per_section)
        alpha_c = np.full(self.n_stations, mom_compaction)
        # program should be shape (n_stations, n_turns+1)
        energy_program = np.array([1 * 25e9, 1 * 25e9, 1.04 * 25e9, 1.05 * 25e9])

        self.ring = Ring(ring_length=sectionlengths, alpha_0=alpha_c,
                         particle=Proton(), n_turns=self.n_turns,
                         n_sections=self.n_stations, synchronous_data=energy_program)
        self.R_shunt = 11897424000
        self.Q_factor = 0.696e6
        fdet = -1320
        self.f_res = 1297263703
        self.harmonic = int(self.f_res * self.ring.t_rev[0])
        self.rf_station = RFStation(ring=self.ring, harmonic=[self.harmonic], voltage=[3e9], phi_rf_d=[0.0], n_rf=1)
        self.n_slices = 2 ** 9
        self.beam = Beam(ring=self.ring,
                         n_macroparticles=1001,
                         intensity=int(1e10))
        cut_options = CutOptions(cut_left=0, cut_right=2 * np.pi, n_slices=self.n_slices,
                                 cuts_unit='rad', rf_station=self.rf_station)
        self.profile = Profile(self.beam, cut_options,
                               FitOptions(fit_option='gaussian'))

        self.resonator = Resonators(self.R_shunt, self.f_res + fdet, self.Q_factor)

    def test_total_induced_voltage(self):

        ivr = InducedVoltageResonator(beam=self.beam, profile=self.profile,
                                      multi_turn_wake=True,
                                      resonators=self.resonator,
                                      rf_station=self.rf_station,
                                      time_decay_factor=0.00001)

        totalvolt = TotalInducedVoltage(beam=self.beam, profile=self.profile, induced_voltage_list=[ivr])
        longitudinal_tracker = RingAndRFTracker(rf_station=self.rf_station, beam=self.beam, profile=self.profile,
                                                total_induced_voltage=totalvolt,
                                                interpolation=True)
        induced_voltages = []
        for _ in range(self.n_turns):
            totalvolt.induced_voltage_sum()
            longitudinal_tracker.track()
            induced_voltages.append(longitudinal_tracker.totalInducedVoltage.induced_voltage)

        induced_voltages = np.array(induced_voltages)

        folder = os.path.abspath(os.path.dirname(__file__)) + "/data/"
        expected_voltages = np.load(folder + "mtw_resonator_induced_voltage.npy")
        np.testing.assert_allclose(induced_voltages, expected_voltages, rtol=1e-5, atol=1e-1)


if __name__ == '__main__':
    unittest.main()
