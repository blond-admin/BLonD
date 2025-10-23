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
from blond.impedances.impedance import InducedVoltageFreq, InducedVoltageTime
from blond.impedances.impedance_sources import Resonators


class TestInducedVoltageFreq(unittest.TestCase):

    def setUp(self):

        # Nyquist frequency 1.6 GHz; frequency spacing 50 MHz
        self.profile = Profile(None,
                               CutOptions=CutOptions(cut_left=0, cut_right=5e-9, n_slices=16))
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
                               CutOptions=CutOptions(cut_left=0, cut_right=5e-9, n_slices=16))
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


if __name__ == '__main__':

    unittest.main()
