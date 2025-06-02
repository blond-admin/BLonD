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
from blond.impedances.impedance import InducedVoltageFreq, InducedVoltageTime, \
    InducedVoltageResonator, TotalInducedVoltage, _InducedVoltage
from blond.impedances.impedance_sources import Resonators


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


class TestInducedVoltageFreq2(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.induced_voltage_freq = InducedVoltageFreq(Beam=None, Profile=None, impedance_source_list=None, frequency_resolution=None, multi_turn_wake=None, front_wake_length=None, RFParams=None, mtw_mode=None, use_regular_fft=None)
    @unittest.skip
    def test___init__(self):
        pass # calls __init__ in  self.setUp

    @unittest.skip
    def test_process(self):
        # TODO: implement test for `process`
        self.induced_voltage_freq.process()

    @unittest.skip
    def test_to_cpu(self):
        # TODO: implement test for `to_cpu`
        self.induced_voltage_freq.to_cpu(recursive=None)

    @unittest.skip
    def test_to_gpu(self):
        # TODO: implement test for `to_gpu`
        self.induced_voltage_freq.to_gpu(recursive=None)


class TestInducedVoltageResonator(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.induced_voltage_resonator = InducedVoltageResonator(Beam=None, Profile=None, Resonators=None, timeArray=None)
    @unittest.skip
    def test___init__(self):
        pass # calls __init__ in  self.setUp

    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.induced_voltage_time = InducedVoltageTime(Beam=None, Profile=None, wake_source_list=None, wake_length=None, multi_turn_wake=None, RFParams=None, mtw_mode=None, use_regular_fft=None)
    @unittest.skip
    def test___init__(self):
        pass # calls __init__ in  self.setUp

    @unittest.skip
    def test_process(self):
        # TODO: implement test for `process`
        self.induced_voltage_time.process()

    @unittest.skip
    def test_to_cpu(self):
        # TODO: implement test for `to_cpu`
        self.induced_voltage_time.to_cpu(recursive=None)

    @unittest.skip
    def test_to_gpu(self):
        # TODO: implement test for `to_gpu`
        self.induced_voltage_time.to_gpu(recursive=None)


class TestTotalInducedVoltage(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.total_induced_voltage = TotalInducedVoltage(Beam=None, Profile=None, induced_voltage_list=None)
    @unittest.skip
    def test___init__(self):
        pass # calls __init__ in  self.setUp

    @unittest.skip
    def test_to_cpu(self):
        # TODO: implement test for `to_cpu`
        self.total_induced_voltage.to_cpu(recursive=None)

    @unittest.skip
    def test_to_gpu(self):
        # TODO: implement test for `to_gpu`
        self.total_induced_voltage.to_gpu(recursive=None)

    @unittest.skip
    def test_track_ghosts_particles(self):
        # TODO: implement test for `track_ghosts_particles`
        self.total_induced_voltage.track_ghosts_particles(ghostBeam=None)


class Test_InducedVoltage(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.__induced_voltage = _InducedVoltage(Beam=None, Profile=None, frequency_resolution=None, wake_length=None, multi_turn_wake=None, mtw_mode=None, RFParams=None, use_regular_fft=None)
    @unittest.skip
    def test___init__(self):
        pass # calls __init__ in  self.setUp

    @unittest.skip
    def test__track(self):
        # TODO: implement test for `_track`
        self.__induced_voltage._track()

    @unittest.skip
    def test_mtw_mode(self):
        # TODO: implement test for `mtw_mode`
        self.__induced_voltage.mtw_mode(mtw_mode=None)

    @unittest.skip
    def test_process(self):
        # TODO: implement test for `process`
        self.__induced_voltage.process()

    @unittest.skip
    def test_shift_trev_freq(self):
        # TODO: implement test for `shift_trev_freq`
        self.__induced_voltage.shift_trev_freq()

if __name__ == '__main__':
    unittest.main()