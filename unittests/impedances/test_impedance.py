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


class TestInducedVoltageResonator_n_rf1(unittest.TestCase):

    def setUp(self):
        import os
        import numpy as np
        from blond.beam.beam import Beam, Proton
        from blond.beam.distributions import bigaussian
        from blond.beam.profile import CutOptions, FitOptions, Profile
        from blond.impedances.impedance_sources import Resonators
        from blond.input_parameters.rf_parameters import RFStation
        from blond.input_parameters.ring import Ring
        import math

        gamma_transition = 1 / np.sqrt(0.00192)  
        mom_compaction = 1 / gamma_transition ** 2
        circ = 6911.56

        ring = Ring(
            ring_length=circ,
            alpha_0=mom_compaction,
            synchronous_data=25.92e9,
            particle=Proton(),
            n_turns=2)
        
        self.rf_station = RFStation(
            ring=ring,
            harmonic=[4620],
            voltage=[0.9e6],
            phi_rf_d=[0.0],
            n_rf=1,
        )

        beam = Beam(
            ring=ring,
            n_macroparticles=1001,
            intensity=int(1e10),
        )
        bigaussian(
            ring=ring,
            rf_station=self.rf_station,
            beam=beam,
            sigma_dt=2e-9 / 4, seed=1,
        )
        self.beam = beam

        cut_options = CutOptions(cut_left=0, cut_right=2 * np.pi, n_slices=2 ** 8,
                                 rf_station=self.rf_station, cuts_unit='rad')
        self.profile = Profile(beam, cut_options,
                               FitOptions(fit_option='gaussian'))
        
        this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
        table = np.loadtxt(this_directory + './data/example_data.dat', comments='!')
        self.R_shunt = table[:, 2] * 10**6
        self.f_res = table[:, 0] * 10**9
        self.Q_factor = table[:, 1]

        self.resonator = Resonators(self.R_shunt, self.f_res, self.Q_factor)


    def test_init(self):
        ivr = InducedVoltageResonator(beam=self.beam, profile=self.profile, resonators=self.resonator)

    def test_init_mtw(self):
        ivr = InducedVoltageResonator(beam=self.beam, profile=self.profile, resonators=self.resonator,
                                      multi_turn_wake=True
                                      )
        
    def test_mtw_false_induced_volt(self):
        ivr = InducedVoltageResonator(beam=self.beam, profile=self.profile, resonators=self.resonator,
                                      multi_turn_wake=False)
        ivr.process()
        ivr.induced_voltage_1turn()
        my_array = ivr.induced_voltage
        self.assertTrue(np.any(my_array != 0.0))

    def test_multi_rf_station(self):
        ivr = InducedVoltageResonator(beam=self.beam, profile=self.profile, resonators=self.resonator, 
                                      rf_station=self.rf_station, multi_turn_wake=False)
        ivr.process()
        ivr.induced_voltage_1turn()
        my_array = ivr.induced_voltage
        self.assertTrue(np.any(my_array != 0.0))

    def test_mtw_true_induced_voltage(self):
        ivr = InducedVoltageResonator(beam=self.beam, profile=self.profile, resonators=self.resonator,
                                      rf_station=self.rf_station, multi_turn_wake=True)
        ivr.process()
        ivr.induced_voltage_mtw()
        my_array = ivr.induced_voltage
        self.assertTrue(np.any(my_array != 0.0))




class TestInducedVoltageResonator_nrf_2(unittest.TestCase):

    def setUp(self):
        import os
        import numpy as np
        from blond.beam.beam import Beam, Proton
        from blond.beam.distributions import bigaussian
        from blond.beam.profile import CutOptions, FitOptions, Profile
        from blond.impedances.impedance_sources import Resonators
        from blond.input_parameters.rf_parameters import RFStation
        from blond.input_parameters.ring import Ring
        import math 
        
        gamma_transition = 1 / np.sqrt(0.00192)  
        mom_compaction = 1 / gamma_transition ** 2
        circ = 6911.56
        n_rf_stations = 2
        l_per_section = circ/ n_rf_stations
        ring_per_section = Ring(ring_length=l_per_section, 
                                alpha_0=mom_compaction,
                        particle=Proton(), synchronous_data=25.92e9)

        sectionlengths = np.full(n_rf_stations, l_per_section)
        alpha_c = np.full(n_rf_stations, mom_compaction)
        n_turns = math.ceil(ring_per_section.n_turns /n_rf_stations)

        ring = Ring(ring_length=sectionlengths, alpha_0=alpha_c, particle=Proton(), n_turns=n_turns, n_sections=n_rf_stations,
                    synchronous_data=25.92e9)

        self.rf_station = RFStation(
            ring=ring,
            harmonic=[4620],
            voltage=[0.9e6],
            phi_rf_d=[0.0],
            n_rf=1,
        )
        
        beam = Beam(
            ring=ring,
            n_macroparticles=1001,
            intensity=int(1e10),
        )
        bigaussian(
            ring=ring,
            rf_station=self.rf_station,
            beam=beam,
            sigma_dt=2e-9 / 4, seed=1,
        )
        self.beam = beam
        cut_options = CutOptions(cut_left=0, cut_right=2 * np.pi, n_slices=2 ** 8,
                                 rf_station=self.rf_station, cuts_unit='rad')
        self.profile = Profile(beam, cut_options,
                               FitOptions(fit_option='gaussian'))
        
        this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
        table = np.loadtxt(this_directory + './data/example_data.dat', comments='!')
        self.R_shunt = table[:, 2] * 10**6
        self.f_res = table[:, 0] * 10
        self.Q_factor = table[:, 1]



        self.resonator = Resonators(self.R_shunt, self.f_res, self.Q_factor)


    def test_init(self):
        ivr = InducedVoltageResonator(beam=self.beam, profile=self.profile, resonators=self.resonator)

    def test_init_mtw(self):
        ivr = InducedVoltageResonator(beam=self.beam, profile=self.profile, resonators=self.resonator,
                                      multi_turn_wake=True
                                      )
        
    def test_mtw_false_induced_volt(self):
        ivr = InducedVoltageResonator(beam=self.beam, profile=self.profile, resonators=self.resonator,
                                      multi_turn_wake=False)
        ivr.process()
        ivr.induced_voltage_1turn()
        my_array = ivr.induced_voltage
        self.assertTrue(np.any(my_array != 0.0))

    def test_multi_rf_station(self):
        ivr = InducedVoltageResonator(beam=self.beam, profile=self.profile, resonators=self.resonator, 
                                      rf_station=self.rf_station, multi_turn_wake=False)
        ivr.process()
        ivr.induced_voltage_1turn()
        my_array = ivr.induced_voltage
        self.assertTrue(np.any(my_array != 0.0))

    def test_mtw_true_induced_voltage(self):
        ivr = InducedVoltageResonator(beam=self.beam, profile=self.profile, resonators=self.resonator,
                                      rf_station=self.rf_station, multi_turn_wake=True)
        ivr.process()
        ivr.induced_voltage_mtw()
        my_array = ivr.induced_voltage
        self.assertTrue(np.any(my_array != 0.0))







if __name__ == '__main__':
    unittest.main()
