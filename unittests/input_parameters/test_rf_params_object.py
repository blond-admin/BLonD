# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Unit-tests for the RFSectionParameters class.
Run as python test_rf_params_object.py in console or via travis
:Authors: **Joel Repond**, **Markus Schwarz**
'''

from __future__ import division, print_function

import unittest

import numpy

from blond.beam.beam import Beam, Electron, Proton
from blond.input_parameters.rf_parameters import RFStation, calculate_phi_s
from blond.input_parameters.ring import Ring
from blond.llrf.rf_modulation import PhaseModulation as PMod


class testRFParamClass(unittest.TestCase):

    # Run before every test
    def setUp(self):

        # Bunch parameters
        # -----------------

        N_turn = 200
        N_b = 1e9  # Intensity
        N_p = int(2e6)  # Macro-particles

        # Machine parameters
        # --------------------
        C = 6911.5038  # Machine circumference [m]
        p = 450e9  # Synchronous momentum [eV/c]
        gamma_t = 17.95142852  # Transition gamma
        alpha = 1. / gamma_t**2  # First order mom. comp. factor

        # Define general parameters
        # --------------------------
        self.ring = Ring(C, alpha, p, Proton(), N_turn)

        # Define beam
        # ------------
        self.beam = Beam(self.ring, N_p, N_b)

        # Define RF section
        # -----------------
        self.rf_params = RFStation(self.ring, [4620], [7e6], [0.])

    # Run after every test

    def tearDown(self):

        del self.ring
        del self.beam
        del self.rf_params

    def test_variables_types(self):

        self.assertIsInstance(self.rf_params.n_turns, int,
                              msg='RFSectionParameters: n_turn is not an int')
        self.assertIsInstance(self.rf_params.ring_circumference, float,
                              msg='RFSectionParameters: ring_circumference is not a float')
        self.assertIsInstance(self.rf_params.section_length, float,
                              msg='RFSectionParameters: section_length is not a float')
        self.assertIsInstance(self.rf_params.length_ratio, float,
                              msg='RFSectionParameters: length_ratio is not a float')

        self.assertIsInstance(self.rf_params.t_rev, numpy.ndarray,
                              msg='RFSectionParameters: t_rev is not a numpy.ndarray')
        self.assertIn('float', type(self.rf_params.t_rev[0]).__name__,
                      msg='RFSectionParameters: t_rev array does not contain float')

        self.assertIsInstance(self.rf_params.momentum, numpy.ndarray,
                              msg='RFSectionParameters: momentum is not a numpy.ndarray')
        self.assertIn('float', type(self.rf_params.momentum[0]).__name__,
                      msg='RFSectionParameters: momentum array does not contain float')

        self.assertIsInstance(self.rf_params.beta, numpy.ndarray,
                              msg='RFSectionParameters: beta is not a numpy.ndarray')
        self.assertIn('float', type(self.rf_params.beta[0]).__name__,
                      msg='RFSectionParameters: beta array does not contain float')

        self.assertIsInstance(self.rf_params.gamma, numpy.ndarray,
                              msg='RFSectionParameters: gamma is not a numpy.ndarray')
        self.assertIn('float', type(self.rf_params.gamma[0]).__name__,
                      msg='RFSectionParameters: gamma array does not contain float')

        self.assertIsInstance(self.rf_params.energy, numpy.ndarray,
                              msg='RFSectionParameters: energy is not a numpy.ndarray')
        self.assertIn('float', type(self.rf_params.energy[0]).__name__,
                      msg='RFSectionParameters: energy array does not contain float')

        self.assertIsInstance(self.rf_params.delta_E, numpy.ndarray,
                              msg='RFSectionParameters: delta_E is not a numpy.ndarray')
        self.assertIn('float', type(self.rf_params.delta_E[0]).__name__,
                      msg='RFSectionParameters: delta_E array does not contain float')

        self.assertIsInstance(self.rf_params.alpha_order, int,
                              msg='RFSectionParameters: alpha_order is not an int')

        self.assertIsInstance(self.rf_params.eta_0, numpy.ndarray,
                              msg='RFSectionParameters: eta_0 is not a numpy.ndarray')
        self.assertIn('float', type(self.rf_params.eta_0[0]).__name__,
                      msg='RFSectionParameters: eta_0 array does not contain float')

        self.assertIsInstance(self.rf_params.n_rf, int,
                              msg='RFSectionParameters: n_rf is not an int')

        self.assertIsInstance(self.rf_params.eta_0, numpy.ndarray,
                              msg='RFSectionParameters: eta_0 is not a numpy.ndarray')
        self.assertIn('float', type(self.rf_params.eta_0[0]).__name__,
                      msg='RFSectionParameters: eta_0 array does not contain float')

        self.assertIsInstance(self.rf_params.harmonic, numpy.ndarray,
                              msg='RFSectionParameters: harmonic is not a numpy.ndarray')
        self.assertIn('float', type(self.rf_params.harmonic[0][0]).__name__,
                      msg='RFSectionParameters: harmonic array does not contain float')

        self.assertIsInstance(self.rf_params.voltage, numpy.ndarray,
                              msg='RFSectionParameters: voltage is not a numpy.ndarray')
        self.assertIn('float', type(self.rf_params.voltage[0][0]).__name__,
                      msg='RFSectionParameters: voltage array does not contain float')

        self.assertIsInstance(self.rf_params.phi_rf_d, numpy.ndarray,
                              msg='RFSectionParameters: phi_rf_d is not a numpy.ndarray')
        self.assertIn('float', type(self.rf_params.phi_rf_d[0][0]).__name__,
                      msg='RFSectionParameters: phi_rf_d array does not contain float')

#        self.assertIsInstance(self.rf_params.phi_noise, numpy.ndarray,
#                      msg='RFSectionParameters: phi_noise is not a numpy.ndarray')
#        self.assertIn('float', type(self.rf_params.phi_noise[0][0]).__name__,
#                              msg='RFSectionParameters: phi_noise array does not contain float')

        self.assertIsInstance(self.rf_params.omega_rf, numpy.ndarray,
                              msg='RFSectionParameters: omega_rf is not a numpy.ndarray')
        self.assertIn('float', type(self.rf_params.omega_rf[0][0]).__name__,
                      msg='RFSectionParameters: omega_rf array does not contain float')

    def test_phi_modulation(self):

        timebase = numpy.linspace(0, 1, 10000)
        frequency = 2E2
        amplitude = numpy.pi / 2
        offset = 0
        harmonic = 36961

        with self.assertRaises(ValueError,
                               msg="""Incorrect harmonic should
                               raise ValueError"""):

            modulator1 = PMod(timebase, frequency, amplitude, offset, harmonic)
            self.rf_params = RFStation(self.ring, [4620, 36960],
                                       [7e6, 1E6], [0., 0],
                                       phi_modulation=modulator1, n_rf=2)

        harmonic = 36960
        modulator1 = PMod(timebase, frequency, amplitude, offset, harmonic)

        with self.assertRaises(TypeError,
                               msg="""Treatment in RFStation requires
                                PhaseModulation not be iterable"""):
            iter(modulator1)

        self.rf_params = RFStation(self.ring, [4620, 36960],
                                   [7e6, 1E6], [0., 0],
                                   phi_modulation=modulator1, n_rf=2)

        self.rf_params = RFStation(self.ring, [4620, 36960],
                                   [7e6, 1E6], [0., 0],
                                   phi_modulation=[modulator1] * 2, n_rf=2)

        with self.assertRaises(RuntimeError,
                               msg="""Two systems with the same harmonic
                               should return RuntimeError when using 
                               PhaseModulation"""):
            self.rf_params = RFStation(self.ring, [4620, 36960, 36960],
                                       [7e6, 1E6, 0], [0., 0, 0],
                                       phi_modulation=[modulator1] * 2, n_rf=3)

        modulator2 = PMod(timebase, frequency * 2, amplitude / 2, offset, harmonic)

        self.rf_params = RFStation(self.ring, [4620, 36960],
                                   [7e6, 1E6], [0., 0],
                                   phi_modulation=[modulator1, modulator2],
                                   n_rf=2)

    def test_RFSectionParameters_eta_tracking(self):

        # To be written
        pass

    def test_rf_parameters_calculate_Q_s(self):

        # To be written
        pass

    def test_rf_parameters_calculate_phi_s(self):

        self.assertEqual(calculate_phi_s(self.rf_params, Particle=Proton())[0], numpy.pi,
                         msg="Wrong phi_s for Proton")
        self.assertEqual(calculate_phi_s(self.rf_params, Particle=Electron())[0], 0.0,
                         msg="Wrong phi_s for Electron")

    # Tests of empty RF station
    def test_rf_parameters_is_empty_station(self):

        # create empty RF station
        rf_params = RFStation(self.ring, [4620], [0], [0.])
        self.assertTrue(rf_params.empty)

    def test_rf_parameters_no_phi_s_for_empty_station(self):

        # create empty RF station
        rf_params = RFStation(self.ring, [4620], [0], [0.])
        self.assertFalse(hasattr(rf_params, 'phi_s'))

    def test_rf_parameters_no_Q_s_for_empty_station(self):

        # create empty RF station
        rf_params = RFStation(self.ring, [4620], [0], [0.])
        self.assertFalse(hasattr(rf_params, 'Q_s'))

    def test_rf_parameters_no_omegaS0_s_for_empty_station(self):

        # create empty RF station
        rf_params = RFStation(self.ring, [4620], [0], [0.])
        self.assertFalse(hasattr(rf_params, 'omega_s0'))


if __name__ == '__main__':

    unittest.main()
