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
Run as python testRFParamsObject.py in console or via travis
:Authors: **Joel Repond**
'''

# General imports
# -----------------
from __future__ import division, print_function
import unittest
import numpy

# BLonD imports
# --------------
from input_parameters.ring import Ring
from input_parameters.rf_parameters import RFStation
from beam.beam import Beam, Proton
#from beam.distributions import matched_from_distribution_function
#from trackers.tracker import FullRingAndRF, RingAndRFTracker

class testRFParamClass(unittest.TestCase):
    
    # Run before every test
    def setUp(self):
        
        # Bunch parameters
        # -----------------

        N_turn = 200
        N_b = 1e9 #  Intensity
        N_p = int(2e6) #  Macro-particles

        # Machine parameters
        # --------------------
        C = 6911.5038 #  Machine circumference [m]
        p = 450e9 #  Synchronous momentum [eV/c]
        gamma_t = 17.95142852 #  Transition gamma
        alpha = 1./gamma_t**2 #  First order mom. comp. factor



        # Define general parameters
        # --------------------------
        self.general_params = Ring(C, alpha, p, Proton(), N_turn)


        # Define beam
        # ------------
        self.beam = Beam(self.general_params, N_p, N_b)
        
        # Define RF section
        # -----------------
        self.rf_params = RFStation(self.general_params, [4620], [7e6], [0.])


    # Run after every test
    def tearDown(self):

        del self.general_params
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


    def test_RFSectionParameters_eta_tracking(self):

        # To be written
        pass

    def test_rf_parameters_calculate_Q_s(self):
        
        # To be written
        pass


    def test_rf_parameters_calculate_phi_s(self):

        # To be written
        pass


if __name__ == '__main__':

    unittest.main()

