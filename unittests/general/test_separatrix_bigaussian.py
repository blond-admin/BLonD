# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example input for simulation of acceleration
No intensity effects

Run as python TestSeparatrixBigaussian.py in console or via travis
'''

from __future__ import division, print_function

import unittest

import numpy as np

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, Profile
from blond.input_parameters.rf_parameters import RFStation, calculate_phi_s
from blond.input_parameters.ring import Ring
from blond.llrf.beam_feedback import BeamFeedback
from blond.trackers.utilities import separatrix


class TestSeparatrixBigaussian(unittest.TestCase):

    # Run before every test
    def setUp(self, negativeEta=True, acceleration=True, singleRF=True):
        # Defining parameters -------------------------------------------------
        # Bunch parameters
        N_b = 1.e9           # Intensity
        N_p = 100000         # Macro-particles
        tau_0 = 50.e-9          # Initial bunch length, 4 sigma [s]

        # Machine parameters
        C = 1567.5           # Machine circumference [m]
        p_1i = 3.e9         # Synchronous momentum [eV/c]
        p_1f = 30.0e9      # Synchronous momentum, final
        p_2f = 40.e9         # Synchronous momentum [eV/c]
        p_2i = 60.e9      # Synchronous momentum, final
        gamma_t = 31.6       # Transition gamma
        alpha_1 = -1. / gamma_t / gamma_t  # First order mom. comp. factor
        alpha_2 = 1. / gamma_t / gamma_t  # First order mom. comp. factor

        # RF parameters
        h = [9, 18]         # Harmonic number
        V = [1800.e3, 110.e3]  # RF voltage [V]
        phi_1 = [np.pi + 1., np.pi / 6 + 2.]  # Phase modulation/offset
        phi_2 = [1., np.pi / 6 + 2.]        # Phase modulation/offset
        N_t = 43857

        # Defining classes ----------------------------------------------------
        # Define general parameters
        if negativeEta:

            if acceleration:
                # eta < 0, acceleration
                general_params = Ring(C, alpha_1,
                                      np.linspace(p_1i, p_1f, N_t + 1), Proton(), N_t)
            elif not acceleration:
                # eta < 0, deceleration
                general_params = Ring(C, alpha_1,
                                      np.linspace(p_1f, p_1i, N_t + 1), Proton(), N_t)

            if singleRF:
                rf_params = RFStation(general_params, 9, 1.8e6, np.pi + 1.,
                                      n_rf=1)
            elif not singleRF:
                rf_params = RFStation(general_params, h, V, phi_1, n_rf=2)
                rf_params.phi_s = calculate_phi_s(
                    rf_params, Particle=general_params.Particle,
                    accelerating_systems='all')

        elif not negativeEta:

            if acceleration:
                # eta > 0, acceleration
                general_params = Ring(C, alpha_2,
                                      np.linspace(p_2i, p_2f, N_t + 1), Proton(), N_t)
            elif not acceleration:
                # eta > 0, deceleration
                general_params = Ring(C, alpha_2,
                                      np.linspace(p_2f, p_2i, N_t + 1), Proton(), N_t)

            if singleRF:
                rf_params = RFStation(general_params, 9, 1.8e6, 1., n_rf=1)
            elif not singleRF:
                rf_params = RFStation(general_params, h, V, phi_2, n_rf=2)
                rf_params.phi_s = calculate_phi_s(
                    rf_params, Particle=general_params.Particle,
                    accelerating_systems='all')

        # Define beam and distribution
        beam = Beam(general_params, N_p, N_b)
        bigaussian(general_params, rf_params, beam, tau_0 / 4,
                   seed=1234)
        # print(np.mean(beam.dt))
        slices = Profile(beam, CutOptions(cut_left=0.e-9, cut_right=600.e-9,
                                          n_slices=1000))
        slices.track()
        configuration = {'machine': 'LHC',
                         'PL_gain': 0.1 * general_params.t_rev[0]}
        PL = BeamFeedback(general_params, rf_params, slices, configuration)
        PL.beam_phase()

        # Quantities to be compared
        self.phi_s = rf_params.phi_s[0]
        self.phi_b = PL.phi_beam
        self.phi_rf = rf_params.phi_rf[0, 0]
        self.dE_sep = separatrix(general_params, rf_params,
                                 [-5.e-7, -3.e-7, 1.e-7, 3.e-7, 7.e-7, 9.e-7])

    # Run after every test

    def tearDown(self):

        del self.phi_s
        del self.phi_b
        del self.phi_rf
        del self.dE_sep

    # Actual tests: compare with expected values

    def test_1(self):

        self.setUp(negativeEta=True, acceleration=True, singleRF=True)

        self.assertAlmostEqual(self.phi_s, 3.4741, places=3,
                               msg='Failed test_1 in TestSeparatrixBigaussian on phi_s')
        self.assertAlmostEqual(self.phi_b, 3.4742, places=3,
                               msg='Failed test_1 in TestSeparatrixBigaussian on phi_b')
        self.assertAlmostEqual(self.phi_rf, 4.1416, places=3,
                               msg='Failed test_1 in TestSeparatrixBigaussian on phi_rf')
        self.assertAlmostEqual(self.dE_sep[0], 21061257.1819199, delta=1.e2,
                               msg='Failed test_1 in TestSeparatrixBigaussian on dE_sep[0]')
        self.assertAlmostEqual(self.dE_sep[1], 41984031.07781138, delta=1.e2,
                               msg='Failed test_1 in TestSeparatrixBigaussian on dE_sep[1]')
        self.assertAlmostEqual(self.dE_sep[2], 15008002.86609393, delta=1.e2,
                               msg='Failed test_1 in TestSeparatrixBigaussian on dE_sep[2]')
        self.assertAlmostEqual(self.dE_sep[3], 43084412.69304573, delta=1.e2,
                               msg='Failed test_1 in TestSeparatrixBigaussian on dE_sep[3]')
        self.assertAlmostEqual(self.dE_sep[5], 44050421.49141181, delta=1.e2,
                               msg='Failed test_1 in TestSeparatrixBigaussian on dE_sep[5]')

    def test_2(self):

        self.setUp(negativeEta=True, acceleration=True, singleRF=False)

        self.assertAlmostEqual(self.phi_s, 3.4152, places=3,
                               msg='Failed test_2 in TestSeparatrixBigaussian on phi_s')
        self.assertAlmostEqual(self.phi_b, 3.4153, places=3,
                               msg='Failed test_2 in TestSeparatrixBigaussian on phi_b')
        self.assertAlmostEqual(self.phi_rf, 4.1416, places=3,
                               msg='Failed test_2 in TestSeparatrixBigaussian on phi_rf')
        self.assertAlmostEqual(self.dE_sep[0], 19644213.74986242, delta=1.e2,
                               msg='Failed test_2 in TestSeparatrixBigaussian on dE_sep[0]')
        self.assertAlmostEqual(self.dE_sep[1], 40597543.48205686, delta=1.e2,
                               msg='Failed test_2 in TestSeparatrixBigaussian on dE_sep[1]')
        self.assertAlmostEqual(self.dE_sep[2], 12508689.67263455, delta=1.e2,
                               msg='Failed test_2 in TestSeparatrixBigaussian on dE_sep[2]')
        self.assertAlmostEqual(self.dE_sep[3], 41811463.57144009, delta=1.e2,
                               msg='Failed test_2 in TestSeparatrixBigaussian on dE_sep[3]')
        self.assertAlmostEqual(self.dE_sep[5], 42898898.52214498, delta=1.e2,
                               msg='Failed test_2 in TestSeparatrixBigaussian on dE_sep[5]')

    def test_3(self):

        self.setUp(negativeEta=True, acceleration=False, singleRF=True)

        self.assertAlmostEqual(self.phi_s, 2.7927, places=3,
                               msg='Failed test_3 in TestSeparatrixBigaussian on phi_s')
        self.assertAlmostEqual(self.phi_b, 2.7928, places=3,
                               msg='Failed test_3 in TestSeparatrixBigaussian on phi_b')
        self.assertAlmostEqual(self.phi_rf, 4.1416, places=3,
                               msg='Failed test_3 in TestSeparatrixBigaussian on phi_rf')
        self.assertAlmostEqual(self.dE_sep[0], 8.33180231e+08, delta=1.e3,
                               msg='Failed test_3 in TestSeparatrixBigaussian on dE_sep[0]')
        self.assertAlmostEqual(self.dE_sep[1], 5.38487881e+08, delta=1.e3,
                               msg='Failed test_3 in TestSeparatrixBigaussian on dE_sep[1]')
        self.assertAlmostEqual(self.dE_sep[2], 8.98061872e+08, delta=1.e3,
                               msg='Failed test_3 in TestSeparatrixBigaussian on dE_sep[2]')
        self.assertAlmostEqual(self.dE_sep[3], 2.42395281e+08, delta=1.e3,
                               msg='Failed test_3 in TestSeparatrixBigaussian on dE_sep[3]')
        self.assertAlmostEqual(self.dE_sep[4], 9.48437947e+08, delta=1.e3,
                               msg='Failed test_3 in TestSeparatrixBigaussian on dE_sep[4]')

    def test_4(self):

        self.setUp(negativeEta=True, acceleration=False, singleRF=False)

        self.assertAlmostEqual(self.phi_s, 2.8051, places=3,
                               msg='Failed test_4 in TestSeparatrixBigaussian on phi_s')
        self.assertAlmostEqual(self.phi_b, 2.8052, places=3,
                               msg='Failed test_4 in TestSeparatrixBigaussian on phi_b')
        self.assertAlmostEqual(self.phi_rf, 4.1416, places=3,
                               msg='Failed test_4 in TestSeparatrixBigaussian on phi_rf')
        self.assertAlmostEqual(self.dE_sep[0], 8.20683627e+08, delta=1.e3,
                               msg='Failed test_4 in TestSeparatrixBigaussian on dE_sep[0]')
        self.assertAlmostEqual(self.dE_sep[1], 5.11428511e+08, delta=1.e3,
                               msg='Failed test_4 in TestSeparatrixBigaussian on dE_sep[1]')
        self.assertAlmostEqual(self.dE_sep[2], 8.92977832e+08, delta=1.e3,
                               msg='Failed test_4 in TestSeparatrixBigaussian on dE_sep[2]')
        self.assertAlmostEqual(self.dE_sep[3], 1.53185755e+08, delta=1.e3,
                               msg='Failed test_4 in TestSeparatrixBigaussian on dE_sep[3]')
        self.assertAlmostEqual(self.dE_sep[4], 9.49799425e+08, delta=1.e3,
                               msg='Failed test_4 in TestSeparatrixBigaussian on dE_sep[4]')

    def test_5(self):

        self.setUp(negativeEta=False, acceleration=True, singleRF=True)

        self.assertAlmostEqual(self.phi_s, 3.3977, places=3,
                               msg='Failed test_5 in TestSeparatrixBigaussian on phi_s')
        self.assertAlmostEqual(self.phi_b, 3.3978, places=3,
                               msg='Failed test_5 in TestSeparatrixBigaussian on phi_b')
        self.assertAlmostEqual(self.phi_rf, 1.0000, places=3,
                               msg='Failed test_5 in TestSeparatrixBigaussian on phi_rf')
        self.assertAlmostEqual(self.dE_sep[0], 1.04542867e+09, delta=1.e3,
                               msg='Failed test_5 in TestSeparatrixBigaussian on dE_sep[0]')
        self.assertAlmostEqual(self.dE_sep[1], 2.34232667e+09, delta=1.e3,
                               msg='Failed test_5 in TestSeparatrixBigaussian on dE_sep[1]')
        self.assertAlmostEqual(self.dE_sep[2], 1.51776652e+09, delta=1.e3,
                               msg='Failed test_5 in TestSeparatrixBigaussian on dE_sep[2]')
        self.assertAlmostEqual(self.dE_sep[3], 2.20889395e+09, delta=1.e3,
                               msg='Failed test_5 in TestSeparatrixBigaussian on dE_sep[3]')
        self.assertAlmostEqual(self.dE_sep[4], 1.84686032e+09, delta=1.e3,
                               msg='Failed test_5 in TestSeparatrixBigaussian on dE_sep[4]')
        self.assertAlmostEqual(self.dE_sep[5], 2.04363122e+09, delta=1.e3,
                               msg='Failed test_5 in TestSeparatrixBigaussian on dE_sep[5]')

    def test_6(self):

        self.setUp(negativeEta=False, acceleration=True, singleRF=False)

        self.assertAlmostEqual(self.phi_s, 3.4529, places=3,
                               msg='Failed test_6 in TestSeparatrixBigaussian on phi_s')
        self.assertAlmostEqual(self.phi_b, 3.4531, places=3,
                               msg='Failed test_6 in TestSeparatrixBigaussian on phi_b')
        self.assertAlmostEqual(self.phi_rf, 1.0000, places=3,
                               msg='Failed test_6 in TestSeparatrixBigaussian on phi_rf')
        self.assertAlmostEqual(self.dE_sep[0], 1.14552779e+09, delta=1.e3,
                               msg='Failed test_6 in TestSeparatrixBigaussian on dE_sep[0]')
        self.assertAlmostEqual(self.dE_sep[1], 2.39697217e+09, delta=1.e3,
                               msg='Failed test_6 in TestSeparatrixBigaussian on dE_sep[1]')
        self.assertAlmostEqual(self.dE_sep[2], 1.56900748e+09, delta=1.e3,
                               msg='Failed test_6 in TestSeparatrixBigaussian on dE_sep[2]')
        self.assertAlmostEqual(self.dE_sep[3], 2.27477221e+09, delta=1.e3,
                               msg='Failed test_6 in TestSeparatrixBigaussian on dE_sep[3]')
        self.assertAlmostEqual(self.dE_sep[4], 1.87276617e+09, delta=1.e3,
                               msg='Failed test_6 in TestSeparatrixBigaussian on dE_sep[4]')
        self.assertAlmostEqual(self.dE_sep[5], 2.11772438e+09, delta=1.e3,
                               msg='Failed test_6 in TestSeparatrixBigaussian on dE_sep[5]')

    def test_7(self):

        self.setUp(negativeEta=False, acceleration=False, singleRF=True)

        self.assertAlmostEqual(self.phi_s, 2.8855, places=3,
                               msg='Failed test_7 in TestSeparatrixBigaussian on phi_s')
        self.assertAlmostEqual(self.phi_b, 2.8857, places=3,
                               msg='Failed test_7 in TestSeparatrixBigaussian on phi_b')
        self.assertAlmostEqual(self.phi_rf, 1.0000, places=3,
                               msg='Failed test_7 in TestSeparatrixBigaussian on phi_rf')
        self.assertAlmostEqual(self.dE_sep[0], 2.19987056e+09, delta=1.e3,
                               msg='Failed test_7 in TestSeparatrixBigaussian on dE_sep[0]')
        self.assertAlmostEqual(self.dE_sep[1], 1.88822988e+09, delta=1.e3,
                               msg='Failed test_7 in TestSeparatrixBigaussian on dE_sep[1]')
        self.assertAlmostEqual(self.dE_sep[2], 2.36696369e+09, delta=1.e3,
                               msg='Failed test_7 in TestSeparatrixBigaussian on dE_sep[2]')
        self.assertAlmostEqual(self.dE_sep[3], 1.51884903e+09, delta=1.e3,
                               msg='Failed test_7 in TestSeparatrixBigaussian on dE_sep[3]')
        self.assertAlmostEqual(self.dE_sep[4], 2.50023666e+09, delta=1.e3,
                               msg='Failed test_7 in TestSeparatrixBigaussian on dE_sep[4]')
        self.assertAlmostEqual(self.dE_sep[5], 9.70872687e+08, delta=1.e3,
                               msg='Failed test_7 in TestSeparatrixBigaussian on dE_sep[5]')

    def test_8(self):

        self.setUp(negativeEta=False, acceleration=False, singleRF=False)

        self.assertAlmostEqual(self.phi_s, 2.8869, places=3,
                               msg='Failed test_8 in TestSeparatrixBigaussian on phi_s')
        self.assertAlmostEqual(self.phi_b, 2.8870, places=3,
                               msg='Failed test_8 in TestSeparatrixBigaussian on phi_b')
        self.assertAlmostEqual(self.phi_rf, 1.0000, places=3,
                               msg='Failed test_8 in TestSeparatrixBigaussian on phi_rf')
        self.assertAlmostEqual(self.dE_sep[0], 2.23770717e+09, delta=1.e3,
                               msg='Failed test_8 in TestSeparatrixBigaussian on dE_sep[0]')
        self.assertAlmostEqual(self.dE_sep[1], 1.94371462e+09, delta=1.e3,
                               msg='Failed test_8 in TestSeparatrixBigaussian on dE_sep[1]')
        self.assertAlmostEqual(self.dE_sep[2], 2.38797174e+09, delta=1.e3,
                               msg='Failed test_8 in TestSeparatrixBigaussian on dE_sep[2]')
        self.assertAlmostEqual(self.dE_sep[3], 1.59999782e+09, delta=1.e3,
                               msg='Failed test_8 in TestSeparatrixBigaussian on dE_sep[3]')
        self.assertAlmostEqual(self.dE_sep[4], 2.50645531e+09, delta=1.e3,
                               msg='Failed test_8 in TestSeparatrixBigaussian on dE_sep[4]')
        self.assertAlmostEqual(self.dE_sep[5], 1.10010190e+09, delta=1.e3,
                               msg='Failed test_8 in TestSeparatrixBigaussian on dE_sep[5]')


if __name__ == '__main__':
    unittest.main()
