# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Unit-tests for the Beam class.

Run as python testBeamObject.py in console or via travis
'''

# General imports
# -----------------
from __future__ import division, print_function
import unittest
import numpy

# BLonD imports
# --------------
from beam.beam import Proton
from input_parameters.ring import Ring
from input_parameters.rf_parameters import RFStation
from beam.beam import Beam
from beam.distributions import matched_from_distribution_function
from trackers.tracker import FullRingAndRF, RingAndRFTracker


class testBeamClass(unittest.TestCase):

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
        alpha = 1./gamma_t**2  # First order mom. comp. factor

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

        self.assertIsInstance(self.beam.Particle.mass, float,
                              msg='Beam: mass is not a float')
        self.assertIsInstance(self.beam.Particle.charge, float,
                              msg='Beam: charge is not an float')
        self.assertIsInstance(self.beam.beta, float,
                              msg='Beam: beta is not a float')
        self.assertIsInstance(self.beam.gamma, float,
                              msg='Beam: gamma is not a float')
        self.assertIsInstance(self.beam.energy, float,
                              msg='Beam: energy is not a float')
        self.assertIsInstance(self.beam.momentum, float,
                              msg='Beam: momentum is not a float')
        self.assertIsInstance(self.beam.mean_dt, float,
                              msg='Beam: mean_dt is not a float')
        self.assertIsInstance(self.beam.mean_dE, float,
                              msg='Beam: mean_dE is not a float')
        self.assertIsInstance(self.beam.sigma_dt, float,
                              msg='Beam: sigma_dt is not a float')
        self.assertIsInstance(self.beam.sigma_dE, float,
                              msg='Beam: sigma_dE is not a float')
        self.assertIsInstance(self.beam.intensity, float,
                              msg='Beam: intensity is not a float')
        self.assertIsInstance(self.beam.n_macroparticles, int,
                              msg='Beam: n_macroparticles is not an int')
        self.assertIsInstance(self.beam.ratio, float,
                              msg='Beam: ratio is not a float')
        self.assertIsInstance(self.beam.id, numpy.ndarray,
                              msg='Beam: id is not a numpy.array')
        self.assertIn('int', type(self.beam.id[0]).__name__,
                              msg='Beam: id array does not contain int')
        self.assertIsInstance(self.beam.n_macroparticles_lost, int,
                              msg='Beam: n_macroparticles_lost is not an int')
        self.assertIsInstance(self.beam.n_macroparticles_alive, int,
                              msg='Beam: n_macroparticles_alive is not an int')
        self.assertIsInstance(self.beam.dt, numpy.ndarray,
                              msg='Beam: dt is not a numpy.array')
        self.assertIsInstance(self.beam.dE, numpy.ndarray,
                              msg='Beam: dE is not a numpy.array')
        self.assertIn('float', type(self.beam.dt[0]).__name__,
                              msg='Beam: dt does not contain float')
        self.assertIn('float', type(self.beam.dE[0]).__name__,
                              msg='Beam: dE does not contain float')

    def test_beam_statistic(self):

        sigma_dt = 1.
        sigma_dE = 1.
        self.beam.dt = sigma_dt*numpy.random.randn(self.beam.n_macroparticles)
        self.beam.dE = sigma_dE*numpy.random.randn(self.beam.n_macroparticles)

        self.beam.statistics()

        self.assertAlmostEqual(self.beam.sigma_dt, sigma_dt, delta=1e-2,
                               msg='Beam: Failed statistic sigma_dt')
        self.assertAlmostEqual(self.beam.sigma_dE, sigma_dE, delta=1e-2,
                               msg='Beam: Failed statistic sigma_dE')
        self.assertAlmostEqual(self.beam.mean_dt, 0., delta=1e-2,
                               msg='Beam: Failed statistic mean_dt')
        self.assertAlmostEqual(self.beam.mean_dE, 0., delta=1e-2,
                               msg='Beam: Failed statistic mean_dE')

    def test_losses_separatrix(self):

        longitudinal_tracker = RingAndRFTracker(self.rf_params, self.beam)
        full_tracker = FullRingAndRF([longitudinal_tracker])
        matched_from_distribution_function(self.beam,
                                           full_tracker,
                                           distribution_exponent=1.5,
                                           distribution_type='binomial',
                                           bunch_length=1.65e-9,
                                           bunch_length_fit='fwhm',
                                           distribution_variable='Hamiltonian')

        self.beam.losses_separatrix(self.general_params, self.rf_params)
        self.assertEqual(len(self.beam.id[self.beam.id == 0]), 0,
                         msg='Beam: Failed losses_sepatrix, first')

        self.beam.dE += 10e8
        self.beam.losses_separatrix(self.general_params, self.rf_params)
        self.assertEqual(len(self.beam.id[self.beam.id == 0]),
                         self.beam.n_macroparticles,
                         msg='Beam: Failed losses_sepatrix, second')

    def test_losses_longitudinal_cut(self):

        longitudinal_tracker = RingAndRFTracker(self.rf_params, self.beam)
        full_tracker = FullRingAndRF([longitudinal_tracker])
        matched_from_distribution_function(self.beam,
                                           full_tracker,
                                           distribution_exponent=1.5,
                                           distribution_type='binomial',
                                           bunch_length=1.65e-9,
                                           bunch_length_fit='fwhm',
                                           distribution_variable='Hamiltonian')

        self.beam.losses_longitudinal_cut(0., 5e-9)
        self.assertEqual(len(self.beam.id[self.beam.id == 0]), 0,
                         msg='Beam: Failed losses_longitudinal_cut, first')

        self.beam.dt += 10e-9
        self.beam.losses_longitudinal_cut(0., 5e-9)
        self.assertEqual(len(self.beam.id[self.beam.id == 0]),
                         self.beam.n_macroparticles,
                         msg='Beam: Failed losses_longitudinal_cut, second')

    def test_losses_energy_cut(self):

        longitudinal_tracker = RingAndRFTracker(self.rf_params, self.beam)
        full_tracker = FullRingAndRF([longitudinal_tracker])
        matched_from_distribution_function(self.beam,
                                           full_tracker,
                                           distribution_exponent=1.5,
                                           distribution_type='binomial',
                                           bunch_length=1.65e-9,
                                           bunch_length_fit='fwhm',
                                           distribution_variable='Hamiltonian')

        self.beam.losses_energy_cut(-3e8, 3e8)
        self.assertEqual(len(self.beam.id[self.beam.id == 0]), 0,
                         msg='Beam: Failed losses_energy_cut, first')

        self.beam.dE += 10e8
        self.beam.losses_energy_cut(-3e8, 3e8)
        self.assertEqual(len(self.beam.id[self.beam.id == 0]),
                         self.beam.n_macroparticles,
                         msg='Beam: Failed losses_energy_cut, second')


if __name__ == '__main__':

    unittest.main()
