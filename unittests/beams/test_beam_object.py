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
from scipy.constants import physical_constants

import blond.utils.exceptions as blExcept
# BLonD imports
# --------------
from blond.beam.beam import Beam, Electron, Particle, Proton
from blond.beam.distributions import matched_from_distribution_function
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.trackers.tracker import FullRingAndRF, RingAndRFTracker


class testParticleClass(unittest.TestCase):

    def setUp(self):
        self.test_particle = Particle(1, 2)

    def test_particle_attributes(self):
        for attribute in ['mass', 'charge', 'radius_cl', 'c_gamma', 'c_q']:
            self.assertTrue(hasattr(self.test_particle, attribute),
                            msg=f"Particle: no '{attribute}' attribute")

    def test_attribute_types(self):

        for attribute in ['mass', 'charge']:
            self.assertIsInstance(getattr(self.test_particle, attribute), float,
                                  msg=f"Particle: {attribute} is not a float")

    def test_negative_restmass_exception(self):
        with self.assertRaises(RuntimeError):
            Particle(-1, 2)


class testElectron(unittest.TestCase):

    def setUp(self):
        self.electron = Electron()

    def test_classical_electron_radius(self):
        self.assertAlmostEqual(self.electron.radius_cl,
                               physical_constants['classical electron radius'][0], delta=1e-24,
                               msg='Electron: wrong classical elctron radius')

    def test_Sand_radiation_constant(self):
        # value from S. Lee: Accelerator Physics, 2nd ed., eq (4.5)
        # convert from GeV^3 to eV^3
        self.assertAlmostEqual(self.electron.c_gamma, 8.846e-5 / (1e9)**3, delta=1e-35,
                               msg='Electron: wrong radiation constant')

    def test_quantum_radiation_constant(self):
        # value from A. Wolski: Beam Dynamics in High Energy Accelerators, p. 233
        self.assertAlmostEqual(self.electron.c_q, 3.832e-13, delta=1e-16,
                               msg='Electron: wrong quantum excitation constant')


class testProton(unittest.TestCase):

    def setUp(self):
        self.proton = Proton()

    def test_classical_proton_radius(self):
        # value from S. Lee: Accelerator Physics, 2nd ed., p. 560
        self.assertAlmostEqual(self.proton.radius_cl, 1.5346986e-18, delta=1e-24,
                               msg='Proton: wrong classical proton radius')

    def test_Sand_radiation_constant(self):
        # value from S. Lee: Accelerator Physics, 2nd ed., eq (4.5)
        # convert from GeV^3 to eV^3
        self.assertAlmostEqual(self.proton.c_gamma, 7.783e-18 / (1e9)**3, delta=1e-48,
                               msg='Proton: wrong radiation constant')


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
        alpha = 1. / gamma_t**2  # First order mom. comp. factor

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
        self.beam.dt = sigma_dt * numpy.random.randn(self.beam.n_macroparticles)
        self.beam.dE = sigma_dE * numpy.random.randn(self.beam.n_macroparticles)

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
        try:
            matched_from_distribution_function(self.beam,
                                               full_tracker,
                                               distribution_exponent=1.5,
                                               distribution_type='binomial',
                                               bunch_length=1.65e-9,
                                               bunch_length_fit='fwhm',
                                               distribution_variable='Hamiltonian')
        except TypeError as te:
            self.skipTest("Skipped because of known bug in deepcopy. Exception message %s"
                          % str(te))
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
        try:
            matched_from_distribution_function(self.beam,
                                               full_tracker,
                                               distribution_exponent=1.5,
                                               distribution_type='binomial',
                                               bunch_length=1.65e-9,
                                               bunch_length_fit='fwhm',
                                               distribution_variable='Hamiltonian')
        except TypeError as te:
            self.skipTest("Skipped because of known bug in deepcopy. Exception message %s"
                          % str(te))
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

        try:
            matched_from_distribution_function(self.beam,
                                               full_tracker,
                                               distribution_exponent=1.5,
                                               distribution_type='binomial',
                                               bunch_length=1.65e-9,
                                               bunch_length_fit='fwhm',
                                               distribution_variable='Hamiltonian')
        except TypeError as te:
            self.skipTest("Skipped because of known bug in deepcopy. Exception message %s"
                          % str(te))

        self.beam.losses_energy_cut(-3e8, 3e8)
        self.assertEqual(len(self.beam.id[self.beam.id == 0]), 0,
                         msg='Beam: Failed losses_energy_cut, first')

        self.beam.dE += 10e8
        self.beam.losses_energy_cut(-3e8, 3e8)
        self.assertEqual(len(self.beam.id[self.beam.id == 0]),
                         self.beam.n_macroparticles,
                         msg='Beam: Failed losses_energy_cut, second')

    def test_addition(self):
        np = numpy

        testdEs = np.linspace(-1E6, 1E6, 2000000)
        testdts = np.linspace(0, 10E-9, 2000000)
        self.beam.dE = testdEs
        self.beam.dt = testdts

        testdEs = np.linspace(-2E6, 2E6, 100000)
        testdts = np.linspace(-1E-9, 12E-9, 100000)

        self.beam.add_particles([testdts, testdEs])

        self.assertEqual(self.beam.n_macroparticles, 2100000,
                         msg="n_macroparticles not incremented correctly")

        testBeam = Beam(self.general_params, 200, 0)

        testBeam.id[:100] = 0

        self.beam.add_beam(testBeam)

        self.assertEqual(self.beam.id[2100000:2100100].tolist(), [0] * 100,
                         msg="particle ids not applied correctly")

        self.assertEqual(self.beam.n_macroparticles, 2100200,
                         msg="Added macroparticles not incremented n_macro correctly")

        self.beam += testBeam

        self.assertEqual(self.beam.n_macroparticles, 2100400,
                         msg="Added macroparticles not incremented n_macro correctly")

        self.beam += (testdts, testdEs)

        self.assertEqual(self.beam.n_macroparticles, 2200400,
                         msg="Added macroparticles not incremented n_macro correctly")

        self.assertEqual(-2E6, np.min(self.beam.dE),
                         msg="coordinates of added beam not used correctly")
        self.assertEqual(2E6, np.max(self.beam.dE),
                         msg="coordinates of added beam not used correctly")
        self.assertEqual(-1E-9, np.min(self.beam.dt),
                         msg="coordinates of added beam not used correctly")
        self.assertEqual(12E-9, np.max(self.beam.dt),
                         msg="coordinates of added beam not used correctly")

        with self.assertRaises(blExcept.ParticleAdditionError,
                               msg="""Unequal length time and energy should raise exception"""):

            self.beam += ([1, 2, 3], [4, 5])

        with self.assertRaises(blExcept.ParticleAdditionError,
                               msg="""Mising time/energy should raise exception"""):

            self.beam += ([1, 2, 3])

        with self.assertRaises(TypeError, msg='Wrong type should raise exception'):
            self.beam.add_beam(([1], [2]))


if __name__ == '__main__':

    unittest.main()
