# coding: utf8
# Copyright 2014-2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for RingAndRFTracker.drift() in trackers.tracker.py

The different implementation are "linear", "exact" and "legacy".

A benchmark is done by comparing the "exact" implementation from the expected
drift which is computed by effectively comparing the revolution period of
an arbitrary particle with the revolution period of the design particle.

:Authors: **Alexandre Lasheen**

"""

# General imports
import unittest

import numpy as np
from scipy.constants import c

from blond.beam.beam import Beam, Proton
from blond.input_parameters.rf_parameters import RFStation
# BLonD imports
from blond.input_parameters.ring import Ring
from blond.trackers.tracker import RingAndRFTracker

# Tests tolerance
absolute_tolerance = 0.    # 0. / 1e-16
relative_tolerance = 1e-9  # 1e-9 / 0.


def linear_drift(original_distribution_dE, eta_0, beta, energy, trev):

    drift_kick = trev * eta_0 * original_distribution_dE / (beta**2. * energy)

    return drift_kick


def legacy_drift(original_distribution_dE, eta_0, eta_1, eta_2, beta, energy,
                 trev):

    drift = trev * (
        1. / (1. - eta_0 * original_distribution_dE / (beta**2. * energy)
              - eta_1 * (original_distribution_dE / (beta**2. * energy))**2.
              - eta_2 * (original_distribution_dE / (beta**2. * energy))**3.) - 1.)

    return drift


def exact_drift(original_distribution_dE, alpha_0, alpha_1, alpha_2, beta,
                energy, trev):

    delta = np.sqrt(
        1. + 1 / beta**2. * (
            (original_distribution_dE / energy)**2.
            + 2. * (original_distribution_dE / energy))) - 1.

    drift = trev * (
        (1. + alpha_0 * delta
         + alpha_1 * delta**2.
         + alpha_2 * delta**3.) *
        (1. + (original_distribution_dE / energy)) / (1. + delta) - 1.)

    return drift


def expected_drift(original_distribution_dE, alpha_0, alpha_1, alpha_2,
                   energy, trev, circumference, mass_energy):

    particle_total_energy = original_distribution_dE + energy

    momentum = np.sqrt(energy**2. - mass_energy**2.)
    particle_momentum = np.sqrt(particle_total_energy**2. - mass_energy**2.)

    particle_dp = particle_momentum - momentum

    particle_beta = particle_momentum / particle_total_energy

    particle_circ = circumference * (
        1. + alpha_0 * particle_dp / momentum
        + alpha_1 * (particle_dp / momentum)**2.
        + alpha_2 * (particle_dp / momentum)**3.)

    particle_trev = particle_circ / (particle_beta * c)

    drift = particle_trev - trev

    return drift


class TestDrift(unittest.TestCase):

    # Using PSB as base for Simulation parameters -----------------------------
    # Bunch parameters
    N_b = 1e9                           # Intensity
    N_p = 50000                         # Macro-particles
    tau_0 = 0.4e-9                      # Initial bunch length, 4 sigma [s]

    # Machine and RF parameters
    C = 2 * np.pi * 25.                     # Machine circumference [m]
    Ek = 160e6                          # Kinetic energy [eV]
    gamma_t = 4.1                       # Transition gamma
    alpha_0 = 1. / gamma_t / gamma_t        # First order mom. comp. factor
    alpha_1 = 10 * alpha_0
    alpha_2 = 100 * alpha_0

    # Tracking details
    N_t = 2000                          # Number of turns to track

    # Run before every test
    def setUp(self):
        pass

    # Run after every test

    def tearDown(self):
        pass

    def test_simple(self):

        self.ring = Ring(self.C, self.alpha_0, self.Ek, Proton(), self.N_t,
                         synchronous_data_type='kinetic energy',
                         alpha_1=None, alpha_2=None)

        self.beam = Beam(self.ring, self.N_p, self.N_b)

        self.rf = RFStation(self.ring, 1, 0, np.pi)

        original_distribution_dt = np.zeros(self.beam.n_macroparticles)
        original_distribution_dE = np.linspace(
            -0.1 * self.beam.energy,
            0.1 * self.beam.energy,
            self.beam.n_macroparticles)

        self.beam.dt[:] = np.array(original_distribution_dt)
        self.beam.dE[:] = np.array(original_distribution_dE)

        self.long_tracker = RingAndRFTracker(
            self.rf, self.beam, solver='simple')

        for i in range(self.ring.n_turns):
            self.long_tracker.track()

        original_distribution_dt += self.ring.n_turns * linear_drift(
            original_distribution_dE, self.ring.eta_0[0, 0],
            self.ring.beta[0, 0], self.ring.energy[0, 0], self.ring.t_rev[0])

        np.testing.assert_allclose(self.beam.dt, original_distribution_dt,
                                   rtol=relative_tolerance,
                                   atol=absolute_tolerance)

    def test_legacy_order0(self):

        self.ring = Ring(self.C, self.alpha_0, self.Ek, Proton(), self.N_t,
                         synchronous_data_type='kinetic energy',
                         alpha_1=None, alpha_2=None)

        self.beam = Beam(self.ring, self.N_p, self.N_b)

        self.rf = RFStation(self.ring, 1, 0, np.pi)

        original_distribution_dt = np.zeros(self.beam.n_macroparticles)
        original_distribution_dE = np.linspace(
            -0.1 * self.beam.energy,
            0.1 * self.beam.energy,
            self.beam.n_macroparticles)

        self.beam.dt[:] = np.array(original_distribution_dt)
        self.beam.dE[:] = np.array(original_distribution_dE)

        self.long_tracker = RingAndRFTracker(
            self.rf, self.beam, solver='legacy')

        # Forcing usage of legacy
        self.long_tracker.solver = 'legacy'
        self.long_tracker.solver = self.long_tracker.solver.encode(
            encoding='utf_8')

        for i in range(self.ring.n_turns):
            self.long_tracker.track()

        original_distribution_dt += self.ring.n_turns * legacy_drift(
            original_distribution_dE, self.ring.eta_0[0, 0],
            self.ring.eta_1[0, 0], self.ring.eta_2[0, 0], self.ring.beta[0, 0],
            self.ring.energy[0, 0], self.ring.t_rev[0])

        np.testing.assert_allclose(self.beam.dt, original_distribution_dt,
                                   rtol=relative_tolerance,
                                   atol=absolute_tolerance)

    def test_legacy_order1(self):

        self.ring = Ring(self.C, self.alpha_0, self.Ek, Proton(), self.N_t,
                         synchronous_data_type='kinetic energy',
                         alpha_1=self.alpha_1, alpha_2=None)

        self.beam = Beam(self.ring, self.N_p, self.N_b)

        self.rf = RFStation(self.ring, 1, 0, np.pi)

        original_distribution_dt = np.zeros(self.beam.n_macroparticles)
        original_distribution_dE = np.linspace(
            -0.1 * self.beam.energy,
            0.1 * self.beam.energy,
            self.beam.n_macroparticles)

        self.beam.dt[:] = np.array(original_distribution_dt)
        self.beam.dE[:] = np.array(original_distribution_dE)

        self.long_tracker = RingAndRFTracker(
            self.rf, self.beam, solver='legacy')

        # Forcing usage of legacy
        self.long_tracker.solver = 'legacy'
        self.long_tracker.solver = self.long_tracker.solver.encode(
            encoding='utf_8')

        for i in range(self.ring.n_turns):
            self.long_tracker.track()

        original_distribution_dt += self.ring.n_turns * legacy_drift(
            original_distribution_dE, self.ring.eta_0[0, 0],
            self.ring.eta_1[0, 0], self.ring.eta_2[0, 0], self.ring.beta[0, 0],
            self.ring.energy[0, 0], self.ring.t_rev[0])

        np.testing.assert_allclose(self.beam.dt, original_distribution_dt,
                                   rtol=relative_tolerance,
                                   atol=absolute_tolerance)

    def test_legacy_order2(self):

        self.ring = Ring(self.C, self.alpha_0, self.Ek, Proton(), self.N_t,
                         synchronous_data_type='kinetic energy',
                         alpha_1=None, alpha_2=self.alpha_2)

        self.beam = Beam(self.ring, self.N_p, self.N_b)

        self.rf = RFStation(self.ring, 1, 0, np.pi)

        original_distribution_dt = np.zeros(self.beam.n_macroparticles)
        original_distribution_dE = np.linspace(
            -0.1 * self.beam.energy,
            0.1 * self.beam.energy,
            self.beam.n_macroparticles)

        self.beam.dt[:] = np.array(original_distribution_dt)
        self.beam.dE[:] = np.array(original_distribution_dE)

        self.long_tracker = RingAndRFTracker(
            self.rf, self.beam, solver='legacy')

        # Forcing usage of legacy
        self.long_tracker.solver = 'legacy'
        self.long_tracker.solver = self.long_tracker.solver.encode(
            encoding='utf_8')

        for i in range(self.ring.n_turns):
            self.long_tracker.track()

        original_distribution_dt += self.ring.n_turns * legacy_drift(
            original_distribution_dE, self.ring.eta_0[0, 0],
            self.ring.eta_1[0, 0], self.ring.eta_2[0, 0], self.ring.beta[0, 0],
            self.ring.energy[0, 0], self.ring.t_rev[0])

        np.testing.assert_allclose(self.beam.dt, original_distribution_dt,
                                   rtol=relative_tolerance,
                                   atol=absolute_tolerance)

    def test_legacy_order1and2(self):

        self.ring = Ring(self.C, self.alpha_0, self.Ek, Proton(), self.N_t,
                         synchronous_data_type='kinetic energy',
                         alpha_1=self.alpha_1, alpha_2=self.alpha_2)

        self.beam = Beam(self.ring, self.N_p, self.N_b)

        self.rf = RFStation(self.ring, 1, 0, np.pi)

        original_distribution_dt = np.zeros(self.beam.n_macroparticles)
        original_distribution_dE = np.linspace(
            -0.1 * self.beam.energy,
            0.1 * self.beam.energy,
            self.beam.n_macroparticles)

        self.beam.dt[:] = np.array(original_distribution_dt)
        self.beam.dE[:] = np.array(original_distribution_dE)

        self.long_tracker = RingAndRFTracker(
            self.rf, self.beam, solver='legacy')

        # Forcing usage of legacy
        self.long_tracker.solver = 'legacy'
        self.long_tracker.solver = self.long_tracker.solver.encode(
            encoding='utf_8')

        for i in range(self.ring.n_turns):
            self.long_tracker.track()

        original_distribution_dt += self.ring.n_turns * legacy_drift(
            original_distribution_dE, self.ring.eta_0[0, 0],
            self.ring.eta_1[0, 0], self.ring.eta_2[0, 0], self.ring.beta[0, 0],
            self.ring.energy[0, 0], self.ring.t_rev[0])

        np.testing.assert_allclose(self.beam.dt, original_distribution_dt,
                                   rtol=relative_tolerance,
                                   atol=absolute_tolerance)

    def test_exact_order0(self):

        self.ring = Ring(self.C, self.alpha_0, self.Ek, Proton(), self.N_t,
                         synchronous_data_type='kinetic energy',
                         alpha_1=None, alpha_2=None)

        self.beam = Beam(self.ring, self.N_p, self.N_b)

        self.rf = RFStation(self.ring, 1, 0, np.pi)

        original_distribution_dt = np.zeros(self.beam.n_macroparticles)
        original_distribution_dE = np.linspace(
            -0.1 * self.beam.energy,
            0.1 * self.beam.energy,
            self.beam.n_macroparticles)

        self.beam.dt[:] = np.array(original_distribution_dt)
        self.beam.dE[:] = np.array(original_distribution_dE)

        self.long_tracker = RingAndRFTracker(
            self.rf, self.beam, solver='exact')

        # Forcing usage of legacy
        self.long_tracker.solver = 'exact'
        self.long_tracker.solver = self.long_tracker.solver.encode(
            encoding='utf_8')

        for i in range(self.ring.n_turns):
            self.long_tracker.track()

        original_distribution_dt += self.ring.n_turns * exact_drift(
            original_distribution_dE, self.ring.alpha_0[0, 0],
            self.ring.alpha_1[0, 0], self.ring.alpha_2[0, 0],
            self.ring.beta[0, 0], self.ring.energy[0, 0], self.ring.t_rev[0])

        np.testing.assert_allclose(self.beam.dt, original_distribution_dt,
                                   rtol=relative_tolerance,
                                   atol=absolute_tolerance)

    def test_exact_order1(self):

        self.ring = Ring(self.C, self.alpha_0, self.Ek, Proton(), self.N_t,
                         synchronous_data_type='kinetic energy',
                         alpha_1=self.alpha_1, alpha_2=None)

        self.beam = Beam(self.ring, self.N_p, self.N_b)

        self.rf = RFStation(self.ring, 1, 0, np.pi)

        original_distribution_dt = np.zeros(self.beam.n_macroparticles)
        original_distribution_dE = np.linspace(
            -0.1 * self.beam.energy,
            0.1 * self.beam.energy,
            self.beam.n_macroparticles)

        self.beam.dt[:] = np.array(original_distribution_dt)
        self.beam.dE[:] = np.array(original_distribution_dE)

        self.long_tracker = RingAndRFTracker(
            self.rf, self.beam, solver='exact')

        # Forcing usage of legacy
        self.long_tracker.solver = 'exact'
        self.long_tracker.solver = self.long_tracker.solver.encode(
            encoding='utf_8')

        for i in range(self.ring.n_turns):
            self.long_tracker.track()

        original_distribution_dt += self.ring.n_turns * exact_drift(
            original_distribution_dE, self.ring.alpha_0[0, 0],
            self.ring.alpha_1[0, 0], self.ring.alpha_2[0, 0],
            self.ring.beta[0, 0], self.ring.energy[0, 0], self.ring.t_rev[0])

        np.testing.assert_allclose(self.beam.dt, original_distribution_dt,
                                   rtol=relative_tolerance,
                                   atol=absolute_tolerance)

    def test_exact_order2(self):

        self.ring = Ring(self.C, self.alpha_0, self.Ek, Proton(), self.N_t,
                         synchronous_data_type='kinetic energy',
                         alpha_1=None, alpha_2=self.alpha_2)

        self.beam = Beam(self.ring, self.N_p, self.N_b)

        self.rf = RFStation(self.ring, 1, 0, np.pi)

        original_distribution_dt = np.zeros(self.beam.n_macroparticles)
        original_distribution_dE = np.linspace(
            -0.1 * self.beam.energy,
            0.1 * self.beam.energy,
            self.beam.n_macroparticles)

        self.beam.dt[:] = np.array(original_distribution_dt)
        self.beam.dE[:] = np.array(original_distribution_dE)

        self.long_tracker = RingAndRFTracker(
            self.rf, self.beam, solver='exact')

        # Forcing usage of legacy
        self.long_tracker.solver = 'exact'
        self.long_tracker.solver = self.long_tracker.solver.encode(
            encoding='utf_8')

        for i in range(self.ring.n_turns):
            self.long_tracker.track()

        original_distribution_dt += self.ring.n_turns * exact_drift(
            original_distribution_dE, self.ring.alpha_0[0, 0],
            self.ring.alpha_1[0, 0], self.ring.alpha_2[0, 0],
            self.ring.beta[0, 0], self.ring.energy[0, 0], self.ring.t_rev[0])

        np.testing.assert_allclose(self.beam.dt, original_distribution_dt,
                                   rtol=relative_tolerance,
                                   atol=absolute_tolerance)

    def test_exact_order1and2(self):

        self.ring = Ring(self.C, self.alpha_0, self.Ek, Proton(), self.N_t,
                         synchronous_data_type='kinetic energy',
                         alpha_1=self.alpha_1, alpha_2=self.alpha_2)

        self.beam = Beam(self.ring, self.N_p, self.N_b)

        self.rf = RFStation(self.ring, 1, 0, np.pi)

        original_distribution_dt = np.zeros(self.beam.n_macroparticles)
        original_distribution_dE = np.linspace(
            -0.1 * self.beam.energy,
            0.1 * self.beam.energy,
            self.beam.n_macroparticles)

        self.beam.dt[:] = np.array(original_distribution_dt)
        self.beam.dE[:] = np.array(original_distribution_dE)

        self.long_tracker = RingAndRFTracker(
            self.rf, self.beam, solver='exact')

        # Forcing usage of legacy
        self.long_tracker.solver = 'exact'
        self.long_tracker.solver = self.long_tracker.solver.encode(
            encoding='utf_8')

        for i in range(self.ring.n_turns):
            self.long_tracker.track()

        original_distribution_dt += self.ring.n_turns * exact_drift(
            original_distribution_dE, self.ring.alpha_0[0, 0],
            self.ring.alpha_1[0, 0], self.ring.alpha_2[0, 0],
            self.ring.beta[0, 0], self.ring.energy[0, 0], self.ring.t_rev[0])

        np.testing.assert_allclose(self.beam.dt, original_distribution_dt,
                                   rtol=relative_tolerance,
                                   atol=absolute_tolerance)

    def test_exact_order1and2_vs_expectation(self):

        self.ring = Ring(self.C, self.alpha_0, self.Ek, Proton(), self.N_t,
                         synchronous_data_type='kinetic energy',
                         alpha_1=self.alpha_1, alpha_2=self.alpha_2)

        self.beam = Beam(self.ring, self.N_p, self.N_b)

        self.rf = RFStation(self.ring, 1, 0, np.pi)

        original_distribution_dt = np.zeros(self.beam.n_macroparticles)
        original_distribution_dE = np.linspace(
            -0.1 * self.beam.energy,
            0.1 * self.beam.energy,
            self.beam.n_macroparticles)

        self.beam.dt[:] = np.array(original_distribution_dt)
        self.beam.dE[:] = np.array(original_distribution_dE)

        self.long_tracker = RingAndRFTracker(
            self.rf, self.beam, solver='exact')

        # Forcing usage of legacy
        self.long_tracker.solver = 'exact'
        self.long_tracker.solver = self.long_tracker.solver.encode(
            encoding='utf_8')

        for i in range(self.ring.n_turns):
            self.long_tracker.track()

        original_distribution_dt += self.ring.n_turns * expected_drift(
            original_distribution_dE, self.ring.alpha_0[0, 0],
            self.ring.alpha_1[0, 0], self.ring.alpha_2[0, 0],
            self.ring.energy[0, 0], self.ring.t_rev[0],
            self.ring.ring_circumference, self.ring.Particle.mass)

        np.testing.assert_allclose(self.beam.dt, original_distribution_dt,
                                   rtol=relative_tolerance,
                                   atol=absolute_tolerance)


if __name__ == '__main__':

    unittest.main()
