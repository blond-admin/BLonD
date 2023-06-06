# coding: utf8
# Copyright 2014-2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Comparison of the different implementations using plots

:Authors: **Alexandre Lasheen**

"""

import matplotlib.pyplot as plt
# General imports
import numpy as np
# Drift test equations import
from test_drift import exact_drift, expected_drift, legacy_drift, linear_drift

from blond.beam.beam import Beam, Proton
from blond.input_parameters.rf_parameters import RFStation
# BLonD imports
from blond.input_parameters.ring import Ring
from blond.trackers.tracker import RingAndRFTracker


class CompareDrift:

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

    def run_simple(self):

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

        return self.beam.dt, original_distribution_dt, original_distribution_dE

    def run_legacy_order0(self):

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

        return self.beam.dt, original_distribution_dt, original_distribution_dE

    def run_legacy_order1(self):

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

        return self.beam.dt, original_distribution_dt, original_distribution_dE

    def run_legacy_order2(self):

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

        return self.beam.dt, original_distribution_dt, original_distribution_dE

    def run_legacy_order1and2(self):

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

        return self.beam.dt, original_distribution_dt, original_distribution_dE

    def run_exact_order0(self):

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

        return self.beam.dt, original_distribution_dt, original_distribution_dE

    def run_exact_order1(self):

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

        return self.beam.dt, original_distribution_dt, original_distribution_dE

    def run_exact_order2(self):

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

        return self.beam.dt, original_distribution_dt, original_distribution_dE

    def run_exact_order1and2(self):

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

        return self.beam.dt, original_distribution_dt, original_distribution_dE

    def run_expected_drift(self):

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

        original_distribution_dt += self.ring.n_turns * expected_drift(
            original_distribution_dE, self.ring.alpha_0[0, 0],
            self.ring.alpha_1[0, 0], self.ring.alpha_2[0, 0],
            self.ring.energy[0, 0], self.ring.t_rev[0],
            self.ring.ring_circumference, self.ring.Particle.mass)

        return original_distribution_dt, original_distribution_dE


if __name__ == '__main__':

    # Linear drift example
    (tracked_distrib_linear, expected_distrib_linear,
     original_distribution_dE) = CompareDrift().run_simple()

    plt.figure('linear')
    plt.clf()
    plt.plot(tracked_distrib_linear, original_distribution_dE,
             label='Linear implemented')
    plt.plot(expected_distrib_linear, original_distribution_dE,
             'k--', linewidth=0.5, label='Linear model')
    plt.xlabel('Time $\\Delta\\tau$ [s]')
    plt.ylabel('Energy $\\Delta E$ [eV]')
    plt.legend(loc='best')

    # Legacy drift example
    (tracked_distrib_legacy, expected_distrib_legacy,
     original_distribution_dE) = CompareDrift().run_legacy_order1and2()

    plt.figure('legacy order 1 and 2')
    plt.clf()
    plt.plot(tracked_distrib_legacy, original_distribution_dE,
             label='Legacy implemented')
    plt.plot(expected_distrib_legacy, original_distribution_dE,
             'k--', linewidth=0.5, label='Legacy model')
    plt.xlabel('Time $\\Delta\\tau$ [s]')
    plt.ylabel('Energy $\\Delta E$ [eV]')
    plt.legend(loc='best')

    # Exact drift example
    (tracked_distrib_exact, expected_distrib_exact,
     original_distribution_dE) = CompareDrift().run_exact_order1and2()

    plt.figure('exact order 1 and 2')
    plt.clf()
    plt.plot(tracked_distrib_exact, original_distribution_dE,
             label='Exact implementation')
    plt.plot(expected_distrib_exact, original_distribution_dE,
             'k--', linewidth=0.5, label='Exact model')
    plt.xlabel('Time $\\Delta\\tau$ [s]')
    plt.ylabel('Energy $\\Delta E$ [eV]')
    plt.legend(loc='best')

    # Comparison with expectation
    (expected_distrib,
     original_distribution_dE) = CompareDrift().run_expected_drift()

    plt.figure('Comparison between methods and expectation')
    plt.clf()
    plt.plot(tracked_distrib_linear, original_distribution_dE,
             label='Linear')
    plt.plot(tracked_distrib_legacy, original_distribution_dE,
             label='Legacy')
    plt.plot(tracked_distrib_exact, original_distribution_dE,
             label='Exact')
    plt.plot(expected_distrib, original_distribution_dE,
             'k--', linewidth=0.5, label='Expected')
    plt.xlabel('Time $\\Delta\\tau$ [s]')
    plt.ylabel('Energy $\\Delta E$ [eV]')
    plt.legend(loc='best')

    plt.show()
