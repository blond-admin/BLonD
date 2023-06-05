# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for trackers.tracker.py

:Authors: **Konstantinos Iliakis**
"""

import unittest

import numpy as np

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, FitOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.llrf.rf_modulation import PhaseModulation as PMod
from blond.trackers.tracker import RingAndRFTracker


def orig_rf_volt_comp(tracker):
    """Function calculating the total, discretised RF voltage seen by the
    beam at a given turn. Requires a Profile object.

    """

    voltages = np.array([])
    omega_rf = np.array([])
    phi_rf = np.array([])

    for rf_system in range(tracker.rf_params.n_rf):
        voltages = np.append(voltages, tracker.rf_params.voltage[rf_system,
                                                                 tracker.counter[0]])
        omega_rf = np.append(omega_rf, tracker.rf_params.omega_rf[rf_system,
                                                                  tracker.counter[0]])
        phi_rf = np.append(phi_rf, tracker.rf_params.phi_rf[rf_system,
                                                            tracker.counter[0]])

    voltages = np.array(voltages, ndmin=2)
    omega_rf = np.array(omega_rf, ndmin=2)
    phi_rf = np.array(phi_rf, ndmin=2)

    # TODO: test with multiple harmonics, think about 800 MHz OTFB
    if tracker.cavityFB:
        rf_voltage = voltages[0, 0] * tracker.cavityFB.V_corr * \
            np.sin(omega_rf[0, 0] * tracker.profile.bin_centers +
                   phi_rf[0, 0] + tracker.cavityFB.phi_corr) + \
            np.sum(voltages.T[1:] * np.sin(omega_rf.T[1:] *
                                           tracker.profile.bin_centers + phi_rf.T[1:]), axis=0)
    else:
        rf_voltage = np.sum(voltages.T *
                            np.sin(omega_rf.T * tracker.profile.bin_centers + phi_rf.T), axis=0)

    return rf_voltage


class TestRfVoltageCalc(unittest.TestCase):
    # Simulation parameters -------------------------------------------------------
    # Bunch parameters
    N_b = 1e9           # Intensity
    N_p = 50000         # Macro-particles
    tau_0 = 0.4e-9          # Initial bunch length, 4 sigma [s]
    # Machine and RF parameters
    C = 26658.883        # Machine circumference [m]
    p_i = 450e9         # Synchronous momentum [eV/c]
    p_f = 460.005e9      # Synchronous momentum, final
    h = 35640            # Harmonic number
    V = 6e6                # RF voltage [V]
    dphi = 0             # Phase modulation/offset
    gamma_t = 55.759505  # Transition gamma
    alpha = 1. / gamma_t / gamma_t        # First order mom. comp. factor
    # Tracking details
    N_t = 2000           # Number of turns to track

    # Run before every test
    def setUp(self):
        self.ring = Ring(self.C, self.alpha, np.linspace(
            self.p_i, self.p_f, self.N_t + 1), Proton(), self.N_t)
        self.beam = Beam(self.ring, self.N_p, self.N_b)
        self.rf = RFStation(
            self.ring, [self.h], self.V * np.linspace(1, 1.1, self.N_t + 1), [self.dphi])
        bigaussian(self.ring, self.rf, self.beam,
                   self.tau_0 / 4, reinsertion=True, seed=1)
        self.profile = Profile(self.beam, CutOptions(n_slices=100, cut_left=0, cut_right=self.rf.t_rf[0, 0]),
                               FitOptions(fit_option='gaussian'))
        self.long_tracker = RingAndRFTracker(
            self.rf, self.beam, Profile=self.profile)

    # Run after every test

    def tearDown(self):
        pass

    def test_rf_voltage_calc_1(self):
        self.long_tracker.rf_voltage_calculation()
        orig_rf_voltage = orig_rf_volt_comp(self.long_tracker)
        np.testing.assert_almost_equal(
            self.long_tracker.rf_voltage, orig_rf_voltage, decimal=8)

    def test_rf_voltage_calc_2(self):
        for i in range(100):
            self.long_tracker.rf_voltage_calculation()
            orig_rf_voltage = orig_rf_volt_comp(self.long_tracker)
        np.testing.assert_almost_equal(
            self.long_tracker.rf_voltage, orig_rf_voltage, decimal=8)

    def test_rf_voltage_calc_3(self):
        for i in range(100):
            self.profile.track()
            self.long_tracker.track()
            self.long_tracker.rf_voltage_calculation()
            orig_rf_voltage = orig_rf_volt_comp(self.long_tracker)
        np.testing.assert_almost_equal(
            self.long_tracker.rf_voltage, orig_rf_voltage, decimal=8)


class CavityFB:
    V_corr = 0
    phi_corr = 0

    def __init__(self, V, phi):
        self.V_corr = V
        self.phi_corr = phi


class TestRfVoltageCalcWCavityFB(unittest.TestCase):
    # Simulation parameters -------------------------------------------------------
    # Bunch parameters
    N_b = 1e9           # Intensity
    N_p = 50000         # Macro-particles
    tau_0 = 0.4e-9          # Initial bunch length, 4 sigma [s]
    # Machine and RF parameters
    C = 26658.883        # Machine circumference [m]
    p_i = 450e9         # Synchronous momentum [eV/c]
    p_f = 460.005e9      # Synchronous momentum, final
    h = 35640            # Harmonic number
    V = 6e6                # RF voltage [V]
    dphi = 0             # Phase modulation/offset
    gamma_t = 55.759505  # Transition gamma
    alpha = 1. / gamma_t / gamma_t        # First order mom. comp. factor
    # Tracking details
    N_t = 2000           # Number of turns to track

    # Run before every test
    def setUp(self):
        self.ring = Ring(self.C, self.alpha, np.linspace(
            self.p_i, self.p_f, self.N_t + 1), Proton(), self.N_t)
        self.beam = Beam(self.ring, self.N_p, self.N_b)
        self.rf = RFStation(
            self.ring, [self.h], self.V * np.linspace(1, 1.1, self.N_t + 1), [self.dphi])
        bigaussian(self.ring, self.rf, self.beam,
                   self.tau_0 / 4, reinsertion=True, seed=1)
        self.profile = Profile(self.beam, CutOptions(n_slices=100, cut_left=0, cut_right=self.rf.t_rf[0, 0]),
                               FitOptions(fit_option='gaussian'))

        self.long_tracker = RingAndRFTracker(
            self.rf, self.beam, Profile=self.profile)

    # Run after every test

    def tearDown(self):
        pass

    def test_rf_voltage_calc_1(self):
        self.long_tracker.cavityFB = CavityFB(1.1, 1.2)
        self.long_tracker.rf_voltage_calculation()
        orig_rf_voltage = orig_rf_volt_comp(self.long_tracker)
        np.testing.assert_almost_equal(
            self.long_tracker.rf_voltage, orig_rf_voltage, decimal=8)

    def test_rf_voltage_calc_2(self):
        self.long_tracker.cavityFB = CavityFB(1.1, 1.2)
        for i in range(100):
            self.long_tracker.rf_voltage_calculation()
            orig_rf_voltage = orig_rf_volt_comp(self.long_tracker)
        np.testing.assert_almost_equal(
            self.long_tracker.rf_voltage, orig_rf_voltage, decimal=8)

    def test_rf_voltage_calc_3(self):
        self.long_tracker.cavityFB = CavityFB(1.1, 1.2)
        for i in range(100):
            self.profile.track()
            self.long_tracker.track()
            self.long_tracker.rf_voltage_calculation()
            orig_rf_voltage = orig_rf_volt_comp(self.long_tracker)
        np.testing.assert_almost_equal(
            self.long_tracker.rf_voltage, orig_rf_voltage, decimal=8)

    def test_rf_voltage_calc_4(self):
        self.long_tracker.cavityFB = CavityFB(np.linspace(
            1, 1.5, self.profile.n_slices), np.linspace(0.1, 0.5, self.profile.n_slices))
        self.long_tracker.rf_voltage_calculation()
        orig_rf_voltage = orig_rf_volt_comp(self.long_tracker)
        np.testing.assert_almost_equal(
            self.long_tracker.rf_voltage, orig_rf_voltage, decimal=8)

    def test_rf_voltage_calc_5(self):
        self.long_tracker.cavityFB = CavityFB(np.linspace(
            1, 1.5, self.profile.n_slices), np.linspace(0.1, 0.5, self.profile.n_slices))
        for i in range(100):
            self.long_tracker.rf_voltage_calculation()
            orig_rf_voltage = orig_rf_volt_comp(self.long_tracker)
        np.testing.assert_almost_equal(
            self.long_tracker.rf_voltage, orig_rf_voltage, decimal=8)

    def test_rf_voltage_calc_6(self):
        self.long_tracker.cavityFB = CavityFB(np.linspace(
            1, 1.5, self.profile.n_slices), np.linspace(0.1, 0.5, self.profile.n_slices))
        for i in range(100):
            self.profile.track()
            self.long_tracker.track()
            self.long_tracker.rf_voltage_calculation()
            orig_rf_voltage = orig_rf_volt_comp(self.long_tracker)
        np.testing.assert_almost_equal(
            self.long_tracker.rf_voltage, orig_rf_voltage, decimal=8)

    def test_phi_modulation(self):

        timebase = np.linspace(0, 0.2, 10000)
        freq = 2E3
        amp = np.pi
        offset = 0
        harmonic = self.h
        phiMod = PMod(timebase, freq, amp, offset, harmonic)

        self.rf = RFStation(
            self.ring, [self.h], self.V * np.linspace(1, 1.1, self.N_t + 1),
            [self.dphi], phi_modulation=phiMod)

        self.long_tracker = RingAndRFTracker(
            self.rf, self.beam, Profile=self.profile)

        for i in range(self.N_t):
            self.long_tracker.track()
            self.assertEqual(
                self.long_tracker.rf_params.phi_rf[:, self.long_tracker.counter[0] - 1],
                self.rf.phi_modulation[0][0][i], msg="""Phi modulation not added correctly in tracker""")


if __name__ == '__main__':

    unittest.main()
