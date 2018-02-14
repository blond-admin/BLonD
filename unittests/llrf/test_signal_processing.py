# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for llrf.filters

:Authors: **Helga Timko**
"""

import unittest
import numpy as np
from llrf.signal_processing import moving_average, modulator
from llrf.signal_processing import polar_to_cartesian, cartesian_to_polar
from llrf.signal_processing import comb_filter, low_pass_filter
from llrf.signal_processing import rf_beam_current


class TestIQ(unittest.TestCase):

    # Run before every test
    def setUp(self, f_rf=200.1e6, T_s=5e-10, n=1000):

        self.f_rf = f_rf  # initial frequency in Hz
        self.T_s = T_s  # sampling time
        self.n = n  # number of points

    # Run after every test
    def tearDown(self):

        del self.f_rf
        del self.T_s
        del self.n

    def test_1(self):

        # Define signal in range (-pi, pi)
        phases = np.pi*(np.fmod(2*np.arange(self.n)*self.f_rf*self.T_s, 2) - 1)
        signal = np.cos(phases) + 1j*np.sin(phases)
        # From IQ to polar
        amplitude, phase = cartesian_to_polar(signal)

        # Drop some digits to avoid rounding errors
        amplitude = np.around(amplitude, 12)
        phase = np.around(phase, 12)
        phases = np.around(phases, 12)
        self.assertSequenceEqual(amplitude.tolist(), np.ones(self.n).tolist(),
            msg="In TestIQ test_1, amplitude is not correct")
        self.assertSequenceEqual(phase.tolist(), phases.tolist(),
            msg="In TestIQ test_1, phase is not correct")

    def test_2(self):

        # Define signal in range (-pi, pi)
        phase = np.pi*(np.fmod(2*np.arange(self.n)*self.f_rf*self.T_s, 2) - 1)
        amplitude = np.ones(self.n)
        # From polar to IQ
        signal = polar_to_cartesian(amplitude, phase)

        # Drop some digits to avoid rounding errors
        signal_real = np.around(signal.real, 12)
        signal_imag = np.around(signal.imag, 12)
        theor_real = np.around(np.cos(phase), 12)  # what it should be
        theor_imag = np.around(np.sin(phase), 12)  # what it should be
        self.assertSequenceEqual(signal_real.tolist(), theor_real.tolist(),
            msg="In TestIQ test_2, real part is not correct")
        self.assertSequenceEqual(signal_imag.tolist(), theor_imag.tolist(),
            msg="In TestIQ test_2, imaginary part is not correct")

    def test_3(self):

        # Define signal in range (-pi, pi)
        phase = np.pi*(np.fmod(2*np.arange(self.n)*self.f_rf*self.T_s, 2) - 1)
        amplitude = np.ones(self.n)
        # Forwards and backwards transform
        signal = polar_to_cartesian(amplitude, phase)
        amplitude_new, phase_new = cartesian_to_polar(signal)

        # Drop some digits to avoid rounding errors
        phase = np.around(phase, 11)
        amplitude = np.around(amplitude, 11)
        amplitude_new = np.around(amplitude_new, 11)
        phase_new = np.around(phase_new, 11)
        self.assertSequenceEqual(phase.tolist(), phase_new.tolist(),
                                 msg="In TestIQ test_3, phase is not correct")
        self.assertSequenceEqual(amplitude.tolist(), amplitude_new.tolist(),
            msg="In TestIQ test_3, amplitude is not correct")

    def test_4(self):

        # Define signal in range (-pi, pi)
        phase = np.pi*(np.fmod(2*np.arange(self.n)*self.f_rf*self.T_s, 2) - 1)
        signal = np.cos(phase) + 1j*np.sin(phase)
        # Forwards and backwards transform
        amplitude, phase = cartesian_to_polar(signal)
        signal_new = polar_to_cartesian(amplitude, phase)

        # Drop some digits to avoid rounding errors
        signal_real = np.around(signal.real, 11)
        signal_imag = np.around(signal.imag, 11)
        signal_real_2 = np.around(np.real(signal_new), 11)
        signal_imag_2 = np.around(np.imag(signal_new), 11)
        self.assertSequenceEqual(signal_real.tolist(), signal_real_2.tolist(),
            msg="In TestIQ test_4, real part is not correct")
        self.assertSequenceEqual(signal_imag.tolist(), signal_imag_2.tolist(),
            msg="In TestIQ test_4, imaginary part is not correct")


class TestModulator(unittest.TestCase):

    def setUp(self, f_rf=200.1e6, f_0=200.222e6, T_s=5e-10, n=1000):
        self.f_rf = f_rf    # initial frequency in Hz
        self.f_0 = f_0      # final frequency in Hz
        self.T_s = T_s      # sampling time
        self.n = n          # number of points

    def test_v1(self):

        # Forwards and backwards transformation of a sine wave
        signal = np.cos(2*np.pi*np.arange(self.n)*self.f_rf*self.T_s) \
            + 1j*np.sin(2*np.pi*np.arange(self.n)*self.f_rf*self.T_s)
        signal_1 = modulator(signal, self.f_rf, self.f_0, self.T_s)
        signal_2 = modulator(signal_1, self.f_0, self.f_rf, self.T_s)

        # Drop some digits to avoid rounding errors
        signal = np.around(signal, 12)
        signal_2 = np.around(signal_2, 12)
        self.assertSequenceEqual(signal.tolist(), signal_2.tolist(),
            msg="In TestModulator, initial and final signals do not match")

    def test_v2(self):

        signal = np.array([42])

        with self.assertRaises(RuntimeError,
            msg="In TestModulator, no exception for wrong signal length"):

            modulator(signal, self.f_rf, self.f_0, self.T_s)


class TestRFCurrent(unittest.TestCase):

    def setUp(self):
        from input_parameters.ring import Ring
        from beam.beam import Beam, Proton
        from beam.profile import Profile, CutOptions

        C = 2*np.pi*1100.009        # Ring circumference [m]
        gamma_t = 18.0              # Gamma at transition
        alpha = 1/gamma_t**2        # Momentum compaction factor
        p_s = 25.92e9               # Synchronous momentum at injection [eV]

        N_m = 1e5                   # Number of macro-particles for tracking
        N_b = 1.0e11                # Bunch intensity [ppb]

        # Set up machine parameters
        ring = Ring(C, alpha, p_s, Proton(), n_turns=1)
        self.t_rev = ring.t_rev[0]

        # Create Gaussian beam
        self.beam = Beam(ring, N_m, N_b)
        self.profile = Profile(self.beam,
                               CutOptions=CutOptions(cut_left=-1.e-9,
                                                 cut_right=6.e-9, n_slices=140))

        self.t = self.profile.bin_centers
        self.profile.n_macroparticles \
            = 2600*np.exp(-(self.t-2.5e-9)**2 / (2*0.5e-9)**2)

    def test_1(self):
        from scipy.constants import e

        omega = 2*np.pi*200.222e6

        rf_current = rf_beam_current(self.profile, omega, self.t_rev,
                                     lpf=False)

        rf_current_real = np.around(rf_current.real, 12)
        rf_current_imag = np.around(rf_current.imag, 12)

        rf_theo_real = 2*self.beam.ratio*self.profile.Beam.Particle.charge*e\
            * 2600*np.exp(-(self.t-2.5e-9)**2/(2*0.5*1e-9)**2)\
            * np.cos(omega*self.t)
        rf_theo_real = np.around(rf_theo_real, 12)

        rf_theo_imag = 2*self.beam.ratio*self.profile.Beam.Particle.charge*e\
            * 2600*np.exp(-(self.t-2.5e-9)**2/(2*0.5*1e-9)**2)\
            * np.sin(omega*self.t)
        rf_theo_imag = np.around(rf_theo_imag, 12)

        self.assertListEqual(rf_current_real.tolist(), rf_theo_real.tolist(),
            msg="In TestRfCurrent test_1, real part not correct")
        self.assertListEqual(rf_current_imag.tolist(), rf_theo_imag.tolist(),
            msg="In TestRfCurrent test_1, imaginary part not correct")


class TestComb(unittest.TestCase):

    def test_1(self):
        y = np.random.rand(42)

        self.assertListEqual(y.tolist(), comb_filter(y, y, 15/16).tolist(),
            msg="In TestComb test_1, filtered signal not correct")

    def test_2(self):

        t = np.arange(0, 2*np.pi, 2*np.pi/120)
        y = np.cos(t)
        # Shift cosine by quarter period
        x = np.roll(y, int(len(t)/4))

        # Drop some digits to avoid rounding errors
        result = np.around(comb_filter(y, x, 0.5), 12)
        result_theo = np.around(np.sin(np.pi/4 + t)/np.sqrt(2), 12)

        self.assertListEqual(result.tolist(), result_theo.tolist(),
            msg="In TestComb test_2, filtered signal not correct")


class TestLowPass(unittest.TestCase):

    def test_1(self):
        # Example based on SciPy.org filtfilt
        t = np.linspace(0, 1.0, 2001)
        xlow = np.sin(2 * np.pi * 5 * t)
        xhigh = np.sin(2 * np.pi * 250 * t)
        x = xlow + xhigh

        y = low_pass_filter(x, cutoff_frequency=1/8)

        # Test for difference between filtered signal and xlow;
        # using signal.butter(8, 0.125) and filtfilt(b, a, x, padlen=15)
        # from the SciPy documentation of filtfilt gives the stated
        # value 9.10862958....e-6
        self.assertAlmostEqual(np.abs(y - xlow).max(), 0.0230316365,
                               places=10)


class TestMovingAverage(unittest.TestCase):

    # Run before every test
    def setUp(self, N=3, x_prev=None):
        self.x = np.array([0, 3, 6, 3, 0, 3, 6, 3, 0], dtype=float)
        self.y = moving_average(self.x, N, x_prev)

    # Run after every test
    def tearDown(self):

        del self.x
        del self.y

    def test_1(self):

        self.setUp(N=3)
        self.assertEqual(len(self.x), len(self.y) + 3 - 1,
            msg="In TestMovingAverage, test_1: wrong array length")
        self.assertSequenceEqual(self.y.tolist(),
            np.array([3, 4, 3, 2, 3, 4, 3], dtype=float).tolist(),
            msg="In TestMovingAverage, test_1: arrays differ")

    def test_2(self):

        self.setUp(N=4)
        self.assertEqual(len(self.x), len(self.y) + 4 - 1,
            msg="In TestMovingAverage, test_2: wrong array length")
        self.assertSequenceEqual(self.y.tolist(),
                                 np.array([3, 3, 3, 3, 3, 3],
                                          dtype=float).tolist(),
            msg="In TestMovingAverage, test_2: arrays differ")

    def test_3(self):

        self.setUp(N=3, x_prev=np.array([0, 3]))
        self.assertEqual(len(self.x), len(self.y),
            msg="In TestMovingAverage, test_3: wrong array length")
        self.assertSequenceEqual(self.y.tolist(),
            np.array([1, 2, 3, 4, 3, 2, 3, 4, 3], dtype=float).tolist(),
            msg="In TestMovingAverage, test_3: arrays differ")


if __name__ == '__main__':

    unittest.main()
