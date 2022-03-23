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

:Authors: **Birk Emil Karlsen-Bæck**, **Helga Timko**
"""

import unittest
import numpy as np
from scipy.constants import e

from blond.llrf.signal_processing import moving_average, modulator
from blond.llrf.signal_processing import polar_to_cartesian, cartesian_to_polar
from blond.llrf.signal_processing import comb_filter, low_pass_filter
from blond.llrf.signal_processing import rf_beam_current, feedforward_filter
from blond.llrf.signal_processing import feedforward_filter_TWC3, \
    feedforward_filter_TWC4, feedforward_filter_TWC5

from blond.llrf.impulse_response import SPS3Section200MHzTWC, \
    SPS4Section200MHzTWC, SPS5Section200MHzTWC

from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions import bigaussian
from blond.input_parameters.rf_parameters import RFStation

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

        C = 2*np.pi*1100.009        # Ring circumference [m]
        gamma_t = 18.0              # Gamma at transition
        alpha = 1/gamma_t**2        # Momentum compaction factor
        p_s = 25.92e9               # Synchronous momentum at injection [eV]

        N_m = 1e5                   # Number of macro-particles for tracking
        N_b = 1.0e11                # Bunch intensity [ppb]

        # Set up machine parameters
        self.ring = Ring(C, alpha, p_s, Proton(), n_turns=1)
        self.rf = RFStation(self.ring, 4620, 4.5e6, 0)

        # RF-frequency at which to compute beam current
        self.omega = 2*np.pi*200.222e6
        
        # Create Gaussian beam
        self.beam = Beam(self.ring, N_m, N_b)
        self.profile = Profile(self.beam, CutOptions=CutOptions(cut_left=-1e-9,
            cut_right=6e-9, n_slices=100))

    # Test charge distribution with analytic functions
    # Compare with theoretical value
    def test_1(self):

        t = self.profile.bin_centers
        self.profile.n_macroparticles \
            = 2600*np.exp(-(t-2.5e-9)**2 / (2*0.5e-9)**2)

        rf_current = rf_beam_current(self.profile, self.omega,
                                     self.ring.t_rev[0], lpf=False)

        rf_current_real = np.around(rf_current.real, 12)
        rf_current_imag = np.around(rf_current.imag, 12)

        rf_theo_real = -2*self.beam.ratio*self.profile.Beam.Particle.charge*e\
            * 2600*np.exp(-(t-2.5e-9)**2/(2*0.5*1e-9)**2)\
            * np.cos(self.omega*t - self.omega * t[0])
        rf_theo_real = np.around(rf_theo_real, 12)

        rf_theo_imag = 2*self.beam.ratio*self.profile.Beam.Particle.charge*e\
            * 2600*np.exp(-(t-2.5e-9)**2/(2*0.5*1e-9)**2)\
            * np.sin(self.omega*t - self.omega * t[0])
        rf_theo_imag = np.around(rf_theo_imag, 12)

        self.assertListEqual(rf_current_real.tolist(), rf_theo_real.tolist(),
            msg="In TestRfCurrent test_1, mismatch in real part of RF current")
        self.assertListEqual(rf_current_imag.tolist(), rf_theo_imag.tolist(),
            msg="In TestRfCurrent test_1, mismatch in real part of RF current")

    # Test charge distribution of a bigaussian profile, without LPF
    # Compare to simulation data
    def test_2(self):

        bigaussian(self.ring, self.rf, self.beam, 3.2e-9/4, seed=1234,
                   reinsertion=True)
        self.profile.track()

        rf_current = rf_beam_current(self.profile, self.omega,
                                     self.ring.t_rev[0], lpf=False)


        Iref_real = np.array(
                [-0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
                 -0.00000000e+00, -4.17276538e-13, -4.58438685e-13,
                 -2.48023978e-13, -5.29812882e-13, -2.79735893e-13,
                 -0.00000000e+00, -1.21117142e-12, -9.32525031e-13,
                 -3.16481491e-13, -6.39337182e-13, -0.00000000e+00,
                 -0.00000000e+00, -4.08671437e-12, -4.92294318e-12,
                 -6.56965581e-12, -1.06279982e-11, -1.36819775e-11,
                 -2.16648779e-11, -3.09847742e-11, -3.52971851e-11,
                 -4.70378846e-11, -4.53538355e-11, -4.87255683e-11,
                 -5.36705233e-11, -5.13609268e-11, -4.32833547e-11,
                 -3.41417626e-11, -1.57452092e-11, 1.09005669e-11,
                 4.60465933e-11, 9.12872561e-11, 1.48257172e-10,
                 2.08540598e-10, 2.77630610e-10, 3.72157670e-10,
                 4.56272790e-10, 5.57978715e-10, 6.46554678e-10,
                 7.48006845e-10, 8.21493949e-10, 9.37522974e-10,
                 1.03729660e-09, 1.06159943e-09, 1.08434838e-09,
                 1.15738772e-09, 1.17887329e-09, 1.17146947e-09,
                 1.10964398e-09, 1.10234199e-09, 1.08852433e-09,
                 9.85866194e-10, 9.11727500e-10, 8.25604186e-10,
                 7.34122908e-10, 6.47294099e-10, 5.30372703e-10,
                 4.40357823e-10, 3.61273448e-10, 2.76871614e-10,
                 2.02227693e-10, 1.45430220e-10, 8.88675659e-11,
                 4.28984529e-11, 8.85451328e-12, -1.79026290e-11,
                 -3.48384214e-11, -4.50190282e-11, -5.62413472e-11,
                 -5.27322597e-11, -4.98163115e-11, -4.83288197e-11,
                 -4.18200851e-11, -3.13334269e-11, -2.44082108e-11,
                 -2.12572805e-11, -1.37397872e-11, -1.00879347e-11,
                 -7.78502213e-12, -4.00790819e-12, -2.51830415e-12,
                 -1.91301490e-12, -0.00000000e+00, -9.58518929e-13,
                 -3.16123809e-13, -1.24116546e-12, -1.20821672e-12,
                 -5.82952183e-13, -8.35917235e-13, -5.27285254e-13,
                 -4.93205919e-13, -0.00000000e+00, -2.06937013e-13,
                 -1.84618142e-13, -1.60868491e-13, -0.00000000e+00,
                 -1.09822743e-13])
        
        I_real = np.around(rf_current.real, 14) # round
        Iref_real = np.around(Iref_real, 14)
        
        self.assertSequenceEqual(I_real.tolist(), Iref_real.tolist(),
            msg="In TestRFCurrent test_2, mismatch in real part of RF current")
        
        Iref_imag = np.array([
                0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                0.00000000e+00,  -4.86410815e-13,  -4.47827158e-13,
                -2.02886432e-13,  -3.60573852e-13,  -1.56290206e-13,
                0.00000000e+00,  -4.19433613e-13,  -2.33465744e-13,
                -5.01823105e-14,  -4.43075921e-14,   0.00000000e+00,
                0.00000000e+00,   8.07144709e-13,   1.43192280e-12,
                2.55659168e-12,   5.25480064e-12,   8.33669524e-12,
                1.59729353e-11,   2.73609511e-11,   3.71844853e-11,
                5.92134758e-11,   6.87376280e-11,   9.02226570e-11,
                1.24465616e-10,   1.55478762e-10,   1.84035433e-10,
                2.37241518e-10,   2.86677989e-10,   3.28265272e-10,
                3.77882012e-10,   4.29727720e-10,   4.83759029e-10,
                5.13978173e-10,   5.41841031e-10,   5.91537968e-10,
                6.00658643e-10,   6.13928028e-10,   5.96367636e-10,
                5.76920099e-10,   5.25297875e-10,   4.89104065e-10,
                4.29776324e-10,   3.33901906e-10,   2.38690921e-10,
                1.49673305e-10,   4.78223853e-11,  -5.57081558e-11,
                -1.51374774e-10,  -2.50724894e-10,  -3.50731761e-10,
                -4.16547058e-10,  -4.83765618e-10,  -5.36075032e-10,
                -5.74421794e-10,  -6.05459147e-10,  -5.91794283e-10,
                -5.88179055e-10,  -5.83222843e-10,  -5.49774151e-10,
                -5.08571646e-10,  -4.86623358e-10,  -4.33179012e-10,
                -3.73737133e-10,  -3.37622742e-10,  -2.89119788e-10,
                -2.30660798e-10,  -1.85597518e-10,  -1.66348322e-10,
                -1.19981335e-10,  -9.07232680e-11,  -7.21467862e-11,
                -5.18977454e-11,  -3.25510912e-11,  -2.12524272e-11,
                -1.54447488e-11,  -8.24107056e-12,  -4.90052047e-12,
                -2.96720377e-12,  -1.13551262e-12,  -4.79152734e-13,
                -1.91861296e-13,   0.00000000e+00,   7.31481456e-14,
                5.23883203e-14,   3.19951675e-13,   4.27870459e-13,
                2.66236636e-13,   4.74712082e-13,   3.64260145e-13,
                4.09222572e-13,   0.00000000e+00,   2.44654594e-13,
                2.61906356e-13,   2.77128356e-13,   0.00000000e+00,
                3.01027843e-13])
        
        I_imag = np.around(rf_current.imag, 14) # round
        Iref_imag = np.around(Iref_imag, 14)
        
        self.assertSequenceEqual(I_imag.tolist(), Iref_imag.tolist(),
            msg="In TestRFCurrent test_2, mismatch in imaginary part of"
            + " RF current")
    
    # Test charge distribution of a bigaussian profile, with LPF
    # Compare to simulation data
    def test_3(self):
        
        bigaussian(self.ring, self.rf, self.beam, 3.2e-9/4, seed=1234,
                   reinsertion=True)
        self.profile.track()
        self.assertEqual(len(self.beam.dt), np.sum(self.profile.n_macroparticles), "In" +
            " TestBeamCurrent: particle number mismatch in Beam vs Profile")

        # RF current calculation with low-pass filter
        rf_current = rf_beam_current(self.profile, self.omega,
                                     self.ring.t_rev[0], lpf=True)

        for i in range(0, len(rf_current), 4):
            print(f'{rf_current[i].imag:.10e}, {rf_current[i+1].imag:.10e}, {rf_current[i+2].imag:.10e}, {rf_current[i+3].imag:.10e},')


        Iref_real = np.array([-7.4760030591e-12, -7.4760667001e-12, -7.4761283585e-12, -7.4761880743e-12,
                            -7.4762458873e-12, -7.4763018373e-12, -7.4763559634e-12, -7.4764083048e-12,
                            -7.4764589002e-12, -7.4765077881e-12, -7.4765550066e-12, -7.4766005934e-12,
                            -7.4766445863e-12, -7.4766870223e-12, -7.4767279383e-12, -7.4767673709e-12,
                            -7.4768053561e-12, -7.4768419298e-12, -7.4768771275e-12, -7.4769109841e-12,
                            -7.4769435345e-12, -7.4769748129e-12, -7.4770048532e-12, -7.4770336890e-12,
                            -7.4770613534e-12, -7.4770878791e-12, -7.4771132983e-12, -7.4771376430e-12,
                            -7.4771609447e-12, -7.4771832343e-12, -7.4772045423e-12, -7.4772248990e-12,
                            -7.4772443341e-12, -7.4772628766e-12, -7.4772805555e-12, -7.4772973991e-12,
                            -7.4773134351e-12, -7.4773286909e-12, -7.4773431935e-12, -7.4773569691e-12,
                            -7.4773700439e-12, -7.4773824432e-12, -7.4773941919e-12, -7.4774053146e-12,
                            -7.4774158351e-12, -7.4774257769e-12, -7.4774351631e-12, -7.4774440160e-12,
                            -7.4774523576e-12, -7.4774602094e-12, -7.4774675923e-12, -7.4774745266e-12,
                            -7.4774810324e-12, -7.4774871291e-12, -7.4774928355e-12, -7.4774981700e-12,
                            -7.4775031506e-12, -7.4775077945e-12, -7.4775121186e-12, -7.4775161393e-12,
                            -7.4775198724e-12, -7.4775233333e-12, -7.4775265368e-12, -7.4775294972e-12,
                            -7.4775322284e-12, -7.4775347438e-12, -7.4775370562e-12, -7.4775391780e-12,
                            -7.4775411210e-12, -7.4775428968e-12, -7.4775445163e-12, -7.4775459899e-12,
                            -7.4775473278e-12, -7.4775485395e-12, -7.4775496341e-12, -7.4775506203e-12,
                            -7.4775515065e-12, -7.4775523004e-12, -7.4775530095e-12, -7.4775536409e-12,
                            -7.4775542012e-12, -7.4775546966e-12, -7.4775551329e-12, -7.4775555159e-12,
                            -7.4775558504e-12, -7.4775561415e-12, -7.4775563934e-12, -7.4775566105e-12,
                            -7.4775567965e-12, -7.4775569550e-12, -7.4775570892e-12, -7.4775572021e-12,
                            -7.4775572965e-12, -7.4775573746e-12, -7.4775574389e-12, -7.4775574913e-12,
                            -7.4775575336e-12, -7.4775575674e-12, -7.4775575940e-12, -7.4775576148e-12])

        np.testing.assert_allclose(rf_current.real, Iref_real, rtol=1e-7,
            atol=0, err_msg="In TestRFCurrent test_3, mismatch in real part of RF current")

        Iref_imag = np.array([3.6350710513e-27, -6.5295290818e-17, -1.2848415740e-16, -1.8961169178e-16,
                            -2.4872263572e-16, -3.0586137201e-16, -3.6107191579e-16, -4.1439790600e-16,
                            -4.6588259694e-16, -5.1556884999e-16, -5.6349912532e-16, -6.0971547386e-16,
                            -6.5425952927e-16, -6.9717250018e-16, -7.3849516241e-16, -7.7826785145e-16,
                            -8.1653045501e-16, -8.5332240575e-16, -8.8868267418e-16, -9.2264976165e-16,
                            -9.5526169356e-16, -9.8655601273e-16, -1.0165697729e-15, -1.0453395325e-15,
                            -1.0729013485e-15, -1.0992907703e-15, -1.1245428344e-15, -1.1486920585e-15,
                            -1.1717724363e-15, -1.1938174325e-15, -1.2148599776e-15, -1.2349324634e-15,
                            -1.2540667385e-15, -1.2722941042e-15, -1.2896453102e-15, -1.3061505509e-15,
                            -1.3218394622e-15, -1.3367411178e-15, -1.3508840265e-15, -1.3642961295e-15,
                            -1.3770047975e-15, -1.3890368291e-15, -1.4004184485e-15, -1.4111753040e-15,
                            -1.4213324665e-15, -1.4309144290e-15, -1.4399451053e-15, -1.4484478299e-15,
                            -1.4564453577e-15, -1.4639598646e-15, -1.4710129478e-15, -1.4776256266e-15,
                            -1.4838183439e-15, -1.4896109678e-15, -1.4950227934e-15, -1.5000725448e-15,
                            -1.5047783786e-15, -1.5091578861e-15, -1.5132280968e-15, -1.5170054827e-15,
                            -1.5205059617e-15, -1.5237449027e-15, -1.5267371298e-15, -1.5294969281e-15,
                            -1.5320380493e-15, -1.5343737170e-15, -1.5365166339e-15, -1.5384789880e-15,
                            -1.5402724599e-15, -1.5419082302e-15, -1.5433969871e-15, -1.5447489349e-15,
                            -1.5459738025e-15, -1.5470808518e-15, -1.5480788876e-15, -1.5489762665e-15,
                            -1.5497809071e-15, -1.5505003002e-15, -1.5511415189e-15, -1.5517112297e-15,
                            -1.5522157035e-15, -1.5526608266e-15, -1.5530521124e-15, -1.5533947134e-15,
                            -1.5536934328e-15, -1.5539527371e-15, -1.5541767678e-15, -1.5543693548e-15,
                            -1.5545340285e-15, -1.5546740327e-15, -1.5547923374e-15, -1.5548916523e-15,
                            -1.5549744390e-15, -1.5550429246e-15, -1.5550991148e-15, -1.5551448063e-15,
                            -1.5551816007e-15, -1.5552109164e-15, -1.5552340019e-15, -1.5552519483e-15])

        np.testing.assert_allclose(rf_current.imag, Iref_imag, rtol=1e-7,
            atol=0, err_msg="In TestRFCurrent test_3, mismatch in imaginary part of RF current")

    # Test RF beam current on coarse grid integrated from fine grid
    # Compare to simulation data for peak RF current
    def test_4(self):

        # Create a batch of 100 equal, short bunches
        bunches = 100
        T_s = 5*self.rf.t_rev[0]/self.rf.harmonic[0, 0]
        N_m = int(1e5)
        N_b = 2.3e11
        bigaussian(self.ring, self.rf, self.beam, 0.1e-9, seed=1234,
                   reinsertion=True)
        beam2 = Beam(self.ring, bunches*N_m, bunches*N_b)
        bunch_spacing = 5*self.rf.t_rf[0, 0]
        buckets = 5*bunches
        for i in range(bunches):
            beam2.dt[i*N_m:(i+1)*N_m] = self.beam.dt + i*bunch_spacing
            beam2.dE[i*N_m:(i+1)*N_m] = self.beam.dE
        profile2 = Profile(beam2, CutOptions=CutOptions(cut_left=0,
            cut_right=bunches*bunch_spacing, n_slices=1000*buckets))
        profile2.track()

        tot_charges = np.sum(profile2.n_macroparticles)/\
                     beam2.n_macroparticles*beam2.intensity
        self.assertAlmostEqual(tot_charges, 2.3000000000e+13, 9)

        # Calculate fine- and coarse-grid RF current
        rf_current_fine, rf_current_coarse = rf_beam_current(profile2,
            self.rf.omega_rf[0, 0], self.ring.t_rev[0], lpf=False,
            downsample={'Ts': T_s, 'points': self.rf.harmonic[0, 0]/5})
        rf_current_coarse /= T_s

        # Peak RF current on coarse grid
        peak_rf_current = np.max(np.absolute(rf_current_coarse))
        self.assertAlmostEqual(peak_rf_current, 2.9285808008, 7)


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


class TestFeedforwardFilter(unittest.TestCase):

    # Run before every test
    def setUp(self):

        # Ring and RF definitions
        ring = Ring(2*np.pi*1100.009, 1/18**2, 25.92e9, Particle=Proton())
        rf = RFStation(ring, [4620], [4.5e6], [0.], n_rf=1)
        self.T_s = 5*rf.t_rf[0, 0]

    def test_1(self):

        # Modified filling time to match reference case
        TWC = SPS3Section200MHzTWC()
        TWC.tau = 420e-9
        filter, n_taps, n_filling, n_fit = feedforward_filter(TWC, 4/125*1e-6,
            debug=False, taps=31, opt_output=True)
        self.assertEqual(n_taps, 31,
            msg="In TestFeedforwardFilter, test_1: n_taps incorrect")
        self.assertEqual(n_filling, 13,
            msg="In TestFeedforwardFilter, test_1: n_filling incorrect")
        self.assertEqual(n_fit, 44,
            msg="In TestFeedforwardFilter, test_1: n_fit incorrect")

        filter_ref = np.array(
            [-0.0227533635, 0.0211514102, 0.0032929202, -0.0026111554,
              0.0119559316, 0.0043905603, 0.0043905603, 0.0040101282,
             -0.0241480816, -0.0237676496, 0.0043905603, 0.0043905603,
              0.0043905603, -0.0107783487, 0.0184915005, 0.0065858404,
             -0.0052223108, 0.0239118633, 0.0087811206, 0.0087811206,
              0.0080202564, 0.0295926259, 0.0237676496, -0.0043905603,
             -0.0043905603, -0.0043905603, -0.0119750148, 0.0026599098,
             -0.0032929202, -0.021005147,  0.022696114])

        np.testing.assert_allclose(filter, filter_ref, rtol=1e-8, atol=1e-9,
            err_msg="In TestFeedforwardFilter, test_1: filter array incorrect")

        del TWC

    def test_2(self):

        TWC = SPS3Section200MHzTWC()
        filter, n_taps, n_filling, n_fit = feedforward_filter(TWC, self.T_s,
            debug=False, opt_output=True)
        self.assertEqual(n_taps, 31,
            msg="In TestFeedforwardFilter, test_2: n_taps incorrect")
        self.assertEqual(n_filling, 18,
            msg="In TestFeedforwardFilter, test_2: n_filling incorrect")
        self.assertEqual(n_fit, 49,
            msg="In TestFeedforwardFilter, test_2: n_fit incorrect")

#        filter_ref = np.array(
#            [-0.0070484734, 0.0161859736, 0.0020289928, 0.0020289928,
#              0.0020289928, -0.0071641302, -0.0162319424, -0.0070388194,
#              0.0020289928, 0.0020289928, 0.0020289928, - 0.0050718734,
#              0.0065971343, 0.0030434892, 0.0030434892, 0.0030434892,
#              0.0030434892, 0.0030434892, -0.0004807475, 0.011136476,
#              0.0040579856, 0.0040579856, 0.0040579856, 0.0132511086,
#              0.019651364, 0.0074147518, -0.0020289928, -0.0020289928,
#             -0.0020289928, -0.0162307252, 0.0071072903])
        filter_ref = np.copy(feedforward_filter_TWC3)

        np.testing.assert_allclose(filter, filter_ref, rtol=1e-8, atol=1e-9,
            err_msg="In TestFeedforwardFilter, test_2: filter array incorrect")

        del TWC

    def test_3(self):

        TWC = SPS4Section200MHzTWC()
        filter, n_taps, n_filling, n_fit = feedforward_filter(TWC, self.T_s,
            debug=False, opt_output=True)
        self.assertEqual(n_taps, 37,
            msg="In TestFeedforwardFilter, test_3: n_taps incorrect")
        self.assertEqual(n_filling, 24,
            msg="In TestFeedforwardFilter, test_3: n_filling incorrect")
        self.assertEqual(n_fit, 61,
            msg="In TestFeedforwardFilter, test_3: n_fit incorrect")

#        filter_ref = np.array(
#            [ 0.0048142895, 0.0035544775, 0.0011144336, 0.0011144336,
#              0.0011144336, -0.0056984584, -0.0122587698, -0.0054458778,
#              0.0011144336, 0.0011144336, 0.0011144336, -0.0001684528,
#             -0.000662115, 0.0016716504, 0.0016716504, 0.0016716504,
#              0.0016716504, 0.0016716504, 0.0016716504, 0.0016716504,
#              0.0016716504, 0.0016716504, 0.0016716504, 0.0016716504,
#              0.0040787952, 0.0034488892, 0.0022288672, 0.0022288672,
#              0.0022288672, 0.0090417593, 0.0146881621, 0.0062036196,
#             -0.0011144336, -0.0011144336, -0.0011144336, -0.0036802064,
#             -0.0046675309])
        filter_ref = np.copy(feedforward_filter_TWC4)

        np.testing.assert_allclose(filter, filter_ref, rtol=1e-8, atol=1e-9,
            err_msg="In TestFeedforwardFilter, test_3: filter array incorrect")

        del TWC

    def test_4(self):

        TWC = SPS5Section200MHzTWC()
        filter, n_taps, n_filling, n_fit = feedforward_filter(TWC, self.T_s,
            debug=False, opt_output=True)
        self.assertEqual(n_taps, 43,
            msg="In TestFeedforwardFilter, test_4: n_taps incorrect")
        self.assertEqual(n_filling, 31,
            msg="In TestFeedforwardFilter, test_4: n_filling incorrect")
        self.assertEqual(n_fit, 74,
            msg="In TestFeedforwardFilter, test_4: n_fit incorrect")

#        filter_ref = np.array(
#            [ 0.0189205535, -0.0105637125, 0.0007262783, 0.0007262783,
#              0.0006531768, -0.0105310359, -0.0104579343, 0.0007262783,
#              0.0007262783, 0.0007262783, 0.0063272331, -0.0083221785,
#              0.0010894175, 0.0010894175, 0.0010894175, 0.0010894175,
#              0.0010894175, 0.0010894175, 0.0010894175, 0.0010894175,
#              0.0010894175, 0.0010894175, 0.0010894175, 0.0010894175,
#              0.0010894175, 0.0010894175, 0.0010894175, 0.0010894175,
#              0.0010894175, 0.0010894175, 0.0010894175, 0.0105496942,
#             -0.0041924387, 0.0014525567, 0.0014525567, 0.0013063535,
#              0.0114011487, 0.0104579343, -0.0007262783, -0.0007262783,
#             -0.0007262783, 0.0104756312, -0.018823192])
        filter_ref = np.copy(feedforward_filter_TWC5)

        np.testing.assert_allclose(filter, filter_ref, rtol=1e-8, atol=1e-9,
            err_msg="In TestFeedforwardFilter, test_4: filter array incorrect")

        del TWC

    #    TWC4 = SPS4Section200MHzTWC()
    #    FF_4 = feedforward_filter(TWC4, 25e-9, debug=True)

    #    TWC5 = SPS5Section200MHzTWC()
    #    FF_5 = feedforward_filter(TWC5, 25e-9, debug=True)


if __name__ == '__main__':

    unittest.main()
