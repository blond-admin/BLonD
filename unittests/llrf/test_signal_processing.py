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

from input_parameters.ring import Ring
from beam.beam import Beam, Proton
from beam.profile import Profile, CutOptions

from scipy.constants import e

from beam.distributions import bigaussian
from input_parameters.rf_parameters import RFStation

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
        
        # RF-frequency at which to compute beam current
        self.omega = 2*np.pi*200.222e6
        
        # Create Gaussian beam
        self.beam = Beam(self.ring, N_m, N_b)
        self.profile = Profile(
                self.beam, CutOptions=CutOptions(cut_left=-1.e-9, n_slices=100,
                                                 cut_right=6.e-9))

    def test_1(self):

        t = self.profile.bin_centers
        self.profile.n_macroparticles \
            = 2600*np.exp(-(t-2.5e-9)**2 / (2*0.5e-9)**2)

        rf_current = rf_beam_current(self.profile, self.omega,
                                     self.ring.t_rev[0], lpf=False)

        rf_current_real = np.around(rf_current.real, 12)
        rf_current_imag = np.around(rf_current.imag, 12)

        rf_theo_real = 2*self.beam.ratio*self.profile.Beam.Particle.charge*e\
            * 2600*np.exp(-(t-2.5e-9)**2/(2*0.5*1e-9)**2)\
            * np.cos(self.omega*t)
        rf_theo_real = np.around(rf_theo_real, 12)

        rf_theo_imag = 2*self.beam.ratio*self.profile.Beam.Particle.charge*e\
            * 2600*np.exp(-(t-2.5e-9)**2/(2*0.5*1e-9)**2)\
            * np.sin(self.omega*t)
        rf_theo_imag = np.around(rf_theo_imag, 12)

        self.assertListEqual(rf_current_real.tolist(), rf_theo_real.tolist(),
            msg="In TestRfCurrent test_1, mismatch in real part of RF current")
        self.assertListEqual(rf_current_imag.tolist(), rf_theo_imag.tolist(),
            msg="In TestRfCurrent test_1, mismatch in real part of RF current")

    def test_2(self):
        
        RF = RFStation(self.ring, 4620, 4.5e6, 0)

        bigaussian(self.ring, RF, self.beam, 3.2e-9/4, seed = 1234,
                   reinsertion = True)
        self.profile.track()
        
        rf_current = rf_beam_current(self.profile, self.omega,
                                     self.ring.t_rev[0], lpf=False)

        Iref_real = np.array(
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                 0.00000000e+00, 4.17276535e-13, 4.58438681e-13,
                 2.48023976e-13, 5.29812878e-13, 2.79735891e-13,
                 0.00000000e+00, 1.21117141e-12, 9.32525023e-13,
                 3.16481489e-13, 6.39337176e-13, 0.00000000e+00,
                 0.00000000e+00, 4.08671434e-12, 4.92294314e-12,
                 6.56965575e-12, 1.06279981e-11, 1.36819774e-11,
                 2.16648778e-11, 3.09847740e-11, 3.52971849e-11,
                 4.70378842e-11, 4.53538351e-11, 4.87255679e-11,
                 5.36705228e-11, 5.13609263e-11, 4.32833543e-11,
                 3.41417624e-11, 1.57452091e-11, -1.09005668e-11,
                 -4.60465929e-11, -9.12872553e-11, -1.48257171e-10,
                 -2.08540597e-10, -2.77630608e-10, -3.72157667e-10,
                 -4.56272786e-10, -5.57978710e-10, -6.46554672e-10,
                 -7.48006839e-10, -8.21493943e-10, -9.37522966e-10,
                 -1.03729659e-09, -1.06159943e-09, -1.08434837e-09,
                 -1.15738771e-09, -1.17887328e-09, -1.17146946e-09,
                 -1.10964397e-09, -1.10234198e-09, -1.08852433e-09,
                 -9.85866185e-10, -9.11727492e-10, -8.25604179e-10,
                 -7.34122902e-10, -6.47294094e-10, -5.30372699e-10,
                 -4.40357820e-10, -3.61273445e-10, -2.76871612e-10,
                 -2.02227691e-10, -1.45430219e-10, -8.88675652e-11,
                 -4.28984525e-11, -8.85451321e-12,  1.79026289e-11,
                 3.48384211e-11,  4.50190278e-11, 5.62413467e-11,
                 5.27322593e-11,  4.98163111e-11, 4.83288193e-11,
                 4.18200848e-11,  3.13334266e-11, 2.44082106e-11,
                 2.12572803e-11,  1.37397871e-11, 1.00879346e-11,
                 7.78502206e-12,  4.00790815e-12, 2.51830412e-12,
                 1.91301488e-12,  0.00000000e+00, 9.58518921e-13,
                 3.16123806e-13,  1.24116545e-12, 1.20821671e-12,
                 5.82952178e-13,  8.35917228e-13, 5.27285250e-13,
                 4.93205915e-13,  0.00000000e+00, 2.06937011e-13,
                 1.84618141e-13,  1.60868490e-13, 0.00000000e+00,
                 1.09822742e-13])
        
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
    
    # Skip this unit test, since its reference values are obsolete;
    # This test used to be in /unittests/general/test_cavity_feedback.py
    @unittest.skip('Skipping because of obsolete reference values!')
    def test_3(self):
        
        # Set up SPS conditions
        ring = Ring(2*np.pi*1100.009, 1/18**2, 25.92e9, Proton(), 1000)
        RF = RFStation(ring, 4620, 4.5e6, 0)
        beam = Beam(ring, 1e5, 1e11)
        bigaussian(ring, RF, beam, 3.2e-9/4, seed = 1234, reinsertion = True) 
        profile = Profile(beam, CutOptions(cut_left=-1.e-9, cut_right=6.e-9, 
                                           n_slices=100))
        profile.track()
        self.assertEqual(len(beam.dt), np.sum(profile.n_macroparticles), "In" +
            " TestBeamCurrent: particle number mismatch in Beam vs Profile")
        
        # RF current calculation with low-pass filter
        rf_current = rf_beam_current(profile, 2*np.pi*200.222e6, ring.t_rev[0])
        Iref_real = np.array([ -9.4646042539e-12,  -7.9596801534e-10,  
            -2.6993572787e-10,
            2.3790828610e-09,   6.4007063190e-09,   9.5444302650e-09,
            9.6957462918e-09,   6.9944771120e-09,   5.0040512366e-09,
            8.2427583408e-09,   1.6487066238e-08,   2.2178930587e-08,
            1.6497620890e-08,   1.9878201568e-09,  -2.4862807497e-09,
            2.0862096916e-08,   6.6115473293e-08,   1.1218114710e-07,
             1.5428441607e-07,   2.1264254596e-07,   3.1213935713e-07,
             4.6339212948e-07,   6.5039440158e-07,   8.2602190806e-07,
             9.4532001396e-07,   1.0161170159e-06,   1.0795840334e-06,
             1.1306004256e-06,   1.1081141333e-06,   9.7040873320e-07,
             7.1863437325e-07,   3.3833950889e-07,  -2.2273124358e-07,
            -1.0035204008e-06,  -1.9962696992e-06,  -3.1751183137e-06,
            -4.5326227784e-06,  -6.0940850385e-06,  -7.9138578879e-06,
            -9.9867317826e-06,  -1.2114906338e-05,  -1.4055138779e-05,
            -1.5925650405e-05,  -1.8096693885e-05,  -2.0418813156e-05,
            -2.2142865862e-05,  -2.3038234657e-05,  -2.3822481250e-05,
            -2.4891969829e-05,  -2.5543384520e-05,  -2.5196086909e-05,
            -2.4415522211e-05,  -2.3869116251e-05,  -2.3182951665e-05,
            -2.1723128723e-05,  -1.9724625363e-05,  -1.7805112266e-05,
            -1.5981218737e-05,  -1.3906226012e-05,  -1.1635865568e-05,
            -9.5381189596e-06,  -7.7236624815e-06,  -6.0416822483e-06,
            -4.4575806261e-06,  -3.0779237834e-06,  -1.9274519396e-06,
            -9.5699993457e-07,  -1.7840768971e-07,   3.7780452612e-07,
             7.5625231388e-07,   1.0158886027e-06,   1.1538975409e-06,
             1.1677937652e-06,   1.1105424636e-06,   1.0216131672e-06,
             8.8605026541e-07,   7.0783694846e-07,   5.4147914020e-07,
             4.1956457226e-07,   3.2130062098e-07,   2.2762751268e-07,
             1.4923020411e-07,   9.5683463322e-08,   5.8942895620e-08,
             3.0515695233e-08,   1.2444834300e-08,   8.9413517889e-09,
             1.6154761941e-08,   2.3261993674e-08,   2.3057968490e-08,
             1.8354179928e-08,   1.4938991667e-08,   1.2506841004e-08,
             8.1230022648e-09,   3.7428821201e-09,   2.8368110506e-09,
             3.6536247240e-09,   2.8429736524e-09,   1.6640835314e-09,
             2.3960087967e-09])
        I_real = np.around(rf_current.real, 9) # round
        Iref_real = np.around(Iref_real, 9) 
        self.assertSequenceEqual(I_real.tolist(), Iref_real.tolist(),
            msg="In TestRFCurrent test_3, mismatch in real part of RF current")
        Iref_imag = np.array([ -1.3134886055e-11,   1.0898262206e-09,   
            3.9806900984e-10,
            -3.0007980073e-09,  -7.4404909183e-09,  -9.5619658077e-09,
            -7.9029982105e-09,  -4.5153699012e-09,  -2.8337010673e-09,
            -4.0605999910e-09,  -5.7035811935e-09,  -4.9421561822e-09,
            -2.6226262365e-09,  -1.0904425703e-09,   1.5886725829e-10,
             3.6061564044e-09,   1.2213233410e-08,   3.0717134774e-08,
             6.2263860975e-08,   1.0789908935e-07,   1.8547368321e-07,
             3.3758410599e-07,   5.8319210090e-07,   8.7586115583e-07,
             1.1744525681e-06,   1.5330067491e-06,   2.0257108185e-06,
             2.6290348930e-06,   3.3065045701e-06,   4.1218136471e-06,
             5.1059358251e-06,   6.1421308306e-06,   7.1521192647e-06,
             8.2164613957e-06,   9.3474086978e-06,   1.0368027059e-05,
             1.1176114701e-05,   1.1892303251e-05,   1.2600522466e-05,
             1.3142991032e-05,   1.3286611961e-05,   1.2972067098e-05,
             1.2344251145e-05,   1.1561930031e-05,   1.0577353622e-05,
             9.1838382917e-06,   7.3302333455e-06,   5.2367297732e-06,
             3.1309520147e-06,   1.0396785645e-06,  -1.1104442284e-06,
            -3.3300486963e-06,  -5.5129705406e-06,  -7.4742790081e-06,
            -9.1003715719e-06,  -1.0458342224e-05,  -1.1632423668e-05,
            -1.2513736332e-05,  -1.2942309414e-05,  -1.2975831165e-05,
            -1.2799952495e-05,  -1.2469945465e-05,  -1.1941176358e-05,
            -1.1222986380e-05,  -1.0349594257e-05,  -9.3491445482e-06,
            -8.2956327726e-06,  -7.2394219079e-06,  -6.1539590898e-06,
            -5.0802321519e-06,  -4.1512021086e-06,  -3.3868884793e-06,
            -2.6850344653e-06,  -2.0327038471e-06,  -1.5048854341e-06,
            -1.0965986189e-06,  -7.4914749272e-07,  -4.7128817088e-07,
            -2.9595396024e-07,  -1.9387567373e-07,  -1.1597751838e-07,
            -5.5766761837e-08,  -2.3991059778e-08,  -1.1910924971e-08,
            -4.7797889603e-09,   9.0715301612e-11,   1.5744084129e-09,
             2.8217939283e-09,   5.5919203984e-09,   7.7259433940e-09,
             8.5033504655e-09,   9.1509256107e-09,   8.6746085156e-09,
             5.8909590412e-09,   3.5957212556e-09,   4.3347189168e-09,
             5.3331969589e-09,   3.9322184713e-09,   3.3616434953e-09,
             6.5154351819e-09])
        I_imag = np.around(rf_current.imag, 9) # round
        Iref_imag = np.around(Iref_imag, 9)
        self.assertSequenceEqual(I_imag.tolist(), Iref_imag.tolist(),
            msg="In TestRFCurrent test_3, mismatch in imaginary part of RF current")



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
