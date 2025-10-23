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

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.llrf.impulse_response import (SPS3Section200MHzTWC,
                                         SPS4Section200MHzTWC,
                                         SPS5Section200MHzTWC)
from blond.llrf.signal_processing import (cartesian_to_polar, comb_filter,
                                          feedforward_filter,
                                          feedforward_filter_TWC3,
                                          feedforward_filter_TWC4,
                                          feedforward_filter_TWC5,
                                          low_pass_filter, modulator,
                                          moving_average, polar_to_cartesian,
                                          rf_beam_current)


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
        phases = np.pi * (np.fmod(2 * np.arange(self.n) * self.f_rf * self.T_s, 2) - 1)
        signal = np.cos(phases) + 1j * np.sin(phases)
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
        phase = np.pi * (np.fmod(2 * np.arange(self.n) * self.f_rf * self.T_s, 2) - 1)
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
        phase = np.pi * (np.fmod(2 * np.arange(self.n) * self.f_rf * self.T_s, 2) - 1)
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
        phase = np.pi * (np.fmod(2 * np.arange(self.n) * self.f_rf * self.T_s, 2) - 1)
        signal = np.cos(phase) + 1j * np.sin(phase)
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
        signal = np.cos(2 * np.pi * np.arange(self.n) * self.f_rf * self.T_s) \
            + 1j * np.sin(2 * np.pi * np.arange(self.n) * self.f_rf * self.T_s)
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

        C = 2 * np.pi * 1100.009        # Ring circumference [m]
        gamma_t = 18.0              # Gamma at transition
        alpha = 1 / gamma_t**2        # Momentum compaction factor
        p_s = 25.92e9               # Synchronous momentum at injection [eV]

        N_m = 1e5                   # Number of macro-particles for tracking
        N_b = 1.0e11                # Bunch intensity [ppb]

        # Set up machine parameters
        self.ring = Ring(C, alpha, p_s, Proton(), n_turns=1)
        self.rf = RFStation(self.ring, 4620, 4.5e6, 0)

        # RF-frequency at which to compute beam current
        self.omega = 2 * np.pi * 200.222e6

        # Create Gaussian beam
        self.beam = Beam(self.ring, N_m, N_b)
        self.profile = Profile(self.beam, CutOptions=CutOptions(cut_left=-2e-9,
            cut_right=8e-9, n_slices=100))

    # Test charge distribution with analytic functions
    # Compare with theoretical value
    def test_1(self):

        t = self.profile.bin_centers
        self.profile.n_macroparticles \
            = 2600 * np.exp(-(t - 2.5e-9)**2 / (2 * 0.5e-9)**2)

        rf_current = rf_beam_current(self.profile, self.omega,
                                     self.ring.t_rev[0], lpf=False,
                                     external_reference=True, dT=0)

        rf_current_real = np.around(rf_current.real, 12)
        rf_current_imag = np.around(rf_current.imag, 12)

        rf_theo_real = 2 * self.beam.ratio * self.profile.Beam.Particle.charge * e\
            * 2600 * np.exp(-(t - 2.5e-9)**2 / (2 * 0.5 * 1e-9)**2)\
            * np.cos(self.omega * t)
        rf_theo_real = np.around(rf_theo_real, 12)

        rf_theo_imag = -2 * self.beam.ratio * self.profile.Beam.Particle.charge * e\
            * 2600 * np.exp(-(t - 2.5e-9)**2 / (2 * 0.5 * 1e-9)**2)\
            * np.sin(self.omega * t)
        rf_theo_imag = np.around(rf_theo_imag, 12)

        self.assertListEqual(rf_current_real.tolist(), rf_theo_real.tolist(),
                             msg="In TestRfCurrent test_1, mismatch in real part of RF current")
        self.assertListEqual(rf_current_imag.tolist(), rf_theo_imag.tolist(),
                             msg="In TestRfCurrent test_1, mismatch in real part of RF current")

    # Test charge distribution of a bigaussian profile, without LPF
    # Compare to simulation data
    def test_2(self):

        bigaussian(self.ring, self.rf, self.beam, 3.2e-9 / 4, seed=1234,
                   reinsertion=True)
        self.profile.track()

        rf_current = rf_beam_current(self.profile, self.omega,
                                     self.ring.t_rev[0], lpf=False, external_reference=True)


        Iref_real = np.array(
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                 7.92343713e-14, 1.17565209e-13, 1.54037861e-13,
                 3.76151710e-13, 4.38282401e-13, 1.48045735e-12,
                 1.35222334e-12, 5.79743820e-13, 9.14152671e-13,
                 9.44240899e-13, 3.19801617e-13, 0.00000000e+00,
                 3.46221663e-12, 6.39906870e-12, 1.56530831e-11,
                 2.35286862e-11, 3.47907477e-11, 4.58005109e-11,
                 5.77392874e-11, 6.62362803e-11, 7.82984295e-11,
                 6.79830906e-11, 4.32594460e-11, -7.71572435e-13,
                 -6.98622431e-11, -1.72921639e-10, -2.93663454e-10,
                 -4.53113976e-10, -6.44608738e-10, -8.27691108e-10,
                 -1.00045780e-09, -1.20532535e-09, -1.37275783e-09,
                 -1.50700319e-09, -1.62579598e-09, -1.66460303e-09,
                 -1.68753194e-09, -1.55757951e-09, -1.50906664e-09,
                 -1.38532852e-09, -1.18191538e-09, -9.77471102e-10,
                 -8.18280669e-10, -6.11484460e-10, -4.58559052e-10,
                 -2.80009598e-10, -1.62320000e-10, -6.47893125e-11,
                 2.23593075e-12, 4.60546387e-11, 6.78858076e-11,
                 7.31008392e-11, 7.22251131e-11, 6.24707337e-11,
                 4.98931974e-11, 4.11950132e-11, 1.73847516e-11,
                 1.80306776e-11, 8.24583997e-12, 6.30317063e-12,
                 9.59802958e-13, 3.19653359e-13, 9.42960274e-13,
                 6.08037625e-13, 8.66737017e-13, 2.69239438e-13,
                 0.00000000e+00, 4.35010763e-13, 0.00000000e+00,
                 1.52074465e-13, 0.00000000e+00, 7.70670378e-14,
                 0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
                 -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
                 -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
                 -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
                 -0.00000000e+00, -0.00000000e+00, 0.00000000e+00,
                 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                 0.00000000e+00])

        I_real = np.around(rf_current.real, 14)
        Iref_real = np.around(Iref_real, 14)

        self.assertSequenceEqual(I_real.tolist(), Iref_real.tolist(),
            msg="In TestRFCurrent test_2, mismatch in real part of RF current")
        
        Iref_imag = np.array(
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             3.10484642e-13, 2.98089282e-13, 2.80982448e-13,
             5.18869045e-13, 4.67572168e-13, 1.22665512e-12,
             8.59338117e-13, 2.73152518e-13, 2.97378683e-13,
             1.80328346e-13, 2.01426047e-14, 0.00000000e+00,
             -6.61203935e-13, -2.08165078e-12, -7.37511800e-12,
             -1.49524832e-11, -2.88263953e-11, -4.88612915e-11,
             -7.96463984e-11, -1.20822453e-10, -1.98527462e-10,
             -2.66395823e-10, -3.46908120e-10, -4.42520514e-10,
             -5.44764748e-10, -6.67874490e-10, -7.37049493e-10,
             -8.19736598e-10, -8.82688669e-10, -8.76856073e-10,
             -8.23077356e-10, -7.60096780e-10, -6.40949423e-10,
             -4.84431348e-10, -3.04617414e-10, -9.90179787e-11,
             1.12198045e-10, 3.03095363e-10, 4.96733215e-10,
             6.58625421e-10, 7.56906322e-10, 8.15663690e-10,
             8.79089063e-10, 8.49710096e-10, 8.43427611e-10,
             7.17289345e-10, 6.45446562e-10, 5.34741143e-10,
             4.27454878e-10, 3.49457174e-10, 2.58477023e-10,
             1.81627143e-10, 1.29594307e-10, 8.49203349e-11,
             5.24889624e-11, 3.36509712e-11, 1.08785464e-11,
             8.34217998e-12, 2.61896305e-12, 1.15825707e-12,
             5.37351873e-14, -2.23725006e-14, -1.86909360e-13,
             -2.02498004e-13, -4.15783756e-13, -1.73749600e-13,
             0.00000000e+00, -4.70617500e-13, 0.00000000e+00,
             -2.82049917e-13, 0.00000000e+00, -3.11029694e-13,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00])
        
        I_imag = np.around(rf_current.imag, 14) # round
        Iref_imag = np.around(Iref_imag, 14)

        self.assertSequenceEqual(I_imag.tolist(), Iref_imag.tolist(),
                                 msg="In TestRFCurrent test_2, mismatch in imaginary part of"
                                 + " RF current")

    # Test charge distribution of a bigaussian profile, with LPF
    # Compare to simulation data
    def test_3(self):

        bigaussian(self.ring, self.rf, self.beam, 3.2e-9 / 4, seed=1234,
                   reinsertion=True)
        self.profile.track()
        self.assertEqual(len(self.beam.dt), np.sum(self.profile.n_macroparticles), "In" +
                         " TestBeamCurrent: particle number mismatch in Beam vs Profile")

        # RF current calculation with low-pass filter
        rf_current = rf_beam_current(self.profile, self.omega,
                                     self.ring.t_rev[0], lpf=True)

        Iref_real = np.array([-4.8900028448e-12, -4.8922909174e-12, -4.8945126435e-12, -4.8966691279e-12,
                            -4.8987614779e-12, -4.9007908029e-12, -4.9027582136e-12, -4.9046648223e-12,
                            -4.9065117417e-12, -4.9083000848e-12, -4.9100309646e-12, -4.9117054934e-12,
                            -4.9133247826e-12, -4.9148899418e-12, -4.9164020789e-12, -4.9178622995e-12,
                            -4.9192717063e-12, -4.9206313987e-12, -4.9219424725e-12, -4.9232060192e-12,
                            -4.9244231261e-12, -4.9255948750e-12, -4.9267223427e-12, -4.9278065999e-12,
                            -4.9288487110e-12, -4.9298497337e-12, -4.9308107186e-12, -4.9317327086e-12,
                            -4.9326167387e-12, -4.9334638356e-12, -4.9342750168e-12, -4.9350512911e-12,
                            -4.9357936571e-12, -4.9365031039e-12, -4.9371806099e-12, -4.9378271427e-12,
                            -4.9384436588e-12, -4.9390311033e-12, -4.9395904092e-12, -4.9401224972e-12,
                            -4.9406282755e-12, -4.9411086395e-12, -4.9415644710e-12, -4.9419966385e-12,
                            -4.9424059964e-12, -4.9427933850e-12, -4.9431596299e-12, -4.9435055423e-12,
                            -4.9438319179e-12, -4.9441395375e-12, -4.9444291661e-12, -4.9447015529e-12,
                            -4.9449574313e-12, -4.9451975184e-12, -4.9454225147e-12, -4.9456331044e-12,
                            -4.9458299549e-12, -4.9460137168e-12, -4.9461850233e-12, -4.9463444911e-12,
                            -4.9464927191e-12, -4.9466302893e-12, -4.9467577662e-12, -4.9468756968e-12,
                            -4.9469846109e-12, -4.9470850208e-12, -4.9471774211e-12, -4.9472622895e-12,
                            -4.9473400860e-12, -4.9474112534e-12, -4.9474762175e-12, -4.9475353868e-12,
                            -4.9475891530e-12, -4.9476378910e-12, -4.9476819591e-12, -4.9477216990e-12,
                            -4.9477574364e-12, -4.9477894810e-12, -4.9478181265e-12, -4.9478436514e-12,
                            -4.9478663190e-12, -4.9478863775e-12, -4.9479040606e-12, -4.9479195881e-12,
                            -4.9479331654e-12, -4.9479449849e-12, -4.9479552256e-12, -4.9479640540e-12,
                            -4.9479716241e-12, -4.9479780782e-12, -4.9479835474e-12, -4.9479881515e-12,
                            -4.9479920001e-12, -4.9479951927e-12, -4.9479978194e-12, -4.9479999612e-12,
                            -4.9480016908e-12, -4.9480030725e-12, -4.9480041636e-12, -4.9480050141e-12,])

        np.testing.assert_allclose(rf_current.real, Iref_real, rtol=1e-5,
                                   atol=0, err_msg="In TestRFCurrent test_3, mismatch in real part of RF current")

        Iref_imag = np.array([-1.4943276366e-12, -1.4949805683e-12, -1.4956135945e-12, -1.4962270785e-12,
                            -1.4968213835e-12, -1.4973968728e-12, -1.4979539091e-12, -1.4984928546e-12,
                            -1.4990140711e-12, -1.4995179197e-12, -1.5000047602e-12, -1.5004749519e-12,
                            -1.5009288526e-12, -1.5013668189e-12, -1.5017892058e-12, -1.5021963668e-12,
                            -1.5025886537e-12, -1.5029664164e-12, -1.5033300028e-12, -1.5036797586e-12,
                            -1.5040160271e-12, -1.5043391493e-12, -1.5046494638e-12, -1.5049473062e-12,
                            -1.5052330094e-12, -1.5055069034e-12, -1.5057693150e-12, -1.5060205680e-12,
                            -1.5062609825e-12, -1.5064908756e-12, -1.5067105605e-12, -1.5069203467e-12,
                            -1.5071205402e-12, -1.5073114427e-12, -1.5074933521e-12, -1.5076665621e-12,
                            -1.5078313621e-12, -1.5079880374e-12, -1.5081368685e-12, -1.5082781317e-12,
                            -1.5084120983e-12, -1.5085390353e-12, -1.5086592047e-12, -1.5087728636e-12,
                            -1.5088802642e-12, -1.5089816536e-12, -1.5090772741e-12, -1.5091673624e-12,
                            -1.5092521503e-12, -1.5093318644e-12, -1.5094067258e-12, -1.5094769503e-12,
                            -1.5095427484e-12, -1.5096043251e-12, -1.5096618800e-12, -1.5097156072e-12,
                            -1.5097656954e-12, -1.5098123277e-12, -1.5098556818e-12, -1.5098959299e-12,
                            -1.5099332387e-12, -1.5099677693e-12, -1.5099996776e-12, -1.5100291139e-12,
                            -1.5100562230e-12, -1.5100811447e-12, -1.5101040131e-12, -1.5101249572e-12,
                            -1.5101441008e-12, -1.5101615624e-12, -1.5101774555e-12, -1.5101918885e-12,
                            -1.5102049649e-12, -1.5102167834e-12, -1.5102274377e-12, -1.5102370168e-12,
                            -1.5102456053e-12, -1.5102532831e-12, -1.5102601257e-12, -1.5102662043e-12,
                            -1.5102715860e-12, -1.5102763337e-12, -1.5102805062e-12, -1.5102841587e-12,
                            -1.5102873427e-12, -1.5102901058e-12, -1.5102924923e-12, -1.5102945432e-12,
                            -1.5102962963e-12, -1.5102977863e-12, -1.5102990449e-12, -1.5103001011e-12,
                            -1.5103009811e-12, -1.5103017088e-12, -1.5103023057e-12, -1.5103027908e-12,
                            -1.5103031812e-12, -1.5103034922e-12, -1.5103037369e-12, -1.5103039271e-12,])

        np.testing.assert_allclose(rf_current.imag, Iref_imag, rtol=1e-5,
                                   atol=0, err_msg="In TestRFCurrent test_3, mismatch in imaginary part of RF current")

    # Test RF beam current on coarse grid integrated from fine grid
    # Compare to simulation data for peak RF current
    def test_4(self):

        # Create a batch of 100 equal, short bunches
        bunches = 100
        T_s = 5 * self.rf.t_rev[0] / self.rf.harmonic[0, 0]
        N_m = int(1e5)
        N_b = 2.3e11
        bigaussian(self.ring, self.rf, self.beam, 0.1e-9, seed=1234,
                   reinsertion=True)
        beam2 = Beam(self.ring, bunches * N_m, bunches * N_b)
        bunch_spacing = 5 * self.rf.t_rf[0, 0]
        buckets = 5 * bunches
        for i in range(bunches):
            beam2.dt[i * N_m:(i + 1) * N_m] = self.beam.dt + i * bunch_spacing
            beam2.dE[i * N_m:(i + 1) * N_m] = self.beam.dE
        profile2 = Profile(beam2, CutOptions=CutOptions(cut_left=0,
                                                        cut_right=bunches * bunch_spacing, n_slices=1000 * buckets))
        profile2.track()

        tot_charges = np.sum(profile2.n_macroparticles) /\
            beam2.n_macroparticles * beam2.intensity
        self.assertAlmostEqual(tot_charges, 2.3000000000e+13, 9)

        # Calculate fine- and coarse-grid RF current
        rf_current_fine, rf_current_coarse = rf_beam_current(profile2,
                                                             self.rf.omega_rf[0, 0], self.ring.t_rev[0], lpf=False,
                                                             downsample={'Ts': T_s, 'points': self.rf.harmonic[0, 0] / 5})
        rf_current_coarse /= T_s

        # Peak RF current on coarse grid
        peak_rf_current = np.max(np.absolute(rf_current_coarse))
        self.assertAlmostEqual(peak_rf_current, 2.9284593979, 7)


class TestComb(unittest.TestCase):

    def test_1(self):
        y = np.random.rand(42)

        self.assertListEqual(y.tolist(), comb_filter(y, y, 15 / 16).tolist(),
                             msg="In TestComb test_1, filtered signal not correct")

    def test_2(self):

        t = np.arange(0, 2 * np.pi, 2 * np.pi / 120)
        y = np.cos(t)
        # Shift cosine by quarter period
        x = np.roll(y, int(len(t) / 4))

        # Drop some digits to avoid rounding errors
        result = np.around(comb_filter(y, x, 0.5), 12)
        result_theo = np.around(np.sin(np.pi / 4 + t) / np.sqrt(2), 12)

        self.assertListEqual(result.tolist(), result_theo.tolist(),
                             msg="In TestComb test_2, filtered signal not correct")


class TestLowPass(unittest.TestCase):

    def test_1(self):
        # Example based on SciPy.org filtfilt
        t = np.linspace(0, 1.0, 2001)
        xlow = np.sin(2 * np.pi * 5 * t)
        xhigh = np.sin(2 * np.pi * 250 * t)
        x = xlow + xhigh

        y = low_pass_filter(x, cutoff_frequency=1 / 8)

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
        ring = Ring(2 * np.pi * 1100.009, 1 / 18**2, 25.92e9, Particle=Proton())
        rf = RFStation(ring, [4620], [4.5e6], [0.], n_rf=1)
        self.T_s = 5 * rf.t_rf[0, 0]

    def test_1(self):

        # Modified filling time to match reference case
        TWC = SPS3Section200MHzTWC()
        TWC.tau = 420e-9
        filter, n_taps, n_filling, n_fit = feedforward_filter(TWC, 4 / 125 * 1e-6,
            taps=31, opt_output=True)
        self.assertEqual(n_taps, 31,
                         msg="In TestFeedforwardFilter, test_1: n_taps incorrect")
        self.assertEqual(n_filling, 13,
                         msg="In TestFeedforwardFilter, test_1: n_filling incorrect")
        self.assertEqual(n_fit, 44,
                         msg="In TestFeedforwardFilter, test_1: n_fit incorrect")

        filter_ref = np.array(
            [-0.02043835,  0.01895906,  0.00295858, -0.00237491,  0.01075756,  0.00394477,
              0.00394477,  0.00394477, -0.02169625, -0.02169625,  0.00394477,  0.00394477,
              0.00394477, -0.0096808,   0.01658415,  0.00591716, -0.00474983,  0.02151511,
              0.00788955,  0.00788955,  0.00788955,  0.03205128,  0.02317554, -0.00394477,
             -0.00394477, -0.00394477, -0.01075756,  0.00237491, -0.00295858, -0.01895906,
              0.02043835])

        np.testing.assert_allclose(np.around(filter, 6), np.around(filter_ref, 6), rtol=1e-8, atol=1e-9,
            err_msg="In TestFeedforwardFilter, test_1: filter array incorrect")

        del TWC

    def test_2(self):

        TWC = SPS3Section200MHzTWC()
        filter, n_taps, n_filling, n_fit = feedforward_filter(TWC, 1 / 40.0444e6, opt_output=True)
        self.assertEqual(n_taps, 31,
                         msg="In TestFeedforwardFilter, test_2: n_taps incorrect")
        self.assertEqual(n_filling, 18,
                         msg="In TestFeedforwardFilter, test_2: n_filling incorrect")
        self.assertEqual(n_fit, 49,
                         msg="In TestFeedforwardFilter, test_2: n_fit incorrect")

        filter_ref = np.copy(feedforward_filter_TWC3)

        np.testing.assert_allclose(np.around(filter, 6), np.around(filter_ref, 6), rtol=1e-8, atol=1e-9,
            err_msg="In TestFeedforwardFilter, test_2: filter array incorrect")

        del TWC

    def test_3(self):

        TWC = SPS4Section200MHzTWC()
        filter, n_taps, n_filling, n_fit = feedforward_filter(TWC, 1 / 40.0444e6, opt_output=True)
        self.assertEqual(n_taps, 37,
            msg="In TestFeedforwardFilter, test_3: n_taps incorrect")
        self.assertEqual(n_filling, 25,
            msg="In TestFeedforwardFilter, test_3: n_filling incorrect")
        self.assertEqual(n_fit, 62,
            msg="In TestFeedforwardFilter, test_3: n_fit incorrect")

        filter_ref = np.copy(feedforward_filter_TWC4)

        np.testing.assert_allclose(np.around(filter, 6), np.around(filter_ref, 6), rtol=1e-8, atol=1e-9,
            err_msg="In TestFeedforwardFilter, test_3: filter array incorrect")

        del TWC

    def test_4(self):

        TWC = SPS5Section200MHzTWC()
        filter, n_taps, n_filling, n_fit = feedforward_filter(TWC, 1 / 40.0444e6, opt_output=True)
        self.assertEqual(n_taps, 43,
                         msg="In TestFeedforwardFilter, test_4: n_taps incorrect")
        self.assertEqual(n_filling, 31,
                         msg="In TestFeedforwardFilter, test_4: n_filling incorrect")
        self.assertEqual(n_fit, 74,
                         msg="In TestFeedforwardFilter, test_4: n_fit incorrect")

        filter_ref = np.copy(feedforward_filter_TWC5)

        np.testing.assert_allclose(np.around(filter, 6), np.around(filter_ref, 6), rtol=1e-8, atol=1e-9,
            err_msg="In TestFeedforwardFilter, test_4: filter array incorrect")

        del TWC


if __name__ == '__main__':

    unittest.main()
