# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for llrf.cavity_feedback

:Authors: **Birk Emil Karlsen-Bæck**, **Helga Timko**
"""

import os
import unittest

import numpy as np
from scipy.constants import c

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, Profile
from blond.impedances.impedance import InducedVoltageTime, TotalInducedVoltage
from blond.impedances.impedance_sources import TravelingWaveCavity
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.llrf.cavity_feedback import (SPSCavityLoopCommissioning,
                                        SPSCavityFeedback, SPSOneTurnFeedback)
from blond.llrf.cavity_feedback import LHCCavityLoop, LHCCavityLoopCommissioning
from blond.trackers.tracker import RingAndRFTracker

this_directory = os.path.dirname(os.path.realpath(__file__))

class TestSPSCavityFeedback(unittest.TestCase):

    def setUp(self):
        C = 2 * np.pi * 1100.009        # Ring circumference [m]
        gamma_t = 18.0              # Gamma at transition
        alpha = 1 / gamma_t**2        # Momentum compaction factor
        p_s = 25.92e9               # Synchronous momentum at injection [eV]
        h = 4620                    # 200 MHz system harmonic
        phi = 0.                    # 200 MHz RF phase

        # With this setting, amplitude in the two four-section, five-section
        # cavities must converge, respectively, to
        # 2.0 MV = 4.5 MV * 4/18 * 2
        # 2.5 MV = 4.5 MV * 5/18 * 2
        V = 4.5e6                   # 200 MHz RF voltage

        N_t = 1                     # Number of turns to track

        self.ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=N_t)
        self.rf = RFStation(self.ring, h, V, phi)

        N_m = int(1e6)                   # Number of macro-particles for tracking
        N_b = 288 * 2.3e11               # Bunch intensity [ppb]

        # Gaussian beam profile
        self.beam = Beam(self.ring, N_m, N_b)
        sigma = 1.0e-9
        bigaussian(self.ring, self.rf, self.beam, sigma, seed=1234,
                   reinsertion=False)

        n_shift = 1550  # how many rf-buckets to shift beam
        self.beam.dt += n_shift * self.rf.t_rf[0, 0]

        self.profile = Profile(
            self.beam, CutOptions=CutOptions(
                cut_left=(n_shift - 1.5) * self.rf.t_rf[0, 0],
                cut_right=(n_shift + 2.5) * self.rf.t_rf[0, 0],
                n_slices=4 * 64))
        self.profile.track()

        # Cavities
        l_cav = 32 * 0.374
        v_g = 0.0946
        tau = l_cav / (v_g * c) * (1 + v_g)
        f_cav = 200.222e6
        n_cav = 4   # factor 2 because of two four/five-sections cavities
        short_cavity = TravelingWaveCavity(l_cav**2 * n_cav * 27.1e3 / 8,
                                           f_cav, 2 * np.pi * tau)
        shortInducedVoltage = InducedVoltageTime(self.beam, self.profile,
                                                 [short_cavity])
        l_cav = 43 * 0.374
        tau = l_cav / (v_g * c) * (1 + v_g)
        n_cav = 2
        long_cavity = TravelingWaveCavity(l_cav**2 * n_cav * 27.1e3 / 8,
                                          f_cav, 2 * np.pi * tau)
        longInducedVoltage = InducedVoltageTime(self.beam, self.profile,
                                                [long_cavity])
        self.induced_voltage = TotalInducedVoltage(
            self.beam, self.profile, [shortInducedVoltage, longInducedVoltage])
        self.induced_voltage.induced_voltage_sum()

        self.cavity_tracker = RingAndRFTracker(
            self.rf, self.beam, Profile=self.profile, interpolation=True,
            TotalInducedVoltage=self.induced_voltage)

        self.OTFB = SPSCavityFeedback(
            self.rf, self.profile, G_llrf=20, G_tx=[1.0355739238973907, 1.078403005653143],
            a_comb=63/64, turns=1000, post_LS2=True, df=[0.18433333e6, 0.2275e6],
            commissioning=SPSCavityLoopCommissioning(open_ff=True, rot_iq=-1))

        self.OTFB_tracker = RingAndRFTracker(self.rf, self.beam,
                                             Profile=self.profile,
                                             TotalInducedVoltage=None,
                                             CavityFeedback=self.OTFB,
                                             interpolation=True)

    def test_FB_pre_tracking(self):

        digit_round = 3

        Vind3_mean = np.mean(np.absolute(self.OTFB.OTFB_1.V_ANT_COARSE[-self.OTFB.OTFB_1.n_coarse:]))/1e6
        Vind3_std = np.std(np.absolute(self.OTFB.OTFB_1.V_ANT_COARSE[-self.OTFB.OTFB_1.n_coarse:]))/1e6
        Vind3_mean_exp = 0.6761952255314454
        Vind3_std_exp = 5.802516784274078e-13

        Vind4_mean = np.mean(np.absolute(self.OTFB.OTFB_2.V_ANT_COARSE[-self.OTFB.OTFB_2.n_coarse:]))/1e6
        Vind4_std = np.std(np.absolute(self.OTFB.OTFB_2.V_ANT_COARSE[-self.OTFB.OTFB_2.n_coarse:]))/1e6
        Vind4_mean_exp = 0.9028493757326258
        Vind4_std_exp = 8.817799015245741e-13

        self.assertAlmostEqual(Vind3_mean, Vind3_mean_exp,
                               places=digit_round,
                               msg='In TestCavityFeedback test_FB_pretracking: ' +
                               'mean value of four-section cavity differs')
        self.assertAlmostEqual(Vind3_std, Vind3_std_exp,
                               places=digit_round,
                               msg='In TestCavityFeedback test_FB_pretracking: standard ' +
                               'deviation of four-section cavity differs')

        self.assertAlmostEqual(Vind4_mean, Vind4_mean_exp,
                               places=digit_round,
                               msg='In TestCavityFeedback test_FB_pretracking: ' +
                               'mean value of five-section cavity differs')
        self.assertAlmostEqual(Vind4_std, Vind4_std_exp,
                               places=digit_round,
                               msg='In TestCavityFeedback test_FB_pretracking: standard '
                               + 'deviation of five-section cavity differs')

    def test_FB_pre_tracking_IQ_v1(self):
        rtol = 1e-2         # relative tolerance
        atol = 0            # absolute tolerance
        # interpolate from coarse mesh to fine mesh
        V_fine_tot_3 = np.interp(
            self.profile.bin_centers, self.OTFB.OTFB_1.rf_centers,
            self.OTFB.OTFB_1.V_IND_COARSE_GEN[-self.OTFB.OTFB_1.n_coarse:])
        V_fine_tot_4 = np.interp(
            self.profile.bin_centers, self.OTFB.OTFB_2.rf_centers,
            self.OTFB.OTFB_2.V_IND_COARSE_GEN[-self.OTFB.OTFB_2.n_coarse:])

        V_tot_3 = V_fine_tot_3 / 1e6 * self.OTFB.OTFB_1.n_cavities
        V_tot_4 = V_fine_tot_4 / 1e6 * self.OTFB.OTFB_2.n_cavities

        V_sum = self.OTFB.V_sum / 1e6

        # expected generator voltage is only in Q
        V_tot_3_exp = 2.7j * np.ones(256)
        V_tot_4_exp = 1.8j * np.ones(256)
        V_sum_exp = 4.5j * np.ones(256)

        np.testing.assert_allclose(V_tot_3, V_tot_3_exp,
                                   rtol=rtol, atol=atol,
                                   err_msg='In TestCavityFeedback test_FB_pretracking_IQ: total voltage ' +
                                   'in four-section cavity differs')

        np.testing.assert_allclose(V_tot_4, V_tot_4_exp,
                                   rtol=rtol, atol=atol,
                                   err_msg='In TestCavityFeedback test_FB_pretracking_IQ: total voltage ' +
                                   'in five-section cavity differs')

        np.testing.assert_allclose(V_sum, V_sum_exp,
                                   rtol=rtol, atol=atol,
                                   err_msg='In TestCavityFeedback test_FB_pretracking_IQ: voltage sum ' +
                                   ' differs')

    def test_rf_voltage(self):

        digit_round = 7

        # compute voltage
        self.cavity_tracker.rf_voltage_calculation()

        # compute voltage after OTFB pre-tracking
        self.OTFB_tracker.rf_voltage_calculation()

        # Since there is a systematic offset between the voltages,
        # compare the maxium of the ratio
        max_ratio = np.max(self.cavity_tracker.rf_voltage
                           / self.OTFB_tracker.rf_voltage)

        max_ratio_exp = 1.0691789378342162

        self.assertAlmostEqual(max_ratio, max_ratio_exp,
                               places=digit_round,
                               msg='In TestCavityFeedback test_rf_voltage: '
                               + 'RF-voltages differ')

    def test_beam_loading(self):
        digit_round = 7
        # Compute voltage with beam loading
        self.cavity_tracker.rf_voltage_calculation()
        cavity_tracker_total_voltage = self.cavity_tracker.rf_voltage \
            + self.cavity_tracker.totalInducedVoltage.induced_voltage

        self.OTFB.track()
        self.OTFB_tracker.rf_voltage_calculation()
        OTFB_tracker_total_voltage = self.OTFB_tracker.rf_voltage

        max_ratio = np.max(cavity_tracker_total_voltage /
                           OTFB_tracker_total_voltage)


        max_ratio_exp = 1.0691789378319636

        self.assertAlmostEqual(max_ratio, max_ratio_exp, places=digit_round,
                               msg='In TestCavityFeedback test_beam_loading: '
                               + 'total voltages differ')

    def test_Vsum_IQ(self):
        rtol = 1e-7         # relative tolerance
        atol = 0              # absolute tolerance

        self.OTFB.track()

        V_sum = self.OTFB.V_sum / 1e6

        V_sum_exp = np.array([1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.481752689e-02 + 4.510454808e+00j,
                            1.481752689e-02 + 4.510454808e+00j, 1.467003927e-02 + 4.510476685e+00j,
                            1.452259724e-02 + 4.510498562e+00j, 1.452264283e-02 + 4.510498562e+00j,
                            1.452268843e-02 + 4.510498561e+00j, 1.439484547e-02 + 4.510575214e+00j,
                            1.426704440e-02 + 4.510651849e+00j, 1.404617816e-02 + 4.510852091e+00j,
                            1.362512627e-02 + 4.511273235e+00j, 1.306981939e-02 + 4.511973098e+00j,
                            1.210167295e-02 + 4.513474984e+00j, 1.078779283e-02 + 4.515980194e+00j,
                            9.836259646e-03 + 4.518163804e+00j, 8.970183292e-03 + 4.521323325e+00j,
                            8.005504272e-03 + 4.526140001e+00j, 7.452065772e-03 + 4.532667025e+00j,
                            7.483017158e-03 + 4.541598489e+00j, 8.506539890e-03 + 4.551668244e+00j,
                            1.096244654e-02 + 4.563623978e+00j, 1.619690543e-02 + 4.580235448e+00j,
                            2.792405194e-02 + 4.607443250e+00j, 4.597134134e-02 + 4.641175739e+00j,
                            7.033402302e-02 + 4.677170985e+00j, 1.099375478e-01 + 4.724391792e+00j,
                            1.621238226e-01 + 4.776614334e+00j, 2.295290786e-01 + 4.831030852e+00j,
                            3.216513566e-01 + 4.891949350e+00j, 4.513066495e-01 + 4.959771077e+00j,
                            6.227821363e-01 + 5.030037721e+00j, 8.394069660e-01 + 5.094274038e+00j,
                            1.113238419e+00 + 5.147372980e+00j, 1.455753644e+00 + 5.179051447e+00j,
                            1.858949185e+00 + 5.177942195e+00j, 2.335009136e+00 + 5.128133857e+00j,
                            2.911053526e+00 + 5.010649906e+00j, 3.590070132e+00 + 4.800842826e+00j,
                            4.374313837e+00 + 4.471893817e+00j, 5.257406144e+00 + 3.994807057e+00j,
                            6.236486342e+00 + 3.334433633e+00j, 7.299218926e+00 + 2.454889827e+00j,
                            8.441367209e+00 + 1.302043808e+00j, 9.619790054e+00 + -1.431878086e-01j,
                            1.078588057e+01 + -1.902936195e+00j, 1.190448503e+01 + -4.012173697e+00j,
                            1.293268248e+01 + -6.523056046e+00j, 1.381664661e+01 + -9.472290261e+00j,
                            1.448239457e+01 + -1.288685295e+01j, 1.485143621e+01 + -1.674102755e+01j,
                            1.483992960e+01 + -2.095675968e+01j, 1.436704002e+01 + -2.559246763e+01j,
                            1.335839107e+01 + -3.059886818e+01j, 1.172864164e+01 + -3.591159047e+01j,
                            9.405852675e+00 + -4.148017912e+01j, 6.362767821e+00 + -4.714251574e+01j,
                            2.551801496e+00 + -5.281570912e+01j, -2.060352411e+00 + -5.840971185e+01j,
                            -7.443164880e+00 + -6.377131279e+01j, -1.356848136e+01 + -6.877704179e+01j,
                            -2.041816782e+01 + -7.333358029e+01j, -2.797169811e+01 + -7.734977038e+01j,
                            -3.617705825e+01 + -8.072875822e+01j, -4.488133996e+01 + -8.335278565e+01j,
                            -5.386298588e+01 + -8.512637679e+01j, -6.304266613e+01 + -8.601255040e+01j,
                            -7.235080816e+01 + -8.599689233e+01j, -8.155032660e+01 + -8.507817883e+01j,
                            -9.043638606e+01 + -8.329863930e+01j, -9.896600061e+01 + -8.069405104e+01j,
                            -1.070668907e+02 + -7.732382369e+01j, -1.146401037e+02 + -7.325710986e+01j,
                            -1.215732229e+02 + -6.861045362e+01j, -1.277377627e+02 + -6.353511085e+01j,
                            -1.331082617e+02 + -5.814733047e+01j, -1.376921000e+02 + -5.253932111e+01j,
                            -1.414931450e+02 + -4.682871685e+01j, -1.445085482e+02 + -4.116149899e+01j,
                            -1.467954368e+02 + -3.560363078e+01j, -1.484015646e+02 + -3.027212706e+01j,
                            -1.493796149e+02 + -2.529825410e+01j, -1.498198307e+02 + -2.070001278e+01j,
                            -1.498059244e+02 + -1.645703513e+01j, -1.494141759e+02 + -1.260122729e+01j,
                            -1.487254135e+02 + -9.194310050e+00j, -1.478140791e+02 + -6.229440009e+00j,
                            -1.467401971e+02 + -3.666445341e+00j, -1.455771042e+02 + -1.509168141e+00j,
                            -1.443839999e+02 + 2.598553928e-01j, -1.431903768e+02 + 1.700762864e+00j,
                            -1.420233660e+02 + 2.856833166e+00j, -1.409337576e+02 + 3.744875834e+00j,
                            -1.399385306e+02 + 4.403406937e+00j, -1.390159460e+02 + 4.891876339e+00j,
                            -1.381811642e+02 + 5.235603747e+00j, -1.374439159e+02 + 5.458671982e+00j,
                            -1.368085639e+02 + 5.586731787e+00j, -1.362659691e+02 + 5.642764332e+00j,
                            -1.358243790e+02 + 5.648552277e+00j, -1.354773565e+02 + 5.620510027e+00j,
                            -1.351877344e+02 + 5.570879849e+00j, -1.349501899e+02 + 5.508917179e+00j,
                            -1.347610938e+02 + 5.442895280e+00j, -1.346137951e+02 + 5.378789620e+00j,
                            -1.344995383e+02 + 5.319627106e+00j, -1.344087722e+02 + 5.265182832e+00j,
                            -1.343376502e+02 + 5.218027112e+00j, -1.342800623e+02 + 5.175638409e+00j,
                            -1.342336381e+02 + 5.140497996e+00j, -1.341996556e+02 + 5.118665146e+00j,
                            -1.341707253e+02 + 5.100847060e+00j, -1.341444026e+02 + 5.084935558e+00j,
                            -1.341211674e+02 + 5.074371007e+00j, -1.340996990e+02 + 5.068345477e+00j,
                            -1.340792378e+02 + 5.064350457e+00j, -1.340592146e+02 + 5.061565710e+00j,
                            -1.340393057e+02 + 5.061197778e+00j, -1.340194638e+02 + 5.061764011e+00j,
                            -1.339997145e+02 + 5.062678068e+00j, -1.339799966e+02 + 5.063934804e+00j,
                            -1.339603403e+02 + 5.065400833e+00j, -1.339403005e+02 + 5.067505666e+00j,
                            -1.339199270e+02 + 5.070108329e+00j, -1.338995535e+02 + 5.072710190e+00j,
                            -1.338790694e+02 + 5.075411344e+00j, -1.338587201e+02 + 5.078047912e+00j,
                            -1.338385110e+02 + 5.080633437e+00j, -1.338181671e+02 + 5.083281895e+00j,
                            -1.337976827e+02 + 5.085979747e+00j, -1.337771983e+02 + 5.088676761e+00j,
                            -1.337567138e+02 + 5.091372938e+00j, -1.337362293e+02 + 5.094068279e+00j,
                            -1.337157447e+02 + 5.096762782e+00j, -1.336952601e+02 + 5.099456448e+00j,
                            -1.336747754e+02 + 5.102149276e+00j, -1.336542906e+02 + 5.104841268e+00j,
                            -1.336338058e+02 + 5.107532423e+00j, -1.336133209e+02 + 5.110222740e+00j,
                            -1.335928360e+02 + 5.112912221e+00j, -1.335723510e+02 + 5.115600864e+00j,
                            -1.335518660e+02 + 5.118288670e+00j, -1.335313809e+02 + 5.120975639e+00j,
                            -1.335108958e+02 + 5.123661771e+00j, -1.334904106e+02 + 5.126347066e+00j,
                            -1.334699254e+02 + 5.129031524e+00j, -1.334494401e+02 + 5.131715144e+00j,
                            -1.334289548e+02 + 5.134397928e+00j, -1.334084694e+02 + 5.137079874e+00j,
                            -1.333879839e+02 + 5.139760983e+00j, -1.333674984e+02 + 5.142441255e+00j,
                            -1.333470128e+02 + 5.145120690e+00j, -1.333265272e+02 + 5.147799288e+00j,
                            -1.333060415e+02 + 5.150477049e+00j, -1.332855558e+02 + 5.153153972e+00j,
                            -1.332650700e+02 + 5.155830059e+00j, -1.332445842e+02 + 5.158505308e+00j,
                            -1.332240983e+02 + 5.161179720e+00j, -1.332036124e+02 + 5.163853295e+00j,
                            -1.331831264e+02 + 5.166526033e+00j, -1.331626403e+02 + 5.169197934e+00j,
                            -1.331421542e+02 + 5.171868997e+00j, -1.331216681e+02 + 5.174539224e+00j,
                            -1.331011819e+02 + 5.177208613e+00j, -1.330806956e+02 + 5.179877165e+00j,
                            -1.330602093e+02 + 5.182544880e+00j, -1.330397229e+02 + 5.185211758e+00j,
                            -1.330192365e+02 + 5.187877799e+00j, -1.329987501e+02 + 5.190543002e+00j,
                            -1.329782635e+02 + 5.193207369e+00j, -1.329577769e+02 + 5.195870898e+00j,
                            -1.329372903e+02 + 5.198533590e+00j, -1.329168036e+02 + 5.201195445e+00j,
                            -1.328963169e+02 + 5.203856463e+00j, -1.328758301e+02 + 5.206516643e+00j,
                            -1.328553432e+02 + 5.209175987e+00j, -1.328348563e+02 + 5.211834493e+00j,
                            -1.328143694e+02 + 5.214492162e+00j, -1.327938824e+02 + 5.217148994e+00j,
                            -1.327733953e+02 + 5.219804989e+00j, -1.327529082e+02 + 5.222460147e+00j,
                            -1.327324210e+02 + 5.225114467e+00j, -1.327119338e+02 + 5.227767951e+00j,
                            -1.326914466e+02 + 5.230420597e+00j, -1.326709592e+02 + 5.233072406e+00j,
                            -1.326504719e+02 + 5.235723378e+00j, -1.326299844e+02 + 5.238373513e+00j,
                            -1.326094969e+02 + 5.241022810e+00j, -1.325890094e+02 + 5.243671271e+00j,
                            -1.325685218e+02 + 5.246318894e+00j, -1.325480342e+02 + 5.248965680e+00j,
                            -1.325275465e+02 + 5.251611629e+00j, -1.325070587e+02 + 5.254256740e+00j,
                            -1.324865709e+02 + 5.256901015e+00j, -1.324660831e+02 + 5.259544452e+00j,])

        np.testing.assert_allclose(np.around(V_sum_exp, 5), np.around(V_sum, 5),
                                   rtol=rtol, atol=atol,
                                   err_msg='In TestCavityFeedback test_Vsum_IQ: total voltage ' +
                                   'is different from expected values!')


class TestSPSOneTurnFeedback(unittest.TestCase):

    def setUp(self):
        # Parameters ----------------------------------------------------------
        C = 2 * np.pi * 1100.009                # Ring circumference [m]
        gamma_t = 18.0                          # Transition Gamma [-]
        alpha = 1 / (gamma_t ** 2)              # Momentum compaction factor [-]
        p_s = 450e9                             # Synchronous momentum [eV]
        h = 4620                                # 200 MHz harmonic number [-]
        V = 10e6                                # 200 MHz RF voltage [V]
        phi = 0                                 # 200 MHz phase [-]

        # Parameters for the Simulation
        N_m = int(1e5)                          # Number of macro-particles for tracking
        N_b = 1.0e11                            # Bunch intensity [ppb]
        N_t = 1                                 # Number of turns to track

        # Objects -------------------------------------------------------------

        # Ring
        self.ring = Ring(C, alpha, p_s, Proton(), N_t)

        # RFStation
        self.rfstation = RFStation(self.ring, [h], [V], [phi], n_rf=1)

        # Beam
        self.beam = Beam(self.ring, N_m, N_b)
        self.profile = Profile(self.beam, CutOptions=CutOptions(cut_left=0.e-9,
                                                                cut_right=self.rfstation.t_rev[0], n_slices=4620))
        self.profile.track()

        # Cavity
        self.Commissioning = SPSCavityLoopCommissioning(open_ff=True, rot_iq=-1, cpp_conv=False)

        self.OTFB = SPSOneTurnFeedback(self.rfstation, self.profile, 3, a_comb=63 / 64,
                                       commissioning=self.Commissioning)

        self.OTFB.update_rf_variables()
        self.OTFB.update_fb_variables()

        self.turn_array = np.linspace(0, 2 * self.rfstation.t_rev[0], 2 * self.OTFB.n_coarse)

    def test_set_point(self):
        self.OTFB.set_point()
        t_sig = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        t_sig[-self.OTFB.n_coarse:] = (1 / 9) * 10e6 * np.exp(1j * (np.pi / 2 - self.rfstation.phi_rf[0, 0]))

        np.testing.assert_allclose(self.OTFB.V_SET, t_sig)

    def test_error_and_gain(self):
        self.OTFB.error_and_gain()

        np.testing.assert_allclose(self.OTFB.DV_GEN, self.OTFB.V_SET * self.OTFB.G_llrf)

    def test_comb(self):
        sig = np.zeros(self.OTFB.n_coarse)
        self.OTFB.DV_COMB_OUT = np.sin(2 * np.pi * self.turn_array / self.rfstation.t_rev[0])
        self.OTFB.DV_GEN = -np.sin(2 * np.pi * self.turn_array / self.rfstation.t_rev[0])
        self.OTFB.a_comb = 0.5

        self.OTFB.comb()

        np.testing.assert_allclose(self.OTFB.DV_COMB_OUT[-self.OTFB.n_coarse:], sig)

    def test_one_turn_delay(self):
        self.OTFB.DV_COMB_OUT = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_COMB_OUT[self.OTFB.n_coarse] = 1

        self.OTFB.one_turn_delay()

        self.assertEqual(np.argmax(self.OTFB.DV_DELAYED), 2 * self.OTFB.n_coarse - self.OTFB.n_mov_av)

    def test_mod_to_fr(self):
        self.OTFB.DV_DELAYED = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_DELAYED[-self.OTFB.n_coarse:] = 1 + 1j * 0

        self.mod_phi = np.copy(self.OTFB.dphi_mod)
        self.OTFB.mod_to_fr()
        ref_DV_MOD_FR = np.load(os.path.join(this_directory, "ref_DV_MOD_FR.npy"))

        # Test real part
        np.testing.assert_allclose(self.OTFB.DV_MOD_FR[-self.OTFB.n_coarse:].real,
                                   ref_DV_MOD_FR.real,
                                   rtol=1e-6, atol=1e-9,
                                   err_msg="In TestSPSOneTurnFeedback test_mod_to_fr(), "
                                           "mismatch in real part of modulated signal")

        # Test imaginary part
        np.testing.assert_allclose(self.OTFB.DV_MOD_FR[-self.OTFB.n_coarse:].imag,
                                   ref_DV_MOD_FR.imag,
                                   rtol=1e-6, atol=1e-9,
                                   err_msg="In TestSPSOneTurnFeedback test_mod_to_fr(), "
                                           "mismatch in imaginary part of modulated signal")

        self.OTFB.DV_DELAYED = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_DELAYED[-self.OTFB.n_coarse:] = 1 + 1j * 0

        self.OTFB.dphi_mod = 0
        self.OTFB.mod_to_fr()

        time_array = self.OTFB.rf_centers - 0.5*self.OTFB.T_s
        ref_sig = np.cos((self.OTFB.omega_carrier - self.OTFB.omega_c) * time_array[:self.OTFB.n_coarse]) + \
                  1j * np.sin((self.OTFB.omega_carrier - self.OTFB.omega_c) * time_array[:self.OTFB.n_coarse])

        np.testing.assert_allclose(self.OTFB.DV_MOD_FR[-self.OTFB.n_coarse:], ref_sig)

        self.OTFB.dphi_mod = self.mod_phi

    def test_mov_avg(self):
        sig = np.zeros(self.OTFB.n_coarse - 1)
        sig[:self.OTFB.n_mov_av] = 1
        self.OTFB.DV_MOD_FR = np.zeros(2 * self.OTFB.n_coarse)
        self.OTFB.DV_MOD_FR[-self.OTFB.n_coarse + 1:] = sig

        self.OTFB.mov_avg()

        sig = np.zeros(self.OTFB.n_coarse)
        sig[:self.OTFB.n_mov_av] = (1 / self.OTFB.n_mov_av) * np.array(range(self.OTFB.n_mov_av))
        sig[self.OTFB.n_mov_av: 2 * self.OTFB.n_mov_av] = (1 / self.OTFB.n_mov_av) * (self.OTFB.n_mov_av
                                                                                      - np.array(range(self.OTFB.n_mov_av)))

        np.testing.assert_allclose(np.abs(self.OTFB.DV_MOV_AVG[-self.OTFB.n_coarse:]), sig)

    def test_mod_to_frf(self):
        self.OTFB.DV_MOV_AVG = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_MOV_AVG[-self.OTFB.n_coarse:] = 1 + 1j * 0

        self.mod_phi = np.copy(self.OTFB.dphi_mod)
        self.OTFB.mod_to_frf()
        ref_DV_MOD_FRF = np.load(os.path.join(this_directory, "ref_DV_MOD_FRF.npy"))

        # Test real part
        np.testing.assert_allclose(self.OTFB.DV_MOD_FRF[-self.OTFB.n_coarse:].real,
                                   ref_DV_MOD_FRF.real,
                                   rtol=1e-6, atol=1e-9,
                                   err_msg="In TestSPSOneTurnFeedback test_mod_to_frf(), "
                                           "mismatch in real part of modulated signal")

        # Test imaginary part
        np.testing.assert_allclose(self.OTFB.DV_MOD_FRF[-self.OTFB.n_coarse:].imag,
                                   ref_DV_MOD_FRF.imag,
                                   rtol=1e-6, atol=1e-9,
                                   err_msg="In TestSPSOneTurnFeedback test_mod_to_frf(), "
                                           "mismatch in imaginary part of modulated signal")

        self.OTFB.DV_MOV_AVG = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_MOV_AVG[-self.OTFB.n_coarse:] = 1 + 1j * 0

        self.OTFB.dphi_mod = 0
        self.OTFB.mod_to_frf()

        time_array = self.OTFB.rf_centers - 0.5*self.OTFB.T_s
        dphi_demod = (self.OTFB.omega_c - self.OTFB.omega_carrier) * self.OTFB.TWC.tau
        ref_sig = np.cos(-(self.OTFB.omega_carrier
                           - self.OTFB.omega_c) * time_array[:self.OTFB.n_coarse] - dphi_demod) + \
                  1j * np.sin(-(self.OTFB.omega_carrier
                                - self.OTFB.omega_c) * time_array[:self.OTFB.n_coarse] - dphi_demod)

        np.testing.assert_allclose(self.OTFB.DV_MOD_FRF[-self.OTFB.n_coarse:], ref_sig)

        self.OTFB.dphi_mod = self.mod_phi

    def test_sum_and_gain(self):
        self.OTFB.V_SET[-self.OTFB.n_coarse:] = np.ones(self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_MOD_FRF[-self.OTFB.n_coarse:] = np.ones(self.OTFB.n_coarse, dtype=complex)

        self.OTFB.sum_and_gain()

        sig = 2 * np.ones(self.OTFB.n_coarse) * self.OTFB.G_tx / self.OTFB.TWC.R_gen

        np.testing.assert_allclose(self.OTFB.I_GEN_COARSE[-self.OTFB.n_coarse:], sig)


    def test_gen_response(self):
        # Tests generator response at resonant frequency.
        self.OTFB.I_GEN_COARSE = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.I_GEN_COARSE[self.OTFB.n_coarse] = 1

        self.OTFB.TWC.impulse_response_gen(self.OTFB.TWC.omega_r, self.OTFB.rf_centers)
        self.OTFB.gen_response()

        sig = np.zeros(self.OTFB.n_coarse)
        sig[1:1 + self.OTFB.n_mov_av] = 4 * self.OTFB.TWC.R_gen / self.OTFB.TWC.tau
        sig[0] = 2 * self.OTFB.TWC.R_gen / self.OTFB.TWC.tau
        sig[self.OTFB.n_mov_av + 1] = 2 * self.OTFB.TWC.R_gen / self.OTFB.TWC.tau
        sig *= self.OTFB.T_s

        np.testing.assert_allclose(
            np.abs(self.OTFB.V_IND_COARSE_GEN[-self.OTFB.n_coarse:]) * self.OTFB.n_cavities,
            sig, atol=5e-5
        )

        # Tests generator response at carrier frequency.
        self.OTFB.TWC.impulse_response_gen(self.OTFB.omega_c, self.OTFB.rf_centers)

        self.OTFB.I_GEN_COARSE = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.I_GEN_COARSE[self.OTFB.n_coarse] = 1

        self.OTFB.gen_response()

        ref_V_IND_COARSE_GEN = np.load(os.path.join(this_directory, "ref_V_IND_COARSE_GEN.npy"))

        # Test real part - sum of cavities
        np.testing.assert_allclose(
            np.around(self.OTFB.V_IND_COARSE_GEN[-self.OTFB.n_coarse:].real, 12) * self.OTFB.n_cavities,
            np.around(ref_V_IND_COARSE_GEN.real, 12),
            rtol=1e-6, atol=0,
           err_msg="In TestSPSOneTurnFeedback test_gen_response(), "
                   "mismatch in real part of generator response"
        )

        # Test imaginary part - sum of cavities
        np.testing.assert_allclose(
            np.around(self.OTFB.V_IND_COARSE_GEN[-self.OTFB.n_coarse:].imag, 12) * self.OTFB.n_cavities,
           np.around(ref_V_IND_COARSE_GEN.imag, 12),
           rtol=1e-6, atol=0,
           err_msg="In TestSPSOneTurnFeedback test_gen_response(), "
                   "mismatch in imaginary part of generator response"
        )


class TestSPSTransmitterGain(unittest.TestCase):

    def setUp(self):
        # Set up machine parameters
        self.ring = Ring(2 * np.pi * 1100.009, 1/18.0**2, 25.92e9, Particle=Proton(),
                    n_turns=1)
        # Set up RF parameters
        self.rf = RFStation(self.ring, [4620], [4.5e6], [0.], n_rf=1)
        self.rf.omega_rf[0, 0] = 200.222e6 * 2 * np.pi
        # Define beam and fill it
        self.beam = Beam(self.ring, int(1e5), 1.e11)
        bigaussian(self.ring, self.rf, self.beam, 3.2e-9/4, seed=1234,
                   reinsertion=True)
        self.profile = Profile(self.beam, CutOptions=CutOptions(cut_left=0.e-9,
            cut_right=self.rf.t_rev[0], n_slices=4620))
        self.profile.track()
        # Commissioning options for the cavity feedback
        self.commissioning = SPSCavityLoopCommissioning(debug=True,
            open_loop=False, open_fb=True, open_drive=False, open_ff=True, rot_iq=-1)

    def init_otfb(self, rf, profile, commissioning,
                  no_sections, no_cavities, V_part, G_tx):

        OTFB = SPSOneTurnFeedback(rf, profile, no_sections,
                                  n_cavities=no_cavities, V_part=V_part,
                                  G_ff=0,
                                  G_llrf=5, G_tx=G_tx, a_comb=15/16,
                                  commissioning=commissioning)
        for i in range(100):
            OTFB.track_no_beam()

        V = np.average(np.absolute(OTFB.V_ANT_COARSE[-10])) * OTFB.n_cavities * 1e-6  # in MV
        I = np.average(np.absolute(OTFB.I_GEN_COARSE[-10])) * 1e-2  # in 100 A

        return OTFB, V, I

    def test_preLS24sec(self):

        OTFB, V, I = self.init_otfb(self.rf, self.profile,
                                    self.commissioning, 4, 2, 4/9, 1.03573985)
        self.assertAlmostEqual(V, 2.00000000, places=7)
        self.assertAlmostEqual(I, 0.78244888, places=7)

    def test_preLS25sec(self):

        OTFB, V, I = self.init_otfb(self.rf, self.profile,
                                    self.commissioning, 5, 2, 5/9, 1.01547845)
        self.assertAlmostEqual(V, 2.50000000, places=7)
        self.assertAlmostEqual(I, 0.76359084, places=7)

    def test_postLS23sec(self):

        OTFB, V, I = self.init_otfb(self.rf, self.profile,
                                    self.commissioning, 3, 4, 6/10, 1.01724955)
        self.assertAlmostEqual(V, 2.70000000, places=7)
        self.assertAlmostEqual(I, 0.69703574, places=7)

    def test_postLS24sec(self):

        OTFB, V, I = self.init_otfb(self.rf, self.profile,
                                    self.commissioning, 4, 2, 4/10, 1.03573985)
        self.assertAlmostEqual(V, 1.80000000, places=7)
        self.assertAlmostEqual(I, 0.70420400, places=7)


class TestLHCOpenDrive(unittest.TestCase):


    def setUp(self):
        # Bunch parameters (dummy)
        N_b = 1e9             # Intensity
        N_p = 50000           # Macro-particles
        # Machine and RF parameters
        C = 26658.883         # Machine circumference [m]
        p_s = 450e9           # Synchronous momentum [eV/c]
        h = 35640             # Harmonic number
        V = 4e6               # RF voltage [V]
        dphi = 0              # Phase modulation/offset
        gamma_t = 53.8        # Transition gamma
        alpha = 1/gamma_t**2  # First order mom. comp. factor

        # Initialise necessary classes
        ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=1)
        self.rf = RFStation(ring, [h], [V], [dphi])
        beam = Beam(ring, N_p, N_b)
        self.profile = Profile(beam)

        # Test in open loop, on tune
        self.RFFB = LHCCavityLoopCommissioning(open_drive=True)
        self.f_c = self.rf.omega_rf[0,0]/(2*np.pi)


    def test_1(self):
        CL = LHCCavityLoop(self.rf, self.profile, f_c=self.f_c, G_gen=1,
                           I_gen_offset=0.2778, n_cavities=8, n_pretrack=0,
                           Q_L=20000, R_over_Q=45, tau_loop=650e-9,
                           tau_otfb=1472e-9, RFFB=self.RFFB)
        CL.track_one_turn()
        # Steady-state antenna voltage [MV]
        V_ant = np.mean(np.absolute(CL.V_ANT_COARSE[-10:])) * 1e-6
        self.assertAlmostEqual(V_ant, 0.49817991, places=7)
        # Updated generator current [A]
        I_gen = np.mean(np.absolute(CL.I_GEN_COARSE[-CL.n_coarse:]))
        self.assertAlmostEqual(I_gen, 0.2778000000, places=10)
        # Generator power [kW]
        P_gen = CL.generator_power()[-1]*1e-3
        self.assertAlmostEqual(P_gen, 34.7277780000, places=10)


    def test_2(self):
        CL = LHCCavityLoop(self.rf, self.profile, f_c=self.f_c, G_gen=1,
                           I_gen_offset=0.2778, n_cavities=8, n_pretrack=0,
                           Q_L=60000, R_over_Q=45, tau_loop=650e-9,
                           tau_otfb=1472e-9, RFFB=self.RFFB)
        CL.track_one_turn()
        # Steady-state antenna voltage [MV]
        V_ant = np.mean(np.absolute(CL.V_ANT_COARSE[-10:]))*1e-6
        self.assertAlmostEqual(V_ant, 1.26745787, places=7)
        # Updated generator current [A]
        I_gen = np.mean(np.absolute(CL.I_GEN_COARSE[-CL.n_coarse:]))
        self.assertAlmostEqual(I_gen, 0.2778000000, places=10)
        # Generator power [kW]
        P_gen = CL.generator_power()[-1]*1e-3
        self.assertAlmostEqual(P_gen, 104.1833340000, places=10)


    def test_3(self):
        CL = LHCCavityLoop(self.rf, self.profile, f_c=self.f_c, G_gen=1,
                           I_gen_offset=0.2778, n_cavities=8, n_pretrack=0,
                           Q_L=20000, R_over_Q=90, tau_loop=650e-9,
                           tau_otfb=1472e-9, RFFB=self.RFFB)
        CL.track_one_turn()
        # Steady-state antenna voltage [MV]
        V_ant = np.mean(np.absolute(CL.V_ANT_COARSE[-10:]))*1e-6
        self.assertAlmostEqual(V_ant, 0.99635982, places=7)
        # Updated generator current [A]
        I_gen = np.mean(np.absolute(CL.I_GEN_COARSE[-CL.n_coarse:]))
        self.assertAlmostEqual(I_gen, 0.2778000000, places=10)
        # Generator power [kW]
        P_gen = CL.generator_power()[-1]*1e-3
        self.assertAlmostEqual(P_gen, 69.4555560000, places=10)


if __name__ == '__main__':
    unittest.main()
