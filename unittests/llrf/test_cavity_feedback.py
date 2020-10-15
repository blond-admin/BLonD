# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:57:49 2017

@author: schwarz, Helga Timko
"""

import unittest
import numpy as np
from scipy.constants import c

from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import Profile, CutOptions
from blond.llrf.cavity_feedback import SPSCavityFeedback, CavityFeedbackCommissioning
from blond.llrf.cavity_feedback import LHCCavityLoop, LHCRFFeedback
from blond.impedances.impedance import InducedVoltageTime, TotalInducedVoltage
from blond.impedances.impedance_sources import TravelingWaveCavity
from blond.trackers.tracker import RingAndRFTracker


class TestCavityFeedback(unittest.TestCase):

    def setUp(self):
        C = 2*np.pi*1100.009        # Ring circumference [m]
        gamma_t = 18.0              # Gamma at transition
        alpha = 1/gamma_t**2        # Momentum compaction factor
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

        N_m = 1e6                   # Number of macro-particles for tracking
        N_b = 72*1.0e11             # Bunch intensity [ppb]

        # Gaussian beam profile
        self.beam = Beam(self.ring, N_m, N_b)
        sigma = 1.0e-9
        bigaussian(self.ring, self.rf, self.beam, sigma, seed=1234,
                   reinsertion=False)

        n_shift = 1550  # how many rf-buckets to shift beam
        self.beam.dt += n_shift * self.rf.t_rf[0, 0]

        self.profile = Profile(
            self.beam, CutOptions=CutOptions(
                cut_left=(n_shift-1.5)*self.rf.t_rf[0, 0],
                cut_right=(n_shift+2.5)*self.rf.t_rf[0, 0],
                n_slices=4*64))
        self.profile.track()

        # Cavities
        l_cav = 43*0.374
        v_g = 0.0946
        tau = l_cav/(v_g*c)*(1 + v_g)
        f_cav = 200.222e6
        n_cav = 2   # factor 2 because of two four/five-sections cavities
        short_cavity = TravelingWaveCavity(l_cav**2 * n_cav * 27.1e3 / 8,
                                           f_cav, 2*np.pi*tau)
        shortInducedVoltage = InducedVoltageTime(self.beam, self.profile,
                                                 [short_cavity])
        l_cav = 54*0.374
        tau = l_cav/(v_g*c)*(1 + v_g)
        long_cavity = TravelingWaveCavity(l_cav**2 * n_cav * 27.1e3 / 8, f_cav,
                                          2*np.pi*tau)
        longInducedVoltage = InducedVoltageTime(self.beam, self.profile,
                                                [long_cavity])
        self.induced_voltage = TotalInducedVoltage(
            self.beam, self.profile, [shortInducedVoltage, longInducedVoltage])
        self.induced_voltage.induced_voltage_sum()

        self.cavity_tracker = RingAndRFTracker(
            self.rf, self.beam, Profile=self.profile, interpolation=True,
            TotalInducedVoltage=self.induced_voltage)

        self.OTFB = SPSCavityFeedback(
            self.rf, self.beam, self.profile, G_llrf=5, a_comb=15/16, turns=50,
            post_LS2=False,
            Commissioning=CavityFeedbackCommissioning(open_FF=True))

        self.OTFB_tracker = RingAndRFTracker(self.rf, self.beam,
                                             Profile=self.profile,
                                             TotalInducedVoltage=None,
                                             CavityFeedback=self.OTFB,
                                             interpolation=True)

    def test_FB_pre_tracking(self):

        digit_round = 3

        Vind4_mean = np.mean(np.absolute(self.OTFB.OTFB_1.V_coarse_tot))/1e6
        Vind4_std = np.std(np.absolute(self.OTFB.OTFB_1.V_coarse_tot))/1e6
        Vind4_mean_exp = 1.99963935
        Vind4_std_exp = 1.686222e-05

        Vind5_mean = np.mean(np.absolute(self.OTFB.OTFB_2.V_coarse_tot))/1e6
        Vind5_std = np.std(np.absolute(self.OTFB.OTFB_2.V_coarse_tot))/1e6
        Vind5_mean_exp = 2.49928000
        Vind5_std_exp = 2.365198e-05

        self.assertAlmostEqual(Vind4_mean, Vind4_mean_exp,
                               places=digit_round,
                               msg='In TestCavityFeedback test_FB_pretracking: ' +
                               'mean value of four-section cavity differs')
        self.assertAlmostEqual(Vind4_std, Vind4_std_exp,
                               places=digit_round+5,
                               msg='In TestCavityFeedback test_FB_pretracking: standard ' +
                               'deviation of four-section cavity differs')

        self.assertAlmostEqual(Vind5_mean, Vind5_mean_exp,
                               places=digit_round,
                               msg='In TestCavityFeedback test_FB_pretracking: ' +
                               'mean value of five-section cavity differs')
        self.assertAlmostEqual(Vind5_std, Vind5_std_exp,
                               places=digit_round+5,
                               msg='In TestCavityFeedback test_FB_pretracking: standard '
                               + 'deviation of five-section cavity differs')

    def test_FB_pre_tracking_IQ_v1(self):
        rtol = 1e-3         # relative tolerance
        atol = 0            # absolute tolerance
        # interpolate from coarse mesh to fine mesh
        V_fine_tot_4 = np.interp(
            self.profile.bin_centers, self.OTFB.OTFB_1.rf_centers,
            self.OTFB.OTFB_1.V_coarse_ind_gen)
        V_fine_tot_5 = np.interp(
            self.profile.bin_centers, self.OTFB.OTFB_2.rf_centers,
            self.OTFB.OTFB_2.V_coarse_ind_gen)

        V_tot_4 = V_fine_tot_4/1e6
        V_tot_5 = V_fine_tot_5/1e6

        V_sum = self.OTFB.V_sum/1e6

        # expected generator voltage is only in Q
        V_tot_4_exp = 2.0j*np.ones(256)
        V_tot_5_exp = 2.5j*np.ones(256)
        V_sum_exp = 4.5j*np.ones(256)

        np.testing.assert_allclose(V_tot_4, V_tot_4_exp,
                                   rtol=rtol, atol=atol,
                                   err_msg='In TestCavityFeedback test_FB_pretracking_IQ: total voltage ' +
                                   'in four-section cavity differs')

        np.testing.assert_allclose(V_tot_5, V_tot_5_exp,
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
        max_ratio = max_ratio

        max_ratio_exp = 1.0010558589
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
        max_ratio_exp = 1.0027985539

        self.assertAlmostEqual(max_ratio, max_ratio_exp, places=digit_round,
                               msg='In TestCavityFeedback test_beam_loading: '
                               + 'total voltages differ')

    def test_Vsum_IQ(self):
        rtol = 1e-7         # relative tolerance
        atol = 0              # absolute tolerance

        self.OTFB.track()

        V_sum = self.OTFB.V_sum/1e6

        V_sum_exp = np.array([1.8006088204e-04 + 4.4989188699j,
            1.8006088204e-04 + 4.4989188699j,
            1.8006088204e-04 + 4.4989188699j,
            1.8006088204e-04 + 4.4989188699j,
            1.8006088204e-04 + 4.4989188699j,
            1.8006088204e-04 + 4.4989188699j,
            1.8006088204e-04 + 4.4989188699j,
            1.8006088204e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088204e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088203e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.8006088202e-04 + 4.4989188699j,
            1.9389407920e-04 + 4.4989238195j,
            2.3428687321e-04 + 4.4989413319j,
            2.7344181363e-04 + 4.4989614462j,
            2.9783150640e-04 + 4.498977748j,
            3.3138859155e-04 + 4.4990062283j,
            3.6300583226e-04 + 4.4990368391j,
            3.9035233998e-04 + 4.4990713163j,
            4.3804027469e-04 + 4.4991453099j,
            5.1218692414e-04 + 4.4992886621j,
            6.0065348834e-04 + 4.4995060871j,
            6.8083218990e-04 + 4.4997730334j,
            7.4878727491e-04 + 4.5001334458j,
            7.9760830694e-04 + 4.5006888033j,
            7.9172729660e-04 + 4.5014809955j,
            6.7224228449e-04 + 4.5025895614j,
            3.6090781970e-04 + 4.5040981769j,
            -2.6422078483e-04 + 4.5061007628j,
            -1.2404611905e-03 + 4.5084433018j,
            -2.7575129178e-03 + 4.5112252199j,
            -5.0743858693e-03 + 4.5146523329j,
            -8.4915705102e-03 + 4.5187554601j,
            -1.3416997682e-02 + 4.5236218551j,
            -2.0372704990e-02 + 4.5292514515j,
            -3.0464748978e-02 + 4.535878316j,
            -4.4016398615e-02 + 4.5430439074j,
            -6.1211036107e-02 + 4.5500603306j,
            -8.2075916843e-02 + 4.5563171j,
            -1.0916742557e-01 + 4.5614742219j,
            -1.4363176116e-01 + 4.564734065j,
            -1.8555427409e-01 + 4.5644897313j,
            -2.3595415789e-01 + 4.5593151167j,
            -2.9458281085e-01 + 4.5473764111j,
            -3.6306373097e-01 + 4.5262389527j,
            -4.4152495128e-01 + 4.4933521937j,
            -5.2872290962e-01 + 4.4463031401j,
            -6.2417323220e-01 + 4.3819600602j,
            -7.2831634287e-01 + 4.2957102202j,
            -8.4115894809e-01 + 4.1818309957j,
            -9.5841268400e-01 + 4.0379560935j,
            -1.0746422543e+00 + 3.8626941171j,
            -1.1866666692e+00 + 3.6512568332j,
            -1.2884272003e+00 + 3.4036672082j,
            -1.3747794340e+00 + 3.114973996j,
            -1.4402226741e+00 + 2.780518999j,
            -1.4761866645e+00 + 2.4039212202j,
            -1.4749726699e+00 + 1.9861480947j,
            -1.4287255608e+00 + 1.5295301962j,
            -1.3290584325e+00 + 1.0359111535j,
            -1.1688752569e+00 + 0.5123676461j,
            -9.4309100792e-01 - 0.0293005543j,
            -6.4428979275e-01 - 0.5848658066j,
            -2.6764425866e-01 - 1.1459352039j,
            1.8790154938e-01 - 1.6986334086j,
            7.1983990062e-01 - 2.2287499711j,
            1.3267855048e+00 - 2.7248599826j,
            2.0071646681e+00 - 3.1777433562j,
            2.7575415884e+00 - 3.5769700492j,
            3.5625639608e+00 - 3.909334337j,
            4.4043403290e+00 - 4.163470107j,
            5.2859811169e+00 - 4.3371213851j,
            6.1974498189e+00 - 4.4256813047j,
            7.1130242624e+00 - 4.4247018578j,
            8.0171807242e+00 - 4.3346245839j,
            8.8979123740e+00 - 4.1584669985j,
            9.7470912574e+00 - 3.8996581899j,
            1.0556254136e+01 - 3.5633748587j,
            1.1302125982e+01 - 3.1642894829j,
            1.1978354729e+01 - 2.7111621368j,
            1.2583574998e+01 - 2.2137894411j,
            1.3111464340e+01 - 1.6846290693j,
            1.3564297133e+01 - 1.1316554366j,
            1.3937409879e+01 - 0.5723901529j,
            1.4232557575e+01 - 0.0187775163j,
            1.4456204487e+01 + 0.5229177995j,
            1.4613664848e+01 + 1.0442963021j,
            1.4710791583e+01 + 1.5352218808j,
            1.4755180774e+01 + 1.9951895756j,
            1.4754797001e+01 + 2.4177085695j,
            1.4716950045e+01 + 2.7953481691j,
            1.4649649758e+01 + 3.1308858002j,
            1.4559557233e+01 + 3.4255721001j,
            1.4454228591e+01 + 3.6788223832j,
            1.4340388358e+01 + 3.8907502067j,
            1.4222744140e+01 + 4.0660881841j,
            1.4106627992e+01 + 4.2070715969j,
            1.3994865385e+01 + 4.318350303j,
            1.3888872669e+01 + 4.4049891133j,
            1.3789728183e+01 + 4.4709671492j,
            1.3699182542e+01 + 4.5193006304j,
            1.3619385862e+01 + 4.5524116957j,
            1.3549816151e+01 + 4.5736301013j,
            1.3490564358e+01 + 4.5857037691j,
            1.3440884732e+01 + 4.5909203273j,
            1.3397941870e+01 + 4.5913267053j,
            1.3361756453e+01 + 4.5883616409j,
            1.3332956859e+01 + 4.58335997j,
            1.3309867262e+01 + 4.5771766578j,
            1.3291429459e+01 + 4.5705420103j,
            1.3277106663e+01 + 4.5640339678j,
            1.3265943526e+01 + 4.5579167377j,
            1.3257346851e+01 + 4.5524153064j,
            1.3250726436e+01 + 4.5476086135j,
            1.3245579538e+01 + 4.5434603377j,
            1.3241725831e+01 + 4.5402649947j,
            1.3238857466e+01 + 4.5379795148j,
            1.3236456451e+01 + 4.536055517j,
            1.3234405717e+01 + 4.5345033148j,
            1.3232616328e+01 + 4.5332475248j,
            1.3230998101e+01 + 4.5322620783j,
            1.3229503536e+01 + 4.531739056j,
            1.3228053607e+01 + 4.5315988404j,
            1.3226636385e+01 + 4.531552134j,
            1.3225244299e+01 + 4.5315117086j,
            1.3223862329e+01 + 4.5315369917j,
            1.3222503779e+01 + 4.531573326j,
            1.3221097430e+01 + 4.5317173599j,
            1.3219642828e+01 + 4.53195298j,
            1.3218190351e+01 + 4.5321904793j,
            1.3216753536e+01 + 4.5324234367j,
            1.3215305822e+01 + 4.5326662074j,
            1.3213822701e+01 + 4.5329351736j,
            1.3212339574e+01 + 4.5332040792j,
            1.3210856442e+01 + 4.5334729241j,
            1.3209373304e+01 + 4.5337417084j,
            1.3207890161e+01 + 4.5340104322j,
            1.3206421687e+01 + 4.5342798162j,
            1.3204953204e+01 + 4.53454914j,
            1.3203470041e+01 + 4.5348176824j,
            1.3201986873e+01 + 4.5350861641j,
            1.3200503699e+01 + 4.5353545852j,
            1.3199020520e+01 + 4.5356229457j,
            1.3197537335e+01 + 4.5358912455j,
            1.3196054145e+01 + 4.5361594848j,
            1.3194570949e+01 + 4.5364276634j,
            1.3193087748e+01 + 4.5366957815j,
            1.3191604542e+01 + 4.5369638389j,
            1.3190121330e+01 + 4.5372318357j,
            1.3188638112e+01 + 4.5374997719j,
            1.3187154889e+01 + 4.5377676475j,
            1.3185671661e+01 + 4.5380354624j,
            1.3184188428e+01 + 4.5383032168j,
            1.3182705188e+01 + 4.5385709105j,
            1.3181221944e+01 + 4.5388385437j,
            1.3179738694e+01 + 4.5391061162j,
            1.3178255438e+01 + 4.5393736281j,
            1.3176772177e+01 + 4.5396410793j,
            1.3175288911e+01 + 4.53990847j,
            1.3173805639e+01 + 4.5401758j,
            1.3172322362e+01 + 4.5404430695j,
            1.3170839080e+01 + 4.5407102783j,
            1.3169355792e+01 + 4.5409774265j,
            1.3167872498e+01 + 4.5412445141j,
            1.3166389199e+01 + 4.5415115411j,
            1.3164905895e+01 + 4.5417785074j,
            1.3163422586e+01 + 4.5420454132j,
            1.3161939271e+01 + 4.5423122583j,
            1.3160455950e+01 + 4.5425790428j,
            1.3158972624e+01 + 4.5428457667j,
            1.3157489293e+01 + 4.54311243j,
            1.3156005956e+01 + 4.5433790326j,
            1.3154522614e+01 + 4.5436455747j,
            1.3153039267e+01 + 4.5439120561j,
            1.3151555914e+01 + 4.5441784769j,
            1.3150072555e+01 + 4.5444448371j,
            1.3148589192e+01 + 4.5447111367j,
            1.3147105823e+01 + 4.5449773756j,
            1.3145622448e+01 + 4.545243554j,
            1.3144139068e+01 + 4.5455096717j,
            1.3142655683e+01 + 4.5457757288j,
            1.3141172293e+01 + 4.5460417253j,
            1.3139688897e+01 + 4.5463076612j,
            1.3138205495e+01 + 4.5465735365j,
            1.3136722088e+01 + 4.5468393511j,
            1.3135238676e+01 + 4.5471051051j,
            1.3133755259e+01 + 4.5473707985j,
            1.3132271836e+01 + 4.5476364313j,
            1.3130788408e+01 + 4.5479020035j,
            1.3129304974e+01 + 4.548167515j,
            1.3127821535e+01 + 4.548432966j,
            1.3126338091e+01 + 4.5486983563j,
            1.3124854641e+01 + 4.548963686j,
            1.3123371186e+01 + 4.549228955j,
            1.3121887726e+01 + 4.5494941635j,
            1.3120404260e+01 + 4.5497593113j,
            1.3118920789e+01 + 4.5500243986j,
            1.3117437312e+01 + 4.5502894252j,
            1.3115953830e+01 + 4.5505543911j,
            1.3114470343e+01 + 4.5508192965j,
            1.3112986850e+01 + 4.5510841412j])

        np.testing.assert_allclose(V_sum_exp, V_sum,
                                   rtol=rtol, atol=atol,
                                   err_msg='In TestCavityFeedback test_Vsum_IQ: total voltage ' +
                                   'is different from expected values!')



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
        self.RFFB = LHCRFFeedback(open_drive=True)
        self.f_c = self.rf.omega_rf[0,0]/(2*np.pi)


    def test_1(self):
        CL = LHCCavityLoop(self.rf, self.profile, f_c=self.f_c, G_gen=1,
                           I_gen_offset=0.2778, n_cav=8, n_pretrack=0,
                           Q_L=20000, R_over_Q=45, tau_loop=650e-9,
                           tau_otfb=1472e-9, RFFB=self.RFFB)
        CL.track_one_turn()
        # Steady-state antenna voltage [MV]
        V_ant = np.mean(np.absolute(CL.V_ANT[-10:]))*1e-6
        self.assertAlmostEqual(V_ant, 0.49817991, places=7)
        # Updated generator current [A]
        I_gen = np.mean(np.absolute(CL.I_GEN[-CL.n_coarse:]))
        self.assertAlmostEqual(I_gen, 0.2778000000, places=10)
        # Generator power [kW]
        P_gen = CL.generator_power()[-1]*1e-3
        self.assertAlmostEqual(P_gen, 34.7277780000, places=10)


    def test_2(self):
        CL = LHCCavityLoop(self.rf, self.profile, f_c=self.f_c, G_gen=1,
                           I_gen_offset=0.2778, n_cav=8, n_pretrack=0,
                           Q_L=60000, R_over_Q=45, tau_loop=650e-9,
                           tau_otfb=1472e-9, RFFB=self.RFFB)
        CL.track_one_turn()
        # Steady-state antenna voltage [MV]
        V_ant = np.mean(np.absolute(CL.V_ANT[-10:]))*1e-6
        self.assertAlmostEqual(V_ant, 1.26745787, places=7)
        # Updated generator current [A]
        I_gen = np.mean(np.absolute(CL.I_GEN[-CL.n_coarse:]))
        self.assertAlmostEqual(I_gen, 0.2778000000, places=10)
        # Generator power [kW]
        P_gen = CL.generator_power()[-1]*1e-3
        self.assertAlmostEqual(P_gen, 104.1833340000, places=10)


    def test_3(self):
        CL = LHCCavityLoop(self.rf, self.profile, f_c=self.f_c, G_gen=1,
                           I_gen_offset=0.2778, n_cav=8, n_pretrack=0,
                           Q_L=20000, R_over_Q=90, tau_loop=650e-9,
                           tau_otfb=1472e-9, RFFB=self.RFFB)
        CL.track_one_turn()
        # Steady-state antenna voltage [MV]
        V_ant = np.mean(np.absolute(CL.V_ANT[-10:]))*1e-6
        self.assertAlmostEqual(V_ant, 0.99635982, places=7)
        # Updated generator current [A]
        I_gen = np.mean(np.absolute(CL.I_GEN[-CL.n_coarse:]))
        self.assertAlmostEqual(I_gen, 0.2778000000, places=10)
        # Generator power [kW]
        P_gen = CL.generator_power()[-1]*1e-3
        self.assertAlmostEqual(P_gen, 69.4555560000, places=10)

if __name__ == '__main__':

    unittest.main()
