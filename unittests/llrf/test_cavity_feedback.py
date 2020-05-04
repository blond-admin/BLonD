# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:57:49 2017

@author: schwarz
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
            self.rf, self.beam, self.profile, G_llrf=5, G_tx=0.5, a_comb=15/16,
            turns=50, post_LS2=False,
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
        Vind4_mean_exp = 1.99886351363
        Vind4_std_exp = 2.148426e-6

        Vind5_mean = np.mean(np.absolute(self.OTFB.OTFB_2.V_coarse_tot))/1e6

        Vind5_std = np.std(np.absolute(self.OTFB.OTFB_2.V_coarse_tot))/1e6

        Vind5_mean_exp = 2.49906605189
        Vind5_std_exp = 2.221665e-6

        self.assertAlmostEqual(Vind4_mean, Vind4_mean_exp,
                               places=digit_round,
                               msg='In TestCavityFeedback test_FB_pretracking: ' +
                               'mean value of four-section cavity differs')
        self.assertAlmostEqual(Vind4_std, Vind4_std_exp,
                               places=digit_round,
                               msg='In TestCavityFeedback test_FB_pretracking: standard ' +
                               'deviation of four-section cavity differs')

        self.assertAlmostEqual(Vind5_mean, Vind5_mean_exp,
                               places=digit_round,
                               msg='In TestCavityFeedback test_FB_pretracking: ' +
                               'mean value of five-section cavity differs')
        self.assertAlmostEqual(Vind5_std, Vind5_std_exp,
                               places=digit_round,
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

        max_ratio_exp = 1.0016540193319539
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
        max_ration_exp = 1.0055233047525063

        self.assertAlmostEqual(max_ratio, max_ration_exp, places=digit_round,
                               msg='In TestCavityFeedback test_beam_loading: '
                               + 'total voltages differ')

    def test_Vsum_IQ(self):
        rtol = 1e-7         # relative tolerance
        atol = 0              # absolute tolerance

        self.OTFB.track()

        V_sum = self.OTFB.V_sum/1e6

        V_sum_exp = np.array([2.5827763187e-04+4.497826566j ,  2.5827763187e-04+4.497826566j ,
            2.5827763187e-04+4.497826566j ,  2.5827763187e-04+4.497826566j ,
            2.5827763187e-04+4.497826566j ,  2.5827763187e-04+4.497826566j ,
            2.5827763187e-04+4.497826566j ,  2.5827763187e-04+4.497826566j ,
            2.5827763187e-04+4.497826566j ,  2.5827763187e-04+4.497826566j ,
            2.5827763187e-04+4.497826566j ,  2.5827763186e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763186e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763186e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763187e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763187e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763186e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763186e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763186e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763186e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763186e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763186e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763186e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763186e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763186e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763186e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763186e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763186e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763186e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763185e-04+4.497826566j ,
            2.5827763185e-04+4.497826566j ,  2.5827763186e-04+4.497826566j ,
            2.5827763185e-04+4.497826566j ,  2.5827763185e-04+4.497826566j ,
            2.5827763185e-04+4.497826566j ,  2.5827763185e-04+4.497826566j ,
            2.5827763186e-04+4.497826566j ,  2.5827763185e-04+4.497826566j ,
            2.5827763185e-04+4.497826566j ,  2.5827763185e-04+4.497826566j ,
            2.5827763185e-04+4.497826566j ,  2.5827763185e-04+4.497826566j ,
            2.5827763185e-04+4.497826566j ,  2.5827763185e-04+4.497826566j ,
            2.5827763185e-04+4.497826566j ,  2.5827763185e-04+4.497826566j ,
            2.5827763185e-04+4.497826566j ,  2.5827763185e-04+4.497826566j ,
            2.5827763185e-04+4.497826566j ,  2.7211082903e-04+4.4978315156j,
            3.1250362304e-04+4.4978490279j,  3.5165856346e-04+4.4978691422j,
            3.7604825623e-04+4.497885444j ,  4.0960534138e-04+4.4979139243j,
            4.4122258209e-04+4.4979445351j,  4.6856908981e-04+4.4979790123j,
            5.1625702452e-04+4.4980530059j,  5.9040367398e-04+4.4981963582j,
            6.7887023817e-04+4.4984137831j,  7.5904893974e-04+4.4986807294j,
            8.2700402474e-04+4.4990411418j,  8.7582505677e-04+4.4995964993j,
            8.6994404643e-04+4.5003886916j,  7.5045903432e-04+4.5014972574j,
            4.3912456953e-04+4.5030058729j, -1.8600403500e-04+4.5050084589j,
           -1.1622444406e-03+4.5073509979j, -2.6792961680e-03+4.510132916j ,
           -4.9961691195e-03+4.5135600289j, -8.4133537603e-03+4.5176631561j,
           -1.3338780932e-02+4.5225295511j, -2.0294488240e-02+4.5281591475j,
           -3.0386532228e-02+4.534786012j , -4.3938181865e-02+4.5419516034j,
           -6.1132819357e-02+4.5489680266j, -8.1997700093e-02+4.555224796j ,
           -1.0908920882e-01+4.5603819179j, -1.4355354441e-01+4.563641761j ,
           -1.8547605734e-01+4.5633974273j, -2.3587594114e-01+4.5582228127j,
           -2.9450459410e-01+4.5462841072j, -3.6298551422e-01+4.5251466487j,
           -4.4144673453e-01+4.4922598897j, -5.2864469287e-01+4.4452108361j,
           -6.2409501545e-01+4.3808677563j, -7.2823812612e-01+4.2946179162j,
           -8.4108073134e-01+4.1807386917j, -9.5833446725e-01+4.0368637895j,
           -1.0745640376e+00+3.8616018131j, -1.1865884525e+00+3.6501645292j,
           -1.2883489836e+00+3.4025749042j, -1.3747012172e+00+3.1138816921j,
           -1.4401444574e+00+2.779426695j , -1.4761084477e+00+2.4028289162j,
           -1.4748944532e+00+1.9850557907j, -1.4286473441e+00+1.5284378922j,
           -1.3289802158e+00+1.0348188495j, -1.1687970402e+00+0.5112753421j,
           -9.4301279117e-01-0.0303928583j, -6.4421157600e-01-0.5859581106j,
           -2.6756604191e-01-1.1470275079j,  1.8797976613e-01-1.6997257125j,
            7.1991811737e-01-2.229842275j ,  1.3268637215e+00-2.7259522866j,
            2.0072428849e+00-3.1788356601j,  2.7576198052e+00-3.5780623532j,
            3.5626421776e+00-3.910426641j ,  4.4044185457e+00-4.164562411j ,
            5.2860593337e+00-4.3382136891j,  6.1975280356e+00-4.4267736087j,
            7.1131024791e+00-4.4257941618j,  8.0172589410e+00-4.3357168878j,
            8.8979905908e+00-4.1595593025j,  9.7471694742e+00-3.9007504938j,
            1.0556332353e+01-3.5644671627j,  1.1302204198e+01-3.1653817869j,
            1.1978432945e+01-2.7122544408j,  1.2583653215e+01-2.214881745j ,
            1.3111542557e+01-1.6857213733j,  1.3564375350e+01-1.1327477406j,
            1.3937488096e+01-0.5734824569j,  1.4232635792e+01-0.0198698203j,
            1.4456282704e+01+0.5218254956j,  1.4613743064e+01+1.0432039982j,
            1.4710869799e+01+1.5341295769j,  1.4755258991e+01+1.9940972717j,
            1.4754875217e+01+2.4166162655j,  1.4717028262e+01+2.7942558651j,
            1.4649727975e+01+3.1297934963j,  1.4559635450e+01+3.4244797961j,
            1.4454306808e+01+3.6777300792j,  1.4340466575e+01+3.8896579027j,
            1.4222822357e+01+4.0649958801j,  1.4106706208e+01+4.2059792929j,
            1.3994943602e+01+4.3172579991j,  1.3888950886e+01+4.4038968093j,
            1.3789806400e+01+4.4698748453j,  1.3699260758e+01+4.5182083265j,
            1.3619464079e+01+4.5513193917j,  1.3549894368e+01+4.5725377973j,
            1.3490642575e+01+4.5846114651j,  1.3440962949e+01+4.5898280233j,
            1.3398020086e+01+4.5902344013j,  1.3361834669e+01+4.587269337j ,
            1.3333035076e+01+4.5822676661j,  1.3309945479e+01+4.5760843538j,
            1.3291507676e+01+4.5694497063j,  1.3277184880e+01+4.5629416638j,
            1.3266021743e+01+4.5568244337j,  1.3257425068e+01+4.5513230024j,
            1.3250804652e+01+4.5465163095j,  1.3245657754e+01+4.5423680337j,
            1.3241804048e+01+4.5391726907j,  1.3238935683e+01+4.5368872108j,
            1.3236534668e+01+4.534963213j ,  1.3234483934e+01+4.5334110108j,
            1.3232694545e+01+4.5321552208j,  1.3231076317e+01+4.5311697743j,
            1.3229581752e+01+4.530646752j ,  1.3228131824e+01+4.5305065364j,
            1.3226714601e+01+4.5304598301j,  1.3225322516e+01+4.5304194047j,
            1.3223940546e+01+4.5304446878j,  1.3222581996e+01+4.530481022j ,
            1.3221175647e+01+4.5306250559j,  1.3219721045e+01+4.530860676j ,
            1.3218268568e+01+4.5310981753j,  1.3216831753e+01+4.5313311328j,
            1.3215384039e+01+4.5315739034j,  1.3213900918e+01+4.5318428696j,
            1.3212417791e+01+4.5321117752j,  1.3210934659e+01+4.5323806201j,
            1.3209451521e+01+4.5326494045j,  1.3207968378e+01+4.5329181282j,
            1.3206499904e+01+4.5331875122j,  1.3205031421e+01+4.533456836j ,
            1.3203548258e+01+4.5337253784j,  1.3202065089e+01+4.5339938601j,
            1.3200581916e+01+4.5342622812j,  1.3199098736e+01+4.5345306417j,
            1.3197615552e+01+4.5347989416j,  1.3196132362e+01+4.5350671808j,
            1.3194649166e+01+4.5353353595j,  1.3193165965e+01+4.5356034775j,
            1.3191682758e+01+4.5358715349j,  1.3190199546e+01+4.5361395317j,
            1.3188716329e+01+4.5364074679j,  1.3187233106e+01+4.5366753435j,
            1.3185749878e+01+4.5369431585j,  1.3184266644e+01+4.5372109128j,
            1.3182783405e+01+4.5374786066j,  1.3181300161e+01+4.5377462397j,
            1.3179816911e+01+4.5380138122j,  1.3178333655e+01+4.5382813241j,
            1.3176850394e+01+4.5385487754j,  1.3175367128e+01+4.538816166j ,
            1.3173883856e+01+4.5390834961j,  1.3172400579e+01+4.5393507655j,
            1.3170917296e+01+4.5396179743j,  1.3169434008e+01+4.5398851225j,
            1.3167950715e+01+4.5401522101j,  1.3166467416e+01+4.5404192371j,
            1.3164984112e+01+4.5406862034j,  1.3163500802e+01+4.5409531092j,
            1.3162017487e+01+4.5412199543j,  1.3160534167e+01+4.5414867388j,
            1.3159050841e+01+4.5417534627j,  1.3157567510e+01+4.542020126j ,
            1.3156084173e+01+4.5422867287j,  1.3154600831e+01+4.5425532707j,
            1.3153117483e+01+4.5428197521j,  1.3151634130e+01+4.5430861729j,
            1.3150150772e+01+4.5433525331j,  1.3148667409e+01+4.5436188327j,
            1.3147184039e+01+4.5438850717j,  1.3145700665e+01+4.54415125j  ,
            1.3144217285e+01+4.5444173677j,  1.3142733900e+01+4.5446834249j,
            1.3141250509e+01+4.5449494213j,  1.3139767113e+01+4.5452153572j,
            1.3138283712e+01+4.5454812325j,  1.3136800305e+01+4.5457470471j,
            1.3135316893e+01+4.5460128011j,  1.3133833476e+01+4.5462784945j,
            1.3132350053e+01+4.5465441273j,  1.3130866624e+01+4.5468096995j,
            1.3129383191e+01+4.5470752111j,  1.3127899752e+01+4.547340662j ,
            1.3126416308e+01+4.5476060523j,  1.3124932858e+01+4.547871382j ,
            1.3123449403e+01+4.5481366511j,  1.3121965942e+01+4.5484018595j,
            1.3120482476e+01+4.5486670074j,  1.3118999005e+01+4.5489320946j,
            1.3117515529e+01+4.5491971212j,  1.3116032047e+01+4.5494620872j,
            1.3114548560e+01+4.5497269925j,  1.3113065067e+01+4.5499918373j])

        np.testing.assert_allclose(V_sum_exp, V_sum,
                                   rtol=rtol, atol=atol,
                                   err_msg='In TestCavityFeedback test_Vsum_IQ: total voltage ' +
                                   'is different from expected values!')


if __name__ == '__main__':

    unittest.main()
