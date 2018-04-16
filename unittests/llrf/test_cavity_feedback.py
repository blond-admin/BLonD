# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:57:49 2017

@author: schwarz
"""

import unittest
import numpy as np
from scipy.constants import c

from input_parameters.ring import Ring
from input_parameters.rf_parameters import RFStation
from beam.beam import Beam, Proton
from beam.distributions import bigaussian
from beam.profile import Profile, CutOptions
from llrf.cavity_feedback import SPSCavityFeedback, CavityFeedbackCommissioning
from impedances.impedance import InducedVoltageTime, TotalInducedVoltage
from impedances.impedance_sources import TravelingWaveCavity
from trackers.tracker import RingAndRFTracker


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
        self.beam.dt += n_shift * self.rf.t_rf[0,0]

        self.profile = Profile(
            self.beam, CutOptions=CutOptions(
                cut_left=(n_shift-1.5)*self.rf.t_rf[0,0],
                cut_right=(n_shift+2.5)*self.rf.t_rf[0,0],
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
            turns=50, Commissioning=CavityFeedbackCommissioning())

        self.OTFB_tracker = RingAndRFTracker(self.rf, self.beam,
                                             Profile=self.profile,
                                             TotalInducedVoltage=None,
                                             CavityFeedback=self.OTFB,
                                             interpolation=True)
        
    def test_FB_pre_tracking(self):

        digit_round = 3

        Vind4_mean = np.around(
                np.mean(np.absolute(self.OTFB.OTFB_4.V_coarse_tot))/1e6,
                digit_round)
        Vind4_std = np.around(
                np.std(np.absolute(self.OTFB.OTFB_4.V_coarse_tot))/1e6,
                digit_round)
        Vind4_mean_exp = np.around(1.99886351363, digit_round)
        Vind4_std_exp = np.around(2.148426e-6, digit_round)

        Vind5_mean = np.around(
                np.mean(np.absolute(self.OTFB.OTFB_5.V_coarse_tot))/1e6,
                digit_round)
        Vind5_std = np.around(
                np.std(np.absolute(self.OTFB.OTFB_5.V_coarse_tot))/1e6,
                digit_round)
        Vind5_mean_exp = np.around(2.49906605189, digit_round)
        Vind5_std_exp = np.around(2.221665e-6, digit_round)

        self.assertEqual(Vind4_mean, Vind4_mean_exp,
            msg='In TestCavityFeedback test_FB_pretracking: '
            +'mean value of four-section cavity differs')
        self.assertEqual(Vind4_std, Vind4_std_exp,
            msg='In TestCavityFeedback test_FB_pretracking: standard '
            +'deviation of four-section cavity differs')

        self.assertEqual(Vind5_mean, Vind5_mean_exp,
            msg='In TestCavityFeedback test_FB_pretracking: '
            +'mean value of five-section cavity differs')
        self.assertEqual(Vind5_std, Vind5_std_exp,
            msg='In TestCavityFeedback test_FB_pretracking: standard '+
            'deviation of five-section cavity differs')
        
    def test_FB_pre_tracking_IQ_v1(self):
        digit_round = 2
        
        # interpolate from coarse mesh to fine mesh
        V_fine_tot_4 = np.interp(
                self.profile.bin_centers, self.OTFB.OTFB_4.rf_centers,
                self.OTFB.OTFB_4.V_coarse_ind_gen)
        V_fine_tot_5 = np.interp(
                self.profile.bin_centers, self.OTFB.OTFB_5.rf_centers,
                self.OTFB.OTFB_5.V_coarse_ind_gen)
        
        V_tot_4 = np.around(V_fine_tot_4/1e6, digit_round)
        V_tot_5 = np.around(V_fine_tot_5/1e6, digit_round)
        
        V_sum = np.around(self.OTFB.V_sum/1e6, digit_round)
        
        # expected generator voltage is only in Q
        V_tot_4_exp = 2.0j*np.ones(256)
        V_tot_5_exp = 2.5j*np.ones(256)
        V_sum_exp = 4.5j*np.ones(256)
        
        self.assertListEqual(V_tot_4.tolist(), V_tot_4_exp.tolist(),
            msg='In TestCavityFeedback test_FB_pretracking_IQ: total voltage '
            +'in four-section cavity differs')
        
        self.assertListEqual(V_tot_5.tolist(), V_tot_5_exp.tolist(),
            msg='In TestCavityFeedback test_FB_pretracking_IQ: total voltage '
            +'in five-section cavity differs')
        
        self.assertListEqual(V_sum.tolist(), V_sum_exp.tolist(),
            msg='In TestCavityFeedback test_FB_pretracking_IQ: voltage sum '
            +' differs')
        
    def test_rf_voltage(self):
        
        digit_round = 8
        
        # compute voltage
        self.cavity_tracker.rf_voltage_calculation()

        # compute voltage after OTFB pre-tracking
        self.OTFB_tracker.rf_voltage_calculation()

        # Since there is a systematic offset between the voltages,
        # compare the maxium of the ratio
        max_ratio = np.max(self.cavity_tracker.rf_voltage
                           / self.OTFB_tracker.rf_voltage)
        max_ratio = np.around(max_ratio, digit_round)
        
        max_ratio_exp = np.around(1.0008217052569774, digit_round)
        self.assertAlmostEqual(max_ratio, max_ratio_exp,
                               places=digit_round,
                         msg='In TestCavityFeedback test_rf_voltage: '
                         + 'RF-voltages differ')

    def test_beam_loading(self):

        digit_round = 10
        
        # Compute voltage with beam loading
        self.cavity_tracker.rf_voltage_calculation()
        cavity_tracker_total_voltage = self.cavity_tracker.rf_voltage \
            + self.cavity_tracker.totalInducedVoltage.induced_voltage

        self.OTFB.track()
        self.OTFB_tracker.rf_voltage_calculation()
        OTFB_tracker_total_voltage = self.OTFB_tracker.rf_voltage

        max_ratio = np.around(np.max(cavity_tracker_total_voltage
                                 / OTFB_tracker_total_voltage), digit_round)
        max_ration_exp = np.around(1.0051759770680779, digit_round)

        self.assertEqual(max_ratio, max_ration_exp,
                         msg='In TestCavityFeedback test_beam_loading: '
                         + 'total voltages differ')
        
    def test_Vsum_IQ(self):
        digit_round = 4
        
        self.OTFB.track()
        
        V_sum = np.around(self.OTFB.V_sum/1e6, digit_round)
        
        V_sum_exp = np.around(np.array([-7.40650823e+01+4497812.99202967j,
        -7.40650823e+01+4497812.99202967j, -7.40650823e+01+4497812.99202967j,
        -7.40650823e+01+4497812.99202967j, -7.40650822e+01+4497812.99202967j,
        -7.40650823e+01+4497812.99202967j, -7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202967j, -7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202967j, -7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202967j, -7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202967j, -7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202967j, -7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202967j, -7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202967j, -7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202967j, -7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202967j, -7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202967j, -7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202967j, -7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202967j, -7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202967j, -7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202967j, -7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202967j,-7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202967j,-7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202967j,-7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202968j,-7.40650822e+01+4497812.99202967j,
        -7.40650822e+01+4497812.99202968j,-7.40650822e+01+4497812.99202968j,
        -7.40650822e+01+4497812.99202968j,-7.40650822e+01+4497812.99202968j,
        -7.40650822e+01+4497812.99202968j,-7.40650822e+01+4497812.99202968j,
        -7.40650822e+01+4497812.99202968j,-7.40650822e+01+4497812.99202968j,
        -7.40650822e+01+4497812.99202968j,-7.40650822e+01+4497812.99202968j,
        -7.40650822e+01+4497812.99202968j,-7.40650822e+01+4497812.99202968j,
        -7.40650822e+01+4497812.99202968j,-7.40650822e+01+4497812.99202968j,
        -7.40650822e+01+4497812.99202968j,-7.40650822e+01+4497812.99202968j,
        -7.40650822e+01+4497812.99202968j,-7.40650822e+01+4497812.99202968j,
        -7.40650822e+01+4497812.99202968j,-7.40650822e+01+4497812.99202968j,
        -7.40650822e+01+4497812.99202968j,-7.40650822e+01+4497812.99202968j,
        -7.40650822e+01+4497812.99202968j,-7.40650822e+01+4497812.99202968j,
        -6.02318851e+01+4497817.94162674j,-1.98390915e+01+4497835.4539936j ,
         1.93158486e+01+4497855.56826354j,4.37055412e+01+4497871.87009121j,
         7.72626261e+01+4497900.35036866j,1.08879867e+02+4497930.96118169j,
         1.36226374e+02+4497965.43834138j,1.83914308e+02+4498039.43198652j,
         2.58060957e+02+4498182.7842137j ,3.46527521e+02+4498400.20917783j,
         4.26706222e+02+4498667.15544146j,4.94661306e+02+4499027.56784064j,
         5.43482338e+02+4499582.92538958j,5.37601327e+02+4500375.11759629j,
         4.18116316e+02+4501483.68340875j,1.06781854e+02+4502992.29896325j,
        -5.18346745e+02+4504994.88486696j,-1.49458714e+03+4507337.42385509j,
        -3.01163886e+03+4510119.3419273j ,-5.32851179e+03+4513546.45481774j,
        -8.74569640e+03+4517649.58201841j,-1.36711235e+04+4522515.97696859j,
        -2.06268308e+04+4528145.57335092j,-3.07188747e+04+4534772.43779405j,
        -4.42705242e+04+4541938.02912512j,-6.14651616e+04+4548954.45228089j,
        -8.23300421e+04+4555211.22162217j,-1.09421551e+05+4560368.34346543j,
        -1.43885886e+05+4563628.1865127j ,-1.85808399e+05+4563383.852869j  ,
        -2.36208282e+05+4558209.23829535j,-2.94836934e+05+4546270.53281669j,
        -3.63317854e+05+4525133.07455768j,-4.41779074e+05+4492246.31582297j,
        -5.28977031e+05+4445197.26263046j,-6.24427353e+05+4380854.18329653j,
        -7.28570463e+05+4294604.34393273j,-8.41413067e+05+4180725.12037253j,
        -9.58666802e+05+4036850.21934613j,-1.07489637e+06+3861588.24443417j,
        -1.18692079e+06+3650150.96222293j,-1.28868132e+06+3402561.33925834j,
        -1.37503355e+06+3113868.12951949j,-1.44047679e+06+2779413.13521708j,
        -1.47644078e+06+2402815.35953829j,-1.47522678e+06+1985042.23744602j,
        -1.42897968e+06+1528424.34271266j,-1.32931255e+06+1034805.30411738j,
        -1.16912937e+06 +511261.80102318j,-9.43345126e+05  -30406.39494472j,
        -6.44543913e+05 -585971.64269612j,-2.67898382e+05-1147041.0353738j ,
         1.87647422e+05-1699739.23544117j,7.19585769e+05-2229855.79356738j,
         1.32653137e+06-2725965.80099238j,2.00691053e+06-3178849.17084715j,
         2.75728744e+06-3578075.86057957j,3.56230981e+06-3910440.14562456j,
         4.40408617e+06-4164575.91355405j,5.28572695e+06-4338227.19020188j,
         6.19719564e+06-4426787.10912655j,7.11277008e+06-4425807.66220696j,
         8.01692653e+06-4335730.38899002j,8.89765817e+06-4159572.80505286j,
         9.74683705e+06-3900763.99853979j,1.05559999e+07-3564480.67012727j,
         1.13018718e+07-3165395.29765105j,1.19781005e+07-2712267.95526922j,
         1.25833208e+07-2214895.26359232j,1.31112101e+07-1685734.89623327j,
         1.35640429e+07-1132761.26804116j,1.39371556e+07 -573495.98893318j,
         1.42323033e+07  -19883.35687916j,1.44559502e+07 +521811.95451886j,
         1.46134106e+07+1043190.45283868j,1.47105373e+07+1534116.02749795j,
         1.47549265e+07+1994083.71851812j,1.47545428e+07+2416602.70892133j,
         1.47166958e+07+2794242.30539414j,1.46493955e+07+3129779.93382244j,
         1.45593030e+07+3424466.23125072j,1.44539743e+07+3677716.51226699j,
         1.43401341e+07+3889644.33404043j,1.42224899e+07+4064982.31004726j,
         1.41063737e+07+4205965.72167177j,1.39946111e+07+4317244.42689922j,
         1.38886184e+07+4403883.23649077j,1.37894739e+07+4469861.27187975j,
         1.36989283e+07+4518194.75268176j,1.36191316e+07+4551305.81768837j,
         1.35495619e+07+4572524.22309931j,1.34903101e+07+4584597.89085099j,
         1.34406305e+07+4589814.4489974j ,1.33976876e+07+4590220.82697034j,
         1.33615022e+07+4587255.76269093j,1.33327026e+07+4582254.09185628j,
         1.33096130e+07+4576070.77968165j,1.32911752e+07+4569436.1321998j ,
         1.32768524e+07+4562928.08977976j,1.32656893e+07+4556810.85976046j,
         1.32570926e+07+4551309.42853408j,1.32504722e+07+4546502.73564931j,
         1.32453253e+07+4542354.45990368j,1.32414716e+07+4539159.11692844j,
         1.32386032e+07+4536873.63706821j,1.32362022e+07+4534949.63932622j,
         1.32341515e+07+4533397.43716764j,1.32323621e+07+4532141.64712087j,
         1.32307439e+07+4531156.20064611j,1.32292493e+07+4530633.17835778j,
         1.32277994e+07+4530492.96280951j,1.32263821e+07+4530446.25647796j,
         1.32249901e+07+4530405.83108728j,1.32236081e+07+4530431.11420123j,
         1.32222495e+07+4530467.44843446j,1.32208432e+07+4530611.48233323j,
         1.32193886e+07+4530847.10244979j,1.32179361e+07+4531084.60180009j,
         1.32164993e+07+4531317.55923836j,1.32150516e+07+4531560.32993118j,
         1.32135685e+07+4531829.29611484j,1.32120853e+07+4532098.20168656j,
         1.32106022e+07+4532367.04664624j,1.32091191e+07+4532635.83099376j,
         1.32076359e+07+4532904.55472902j,1.32061675e+07+4533173.93875581j,
         1.32046990e+07+4533443.26260922j,1.32032158e+07+4533711.80494598j,
         1.32017326e+07+4533980.28666989j,1.32002495e+07+4534248.70778084j,
         1.31987663e+07+4534517.06827872j,1.31972831e+07+4534785.36816343j,
         1.31957999e+07+4535053.60743485j,1.31943167e+07+4535321.78609287j,
         1.31928335e+07+4535589.90413737j,1.31913503e+07+4535857.96156826j,
         1.31898671e+07+4536125.95838542j,1.31883839e+07+4536393.89458873j,
         1.31869007e+07+4536661.77017809j,1.31854174e+07+4536929.58515338j,
         1.31839342e+07+4537197.33951451j,1.31824510e+07+4537465.03326134j,
         1.31809677e+07+4537732.66639378j,1.31794845e+07+4538000.23891172j,
         1.31780012e+07+4538267.75081504j,1.31765179e+07+4538535.20210364j,
         1.31750347e+07+4538802.5927774j ,1.31735514e+07+4539069.92283622j,
         1.31720681e+07+4539337.19227997j,1.31705848e+07+4539604.40110857j,
         1.31691016e+07+4539871.54932188j,1.31676183e+07+4540138.63691982j,
         1.31661350e+07+4540405.66390225j,1.31646517e+07+4540672.63026908j,
         1.31631684e+07+4540939.53602019j,1.31616850e+07+4541206.38115549j,
         1.31602017e+07+4541473.16567484j,1.31587184e+07+4541739.88957815j,
         1.31572351e+07+4542006.55286531j,1.31557517e+07+4542273.1555362j ,
         1.31542684e+07+4542539.69759073j,1.31527850e+07+4542806.17902876j,
         1.31513017e+07+4543072.59985021j,1.31498183e+07+4543338.96005496j,
         1.31483350e+07+4543605.2596429j ,1.31468516e+07+4543871.49861392j,
         1.31453682e+07+4544137.6769679j ,1.31438848e+07+4544403.79470476j,
         1.31424014e+07+4544669.85182436j,1.31409181e+07+4544935.84832662j,
         1.31394347e+07+4545201.7842114j ,1.31379513e+07+4545467.65947862j,
         1.31364679e+07+4545733.47412815j,1.31349844e+07+4545999.22815989j,
         1.31335010e+07+4546264.92157373j,1.31320176e+07+4546530.55436956j,
         1.31305342e+07+4546796.12654728j,1.31290507e+07+4547061.63810677j,
         1.31275673e+07+4547327.08904792j,1.31260839e+07+4547592.47937064j,
         1.31246004e+07+4547857.8090748j ,1.31231170e+07+4548123.0781603j ,
         1.31216335e+07+4548388.28662704j,1.31201500e+07+4548653.4344749j ,
         1.31186666e+07+4548918.52170378j,1.31171831e+07+4549183.54831356j,
         1.31156996e+07+4549448.51430414j,1.31142161e+07+4549713.41967541j,
         1.31127326e+07+4549978.26442727j])/1e6, digit_round)

        self.assertListEqual(V_sum.tolist(), V_sum_exp.tolist(),
            msg='In TestCavityFeedback test_Vsum_IQ: total voltage '
            +'is different from expected values!')

if __name__ == '__main__':

    unittest.main()
