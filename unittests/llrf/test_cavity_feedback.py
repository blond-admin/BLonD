# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Unittest for llrf.cavity_feedback

:Authors: **Birk Emil Karlsen-Bæck**, **Helga Timko**
'''

import unittest
import numpy as np
from scipy.constants import c

from blond.llrf.cavity_feedback import SPSOneTurnFeedback, SPSCavityFeedback, CavityFeedbackCommissioning
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions import bigaussian
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.trackers.tracker import RingAndRFTracker
from blond.impedances.impedance import TotalInducedVoltage, InducedVoltageTime
from blond.impedances.impedance_sources import TravelingWaveCavity

class TestSPSCavityFeedback(unittest.TestCase):

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
                cut_left=(n_shift-1.5)*self.rf.t_rf[0, 0],
                cut_right=(n_shift+2.5)*self.rf.t_rf[0, 0],
                n_slices=4*64))
        self.profile.track()

        # Cavities
        l_cav = 32*0.374
        v_g = 0.0946
        tau = l_cav/(v_g*c)*(1 + v_g)
        f_cav = 200.222e6
        n_cav = 4   # factor 2 because of two four/five-sections cavities
        short_cavity = TravelingWaveCavity(l_cav**2 * n_cav * 27.1e3 / 8,
                                            f_cav, 2*np.pi*tau)
        shortInducedVoltage = InducedVoltageTime(self.beam, self.profile,
                                                 [short_cavity])
        l_cav = 43*0.374
        tau = l_cav/(v_g*c)*(1 + v_g)
        n_cav = 2
        long_cavity = TravelingWaveCavity(l_cav**2 * n_cav * 27.1e3 / 8,
                                           f_cav, 2*np.pi*tau)
        longInducedVoltage = InducedVoltageTime(self.beam, self.profile,
                                                [long_cavity])
        self.induced_voltage = TotalInducedVoltage(
            self.beam, self.profile, [shortInducedVoltage, longInducedVoltage])
        self.induced_voltage.induced_voltage_sum()

        self.cavity_tracker = RingAndRFTracker(
            self.rf, self.beam, Profile=self.profile, interpolation=True,
            TotalInducedVoltage=self.induced_voltage)

        self.OTFB = SPSCavityFeedback(
            self.rf, self.beam, self.profile, G_llrf=20, G_tx=[1.0355739238973907, 1.078403005653143],
            a_comb=63/64, turns=1000, post_LS2=True, df=[0.18433333e6, 0.2275e6],
            Commissioning=CavityFeedbackCommissioning(open_FF=True))


        self.OTFB_tracker = RingAndRFTracker(self.rf, self.beam,
                                             Profile=self.profile,
                                             TotalInducedVoltage=None,
                                             CavityFeedback=self.OTFB,
                                             interpolation=True)

    def test_FB_pre_tracking(self):

        digit_round = 3

        Vind3_mean = np.mean(np.absolute(self.OTFB.OTFB_1.V_ANT[-self.OTFB.OTFB_1.n_coarse:]))/1e6
        Vind3_std = np.std(np.absolute(self.OTFB.OTFB_1.V_ANT[-self.OTFB.OTFB_1.n_coarse:]))/1e6
        Vind3_mean_exp = 2.7047955940118764
        Vind3_std_exp = 2.4121534046270847e-12

        Vind4_mean = np.mean(np.absolute(self.OTFB.OTFB_2.V_ANT[-self.OTFB.OTFB_2.n_coarse:]))/1e6
        Vind4_std = np.std(np.absolute(self.OTFB.OTFB_2.V_ANT[-self.OTFB.OTFB_2.n_coarse:]))/1e6
        Vind4_mean_exp = 1.8057100857806163
        Vind4_std_exp = 1.89451253314611e-12

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

        V_tot_3 = V_fine_tot_3/1e6
        V_tot_4 = V_fine_tot_4/1e6

        V_sum = self.OTFB.V_sum/1e6

        # expected generator voltage is only in Q
        V_tot_3_exp = 2.7j*np.ones(256)
        V_tot_4_exp = 1.8j*np.ones(256)
        V_sum_exp = 4.5j*np.ones(256)

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

        max_ratio_exp = 1.0690779399272086#1.0001336336515099
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


        max_ration_exp = 1.0690779399245092 #1.0055233047525063

        self.assertAlmostEqual(max_ratio, max_ration_exp, places=digit_round,
                               msg='In TestCavityFeedback test_beam_loading: '
                               + 'total voltages differ')

    def test_Vsum_IQ(self):
        rtol = 1e-7         # relative tolerance
        atol = 0              # absolute tolerance

        self.OTFB.track()

        V_sum = self.OTFB.V_sum/1e6


        V_sum_exp = np.array([0.014799274843711883+4.510480895814642j, 0.014799274843691887+4.510480895814651j,
                            0.014799274843711838+4.510480895814639j, 0.014799274843711817+4.510480895814646j,
                            0.014799274843701804+4.5104808958146405j, 0.014799274843686804+4.510480895814651j,
                            0.014799274843716742+4.5104808958146485j, 0.01479927484370174+4.5104808958146485j,
                            0.014799274843721691+4.510480895814649j, 0.014799274843711683+4.5104808958146565j,
                            0.014799274843706668+4.510480895814642j, 0.014799274843731612+4.510480895814646j,
                            0.014799274843731593+4.510480895814661j, 0.014799274843696617+4.510480895814666j,
                            0.014799274843721559+4.510480895814642j, 0.014799274843721536+4.510480895814652j,
                            0.014799274843706536+4.510480895814652j, 0.01479927484371151+4.510480895814663j,
                            0.01479927484371648+4.510480895814657j, 0.014799274843711463+4.510480895814652j,
                            0.014799274843716436+4.510480895814651j, 0.014799274843701433+4.510480895814666j,
                            0.014799274843721384+4.510480895814657j, 0.01479927484370139+4.510480895814652j,
                            0.014799274843706361+4.510480895814659j, 0.01479927484372132+4.510480895814663j,
                            0.014799274843716305+4.510480895814658j, 0.01479927484371129+4.510480895814655j,
                            0.014799274843736233+4.510480895814659j, 0.014799274843711246+4.510480895814666j,
                            0.01479927484370623+4.5104808958146645j, 0.014799274843716195+4.510480895814659j,
                            0.014799274843706184+4.510480895814668j, 0.014799274843696177+4.510480895814672j,
                            0.014799274843711137+4.510480895814659j, 0.014799274843701127+4.510480895814666j,
                            0.014799274843716084+4.510480895814665j, 0.014799274843701082+4.510480895814676j,
                            0.014799274843706054+4.510480895814664j, 0.014799274843711024+4.51048089581467j,
                            0.014799274843730977+4.510480895814666j, 0.01479927484372596+4.510480895814677j,
                            0.01479927484372594+4.510480895814669j, 0.014799274843710937+4.510480895814666j,
                            0.014799274843725896+4.51048089581468j, 0.014799274843705901+4.510480895814686j,
                            0.014799274843715866+4.510480895814668j, 0.014799274843720837+4.510480895814669j,
                            0.014799274843710826+4.510480895814677j, 0.014799274843710807+4.5104808958146805j,
                            0.014799274843695804+4.510480895814671j, 0.014799274843710762+4.51048089581468j,
                            0.014799274843705749+4.51048089581468j, 0.014799274843695738+4.510480895814682j,
                            0.01479927484371569+4.510480895814678j, 0.014799274843695694+4.510480895814668j,
                            0.01479927484372064+4.5104808958146805j, 0.01479927484371063+4.510480895814674j,
                            0.01479927484371061+4.510480895814678j, 0.01479927484373056+4.510480895814678j,
                            0.014799274843735532+4.5104808958146805j, 0.014799274843710544+4.510480895814687j,
                            0.014799274843730495+4.51048089581469j, 0.014799274843720485+4.5104808958146725j,
                            0.014799274843725466+4.510480895814684j, 0.014799274843715472+4.510480895814689j,
                            0.014799274843725455+4.510480895814673j, 0.014658889283833644+4.5105311265711965j,
                            0.014248976146476367+4.510708845986652j, 0.013851646927663488+4.510912960468902j,
                            0.01360417239558314+4.511078378408325j, 0.013263672610738925+4.511367381656121j,
                            0.012942876965929885+4.511677990482388j, 0.012665430685210689+4.51202782788558j,
                            0.012181564620148118+4.512778677298704j, 0.011429208102346598+4.514233368039539j,
                            0.010531562376192436+4.51643970684789j, 0.009718061885618317+4.519148508935177j,
                            0.00902864124141807+4.522805731768613j, 0.008533427944443032+4.5284411689923525j,
                            0.008593369252332348+4.536479837065873j, 0.009806197270317413+4.547728837772571j,
                            0.012965921541985453+4.563037192477819j, 0.019309996667349005+4.583357886430505j,
                            0.029216979136955694+4.607127644048769j, 0.04461189992360321+4.635355447107408j,
                            0.06812299450682953+4.6701296876299825j, 0.10279943973905167+4.711762929851321j,
                            0.15278054368861294+4.761140316724495j, 0.223363511926328+4.8182608982066j,
                            0.32577212378488807+4.885499744128283j, 0.46328549009994474+4.958202903070531j,
                            0.63776340903079+5.0293893324729275j, 0.8494807617866049+5.09286363948396j,
                            1.1243796301342157+5.145175939526339j, 1.4740883605058082+5.178231877645256j,
                            1.8994709242847085+5.175725099883707j, 2.410866690096992+5.123184531009498j,
                            3.0057497008307137+5.002002171506081j, 3.7005922826289663+4.787473138877127j,
                            4.496689416706626+4.453717636455056j, 5.381417097037591+3.976252754341861j,
                            6.349856334092241+3.3233025423270495j, 7.406473369726965+2.448062682383963j,
                            8.551332560395378+1.2924674280781654j, 9.740910259580367+-0.16748498004641105j,
                            10.920046401272337+-1.9459030954391583j, 12.056459897411518+-4.091366489783575j,
                            13.088666399645325+-6.603626219369836j, 13.964464036247506+-9.532916197077226j,
                            14.628035814593206+-12.92649002604683j, 14.992417335426559+-16.747603046562297j,
                            14.979491887144365+-20.986420330314765j, 14.50955877141081+-25.61927356800518j,
                            13.497523815343527+-30.62743771277803j, 11.871396859940578+-35.939080320516304j,
                            9.579598474932784+-41.434442646983435j, 6.5468965132122+-47.0706151837596j,
                            2.7243325738122173+-52.76241703674196j, -1.8987780215315229+-58.36903632775433j,
                            -7.296953052629184+-63.74626439694516j, -13.456104367254394+-68.77816899734592j,
                            -20.3602298975143+-73.37119523433002j, -27.974433240235513+-77.41951472320127j,
                            -36.14288352255647+-80.7891330044604j, -44.68399429644744+-83.36473278576285j,
                            -53.62931739590642+-85.12345190110167j, -62.87697944028692+-86.01857079708142j,
                            -72.16593416179133+-86.0049853098913j, -81.33864234399393+-85.08721302911619j,
                            -90.2732598689588+-83.2959129612618j, -98.88731291126743+-80.66591924629697j,
                            -107.09492595352097+-77.24980049700667j, -114.65991268048072+-73.19648966642923j,
                            -121.51784166930014+-68.59490913112872j, -127.6548808424254+-63.54450230690343j,
                            -133.0069004448795+-58.171708517504136j, -137.59701146222258+-52.55746847059092j,
                            -141.37791809088407+-46.8796061241762j, -144.3674642000868+-41.2593361004134j,
                            -146.63128216197546+-35.760232137413375j, -148.22333134029688+-30.467529461669727j,
                            -149.20303561833512+-25.484084071308775j, -149.64750684568196+-20.81500564293486j,
                            -149.63759273456972+-16.526155666102802j, -149.24749316692285+-12.69292401550595j,
                            -148.55850935618002+-9.287107794223967j, -147.63825147048664+-6.295997054164775j,
                            -146.5634106401422+-3.7255101426639063j, -145.40223462457695+-1.5744762345069667j,
                            -144.2025010764714+0.2051485963983133j, -143.0183224858328+1.6360635468544744j,
                            -141.87837141087462+2.765470886668634j, -140.7970196851828+3.644781378958509j,
                            -139.78520801754027+4.314386234793984j, -138.86069964021908+4.804902948304536j,
                            -138.0453107284997+5.1409193109713565j, -137.3337393155524+5.356232758441748j,
                            -136.72690461905825+5.478734568362141j, -136.21723381103192+5.531643951366212j,
                            -135.7759487486148+5.535736333844168j, -135.40325701858524+5.505614338603859j,
                            -135.10553249910936+5.454825571005634j, -134.86576531565393+5.392047177307719j,
                            -134.67321453146758+5.324691025739685j, -134.52243061563024+5.258622294334678j,
                            -134.40371688867938+5.196522199860245j, -134.31105212942018+5.140673768141151j,
                            -134.2384457880603+5.091877851140164j, -134.1807951057739+5.049765542312797j,
                            -134.1362695689274+5.0173254869787876j, -134.10174439936466+4.994120052449075j,
                            -134.0719625613446+4.974583821565376j, -134.04573583172086+4.958821364030993j,
                            -134.02216151881981+4.94606744774473j, -134.0003243121118+4.936057482621122j,
                            -133.97974210967863+4.9307406046551465j, -133.95961288697004+4.929308628785965j,
                            -133.939815571267+4.928825512025661j, -133.92027331157462+4.928406066592057j,
                            -133.90083367978656+4.928653340899114j, -133.88163167254706+4.929012642959638j,
                            -133.86194452583945+4.9304648010887755j, -133.84176768147861+4.932846206003178j,
                            -133.8216123873659+4.9352464786444745j, -133.80161602591124+4.937600444621707j,
                            -133.78150903429483+4.940053800016984j, -133.76104270925114+4.942772777826079j,
                            -133.74057632896336+4.94549091920606j, -133.72010989345716+4.94820822415581j,
                            -133.6996434027582+4.950924692674183j, -133.67917685689204+4.9536403247600385j,
                            -133.65885917770817+4.956362436472391j, -133.63854139797277+4.959083715619173j,
                            -133.61807464132346+4.961796842268201j, -133.59760782960947+4.964509132478421j,
                            -133.5771409628565+4.967220586248702j, -133.55667404109022+4.969931203577941j,
                            -133.53620706433625+4.972640984465001j, -133.51574003262024+4.9753499289087815j,
                            -133.49527294596786+4.9780580369081795j, -133.47480580440467+4.980765308462055j,
                            -133.45433860795637+4.983471743569327j, -133.43387135664858+4.986177342228841j,
                            -133.41340405050698+4.988882104439528j, -133.39293668955716+4.991586030200257j,
                            -133.37246927382483+4.994289119509905j, -133.35200180333555+4.996991372367376j,
                            -133.33153427811502+4.99969278877157j, -133.31106669818885+5.002393368721354j,
                            -133.29059906358273+5.005093112215658j, -133.27013137432223+5.007792019253332j,
                            -133.24966363043305+5.010490089833296j, -133.22919583194079+5.013187323954451j,
                            -133.20872797887114+5.015883721615655j, -133.1882600712497+5.018579282815844j,
                            -133.16779210910212+5.021274007553907j, -133.14732409245406+5.023967895828711j,
                            -133.12685602133115+5.026660947639206j, -133.10638789575907+5.029353162984246j,
                            -133.08591971576337+5.032044541862752j, -133.06545148136976+5.034735084273613j,
                            -133.0449831926039+5.037424790215743j, -133.02451484949137+5.040113659688021j,
                            -133.00404645205785+5.042801692689391j, -132.98357800032898+5.045488889218693j,
                            -132.96310949433044+5.048175249274901j, -132.9426409340878+5.050860772856868j,
                            -132.92217231962672+5.053545459963516j, -132.90170365097285+5.056229310593759j,
                            -132.88123492815188+5.0589123247464824j, -132.8607661511894+5.061594502420611j,
                            -132.84029732011106+5.064275843615067j, -132.81982843494248+5.066956348328711j,
                            -132.79935949570938+5.069636016560496j, -132.77889050243732+5.072314848309307j,
                            -132.75842145515193+5.074992843574076j, -132.7379523538789+5.077670002353692j,
                            -132.71748319864398+5.08034632464708j, -132.69701398947262+5.083021810453148j,
                            -132.67654472639055+5.085696459770818j, -132.6560754094234+5.088370272598987j,
                            -132.63560603859682+5.091043248936602j, -132.61513661393647+5.0937153887825355j,
                            -132.59466713546797+5.0963866921357415j, -132.57419760321696+5.099057158995106j,
                            -132.55372801720912+5.101726789359567j, -132.53325837747002+5.104395583228046j,
                            -132.51278868402537+5.107063540599446j, -132.49231893690074+5.109730661472694j,
                            -132.47184913612193+5.11239694584672j, -132.4513792817144+5.115062393720419j,
                            -132.43090937370386+5.117727005092745j, -132.41043941211598+5.120390779962594j,
                            -132.38996939697637+5.1230537183289j, -132.36949932831072+5.125715820190601j])

        np.testing.assert_allclose(V_sum_exp, V_sum,
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
        N_m = 1e5                               # Number of macro-particles for tracking
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
        self.Commissioning = CavityFeedbackCommissioning()


        self.OTFB = SPSOneTurnFeedback(self.rfstation, self.beam, self.profile, 3, a_comb=63 / 64,
                                          Commissioning=self.Commissioning)

        self.OTFB.update_variables()

        self.turn_array = np.linspace(0, 2 * self.rfstation.t_rev[0], 2 * self.OTFB.n_coarse)

    def test_set_point(self):
        self.OTFB.set_point()
        t_sig = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        t_sig[-self.OTFB.n_coarse:] = (4/9) * 10e6 * np.exp(1j * (np.pi/2 - self.rfstation.phi_rf[0,0]))

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
        ref_DV_MOD_FR = np.load("ref_DV_MOD_FR.npy")

        np.testing.assert_allclose(self.OTFB.DV_MOD_FR[-self.OTFB.n_coarse:], ref_DV_MOD_FR)

        self.OTFB.DV_DELAYED = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_DELAYED[-self.OTFB.n_coarse:] = 1 + 1j * 0

        self.OTFB.dphi_mod = 0
        self.OTFB.mod_to_fr()

        time_array = self.OTFB.rf_centers - 0.5*self.OTFB.T_s
        ref_sig = np.cos((self.OTFB.omega_c - self.OTFB.omega_r) * time_array[:self.OTFB.n_coarse]) - \
                  1j * np.sin((self.OTFB.omega_c - self.OTFB.omega_r) * time_array[:self.OTFB.n_coarse])

        np.testing.assert_allclose(self.OTFB.DV_MOD_FR[-self.OTFB.n_coarse:], ref_sig)

        self.OTFB.dphi_mod = self.mod_phi


    def test_mov_avg(self):
        sig = np.zeros(self.OTFB.n_coarse-1)
        sig[:self.OTFB.n_mov_av] = 1
        self.OTFB.DV_MOD_FR = np.zeros(2 * self.OTFB.n_coarse)
        self.OTFB.DV_MOD_FR[-self.OTFB.n_coarse + 1:] = sig

        self.OTFB.mov_avg()

        sig = np.zeros(self.OTFB.n_coarse)
        sig[:self.OTFB.n_mov_av] = (1/self.OTFB.n_mov_av) * np.array(range(self.OTFB.n_mov_av))
        sig[self.OTFB.n_mov_av: 2 * self.OTFB.n_mov_av] = (1/self.OTFB.n_mov_av) * (self.OTFB.n_mov_av
                                                                                    - np.array(range(self.OTFB.n_mov_av)))

        np.testing.assert_allclose(np.abs(self.OTFB.DV_MOV_AVG[-self.OTFB.n_coarse:]), sig)


    def test_mod_to_frf(self):
        self.OTFB.DV_MOV_AVG = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_MOV_AVG[-self.OTFB.n_coarse:] = 1 + 1j * 0

        self.mod_phi = np.copy(self.OTFB.dphi_mod)
        self.OTFB.mod_to_frf()
        ref_DV_MOD_FRF = np.load("ref_DV_MOD_FRF.npy")

        np.testing.assert_allclose(self.OTFB.DV_MOD_FRF[-self.OTFB.n_coarse:], ref_DV_MOD_FRF)

        self.OTFB.DV_MOV_AVG = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_MOV_AVG[-self.OTFB.n_coarse:] = 1 + 1j * 0

        self.OTFB.dphi_mod = 0
        self.OTFB.mod_to_frf()

        time_array = self.OTFB.rf_centers - 0.5*self.OTFB.T_s
        ref_sig = np.cos(-(self.OTFB.omega_c - self.OTFB.omega_r) * time_array[:self.OTFB.n_coarse]) - \
                  1j * np.sin(-(self.OTFB.omega_c - self.OTFB.omega_r) * time_array[:self.OTFB.n_coarse])

        np.testing.assert_allclose(self.OTFB.DV_MOD_FRF[-self.OTFB.n_coarse:], ref_sig)

        self.OTFB.dphi_mod = self.mod_phi

    def test_sum_and_gain(self):
        self.OTFB.V_SET[-self.OTFB.n_coarse:] = np.ones(self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_MOD_FRF[-self.OTFB.n_coarse:] = np.ones(self.OTFB.n_coarse, dtype=complex)

        self.OTFB.sum_and_gain()

        sig = 2 * np.ones(self.OTFB.n_coarse) * self.OTFB.G_tx / self.OTFB.TWC.R_gen

        np.testing.assert_allclose(self.OTFB.I_GEN[-self.OTFB.n_coarse:], sig)


    def test_gen_response(self):
        # Tests generator response at resonant frequency.
        self.OTFB.I_GEN = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.I_GEN[self.OTFB.n_coarse] = 1

        self.OTFB.TWC.impulse_response_gen(self.OTFB.TWC.omega_r, self.OTFB.rf_centers)
        self.OTFB.gen_response()

        sig = np.zeros(self.OTFB.n_coarse)
        sig[1:1 + self.OTFB.n_mov_av] = 4 * self.OTFB.TWC.R_gen / self.OTFB.TWC.tau
        sig[0] = 2 * self.OTFB.TWC.R_gen / self.OTFB.TWC.tau
        sig[self.OTFB.n_mov_av + 1] = 2 * self.OTFB.TWC.R_gen / self.OTFB.TWC.tau
        sig *= self.OTFB.T_s

        np.testing.assert_allclose(np.abs(self.OTFB.V_IND_COARSE_GEN[-self.OTFB.n_coarse:]), sig,
                                   atol=5e-5)

        # Tests generator response at carrier frequency.
        self.OTFB.TWC.impulse_response_gen(self.OTFB.omega_c, self.OTFB.rf_centers)

        self.OTFB.I_GEN = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.I_GEN[self.OTFB.n_coarse] = 1

        self.OTFB.gen_response()

        ref_V_IND_COARSE_GEN = np.load("ref_V_IND_COARSE_GEN.npy")
        np.testing.assert_allclose(self.OTFB.V_IND_COARSE_GEN[-self.OTFB.n_coarse:], ref_V_IND_COARSE_GEN)



if __name__ == '__main__':
    unittest.main()
