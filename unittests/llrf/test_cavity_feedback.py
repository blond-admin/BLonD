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
import matplotlib.pyplot as plt
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
            self.rf, self.beam, self.profile, G_llrf=20, G_tx=[0.22909261332041, 0.429420301179296],
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
        Vind3_mean_exp = 2.689700267145197
        Vind3_std_exp = 2.388168848303381e-12

        Vind4_mean = np.mean(np.absolute(self.OTFB.OTFB_2.V_ANT[-self.OTFB.OTFB_2.n_coarse:]))/1e6
        Vind4_std = np.std(np.absolute(self.OTFB.OTFB_2.V_ANT[-self.OTFB.OTFB_2.n_coarse:]))/1e6
        Vind4_mean_exp = 1.7866100616589826
        Vind4_std_exp = 1.854813485950122e-12

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

        max_ratio_exp = 1.0953022998660298#1.0001336336515099
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

        #plt.plot(OTFB_tracker_total_voltage)
        #plt.plot(cavity_tracker_total_voltage)
        #plt.show()

        max_ration_exp = 1.0953022998625235 #1.0055233047525063

        self.assertAlmostEqual(max_ratio, max_ration_exp, places=digit_round,
                               msg='In TestCavityFeedback test_beam_loading: '
                               + 'total voltages differ')

    def test_Vsum_IQ(self):
        rtol = 1e-7         # relative tolerance
        atol = 0              # absolute tolerance

        self.OTFB.track()

        V_sum = self.OTFB.V_sum/1e6

        V_sum_exp = np.array([-0.0001722441841138659+4.502904839895534j, -0.0001722441841105205+4.502904839895534j,
                            -0.000172244184113384+4.502904839895534j, -0.00017224418411252218+4.502904839895533j,
                            -0.00017224418411166032+4.502904839895533j, -0.00017224418411204026+4.502904839895531j,
                            -0.00017224418411366198+4.502904839895531j, -0.0001722441841115584+4.502904839895533j,
                            -0.00017224418411131742+4.50290483989553j, -0.00017224418411231822+4.50290483989553j,
                            -0.00017224418411021465+4.50290483989553j, -0.00017224418411121548+4.502904839895531j,
                            -0.0001722441841115954+4.50290483989553j, -0.00017224418411073358+4.502904839895527j,
                            -0.0001722441841111135+4.50290483989553j, -0.00017224418410900994+4.502904839895529j,
                            -0.0001722441841106316+4.502904839895528j, -0.00017224418410790714+4.502904839895528j,
                            -0.0001722441841076662+4.502904839895527j, -0.00017224418410742527+4.502904839895527j,
                            -0.00017224418410780517+4.502904839895527j, -0.00017224418410756424+4.502904839895527j,
                            -0.00017224418410856506+4.5029048398955265j, -0.00017224418410584057+4.502904839895528j,
                            -0.00017224418410808313+4.502904839895527j, -0.0001722441841072213+4.502904839895527j,
                            -0.00017224418410511773+4.502904839895527j, -0.00017224418410611853+4.502904839895528j,
                            -0.00017224418410711936+4.502904839895527j, -0.00017224418410687842+4.502904839895526j,
                            -0.0001722441841066375+4.5029048398955265j, -0.0001722441841045339+4.502904839895527j,
                            -0.00017224418410677645+4.5029048398955265j, -0.00017224418410467286+4.502904839895526j,
                            -0.00017224418410381105+4.502904839895526j, -0.0001722441841035701+4.502904839895526j,
                            -0.00017224418410519176+4.502904839895526j, -0.00017224418410432998+4.502904839895524j,
                            -0.00017224418410595165+4.502904839895525j, -0.00017224418410260631+4.502904839895526j,
                            -0.00017224418410422798+4.502904839895526j, -0.00017224418410274526+4.502904839895525j,
                            -0.00017224418410312521+4.502904839895525j, -0.00017224418410288428+4.502904839895523j,
                            -0.0001722441841032642+4.502904839895525j, -0.0001722441841017815+4.502904839895522j,
                            -0.00017224418410154057+4.502904839895524j, -0.0001722441841012996+4.502904839895525j,
                            -0.00017224418410167954+4.502904839895523j, -0.0001722441841014386+4.502904839895523j,
                            -0.0001722441841018185+4.502904839895523j, -0.00017224418409909404+4.502904839895523j,
                            -0.00017224418410133663+4.502904839895523j, -0.0001722441841004748+4.502904839895521j,
                            -0.00017224418410147563+4.502904839895522j, -0.00017224418409999293+4.502904839895524j,
                            -0.0001722441841003728+4.502904839895522j, -0.00017224418409826926+4.502904839895522j,
                            -0.00017224418409989096+4.502904839895522j, -0.00017224418409778733+4.502904839895522j,
                            -0.0001722441841006508+4.502904839895522j, -0.00017224418409792636+4.502904839895519j,
                            -0.00017224418409954805+4.502904839895521j, -0.00017224418409868622+4.50290483989552j,
                            -0.00017224418409968732+4.502904839895522j, -0.00017224418409696345+4.502904839895522j,
                            -0.00017224418409858577+4.50290483989552j, -0.00018750348408254718+4.502899380030682j,
                            -0.0002320597061889474+4.502880063950246j, -0.0002752492437424446+4.502857881235587j,
                            -0.0003021509089870766+4.502839905802075j, -0.00033916480322379334+4.502808498245488j,
                            -0.00037403872151100896+4.502774744217872j, -0.0004042017325190213+4.502736727101794j,
                            -0.00045680380298011874+4.50265512329946j, -0.0005385941610718934+4.502497017726146j,
                            -0.0006361852036852637+4.50225721523687j, -0.00072464124467401+4.501962801185369j,
                            -0.000799623186800918+4.501565301134899j, -0.0008535152775136621+4.5009527806895635j,
                            -0.0008470948898293645+4.500079040986549j, -0.0007154017189060054+4.4988563500604934j,
                            -0.0003721483181615157+4.49719241624881j, 0.0003171531303819703+4.494983643634726j,
                            0.0013936236704705015+4.492399935753529j, 0.0030665015768633805+4.489331606750335j,
                            0.005621429260809986+4.485551622670133j, 0.009389816211034865+4.481025971870464j,
                            0.014821549091373511+4.4756583812582305j, 0.02249238047807154+4.469448880064851j,
                            0.033622248276346786+4.462139199577795j, 0.04856752584134512+4.454235030348051j,
                            0.06753036839265683+4.446495043301431j, 0.09054066462373754+4.439592460498928j,
                            0.12041826431423408+4.433902128945606j, 0.1584271591079335+4.43030344918596j,
                            0.2046613891047944+4.43056860727175j, 0.2602449845253899+4.436270117962638j,
                            0.324903643500024+4.449430254376412j, 0.40042818732888597+4.472733924190136j,
                            0.4869598529341774+4.508993666985897j, 0.5831270466316344+4.560870251582143j,
                            0.6883957911100752+4.631817216102861j, 0.8032524104635083+4.726921613184165j,
                            0.9277050110538022+4.852494527890535j, 1.0570241403976157+5.011144473379337j,
                            1.1852154631451042+5.204404965732725j, 1.3087716689721332+5.437556509595485j,
                            1.4210114317687046+5.710572310296637j, 1.5162621359806572+6.028913238359752j,
                            1.5884589182487723+6.397716283097762j, 1.6281510468411333+6.812988618005941j,
                            1.6268494488024967+7.273662729621871j, 1.5758928372038423+7.777167827519203j,
                            1.4660335085981533+8.3214707886155j, 1.2894481113183893+8.898766345873725j,
                            1.040531210090549+9.496040091479998j, 0.7111059987857364+10.108631013207674j,
                            0.2958515551797077+10.727283832765389j, -0.20639365450976202+11.336696410036943j,
                            -0.7928604318137645+11.921197499109953j, -1.4620188926537296+12.468189040160606j,
                            -2.2121313196428343+12.967505505112111j, -3.039406892957403+13.407646936934066j,
                            -3.926910051727683+13.774051891651292j, -4.8549065957663196+14.054187623820702j,
                            -5.8268296734781995+14.24557067236217j, -6.831609750542692+14.343124847759796j,
                            -7.84087624779641+14.341948175002761j, -8.837509396211487+14.242531399390899j,
                            -9.808269243744233+14.048206273346036j, -10.744195029392074+13.76275773704397j,
                            -11.635956120953596+13.391898289445498j, -12.457883409936708+12.951814153183113j,
                            -13.20297581968524+12.452167225339492j, -13.869732389647497+11.903763859287631j,
                            -14.451183948995876+11.320343759438753j, -14.949843963483884+10.710701546322397j,
                            -15.360572320141388+10.094162417200819j, -15.685308391821321+9.483898172803361j,
                            -15.931185790192602+8.886815667822287j, -16.10406892067257+8.312176483362908j,
                            -16.21041580209582+7.771157765398302j, -16.258606500889243+7.2643123250147825j,
                            -16.257427667984185+6.798796851887058j, -16.214942266990295+6.382804028139302j,
                            -16.13998557302848+6.013266813445359j, -16.03990452779425+5.688802545091459j,
                            -15.92303342393833+5.4100536147430045j, -15.796788454610883+5.176892293854129j,
                            -15.666361154092103+4.984096116162293j, -15.537631700705251+4.829197737309734j,
                            -15.413715190223305+4.707066656229253j, -15.296172584276661+4.61211496084594j,
                            -15.186192060927535+4.539952610155151j, -15.085703571324288+4.487251927117381j,
                            -14.997077681789037+4.451340791426509j, -14.919737475308239+4.428546073938241j,
                            -14.853782410269092+4.415836532489148j, -14.798388984908126+4.410688852997459j,
                            -14.750428890061496+4.410845187966397j, -14.709924505910116+4.414718553087287j,
                            -14.67756852495068+4.420836735314705j, -14.651511984590146+4.428256880427315j,
                            -14.63058734847626+4.436173552086109j, -14.61420227896996+4.44394944925956j,
                            -14.601302798062521+4.451293288374408j, -14.591234451683261+4.457957046548599j,
                            -14.583346122281695+4.463853774143679j, -14.577083195756858+4.469023648530971j,
                            -14.572246729111795+4.473141871079364j, -14.56849713268627+4.476256073106302j,
                            -14.565263011529401+4.478971217602783j, -14.56241522162252+4.4812759541166445j,
                            -14.559855667204763+4.483253475828608j, -14.557484861869344+4.484932550326217j,
                            -14.555250423775016+4.486101322642555j, -14.55306519752505+4.486847642490516j,
                            -14.550916036395344+4.48749064549126j, -14.54879458210846+4.488126550142445j,
                            -14.546684271876948+4.488689807534013j, -14.544599779700267+4.489240711352808j,
                            -14.542462544827076+4.489672650855556j, -14.540272080968153+4.490003404317468j,
                            -14.538083957648757+4.4903319269736945j, -14.535913108813647+4.49066530219865j,
                            -14.533730232678755+4.49098769567174j, -14.531508298648646+4.491281035395668j,
                            -14.529286358699757+4.4915742842879025j, -14.527064412834864+4.491867442348324j,
                            -14.524842461056764+4.492160509576817j, -14.522620503368236+4.492453485973261j,
                            -14.520414726926822+4.492745576313617j, -14.518208939706891+4.493037576724294j,
                            -14.515986959430112+4.493330281526286j, -14.513764973254025+4.493622895495552j,
                            -14.511542981181405+4.493915418631974j, -14.509320983215046+4.494207850935429j,
                            -14.507098979357727+4.494500192405804j, -14.504876969612233+4.494792443042979j,
                            -14.502654953981356+4.495084602846834j, -14.500432932467877+4.495376671817251j,
                            -14.498210905074576+4.495668649954116j, -14.495988871804242+4.4959605372573j,
                            -14.49376683265966+4.496252333726695j, -14.49154478764361+4.4965440393621785j,
                            -14.489322736758886+4.496835654163633j, -14.487100680008265+4.497127178130941j,
                            -14.484878617394537+4.497418611263987j, -14.482656548920481+4.497709953562644j,
                            -14.480434474588884+4.498001205026805j, -14.478212394402536+4.498292365656345j,
                            -14.475990308364215+4.498583435451152j, -14.473768216476712+4.498874414411102j,
                            -14.471546118742808+4.499165302536084j, -14.469324015165284+4.499456099825975j,
                            -14.467101905746935+4.499746806280662j, -14.464879790490535+4.500037421900024j,
                            -14.462657669398874+4.50032794668395j, -14.460435542474741+4.500618380632315j,
                            -14.458213409720912+4.500908723745009j, -14.45599127114018+4.501198976021907j,
                            -14.453769126735324+4.5014891374629j, -14.45154697650913+4.501779208067867j,
                            -14.449324820464387+4.502069187836693j, -14.447102658603873+4.50235907676926j,
                            -14.44488049093038+4.502648874865453j, -14.442658317446687+4.502938582125154j,
                            -14.440436138155583+4.503228198548248j, -14.438213953059849+4.503517724134615j,
                            -14.435991762162274+4.503807158884143j, -14.43376956546564+4.504096502796715j,
                            -14.431547362972736+4.504385755872213j, -14.429325154686339+4.504674918110523j,
                            -14.427102940609242+4.5049639895115305j, -14.424880720744225+4.505252970075113j,
                            -14.422658495094074+4.505541859801163j, -14.420436263661573+4.505830658689557j,
                            -14.418214026449512+4.506119366740186j, -14.41599178346067+4.506407983952928j,
                            -14.413769534697838+4.506696510327675j, -14.41154728016379+4.506984945864304j,
                            -14.409325019861319+4.507273290562708j, -14.407102753793213+4.507561544422762j,
                            -14.404880481962248+4.507849707444358j, -14.402658204371217+4.508137779627375j,
                            -14.400435921022899+4.508425760971707j, -14.39821363192008+4.50871365147723j,
                            -14.395991337065551+4.509001451143833j, -14.39376903646209+4.5092891599714005j,
                            -14.39154673011248+4.509576777959819j, -14.389324418019518+4.50986430510897j,
                            -14.387102100185976+4.5101517414187455j, -14.384879776614639+4.510439086889024j,
                            -14.382657447308304+4.510726341519695j, -14.380435112269748+4.511013505310644j])

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
        ref_sig = np.cos((self.OTFB.omega_c - self.OTFB.omega_r) * time_array[:self.OTFB.n_coarse]) + \
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
        ref_sig = np.cos(-(self.OTFB.omega_c - self.OTFB.omega_r) * time_array[:self.OTFB.n_coarse]) + \
                  1j * np.sin(-(self.OTFB.omega_c - self.OTFB.omega_r) * time_array[:self.OTFB.n_coarse])

        np.testing.assert_allclose(self.OTFB.DV_MOD_FRF[-self.OTFB.n_coarse:], ref_sig)

        self.OTFB.dphi_mod = self.mod_phi

    def test_sum_and_gain(self):
        self.OTFB.V_SET[-self.OTFB.n_coarse:] = np.ones(self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_MOD_FRF[-self.OTFB.n_coarse:] = np.ones(self.OTFB.n_coarse, dtype=complex)

        self.OTFB.sum_and_gain()

        sig = 2 * np.ones(self.OTFB.n_coarse) * self.OTFB.G_tx * self.OTFB.T_s / self.OTFB.TWC.R_gen

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
