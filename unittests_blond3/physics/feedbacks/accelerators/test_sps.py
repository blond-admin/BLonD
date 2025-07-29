import os
import unittest
from copy import deepcopy

import numpy as np

from blond3 import (
    Ring,
    MultiHarmonicCavity,
    DriftSimple,
    ConstantMagneticCycle,
    proton,
    Simulation,
    Beam,
    BiGaussian,
    StaticProfile,
)
from blond3._core.backends.backend import backend, Numpy64Bit
from blond3.physics.feedbacks.accelerators.sps.cavity_feedback import (
    SPSCavityFeedback,
    SPSCavityLoopCommissioning,
    SPSOneTurnFeedback,
)

this_directory = os.path.dirname(os.path.realpath(__file__))


class TestSPSCavityFeedback(unittest.TestCase):
    def setUp(self):
        backend.change_backend(Numpy64Bit)
        #backend.set_specials("fortran")
        C = 2 * np.pi * 1100.009  # Ring circumference [m]
        # Gamma at transition
        p_s = 25.92e9  # Synchronous momentum at injection [eV]
        h = 4620  # 200 MHz system harmonic
        phi = 0.0  # 200 MHz RF phase

        # With this setting, amplitude in the two four-section, five-section
        # cavities must converge, respectively, to
        # 2.0 MV = 4.5 MV * 4/18 * 2
        # 2.5 MV = 4.5 MV * 5/18 * 2
        V = 4.5e6  # 200 MHz RF voltage

        N_t = 1  # Number of turns to track

        # self.ring = Ring(C, alpha, p_s, particle=Proton(), n_turns=N_t)
        ring = Ring(circumference=C)
        self.ring = ring
        rf = MultiHarmonicCavity(
            n_harmonics=1,
            main_harmonic_idx=0,
        )
        rf.harmonic = np.array([h])
        rf.voltage = np.array([V])
        rf.phi_rf = np.array([phi])
        self.ring.add_element(rf)
        self.ring.add_drifts(
            n_drifts_per_section=1,
            n_sections=1,
            driftclass=DriftSimple,
            transition_gamma=18.0,
        )
        magnetic_cycle = ConstantMagneticCycle(
            reference_particle=proton,
            value=p_s,
            in_unit="momentum",
        )
        simulation = Simulation(
            ring=ring,
            magnetic_cycle=magnetic_cycle,
        )
        N_m = int(1e6)  # Number of macro-particles for tracking
        N_b = 288 * 2.3e11  # Bunch intensity [ppb]

        # Gaussian beam profile
        self.beam = Beam(
            n_particles=N_b,
            particle_type=proton,
        )

        simulation.prepare_beam(
            beam=self.beam,
            preparation_routine=BiGaussian(
                n_macroparticles=N_m,
                sigma_dt=1.0e-9,
                seed=1234,
                reinsertion=False,
            ),
        )

        n_shift = 1550  # how many rf-buckets to shift beam
        omega_rf = float(
            rf.calc_omega(
                beam_beta=self.beam.reference_beta,
                ring_circumference=ring.circumference,
            )
        )
        t_rf = (2 * np.pi) / omega_rf
        self.beam._dt += n_shift * t_rf

        self.profile = StaticProfile(
            cut_left=(n_shift - 1.5) * t_rf,
            cut_right=(n_shift + 2.5) * t_rf,
            n_bins=4 * 64,
        )
        self.profile.track(beam=self.beam)

        # todo remove hacky bugfix.
        # SPSCavityFeedback needs better init
        beam_2 = deepcopy(self.beam)
        rf.track(beam_2)
        del beam_2
        """

        # Cavities
        l_cav = 32 * 0.374
        v_g = 0.0946
        tau = l_cav / (v_g * c) * (1 + v_g)
        f_cav = 200.222e6
        n_cav = 4  # factor 2 because of two four/five-sections cavities
        short_cavity = TravelingWaveCavity(
            l_cav**2 * n_cav * 27.1e3 / 8, f_cav, 2 * np.pi * tau
        )
        shortInducedVoltage = InducedVoltageTime(
            self.beam, self.profile, [short_cavity]
        )
        l_cav = 43 * 0.374
        tau = l_cav / (v_g * c) * (1 + v_g)
        n_cav = 2
        long_cavity = TravelingWaveCavity(
            l_cav**2 * n_cav * 27.1e3 / 8, f_cav, 2 * np.pi * tau
        )
        longInducedVoltage = InducedVoltageTime(self.beam, self.profile, [long_cavity])
        self.induced_voltage = TotalInducedVoltage(
            self.beam, self.profile, [shortInducedVoltage, longInducedVoltage]
        )
        self.induced_voltage.induced_voltage_sum()

        self.cavity_tracker = RingAndRFTracker(
            self.rf,
            self.beam,
            profile=self.profile,
            interpolation=True,
            total_induced_voltage=self.induced_voltage,
        )

        """
        self.OTFB = SPSCavityFeedback(
            rf,
            self.profile,
            G_llrf=20,
            G_tx=[1.0355739238973907, 1.078403005653143],
            a_comb=63 / 64,
            turns=1000,
            post_LS2=True,
            df=[0.18433333e6, 0.2275e6],
            commissioning=SPSCavityLoopCommissioning(open_ff=True, rot_iq=-1),
        )
        """

        self.OTFB_tracker = RingAndRFTracker(
            self.rf,
            self.beam,
            profile=self.profile,
            total_induced_voltage=None,
            cavity_feedback=self.OTFB,
            interpolation=True,
        )"""

    def test_setUp(self):
        pass  # will fail if something in setUp() is wrong

    def test_FB_pre_tracking(self):
        digit_round = 3

        Vind3_mean = (
            np.mean(
                np.absolute(self.OTFB.OTFB_1.V_ANT_COARSE[-self.OTFB.OTFB_1.n_coarse :])
            )
            / 1e6
        )
        Vind3_std = (
            np.std(
                np.absolute(self.OTFB.OTFB_1.V_ANT_COARSE[-self.OTFB.OTFB_1.n_coarse :])
            )
            / 1e6
        )
        Vind3_mean_exp = 0.6761952255314454
        Vind3_std_exp = 5.802516784274078e-13

        Vind4_mean = (
            np.mean(
                np.absolute(self.OTFB.OTFB_2.V_ANT_COARSE[-self.OTFB.OTFB_2.n_coarse :])
            )
            / 1e6
        )
        Vind4_std = (
            np.std(
                np.absolute(self.OTFB.OTFB_2.V_ANT_COARSE[-self.OTFB.OTFB_2.n_coarse :])
            )
            / 1e6
        )
        Vind4_mean_exp = 0.9028493757326258
        Vind4_std_exp = 8.817799015245741e-13

        self.assertAlmostEqual(
            Vind3_mean,
            Vind3_mean_exp,
            places=digit_round,
            msg="In TestCavityFeedback test_FB_pretracking: "
            + "mean value of four-section cavity differs",
        )
        self.assertAlmostEqual(
            Vind3_std,
            Vind3_std_exp,
            places=digit_round,
            msg="In TestCavityFeedback test_FB_pretracking: standard "
            + "deviation of four-section cavity differs",
        )

        self.assertAlmostEqual(
            Vind4_mean,
            Vind4_mean_exp,
            places=digit_round,
            msg="In TestCavityFeedback test_FB_pretracking: "
            + "mean value of five-section cavity differs",
        )
        self.assertAlmostEqual(
            Vind4_std,
            Vind4_std_exp,
            places=digit_round,
            msg="In TestCavityFeedback test_FB_pretracking: standard "
            + "deviation of five-section cavity differs",
        )

    def test_FB_pre_tracking_IQ_v1(self):
        rtol = 1e-2  # relative tolerance
        atol = 0  # absolute tolerance
        # interpolate from coarse mesh to fine mesh
        V_fine_tot_3 = np.interp(
            self.profile.hist_x,
            self.OTFB.OTFB_1.rf_centers,
            self.OTFB.OTFB_1.V_IND_COARSE_GEN[-self.OTFB.OTFB_1.n_coarse :],
        )
        V_fine_tot_4 = np.interp(
            self.profile.hist_x,
            self.OTFB.OTFB_2.rf_centers,
            self.OTFB.OTFB_2.V_IND_COARSE_GEN[-self.OTFB.OTFB_2.n_coarse :],
        )

        V_tot_3 = V_fine_tot_3 / 1e6 * self.OTFB.OTFB_1.n_cavities
        V_tot_4 = V_fine_tot_4 / 1e6 * self.OTFB.OTFB_2.n_cavities

        V_sum = self.OTFB.V_sum / 1e6

        # expected generator voltage is only in Q
        V_tot_3_exp = 2.7j * np.ones(256)
        V_tot_4_exp = 1.8j * np.ones(256)
        V_sum_exp = 4.5j * np.ones(256)

        np.testing.assert_allclose(
            V_tot_3,
            V_tot_3_exp,
            rtol=rtol,
            atol=atol,
            err_msg="In TestCavityFeedback test_FB_pretracking_IQ: total voltage "
            + "in four-section cavity differs",
        )

        np.testing.assert_allclose(
            V_tot_4,
            V_tot_4_exp,
            rtol=rtol,
            atol=atol,
            err_msg="In TestCavityFeedback test_FB_pretracking_IQ: total voltage "
            + "in five-section cavity differs",
        )

        np.testing.assert_allclose(
            V_sum,
            V_sum_exp,
            rtol=rtol,
            atol=atol,
            err_msg="In TestCavityFeedback test_FB_pretracking_IQ: voltage sum "
            + " differs",
        )

    def test_rf_voltage(self):
        digit_round = 7

        # compute voltage
        self.cavity_tracker.rf_voltage_calculation()

        # compute voltage after OTFB pre-tracking
        self.OTFB_tracker.rf_voltage_calculation()

        # Since there is a systematic offset between the voltages,
        # compare the maxium of the ratio
        max_ratio = np.max(
            self.cavity_tracker.rf_voltage / self.OTFB_tracker.rf_voltage
        )

        max_ratio_exp = 1.0691789378342162

        self.assertAlmostEqual(
            max_ratio,
            max_ratio_exp,
            places=digit_round,
            msg="In TestCavityFeedback test_rf_voltage: " + "RF-voltages differ",
        )

    def test_beam_loading(self):
        digit_round = 7
        # Compute voltage with beam loading
        self.cavity_tracker.rf_voltage_calculation()
        cavity_tracker_total_voltage = (
            self.cavity_tracker.rf_voltage
            + self.cavity_tracker.totalInducedVoltage.induced_voltage
        )

        self.OTFB.track()
        self.OTFB_tracker.rf_voltage_calculation()
        OTFB_tracker_total_voltage = self.OTFB_tracker.rf_voltage

        max_ratio = np.max(cavity_tracker_total_voltage / OTFB_tracker_total_voltage)

        max_ratio_exp = 1.0691789378319636

        self.assertAlmostEqual(
            max_ratio,
            max_ratio_exp,
            places=digit_round,
            msg="In TestCavityFeedback test_beam_loading: " + "total voltages differ",
        )

    def test_Vsum_IQ(self):
        rtol = 1e-7  # relative tolerance
        atol = 0  # absolute tolerance

        self.OTFB.track(self.beam)

        V_sum = self.OTFB.V_sum / 1e6

        V_sum_exp = np.array(
            [
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.481752689e-02 + 4.510454808e00j,
                1.467003927e-02 + 4.510476685e00j,
                1.452259724e-02 + 4.510498562e00j,
                1.452264283e-02 + 4.510498562e00j,
                1.452268843e-02 + 4.510498561e00j,
                1.439484547e-02 + 4.510575214e00j,
                1.426704440e-02 + 4.510651849e00j,
                1.404617816e-02 + 4.510852091e00j,
                1.362512627e-02 + 4.511273235e00j,
                1.306981939e-02 + 4.511973098e00j,
                1.210167295e-02 + 4.513474984e00j,
                1.078779283e-02 + 4.515980194e00j,
                9.836259646e-03 + 4.518163804e00j,
                8.970183292e-03 + 4.521323325e00j,
                8.005504272e-03 + 4.526140001e00j,
                7.452065772e-03 + 4.532667025e00j,
                7.483017158e-03 + 4.541598489e00j,
                8.506539890e-03 + 4.551668244e00j,
                1.096244654e-02 + 4.563623978e00j,
                1.619690543e-02 + 4.580235448e00j,
                2.792405194e-02 + 4.607443250e00j,
                4.597134134e-02 + 4.641175739e00j,
                7.033402302e-02 + 4.677170985e00j,
                1.099375478e-01 + 4.724391792e00j,
                1.621238226e-01 + 4.776614334e00j,
                2.295290786e-01 + 4.831030852e00j,
                3.216513566e-01 + 4.891949350e00j,
                4.513066495e-01 + 4.959771077e00j,
                6.227821363e-01 + 5.030037721e00j,
                8.394069660e-01 + 5.094274038e00j,
                1.113238419e00 + 5.147372980e00j,
                1.455753644e00 + 5.179051447e00j,
                1.858949185e00 + 5.177942195e00j,
                2.335009136e00 + 5.128133857e00j,
                2.911053526e00 + 5.010649906e00j,
                3.590070132e00 + 4.800842826e00j,
                4.374313837e00 + 4.471893817e00j,
                5.257406144e00 + 3.994807057e00j,
                6.236486342e00 + 3.334433633e00j,
                7.299218926e00 + 2.454889827e00j,
                8.441367209e00 + 1.302043808e00j,
                9.619790054e00 + -1.431878086e-01j,
                1.078588057e01 + -1.902936195e00j,
                1.190448503e01 + -4.012173697e00j,
                1.293268248e01 + -6.523056046e00j,
                1.381664661e01 + -9.472290261e00j,
                1.448239457e01 + -1.288685295e01j,
                1.485143621e01 + -1.674102755e01j,
                1.483992960e01 + -2.095675968e01j,
                1.436704002e01 + -2.559246763e01j,
                1.335839107e01 + -3.059886818e01j,
                1.172864164e01 + -3.591159047e01j,
                9.405852675e00 + -4.148017912e01j,
                6.362767821e00 + -4.714251574e01j,
                2.551801496e00 + -5.281570912e01j,
                -2.060352411e00 + -5.840971185e01j,
                -7.443164880e00 + -6.377131279e01j,
                -1.356848136e01 + -6.877704179e01j,
                -2.041816782e01 + -7.333358029e01j,
                -2.797169811e01 + -7.734977038e01j,
                -3.617705825e01 + -8.072875822e01j,
                -4.488133996e01 + -8.335278565e01j,
                -5.386298588e01 + -8.512637679e01j,
                -6.304266613e01 + -8.601255040e01j,
                -7.235080816e01 + -8.599689233e01j,
                -8.155032660e01 + -8.507817883e01j,
                -9.043638606e01 + -8.329863930e01j,
                -9.896600061e01 + -8.069405104e01j,
                -1.070668907e02 + -7.732382369e01j,
                -1.146401037e02 + -7.325710986e01j,
                -1.215732229e02 + -6.861045362e01j,
                -1.277377627e02 + -6.353511085e01j,
                -1.331082617e02 + -5.814733047e01j,
                -1.376921000e02 + -5.253932111e01j,
                -1.414931450e02 + -4.682871685e01j,
                -1.445085482e02 + -4.116149899e01j,
                -1.467954368e02 + -3.560363078e01j,
                -1.484015646e02 + -3.027212706e01j,
                -1.493796149e02 + -2.529825410e01j,
                -1.498198307e02 + -2.070001278e01j,
                -1.498059244e02 + -1.645703513e01j,
                -1.494141759e02 + -1.260122729e01j,
                -1.487254135e02 + -9.194310050e00j,
                -1.478140791e02 + -6.229440009e00j,
                -1.467401971e02 + -3.666445341e00j,
                -1.455771042e02 + -1.509168141e00j,
                -1.443839999e02 + 2.598553928e-01j,
                -1.431903768e02 + 1.700762864e00j,
                -1.420233660e02 + 2.856833166e00j,
                -1.409337576e02 + 3.744875834e00j,
                -1.399385306e02 + 4.403406937e00j,
                -1.390159460e02 + 4.891876339e00j,
                -1.381811642e02 + 5.235603747e00j,
                -1.374439159e02 + 5.458671982e00j,
                -1.368085639e02 + 5.586731787e00j,
                -1.362659691e02 + 5.642764332e00j,
                -1.358243790e02 + 5.648552277e00j,
                -1.354773565e02 + 5.620510027e00j,
                -1.351877344e02 + 5.570879849e00j,
                -1.349501899e02 + 5.508917179e00j,
                -1.347610938e02 + 5.442895280e00j,
                -1.346137951e02 + 5.378789620e00j,
                -1.344995383e02 + 5.319627106e00j,
                -1.344087722e02 + 5.265182832e00j,
                -1.343376502e02 + 5.218027112e00j,
                -1.342800623e02 + 5.175638409e00j,
                -1.342336381e02 + 5.140497996e00j,
                -1.341996556e02 + 5.118665146e00j,
                -1.341707253e02 + 5.100847060e00j,
                -1.341444026e02 + 5.084935558e00j,
                -1.341211674e02 + 5.074371007e00j,
                -1.340996990e02 + 5.068345477e00j,
                -1.340792378e02 + 5.064350457e00j,
                -1.340592146e02 + 5.061565710e00j,
                -1.340393057e02 + 5.061197778e00j,
                -1.340194638e02 + 5.061764011e00j,
                -1.339997145e02 + 5.062678068e00j,
                -1.339799966e02 + 5.063934804e00j,
                -1.339603403e02 + 5.065400833e00j,
                -1.339403005e02 + 5.067505666e00j,
                -1.339199270e02 + 5.070108329e00j,
                -1.338995535e02 + 5.072710190e00j,
                -1.338790694e02 + 5.075411344e00j,
                -1.338587201e02 + 5.078047912e00j,
                -1.338385110e02 + 5.080633437e00j,
                -1.338181671e02 + 5.083281895e00j,
                -1.337976827e02 + 5.085979747e00j,
                -1.337771983e02 + 5.088676761e00j,
                -1.337567138e02 + 5.091372938e00j,
                -1.337362293e02 + 5.094068279e00j,
                -1.337157447e02 + 5.096762782e00j,
                -1.336952601e02 + 5.099456448e00j,
                -1.336747754e02 + 5.102149276e00j,
                -1.336542906e02 + 5.104841268e00j,
                -1.336338058e02 + 5.107532423e00j,
                -1.336133209e02 + 5.110222740e00j,
                -1.335928360e02 + 5.112912221e00j,
                -1.335723510e02 + 5.115600864e00j,
                -1.335518660e02 + 5.118288670e00j,
                -1.335313809e02 + 5.120975639e00j,
                -1.335108958e02 + 5.123661771e00j,
                -1.334904106e02 + 5.126347066e00j,
                -1.334699254e02 + 5.129031524e00j,
                -1.334494401e02 + 5.131715144e00j,
                -1.334289548e02 + 5.134397928e00j,
                -1.334084694e02 + 5.137079874e00j,
                -1.333879839e02 + 5.139760983e00j,
                -1.333674984e02 + 5.142441255e00j,
                -1.333470128e02 + 5.145120690e00j,
                -1.333265272e02 + 5.147799288e00j,
                -1.333060415e02 + 5.150477049e00j,
                -1.332855558e02 + 5.153153972e00j,
                -1.332650700e02 + 5.155830059e00j,
                -1.332445842e02 + 5.158505308e00j,
                -1.332240983e02 + 5.161179720e00j,
                -1.332036124e02 + 5.163853295e00j,
                -1.331831264e02 + 5.166526033e00j,
                -1.331626403e02 + 5.169197934e00j,
                -1.331421542e02 + 5.171868997e00j,
                -1.331216681e02 + 5.174539224e00j,
                -1.331011819e02 + 5.177208613e00j,
                -1.330806956e02 + 5.179877165e00j,
                -1.330602093e02 + 5.182544880e00j,
                -1.330397229e02 + 5.185211758e00j,
                -1.330192365e02 + 5.187877799e00j,
                -1.329987501e02 + 5.190543002e00j,
                -1.329782635e02 + 5.193207369e00j,
                -1.329577769e02 + 5.195870898e00j,
                -1.329372903e02 + 5.198533590e00j,
                -1.329168036e02 + 5.201195445e00j,
                -1.328963169e02 + 5.203856463e00j,
                -1.328758301e02 + 5.206516643e00j,
                -1.328553432e02 + 5.209175987e00j,
                -1.328348563e02 + 5.211834493e00j,
                -1.328143694e02 + 5.214492162e00j,
                -1.327938824e02 + 5.217148994e00j,
                -1.327733953e02 + 5.219804989e00j,
                -1.327529082e02 + 5.222460147e00j,
                -1.327324210e02 + 5.225114467e00j,
                -1.327119338e02 + 5.227767951e00j,
                -1.326914466e02 + 5.230420597e00j,
                -1.326709592e02 + 5.233072406e00j,
                -1.326504719e02 + 5.235723378e00j,
                -1.326299844e02 + 5.238373513e00j,
                -1.326094969e02 + 5.241022810e00j,
                -1.325890094e02 + 5.243671271e00j,
                -1.325685218e02 + 5.246318894e00j,
                -1.325480342e02 + 5.248965680e00j,
                -1.325275465e02 + 5.251611629e00j,
                -1.325070587e02 + 5.254256740e00j,
                -1.324865709e02 + 5.256901015e00j,
                -1.324660831e02 + 5.259544452e00j,
            ]
        )

        np.testing.assert_allclose(
            np.around(V_sum_exp, 5),
            np.around(V_sum, 5),
            rtol=rtol,
            atol=atol,
            err_msg="In TestCavityFeedback test_Vsum_IQ: total voltage "
            + "is different from expected values!",
        )


class TestSPSOneTurnFeedback(unittest.TestCase):
    def setUp(self):
        # Parameters ----------------------------------------------------------
        C = 2 * np.pi * 1100.009  # Ring circumference [m]
        gamma_t = 18.0  # Transition Gamma [-]
        alpha = 1 / (gamma_t**2)  # Momentum compaction factor [-]
        p_s = 450e9  # Synchronous momentum [eV]
        h = 4620  # 200 MHz harmonic number [-]
        V = 10e6  # 200 MHz RF voltage [V]
        phi = 0  # 200 MHz phase [-]

        # Parameters for the Simulation
        N_m = int(1e5)  # Number of macro-particles for tracking
        N_b = 1.0e11  # Bunch intensity [ppb]
        N_t = 1  # Number of turns to track

        # Objects -------------------------------------------------------------

        # Ring
        # self.ring = Ring(C, alpha, p_s, Proton(), N_t)
        self.ring = Ring(circumference=C)

        # RFStation
        self.rfstation = MultiHarmonicCavity(n_harmonics=1, main_harmonic_idx=0)
        self.rfstation.voltage = np.array([V])
        self.rfstation.phi_rf = np.array([phi])
        self.rfstation.harmonic = np.array([h])
        self.magnetic_cycle = ConstantMagneticCycle(
            reference_particle=proton,
            value=p_s,
            in_unit="momentum",
        )
        self.ring.add_element(self.rfstation)
        self.ring.add_drifts(
            n_drifts_per_section=1,
            n_sections=1,
            driftclass=DriftSimple,
            transition_gamma=gamma_t,
        )
        sim = Simulation(ring=self.ring, magnetic_cycle=self.magnetic_cycle)

        # Beam
        self.beam = Beam(n_particles=N_b, particle_type=proton)
        self.beam.setup_beam(
            dt=np.zeros(N_m),
            dE=np.zeros(N_m),
            reference_total_energy=self.magnetic_cycle.get_total_energy_init(
                0,
                0,
                particle_type=proton,
            ),
        )
        beam_2 = deepcopy(self.beam)
        self.rfstation.track(beam_2)
        del beam_2

        self.profile = StaticProfile(
            cut_left=0.0e-9, cut_right=self.rfstation._t_rev, n_bins=4620
        )
        self.profile.track(self.beam)

        # Cavity
        self.Commissioning = SPSCavityLoopCommissioning(
            open_ff=True, rot_iq=-1, cpp_conv=False
        )

        self.OTFB = SPSOneTurnFeedback(
            self.rfstation,
            self.profile,
            3,
            a_comb=63 / 64,
            commissioning=self.Commissioning,
        )

        self.OTFB.update_rf_variables()
        self.OTFB.update_fb_variables()

        self.turn_array = np.linspace(
            0, 2 * self.rfstation._t_rev, 2 * self.OTFB.n_coarse
        )

    def test_setup(self):
        pass

    def test_set_point(self):
        self.OTFB.set_point()
        t_sig = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        t_sig[-self.OTFB.n_coarse :] = (
            (1 / 9) * 10e6 * np.exp(1j * (np.pi / 2 - self.rfstation.phi_rf[0]))
        )

        np.testing.assert_allclose(self.OTFB.V_SET, t_sig)

    def test_error_and_gain(self):
        self.OTFB.error_and_gain()

        np.testing.assert_allclose(self.OTFB.DV_GEN, self.OTFB.V_SET * self.OTFB.G_llrf)

    def test_comb(self):
        sig = np.zeros(self.OTFB.n_coarse)
        self.OTFB.DV_COMB_OUT = np.sin(
            2 * np.pi * self.turn_array / self.rfstation._t_rev
        )
        self.OTFB.DV_GEN = -np.sin(2 * np.pi * self.turn_array / self.rfstation._t_rev)
        self.OTFB.a_comb = 0.5

        self.OTFB.comb()

        np.testing.assert_allclose(self.OTFB.DV_COMB_OUT[-self.OTFB.n_coarse :], sig)

    def test_one_turn_delay(self):
        self.OTFB.DV_COMB_OUT = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_COMB_OUT[self.OTFB.n_coarse] = 1

        self.OTFB.one_turn_delay()

        self.assertEqual(
            np.argmax(self.OTFB.DV_DELAYED), 2 * self.OTFB.n_coarse - self.OTFB.n_mov_av
        )

    def test_mod_to_fr(self):
        self.OTFB.DV_DELAYED = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_DELAYED[-self.OTFB.n_coarse :] = 1 + 1j * 0

        self.mod_phi = np.copy(self.OTFB.dphi_mod)
        self.OTFB.mod_to_fr()
        ref_DV_MOD_FR = np.load(
            os.path.join(this_directory, "resources/ref_DV_MOD_FR.npy")
        )

        # Test real part
        np.testing.assert_allclose(
            self.OTFB.DV_MOD_FR[-self.OTFB.n_coarse :].real,
            ref_DV_MOD_FR.real,
            rtol=1e-6,
            atol=1e-9,
            err_msg="In TestSPSOneTurnFeedback test_mod_to_fr(), "
            "mismatch in real part of modulated signal",
        )

        # Test imaginary part
        np.testing.assert_allclose(
            self.OTFB.DV_MOD_FR[-self.OTFB.n_coarse :].imag,
            ref_DV_MOD_FR.imag,
            rtol=1e-6,
            atol=1e-9,
            err_msg="In TestSPSOneTurnFeedback test_mod_to_fr(), "
            "mismatch in imaginary part of modulated signal",
        )

        self.OTFB.DV_DELAYED = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_DELAYED[-self.OTFB.n_coarse :] = 1 + 1j * 0

        self.OTFB.dphi_mod = 0
        self.OTFB.mod_to_fr()

        time_array = self.OTFB.rf_centers - 0.5 * self.OTFB.T_s
        ref_sig = np.cos(
            (self.OTFB.omega_carrier - self.OTFB.omega_c)
            * time_array[: self.OTFB.n_coarse]
        ) + 1j * np.sin(
            (self.OTFB.omega_carrier - self.OTFB.omega_c)
            * time_array[: self.OTFB.n_coarse]
        )

        np.testing.assert_allclose(self.OTFB.DV_MOD_FR[-self.OTFB.n_coarse :], ref_sig)

        self.OTFB.dphi_mod = self.mod_phi

    def test_mov_avg(self):
        sig = np.zeros(self.OTFB.n_coarse - 1)
        sig[: self.OTFB.n_mov_av] = 1
        self.OTFB.DV_MOD_FR = np.zeros(2 * self.OTFB.n_coarse)
        self.OTFB.DV_MOD_FR[-self.OTFB.n_coarse + 1 :] = sig

        self.OTFB.mov_avg()

        sig = np.zeros(self.OTFB.n_coarse)
        sig[: self.OTFB.n_mov_av] = (1 / self.OTFB.n_mov_av) * np.array(
            range(self.OTFB.n_mov_av)
        )
        sig[self.OTFB.n_mov_av : 2 * self.OTFB.n_mov_av] = (1 / self.OTFB.n_mov_av) * (
            self.OTFB.n_mov_av - np.array(range(self.OTFB.n_mov_av))
        )

        np.testing.assert_allclose(
            np.abs(self.OTFB.DV_MOV_AVG[-self.OTFB.n_coarse :]), sig
        )

    def test_mod_to_frf(self):
        self.OTFB.DV_MOV_AVG = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_MOV_AVG[-self.OTFB.n_coarse :] = 1 + 1j * 0

        self.mod_phi = np.copy(self.OTFB.dphi_mod)
        self.OTFB.mod_to_frf()
        ref_DV_MOD_FRF = np.load(
            os.path.join(this_directory, "resources/ref_DV_MOD_FRF.npy")
        )

        # Test real part
        np.testing.assert_allclose(
            self.OTFB.DV_MOD_FRF[-self.OTFB.n_coarse :].real,
            ref_DV_MOD_FRF.real,
            rtol=1e-6,
            atol=1e-9,
            err_msg="In TestSPSOneTurnFeedback test_mod_to_frf(), "
            "mismatch in real part of modulated signal",
        )

        # Test imaginary part
        np.testing.assert_allclose(
            self.OTFB.DV_MOD_FRF[-self.OTFB.n_coarse :].imag,
            ref_DV_MOD_FRF.imag,
            rtol=1e-6,
            atol=1e-9,
            err_msg="In TestSPSOneTurnFeedback test_mod_to_frf(), "
            "mismatch in imaginary part of modulated signal",
        )

        self.OTFB.DV_MOV_AVG = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.DV_MOV_AVG[-self.OTFB.n_coarse :] = 1 + 1j * 0

        self.OTFB.dphi_mod = 0
        self.OTFB.mod_to_frf()

        time_array = self.OTFB.rf_centers - 0.5 * self.OTFB.T_s
        dphi_demod = (self.OTFB.omega_c - self.OTFB.omega_carrier) * self.OTFB.TWC.tau
        ref_sig = np.cos(
            -(self.OTFB.omega_carrier - self.OTFB.omega_c)
            * time_array[: self.OTFB.n_coarse]
            - dphi_demod
        ) + 1j * np.sin(
            -(self.OTFB.omega_carrier - self.OTFB.omega_c)
            * time_array[: self.OTFB.n_coarse]
            - dphi_demod
        )

        np.testing.assert_allclose(self.OTFB.DV_MOD_FRF[-self.OTFB.n_coarse :], ref_sig)

        self.OTFB.dphi_mod = self.mod_phi

    def test_sum_and_gain(self):
        self.OTFB.V_SET[-self.OTFB.n_coarse :] = np.ones(
            self.OTFB.n_coarse, dtype=complex
        )
        self.OTFB.DV_MOD_FRF[-self.OTFB.n_coarse :] = np.ones(
            self.OTFB.n_coarse, dtype=complex
        )

        self.OTFB.sum_and_gain()

        sig = 2 * np.ones(self.OTFB.n_coarse) * self.OTFB.G_tx / self.OTFB.TWC.R_gen

        np.testing.assert_allclose(self.OTFB.I_GEN_COARSE[-self.OTFB.n_coarse :], sig)

    def test_gen_response(self):
        # Tests generator response at resonant frequency.
        self.OTFB.I_GEN_COARSE = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.I_GEN_COARSE[self.OTFB.n_coarse] = 1

        self.OTFB.TWC.impulse_response_gen(self.OTFB.TWC.omega_r, self.OTFB.rf_centers)
        self.OTFB.gen_response()

        sig = np.zeros(self.OTFB.n_coarse)
        sig[1 : 1 + self.OTFB.n_mov_av] = 4 * self.OTFB.TWC.R_gen / self.OTFB.TWC.tau
        sig[0] = 2 * self.OTFB.TWC.R_gen / self.OTFB.TWC.tau
        sig[self.OTFB.n_mov_av + 1] = 2 * self.OTFB.TWC.R_gen / self.OTFB.TWC.tau
        sig *= self.OTFB.T_s

        np.testing.assert_allclose(
            np.abs(self.OTFB.V_IND_COARSE_GEN[-self.OTFB.n_coarse :])
            * self.OTFB.n_cavities,
            sig,
            atol=5e-5,
        )

        # Tests generator response at carrier frequency.
        self.OTFB.TWC.impulse_response_gen(self.OTFB.omega_c, self.OTFB.rf_centers)

        self.OTFB.I_GEN_COARSE = np.zeros(2 * self.OTFB.n_coarse, dtype=complex)
        self.OTFB.I_GEN_COARSE[self.OTFB.n_coarse] = 1

        self.OTFB.gen_response()

        ref_V_IND_COARSE_GEN = np.load(
            os.path.join(this_directory, "resources/ref_V_IND_COARSE_GEN.npy")
        )

        # Test real part - sum of cavities
        np.testing.assert_allclose(
            np.around(self.OTFB.V_IND_COARSE_GEN[-self.OTFB.n_coarse :].real, 12)
            * self.OTFB.n_cavities,
            np.around(ref_V_IND_COARSE_GEN.real, 12),
            rtol=1e-6,
            atol=0,
            err_msg="In TestSPSOneTurnFeedback test_gen_response(), "
            "mismatch in real part of generator response",
        )

        # Test imaginary part - sum of cavities
        np.testing.assert_allclose(
            np.around(self.OTFB.V_IND_COARSE_GEN[-self.OTFB.n_coarse :].imag, 12)
            * self.OTFB.n_cavities,
            np.around(ref_V_IND_COARSE_GEN.imag, 12),
            rtol=1e-6,
            atol=0,
            err_msg="In TestSPSOneTurnFeedback test_gen_response(), "
            "mismatch in imaginary part of generator response",
        )


class TestSPSTransmitterGain(unittest.TestCase):
    def setUp(self):
        self.ring = Ring(circumference=2 * np.pi * 1100.009)
        cavity = MultiHarmonicCavity(
            n_harmonics=1,
            main_harmonic_idx=0,
        )
        cavity.harmonic = np.array([4620])
        cavity.phi_rf = np.array([0])
        cavity.voltage = np.array([4.5e6])
        self.rf = cavity
        drift = DriftSimple(
            orbit_length=2 * np.pi * 1100.009,
            transition_gamma=18.0,
        )

        self.ring.add_element(cavity)
        self.ring.add_element(drift)
        self.magnetic_cycle = ConstantMagneticCycle(
            reference_particle=proton,
            value=25.92e9,
            in_unit="momentum",
        )
        self.beam = Beam(
            n_particles=1.0e11,
            particle_type=proton,
        )
        sim = Simulation(
            ring=self.ring,
            magnetic_cycle=self.magnetic_cycle,
        )
        sim.prepare_beam(
            self.beam,
            BiGaussian(
                n_macroparticles=int(1e5),
                sigma_dt=3.2e-9 / 4,
                seed=1234,
                reinsertion=True,
            ),
        )
        beam_2 = deepcopy(self.beam)
        self.rf.track(beam_2)
        del beam_2
        cavity._omega = np.array([200.222e6 * 2 * np.pi])

        self.profile = StaticProfile(
            cut_left=0.0e-9,
            cut_right=self.rf._t_rev,
            n_bins=4620,
        )
        self.profile.track(self.beam)
        # Commissioning options for the cavity feedback
        self.commissioning = SPSCavityLoopCommissioning(
            debug=True,
            open_loop=False,
            open_fb=True,
            open_drive=False,
            open_ff=True,
            rot_iq=-1,
        )

    def test_setup(self):
        pass

    def init_otfb(
        self, rf, profile, commissioning, no_sections, no_cavities, V_part, G_tx
    ):
        OTFB = SPSOneTurnFeedback(
            rf,
            profile,
            no_sections,
            n_cavities=no_cavities,
            V_part=V_part,
            G_ff=0,
            G_llrf=5,
            G_tx=G_tx,
            a_comb=15 / 16,
            commissioning=commissioning,
        )
        for i in range(100):
            OTFB.track_no_beam()

        V = (
            np.average(np.absolute(OTFB.V_ANT_COARSE[-10])) * OTFB.n_cavities * 1e-6
        )  # in MV
        I = np.average(np.absolute(OTFB.I_GEN_COARSE[-10])) * 1e-2  # in 100 A

        return OTFB, V, I

    def test_preLS24sec(self):
        OTFB, V, I = self.init_otfb(
            self.rf, self.profile, self.commissioning, 4, 2, 4 / 9, 1.03573985
        )
        self.assertAlmostEqual(
            V,
            2.00000000,
            places=7,
        )
        self.assertAlmostEqual(
            I,
            0.78244888,
            places=7,
        )

    def test_preLS25sec(self):
        OTFB, V, I = self.init_otfb(
            self.rf, self.profile, self.commissioning, 5, 2, 5 / 9, 1.01547845
        )
        self.assertAlmostEqual(
            V,
            2.50000000,
            places=7,
        )
        self.assertAlmostEqual(
            I,
            0.76359084,
            places=7,
        )

    def test_postLS23sec(self):
        OTFB, V, I = self.init_otfb(
            self.rf, self.profile, self.commissioning, 3, 4, 6 / 10, 1.01724955
        )
        self.assertAlmostEqual(
            V,
            2.70000000,
            places=7,
        )
        self.assertAlmostEqual(
            I,
            0.69703574,
            places=7,
        )

    def test_postLS24sec(self):
        OTFB, V, I = self.init_otfb(
            self.rf, self.profile, self.commissioning, 4, 2, 4 / 10, 1.03573985
        )
        self.assertAlmostEqual(
            V,
            1.80000000,
            places=7,
        )
        self.assertAlmostEqual(
            I,
            0.70420400,
            places=7,
        )
