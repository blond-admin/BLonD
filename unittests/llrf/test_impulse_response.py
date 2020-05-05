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
from scipy.constants import c

from blond.llrf.impulse_response import rectangle, triangle, \
    SPS4Section200MHzTWC
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import Profile, CutOptions
from blond.impedances.impedance import InducedVoltageTime, TotalInducedVoltage
from blond.llrf.cavity_feedback import SPSOneTurnFeedback, \
    CavityFeedbackCommissioning
from blond.impedances.impedance_sources import TravelingWaveCavity


class TestRectangle(unittest.TestCase):

    def test_1(self):

        tau = 1.
        time = np.array([-1, -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, 1])
        rect_exp = np.array([0., 0., 0.5, 1., 1., 1., 0.5, 0., 0.])
        rect_act = rectangle(time, tau)

        rect_exp = np.around(rect_exp, 12)
        rect_act = np.around(rect_act, 12)
        self.assertSequenceEqual(rect_exp.tolist(), rect_act.tolist(),
            msg="In TestRectangle test 1: rectangle arrays differ")

    def test_2(self):

        tau = 1.
        time = np.array([-0.51, -0.26, 0.01, 0.26, 0.51, 0.76, 1.01])
        rect_exp = np.array([0.5, 1., 1., 1., 0.5, 0., 0.])
        rect_act = rectangle(time, tau)

        rect_exp = np.around(rect_exp, 12)
        rect_act = np.around(rect_act, 12)
        self.assertSequenceEqual(rect_exp.tolist(), rect_act.tolist(),
            msg="In TestRectangle test 2: rectangle arrays differ")


class TestTriangle(unittest.TestCase):

    def test_1(self):

        tau = 1.
        time = np.array([-0.5, -0.25, 0., 0.25, 0.5, 0.75, 1, 1.25, 1.5])
        tri_exp = np.array([0., 0., 0.5, 0.75, 0.5, 0.25, 0., 0., 0.])
        tri_act = triangle(time, tau)

        tri_exp = np.around(tri_exp, 12)
        tri_act = np.around(tri_act, 12)
        self.assertSequenceEqual(tri_exp.tolist(), tri_act.tolist(),
            msg="In TestTriangle test 1: triangle arrays differ")

    def test_2(self):

        tau = 1.
        time = np.array([-0.01, 0.26, 0.51, 0.76, 1.01, 1.26, 1.51])
        tri_exp = np.array([0.5, 0.74, 0.49, 0.24, 0., 0., 0.])
        tri_act = triangle(time, tau)

        tri_exp = np.around(tri_exp, 12)
        tri_act = np.around(tri_act, 12)
        self.assertSequenceEqual(tri_exp.tolist(), tri_act.tolist(),
            msg="In TestTriangle test 2: triangle arrays differ")


class TestTravelingWaveCavity(unittest.TestCase):

    def test_vg(self):
        from blond.llrf.impulse_response import TravellingWaveCavity
        v_g = 0.0946+1

        with self.assertRaises(RuntimeError, msg="In TestTravelingWaveCavity,"
                               + " no exception for group velocity > 1"):

            TravellingWaveCavity(0.374, 43, 2.71e4, v_g, 2*np.pi*200.222e6)

    def test_wake(self):

        time = np.linspace(-0.1e-6, 0.7e-6, 1000)

        l_cav = 16.082
        v_g = 0.0946
        tau = l_cav/(v_g*c)*(1 + v_g)

        TWC_impedance_source = TravelingWaveCavity(l_cav**2 * 27.1e3 / 8,
                                                   200.222e6, 2*np.pi*tau)

        TWC_impedance_source.wake_calc(time-time[0])
        wake_impSource = np.around(TWC_impedance_source.wake/1e12, 12)

        TWC_impulse_response = SPS4Section200MHzTWC()
        # omega_c not need for computation of wake function
        TWC_impulse_response.impulse_response_beam(2*np.pi*200.222e6, time)
        TWC_impulse_response.impulse_response_gen(2*np.pi*200.222e6, time)
        TWC_impulse_response.compute_wakes(time)
        wake_impResp = np.around(TWC_impulse_response.W_beam/1e12, 12)

        self.assertListEqual(wake_impSource.tolist(), wake_impResp.tolist(),
            msg="In TestTravelingWaveCavity test_wake: wake fields differ")

    def test_vind(self):

        # randomly chose omega_c from allowed range
        np.random.seed(1980)
        factor = np.random.uniform(0.9, 1.1)

        # round results to this digits
        digit_round = 8

        # SPS parameters
        C = 2*np.pi*1100.009        # Ring circumference [m]
        gamma_t = 18.0              # Gamma at transition
        alpha = 1/gamma_t**2        # Momentum compaction factor
        p_s = 25.92e9               # Synchronous momentum at injection [eV]
        h = 4620                    # 200 MHz system harmonic
        V = 4.5e6                   # 200 MHz RF voltage
        phi = 0.                    # 200 MHz RF phase

        # Beam and tracking parameters
        N_m = 1e5                   # Number of macro-particles for tracking
        N_b = 1.0e11                 # Bunch intensity [ppb]
        N_t = 1                  # Number of turns to track

        ring = Ring(C, alpha, p_s, Proton(), n_turns=N_t)
        rf = RFStation(ring, h, V, phi)
        beam = Beam(ring, N_m, N_b)
        bigaussian(ring, rf, beam, 3.2e-9/4, seed=1234, reinsertion=True)

        n_shift = 5     # how many rf-buckets to shift beam
        beam.dt += n_shift * rf.t_rf[0,0]
        profile = Profile(beam, CutOptions=
                          CutOptions(cut_left=(n_shift-1.5)*rf.t_rf[0,0],
                                     cut_right=(n_shift+1.5)*rf.t_rf[0,0],
                                     n_slices=140))
        profile.track()

        l_cav = 16.082
        v_g = 0.0946
        tau = l_cav/(v_g*c)*(1 + v_g)
        TWC_impedance_source = TravelingWaveCavity(l_cav**2 * 27.1e3 / 8,
                                                   200.222e6, 2*np.pi*tau)

        # Beam loading by convolution of beam and wake from cavity
        inducedVoltageTWC = InducedVoltageTime(beam, profile,
                                               [TWC_impedance_source])
        induced_voltage = TotalInducedVoltage(beam, profile,
                                              [inducedVoltageTWC])
        induced_voltage.induced_voltage_sum()
        V_ind_impSource = np.around(induced_voltage.induced_voltage,
                                    digit_round)

        # Beam loading via feed-back system
        OTFB_4 = SPSOneTurnFeedback(rf, beam, profile, 4, n_cavities=1,
            Commissioning=CavityFeedbackCommissioning(open_FF=True))
        OTFB_4.counter = 0  # First turn

        OTFB_4.omega_c = factor * OTFB_4.TWC.omega_r
        # Compute impulse response
        OTFB_4.TWC.impulse_response_beam(OTFB_4.omega_c, profile.bin_centers)

        # Compute induced voltage in (I,Q) coordinates
        OTFB_4.beam_induced_voltage(lpf=False)
        # convert back to time
        V_ind_OTFB \
            = OTFB_4.V_fine_ind_beam.real \
                * np.cos(OTFB_4.omega_c*profile.bin_centers) \
            + OTFB_4.V_fine_ind_beam.imag \
                * np.sin(OTFB_4.omega_c*profile.bin_centers)
        V_ind_OTFB = np.around(V_ind_OTFB, digit_round)

        self.assertListEqual(V_ind_impSource.tolist(), V_ind_OTFB.tolist(),
            msg="In TravelingWaveCavity test_vind: induced voltages differ")

    def test_beam_fine_coarse(self):

        # Test beam impulse response and induced voltage
        # Compare on coarse and fine grid

        # Create a batch of 100 equal, short bunches at HL-LHC intensity
        ring = Ring(2*np.pi*1100.009, 1/18**2, 25.92e9, Particle=Proton())
        rf = RFStation(ring, [4620], [4.5e6], [0], n_rf=1)
        bunches = 100
        N_m = int(1e5)
        N_b = 2.3e11
        beam = Beam(ring, N_m, N_b)
        bigaussian(ring, rf, beam, 1.8e-9/4, seed=1234, reinsertion=True)
        beam2 = Beam(ring, bunches*N_m, bunches*N_b)
        bunch_spacing = 5 * rf.t_rf[0, 0]
        buckets = 5 * bunches
        for i in range(bunches):
            beam2.dt[i * N_m:(i + 1) * N_m] = beam.dt + i * bunch_spacing
            beam2.dE[i * N_m:(i + 1) * N_m] = beam.dE
        profile2 = Profile(beam2, CutOptions=CutOptions(cut_left=0,
            cut_right=bunches*bunch_spacing,n_slices=1000*buckets))
        profile2.track()

        # Calculate impulse response and induced voltage
        OTFB = SPSOneTurnFeedback(rf, beam2, profile2, 3, n_cavities=1,
            Commissioning=CavityFeedbackCommissioning(open_FF=True))
        OTFB.TWC.impulse_response_beam(OTFB.omega_c, OTFB.profile.bin_centers,
                                       OTFB.rf_centers)
        OTFB.beam_induced_voltage(lpf=False)
        imp_fine_meas = (OTFB.TWC.h_beam[::1000])[:100]
        imp_coarse_meas = OTFB.TWC.h_beam_coarse[:100]

        imp_fine_ref = np.array([1.0504062083e+12+0.0000000000e+00j,
            2.0781004955e+12+2.7183115978e+09j,
            2.0553850965e+12+5.3772054987e+09j,
            2.0326663360e+12+7.9766773057e+09j,
            2.0099443306e+12+1.0516722825e+10j,
            1.9872191969e+12+1.2997338066e+10j,
            1.9644910516e+12+1.5418519242e+10j,
            1.9417600113e+12+1.7780262770e+10j,
            1.9190261924e+12+2.0082565269e+10j,
            1.8962897118e+12+2.2325423561e+10j,
            1.8735506859e+12+2.4508834674e+10j,
            1.8508092314e+12+2.6632795838e+10j,
            1.8280654649e+12+2.8697304485e+10j,
            1.8053195030e+12+3.0702358252e+10j,
            1.7825714624e+12+3.2647954978e+10j,
            1.7598214597e+12+3.4534092708e+10j,
            1.7370696115e+12+3.6360769688e+10j,
            1.7143160345e+12+3.8127984368e+10j,
            1.6915608452e+12+3.9835735402e+10j,
            1.6688041604e+12+4.1484021645e+10j,
            1.6460460966e+12+4.3072842159e+10j,
            1.6232867705e+12+4.4602196207e+10j,
            1.6005262987e+12+4.6072083256e+10j,
            1.5777647978e+12+4.7482502976e+10j,
            1.5550023845e+12+4.8833455241e+10j,
            1.5322391754e+12+5.0124940128e+10j,
            1.5094752871e+12+5.1356957918e+10j,
            1.4867108362e+12+5.2529509093e+10j,
            1.4639459395e+12+5.3642594342e+10j,
            1.4411807134e+12+5.4696214555e+10j,
            1.4184152746e+12+5.5690370826e+10j,
            1.3956497397e+12+5.6625064451e+10j,
            1.3728842254e+12+5.7500296932e+10j,
            1.3501188481e+12+5.8316069972e+10j,
            1.3273537246e+12+5.9072385477e+10j,
            1.3045889714e+12+5.9769245560e+10j,
            1.2818247051e+12+6.0406652532e+10j,
            1.2590610424e+12+6.0984608912e+10j,
            1.2362980996e+12+6.1503117419e+10j,
            1.2135359936e+12+6.1962180977e+10j,
            1.1907748407e+12+6.2361802713e+10j,
            1.1680147576e+12+6.2701985956e+10j,
            1.1452558608e+12+6.2982734240e+10j,
            1.1224982669e+12+6.3204051301e+10j,
            1.0997420924e+12+6.3365941080e+10j,
            1.0769874538e+12+6.3468407718e+10j,
            1.0542344676e+12+6.3511455561e+10j,
            1.0314832504e+12+6.3495089159e+10j,
            1.0087339187e+12+6.3419313265e+10j,
            9.8598658892e+11+6.3284132832e+10j,
            9.6324137757e+11+6.3089553021e+10j,
            9.4049840113e+11+6.2835579191e+10j,
            9.1775777605e+11+6.2522216909e+10j,
            8.9501961879e+11+6.2149471941e+10j,
            8.7228404579e+11+6.1717350259e+10j,
            8.4955117347e+11+6.1225858036e+10j,
            8.2682111826e+11+6.0675001648e+10j,
            8.0409399656e+11+6.0064787676e+10j,
            7.8136992476e+11+5.9395222903e+10j,
            7.5864901923e+11+5.8666314312e+10j,
            7.3593139635e+11+5.7878069094e+10j,
            7.1321717247e+11+5.7030494640e+10j,
            6.9050646392e+11+5.6123598543e+10j,
            6.6779938704e+11+5.5157388601e+10j,
            6.4509605813e+11+5.4131872814e+10j,
            6.2239659348e+11+5.3047059384e+10j,
            5.9970110939e+11+5.1902956716e+10j,
            5.7700972210e+11+5.0699573420e+10j,
            5.5432254788e+11+4.9436918305e+10j,
            5.3163970295e+11+4.8115000386e+10j,
            5.0896130353e+11+4.6733828878e+10j,
            4.8628746583e+11+4.5293413201e+10j,
            4.6361830601e+11+4.3793762975e+10j,
            4.4095394026e+11+4.2234888026e+10j,
            4.1829448472e+11+4.0616798379e+10j,
            3.9564005551e+11+3.8939504264e+10j,
            3.7299076875e+11+3.7203016111e+10j,
            3.5034674052e+11+3.5407344556e+10j,
            3.2770808692e+11+3.3552500435e+10j,
            3.0507492397e+11+3.1638494786e+10j,
            2.8244736773e+11+2.9665338851e+10j,
            2.5982553421e+11+2.7633044074e+10j,
            2.3720953939e+11+2.5541622099e+10j,
            2.1459949925e+11+2.3391084776e+10j,
            1.9199552975e+11+2.1181444154e+10j,
            1.6939774681e+11+1.8912712486e+10j,
            1.4680626634e+11+1.6584902227e+10j,
            1.2422120423e+11+1.4198026033e+10j,
            1.0164267634e+11+1.1752096764e+10j,
            7.9070798521e+10+9.2471274799e+09j,
            5.6505686581e+10+6.6831314440e+09j,
            3.3947456317e+10+4.0601221211e+09j,
            1.1396223503e+10+1.3781131781e+09j,
            0.0000000000e+00+0.0000000000e+00j,
            0.0000000000e+00+0.0000000000e+00j,
            0.0000000000e+00+0.0000000000e+00j,
            0.0000000000e+00+0.0000000000e+00j,
            0.0000000000e+00+0.0000000000e+00j,
            0.0000000000e+00+0.0000000000e+00j,
            0.0000000000e+00+0.0000000000e+00j])

        imp_coarse_ref = np.array([1.0504062083e+12+0.0000000000e+00j,
            2.0781004955e+12+2.7183115978e+09j,
            2.0553850965e+12+5.3772054987e+09j,
            2.0326663360e+12+7.9766773057e+09j,
            2.0099443306e+12+1.0516722825e+10j,
            1.9872191969e+12+1.2997338066e+10j,
            1.9644910516e+12+1.5418519242e+10j,
            1.9417600113e+12+1.7780262770e+10j,
            1.9190261924e+12+2.0082565269e+10j,
            1.8962897118e+12+2.2325423561e+10j,
            1.8735506859e+12+2.4508834674e+10j,
            1.8508092314e+12+2.6632795838e+10j,
            1.8280654649e+12+2.8697304485e+10j,
            1.8053195030e+12+3.0702358252e+10j,
            1.7825714624e+12+3.2647954978e+10j,
            1.7598214597e+12+3.4534092708e+10j,
            1.7370696115e+12+3.6360769688e+10j,
            1.7143160345e+12+3.8127984368e+10j,
            1.6915608452e+12+3.9835735402e+10j,
            1.6688041604e+12+4.1484021645e+10j,
            1.6460460966e+12+4.3072842159e+10j,
            1.6232867705e+12+4.4602196207e+10j,
            1.6005262987e+12+4.6072083256e+10j,
            1.5777647978e+12+4.7482502976e+10j,
            1.5550023845e+12+4.8833455241e+10j,
            1.5322391754e+12+5.0124940128e+10j,
            1.5094752871e+12+5.1356957918e+10j,
            1.4867108362e+12+5.2529509093e+10j,
            1.4639459395e+12+5.3642594342e+10j,
            1.4411807134e+12+5.4696214555e+10j,
            1.4184152746e+12+5.5690370826e+10j,
            1.3956497397e+12+5.6625064451e+10j,
            1.3728842254e+12+5.7500296932e+10j,
            1.3501188481e+12+5.8316069972e+10j,
            1.3273537246e+12+5.9072385477e+10j,
            1.3045889714e+12+5.9769245560e+10j,
            1.2818247051e+12+6.0406652532e+10j,
            1.2590610424e+12+6.0984608912e+10j,
            1.2362980996e+12+6.1503117419e+10j,
            1.2135359936e+12+6.1962180977e+10j,
            1.1907748407e+12+6.2361802713e+10j,
            1.1680147576e+12+6.2701985956e+10j,
            1.1452558608e+12+6.2982734240e+10j,
            1.1224982669e+12+6.3204051301e+10j,
            1.0997420924e+12+6.3365941080e+10j,
            1.0769874538e+12+6.3468407718e+10j,
            1.0542344676e+12+6.3511455561e+10j,
            1.0314832504e+12+6.3495089159e+10j,
            1.0087339187e+12+6.3419313265e+10j,
            9.8598658892e+11+6.3284132832e+10j,
            9.6324137757e+11+6.3089553021e+10j,
            9.4049840113e+11+6.2835579191e+10j,
            9.1775777605e+11+6.2522216909e+10j,
            8.9501961879e+11+6.2149471941e+10j,
            8.7228404579e+11+6.1717350259e+10j,
            8.4955117347e+11+6.1225858036e+10j,
            8.2682111826e+11+6.0675001648e+10j,
            8.0409399656e+11+6.0064787676e+10j,
            7.8136992476e+11+5.9395222903e+10j,
            7.5864901923e+11+5.8666314312e+10j,
            7.3593139635e+11+5.7878069094e+10j,
            7.1321717247e+11+5.7030494640e+10j,
            6.9050646392e+11+5.6123598543e+10j,
            6.6779938704e+11+5.5157388601e+10j,
            6.4509605813e+11+5.4131872814e+10j,
            6.2239659348e+11+5.3047059384e+10j,
            5.9970110939e+11+5.1902956716e+10j,
            5.7700972210e+11+5.0699573420e+10j,
            5.5432254788e+11+4.9436918305e+10j,
            5.3163970295e+11+4.8115000386e+10j,
            5.0896130353e+11+4.6733828878e+10j,
            4.8628746583e+11+4.5293413201e+10j,
            4.6361830601e+11+4.3793762975e+10j,
            4.4095394026e+11+4.2234888026e+10j,
            4.1829448472e+11+4.0616798379e+10j,
            3.9564005551e+11+3.8939504264e+10j,
            3.7299076875e+11+3.7203016111e+10j,
            3.5034674052e+11+3.5407344556e+10j,
            3.2770808692e+11+3.3552500435e+10j,
            3.0507492397e+11+3.1638494786e+10j,
            2.8244736773e+11+2.9665338851e+10j,
            2.5982553421e+11+2.7633044074e+10j,
            2.3720953939e+11+2.5541622099e+10j,
            2.1459949925e+11+2.3391084776e+10j,
            1.9199552975e+11+2.1181444154e+10j,
            1.6939774681e+11+1.8912712486e+10j,
            1.4680626634e+11+1.6584902227e+10j,
            1.2422120423e+11+1.4198026033e+10j,
            1.0164267634e+11+1.1752096764e+10j,
            7.9070798521e+10+9.2471274799e+09j,
            5.6505686581e+10+6.6831314440e+09j,
            3.3947456317e+10+4.0601221211e+09j,
            1.1396223503e+10+1.3781131781e+09j,
            0.0000000000e+00+0.0000000000e+00j,
            0.0000000000e+00+0.0000000000e+00j,
            0.0000000000e+00+0.0000000000e+00j,
            0.0000000000e+00+0.0000000000e+00j,
            0.0000000000e+00+0.0000000000e+00j,
            0.0000000000e+00+0.0000000000e+00j,
            0.0000000000e+00+0.0000000000e+00j])

        np.testing.assert_allclose(imp_fine_meas[:-7], imp_fine_ref[:-7], rtol=1e-8,
            atol=0, err_msg="In TestTravelingWaveCavity test_beam_fine_coarse,"
                            "mismatch in beam impulse response on fine grid")
        np.testing.assert_allclose(imp_coarse_meas[:-7], imp_coarse_ref[:-7], rtol=1e-8,
            atol=0, err_msg="In TestTravelingWaveCavity test_beam_fine_coarse,"
                            "mismatch in beam impulse response on coarse grid")

        Vind_fine_meas = (OTFB.V_fine_ind_beam[::1000])[:100]
        Vind_coarse_meas = OTFB.V_coarse_ind_beam[:100]

        Vind_fine_ref = np.array([-3.0517578125e-11+2.0599365234e-10j,
            1.3119591242e+05+2.5059688582e+02j,
            1.2976951147e+05+4.1841300342e+02j,
            1.2834289840e+05+5.8249788025e+02j,
            1.2691608056e+05+7.4285125172e+02j,
            1.2548906524e+05+8.9947286604e+02j,
            2.5525777220e+05+1.3029593700e+03j,
            2.5240398296e+05+1.6199328833e+03j,
            2.4954980612e+05+1.9294427198e+03j,
            2.4669525631e+05+2.2314884143e+03j,
            2.4384034818e+05+2.5260695271e+03j,
            3.7218100878e+05+3.0637825296e+03j,
            3.6789902698e+05+3.5112493792e+03j,
            3.6361651869e+05+3.9475192402e+03j,
            3.5933350586e+05+4.3725915103e+03j,
            3.5505001047e+05+4.7864656258e+03j,
            4.8196196690e+05+5.4397379469e+03j,
            4.7625117134e+05+5.9990303323e+03j,
            4.7053974700e+05+6.5433918605e+03j,
            4.6482772318e+05+7.0728218563e+03j,
            4.5911512918e+05+7.5873196955e+03j,
            5.8459790670e+05+8.3374816909e+03j,
            5.7745785924e+05+8.9899296666e+03j,
            5.7031711736e+05+9.6237126791e+03j,
            5.6317571767e+05+1.0238830044e+04j,
            5.5603369679e+05+1.0835281141e+04j,
            6.8008700374e+05+1.1663662299e+04j,
            6.7151744935e+05+1.2390595370e+04j,
            6.6294717151e+05+1.3095129455e+04j,
            6.5437621413e+05+1.3777263922e+04j,
            6.4580462118e+05+1.4436998220e+04j,
            7.6842834898e+05+1.5324928756e+04j,
            7.5842921570e+05+1.6107677477e+04j,
            7.4842936654e+05+1.6864293588e+04j,
            7.3842885274e+05+1.7594776578e+04j,
            7.2842772557e+05+1.8299126023e+04j,
            8.4962194870e+05+1.9227938478e+04j,
            8.3819334760e+05+2.0047836046e+04j,
            8.2676407480e+05+2.0837868101e+04j,
            8.1533418885e+05+2.1598034313e+04j,
            8.0390374836e+05+2.2328334456e+04j,
            9.2366872430e+05+2.3279365291e+04j,
            9.1081094946e+05+2.4117749141e+04j,
            8.9795258369e+05+2.4922535614e+04j,
            8.8509369287e+05+2.5693724627e+04j,
            8.7223434290e+05+2.6431316212e+04j,
            9.9057051210e+05+2.7385907402e+04j,
            9.7628404055e+05+2.8224120805e+04j,
            9.6199709542e+05+2.9025006325e+04j,
            9.4770974992e+05+2.9788564189e+04j,
            9.3342207725e+05+3.0514794752e+04j,
            1.0503300630e+06+3.1454295383e+04j,
            1.0346155547e+06+3.2273689037e+04j,
            1.0189007267e+06+3.3052025980e+04j,
            1.0031856595e+06+3.3789306812e+04j,
            9.8747043377e+05+3.4485532275e+04j,
            1.1029510423e+06+3.5391300137e+04j,
            1.0858093399e+06+3.6173233764e+04j,
            1.0686675083e+06+3.6910383847e+04j,
            1.0515256353e+06+3.7602751424e+04j,
            1.0343838088e+06+3.8250337687e+04j,
            1.1484380290e+06+3.9103740866e+04j,
            1.1298701579e+06+3.9829584804e+04j,
            1.1113023846e+06+4.0506920680e+04j,
            1.0927348043e+06+4.1135750033e+04j,
            1.0741675120e+06+4.1716074567e+04j,
            1.1867965154e+06+4.2498493041e+04j,
            1.1668036837e+06+4.3149629836e+04j,
            1.1468112133e+06+4.3748536684e+04j,
            1.1268192067e+06+4.4295215687e+04j,
            1.1068277662e+06+4.4789669129e+04j,
            1.2180329068e+06+4.5482496357e+04j,
            1.1966165050e+06+4.6040322356e+04j,
            1.1752007645e+06+4.6542199472e+04j,
            1.1537857950e+06+4.6988130437e+04j,
            1.1323717063e+06+4.7378118176e+04j,
            1.2421545205e+06+4.7962762689e+04j,
            1.2193161215e+06+4.8408689628e+04j,
            1.1964787202e+06+4.8794952019e+04j,
            1.1736424337e+06+4.9121553285e+04j,
            1.1508073790e+06+4.9388497055e+04j,
            1.2591695854e+06+4.9846384046e+04j,
            1.2349109442e+06+5.0161840642e+04j,
            1.2106536737e+06+5.0413920610e+04j,
            1.1863978981e+06+5.0602628129e+04j,
            1.1621437417e+06+5.0727967597e+04j,
            1.2690872411e+06+5.1040540511e+04j,
            1.2434102950e+06+5.1206974048e+04j,
            1.2177351287e+06+5.1306322781e+04j,
            1.1920618739e+06+5.1338591709e+04j,
            1.1663906621e+06+5.1303786060e+04j,
            1.2719175372e+06+5.1452508178e+04j,
            1.2448244051e+06+5.1451386094e+04j,
            1.2177768552e+06+5.1349872757e+04j,
            1.1920618739e+06+5.1338591709e+04j,
            1.1663906621e+06+5.1303786060e+04j,
            1.2719175372e+06+5.1452508178e+04j,
            1.2448244051e+06+5.1451386094e+04j,
            1.2177768552e+06+5.1349872757e+04j,
            1.1920618739e+06+5.1338591709e+04j])

        Vind_coarse_ref = np.array([65950.402941899 +4.6204946754e+01j,
            130474.7043137922+2.6208172207e+02j,
            129048.3869868867+4.2802282083e+02j,
            127621.8612230706+5.9023291883e+02j,
            126195.13434267  +7.4871175790e+02j,
            190718.6166082232+9.4966403938e+02j,
            253815.81082876  +1.3165564125e+03j,
            250962.2071966935+1.6297811521e+03j,
            248108.2232953706+1.9355427270e+03j,
            245253.8737668068+2.2338406848e+03j,
            308349.5761954412+2.5708795450e+03j,
            370018.8407123919+3.0701257882e+03j,
            365737.1648323309+3.5119715333e+03j,
            361454.9734610094+3.9426211057e+03j,
            357172.2885629754+4.3620739225e+03j,
            418839.5350453047+4.8165343856e+03j,
            479080.2303618401+5.4294688706e+03j,
            473369.8793500392+5.9812694041e+03j,
            467658.9142384843+6.5181402322e+03j,
            461947.3643145755+7.0400807053e+03j,
            522185.6618082431+7.5932951716e+03j,
            580997.3314961634+8.3012499653e+03j,
            573857.8855385883+8.9443370849e+03j,
            566717.7634868335+9.5687607609e+03j,
            559577.0019509497+1.0174520341e+04j,
            618386.0404834162+1.0807820183e+04j,
            675768.4111833326+1.1592126644e+04j,
            667199.6335332314+1.2307831758e+04j,
            658630.1544065415+1.3001139804e+04j,
            650060.0177352416+1.3672050190e+04j,
            707439.6703935305+1.4366767349e+04j,
            763392.6518020012+1.5208757723e+04j,
            753394.4887644283+1.5978413447e+04j,
            743395.635475215 +1.6721938911e+04j,
            733396.1431870223+1.7439333649e+04j,
            789346.4660944196+1.8176802229e+04j,
            843870.1509380381+1.9057811247e+04j,
            832442.7318413376+1.9862752998e+04j,
            821014.6703180319+2.0637832050e+04j,
            809586.0249396971+2.1383048124e+04j,
            864107.2572194036+2.2144605992e+04j,
            917201.9212158447+2.3045972461e+04j,
            904345.5583700862+2.3867540055e+04j,
            891488.6375129672+2.4655513581e+04j,
            878631.2245326925+2.5409893014e+04j,
            931723.7882584379+2.6176883388e+04j,
            983389.8900644641+2.7079951792e+04j,
            969105.0787068398+2.7899491038e+04j,
            954819.8303308268+2.8681706237e+04j,
            940534.2181384464+2.9426597680e+04j,
            992198.7182720678+3.0180370731e+04j,
            1042436.8994184991+3.1066492820e+04j,
            1026724.3176456908+3.1865357114e+04j,
            1011011.4564100985+3.2623169090e+04j,
            995298.3962242305+3.3339929421e+04j,
            1045535.6205402148+3.4061843863e+04j,
            1094346.7053538668+3.4912380250e+04j,
            1077207.2140413758+3.5671932169e+04j,
            1060067.6373666434+3.6386705528e+04j,
            1042928.0631488134+3.7056701442e+04j,
            1091738.982145815 +3.7728126125e+04j,
            1139123.977658412 +3.8524447879e+04j,
            1120558.6203668672+3.9226060775e+04j,
            1101993.4083382622+3.9879171214e+04j,
            1083428.4366940013+4.0483780818e+04j,
            1130814.2034933397+4.1086096322e+04j,
            1176774.2993374085+4.1809586562e+04j,
            1156784.3022058595+4.2434646152e+04j,
            1136794.717464163 +4.3007482052e+04j,
            1116805.6475310936+4.3528096457e+04j,
            1162767.5977622345+4.4042696684e+04j,
            1207304.1660539836+4.4674752165e+04j,
            1185890.9376801767+4.5204658125e+04j,
            1164478.425299364 +4.5678622145e+04j,
            1143066.738622277 +4.6096647054e+04j,
            1187606.39029531  +4.6504940818e+04j,
            1230720.9855045064+4.7026973527e+04j,
            1207886.116812161 +4.7443141080e+04j,
            1185052.3041640767+4.7799651744e+04j,
            1162219.6645570078+4.8096509044e+04j,
            1205338.7179221238+4.8379921659e+04j,
            1247033.0767289733+4.8773360403e+04j,
            1222778.3408220657+4.9057221909e+04j,
            1198525.0374281076+4.9277715194e+04j,
            1174273.2908234247+4.9434844546e+04j,
            1215973.628217406 +4.9574819417e+04j,
            1256249.6693564458+4.9821111408e+04j,
            1230577.0213605044+4.9954117952e+04j,
            1204906.2187303253+5.0020048879e+04j,
            1179237.3930148352+5.0018909302e+04j,
            1219521.0786945666+4.9996909510e+04j,
            1258380.9027855818+5.0077521955e+04j,
            1231292.4796760113+5.0041144933e+04j,
            1204906.2187303249+5.0020048879e+04j,
            1179237.3930148345+5.0018909302e+04j,
            1219521.0786945664+4.9996909510e+04j,
            1258380.9027855818+5.0077521955e+04j,
            1231292.4796760113+5.0041144933e+04j,
            1204906.2187303246+5.0020048879e+04j,
            1179237.3930148345+5.0018909302e+04j])

        np.testing.assert_allclose(Vind_fine_meas, Vind_fine_ref, rtol=1e-8,
            atol=1e-9, err_msg="In TestTravelingWaveCavity test_beam_fine_coarse,"
                            "mismatch in beam-induced voltage on fine grid")
        np.testing.assert_allclose(Vind_coarse_meas, Vind_coarse_ref, rtol=1e-8,
            atol=0, err_msg="In TestTravelingWaveCavity test_beam_fine_coarse,"
                            "mismatch in beam-induced voltage on coarse grid")


if __name__ == '__main__':

    unittest.main()
