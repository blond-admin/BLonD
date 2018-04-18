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

from llrf.impulse_response import rectangle, triangle, SPS4Section200MHzTWC
from input_parameters.ring import Ring
from input_parameters.rf_parameters import RFStation
from beam.beam import Beam, Proton
from beam.distributions import bigaussian
from beam.profile import Profile, CutOptions
from impedances.impedance import InducedVoltageTime, TotalInducedVoltage
from llrf.cavity_feedback import SPSOneTurnFeedback
from impedances.impedance_sources import TravelingWaveCavity


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
        from llrf.impulse_response import TravellingWaveCavity
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
        OTFB_4 = SPSOneTurnFeedback(rf, beam, profile, 4, n_cavities=1)
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


if __name__ == '__main__':

    unittest.main()
