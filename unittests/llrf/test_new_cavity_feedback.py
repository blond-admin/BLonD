'''
25.08.2021

Author: Birk Emil Karlsen-Bæck
'''

# Imports ---------------------------------------------------------------------
import unittest
import numpy as np
import matplotlib.pyplot as plt

from blond.llrf.new_SPS_OTFB import SPSOneTurnFeedback_new, CavityFeedbackCommissioning_new
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring


class TestNewCavityFeedback(unittest.TestCase):

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
        self.Commissioning_new = CavityFeedbackCommissioning_new()


        self.OTFB_new = SPSOneTurnFeedback_new(self.rfstation, self.beam, self.profile, 3, a_comb=63 / 64,
                                          Commissioning=self.Commissioning_new)

        self.OTFB_new.update_variables()

        self.turn_array = np.linspace(0, 2 * self.rfstation.t_rev[0], 2 * self.OTFB_new.n_coarse)

    def test_set_point(self):
        self.OTFB_new.set_point()
        t_sig = np.zeros(2 * self.OTFB_new.n_coarse, dtype=complex)
        t_sig[-self.OTFB_new.n_coarse:] = (4/9) * 10e6 * np.exp(1j * (np.pi/2 - self.rfstation.phi_rf[0,0]))

        np.testing.assert_allclose(self.OTFB_new.V_SET, t_sig)


    def test_error_and_gain(self):
        self.OTFB_new.error_and_gain()

        np.testing.assert_allclose(self.OTFB_new.DV_GEN, self.OTFB_new.V_SET * self.OTFB_new.G_llrf)


    def test_comb(self):
        sig = np.zeros(self.OTFB_new.n_coarse)
        self.OTFB_new.DV_COMB_OUT = np.sin(2 * np.pi * self.turn_array / self.rfstation.t_rev[0])
        self.OTFB_new.DV_GEN = -np.sin(2 * np.pi * self.turn_array / self.rfstation.t_rev[0])
        self.OTFB_new.a_comb = 0.5

        self.OTFB_new.comb()

        np.testing.assert_allclose(self.OTFB_new.DV_COMB_OUT[-self.OTFB_new.n_coarse:], sig)


    def test_one_turn_delay(self):
        self.OTFB_new.DV_COMB_OUT = np.zeros(2 * self.OTFB_new.n_coarse, dtype=complex)
        self.OTFB_new.DV_COMB_OUT[self.OTFB_new.n_coarse] = 1

        self.OTFB_new.one_turn_delay()

        self.assertEqual(np.argmax(self.OTFB_new.DV_DELAYED), 2 * self.OTFB_new.n_coarse - self.OTFB_new.n_mov_av)


    def test_mod_to_fr(self):
        self.OTFB_new.DV_DELAYED = np.zeros(2 * self.OTFB_new.n_coarse, dtype=complex)
        self.OTFB_new.DV_DELAYED[-self.OTFB_new.n_coarse:] = 1 + 1j * 0

        self.mod_phi = np.copy(self.OTFB_new.dphi_mod)
        self.OTFB_new.mod_to_fr()

        ref_DV_MOD_FR = np.load("ref_DV_MOD_FR.npy")

        np.testing.assert_allclose(self.OTFB_new.DV_MOD_FR[-self.OTFB_new.n_coarse:], ref_DV_MOD_FR)

        self.OTFB_new.DV_DELAYED = np.zeros(2 * self.OTFB_new.n_coarse, dtype=complex)
        self.OTFB_new.DV_DELAYED[-self.OTFB_new.n_coarse:] = 1 + 1j * 0

        self.OTFB_new.dphi_mod = 0
        self.OTFB_new.mod_to_fr()

        time_array = self.OTFB_new.rf_centers - 0.5*self.OTFB_new.T_s
        ref_sig = np.cos((self.OTFB_new.omega_c - self.OTFB_new.omega_r) * time_array[:self.OTFB_new.n_coarse]) + \
                  1j * np.sin((self.OTFB_new.omega_c - self.OTFB_new.omega_r) * time_array[:self.OTFB_new.n_coarse])

        np.testing.assert_allclose(self.OTFB_new.DV_MOD_FR[-self.OTFB_new.n_coarse:], ref_sig)

        self.OTFB_new.dphi_mod = self.mod_phi


    def test_mov_avg(self):
        sig = np.zeros(self.OTFB_new.n_coarse-1)
        sig[:self.OTFB_new.n_mov_av] = 1
        self.OTFB_new.DV_MOD_FR = np.zeros(2 * self.OTFB_new.n_coarse)
        self.OTFB_new.DV_MOD_FR[-self.OTFB_new.n_coarse + 1:] = sig

        self.OTFB_new.mov_avg()

        sig = np.zeros(self.OTFB_new.n_coarse)
        sig[:self.OTFB_new.n_mov_av] = (1/self.OTFB_new.n_mov_av) * np.array(range(self.OTFB_new.n_mov_av))
        sig[self.OTFB_new.n_mov_av: 2 * self.OTFB_new.n_mov_av] = (1/self.OTFB_new.n_mov_av) * (self.OTFB_new.n_mov_av - np.array(range(self.OTFB_new.n_mov_av)))

        np.testing.assert_allclose(np.abs(self.OTFB_new.DV_MOV_AVG[-self.OTFB_new.n_coarse:]), sig)


    def test_mod_to_frf(self):
        self.OTFB_new.DV_MOV_AVG = np.zeros(2 * self.OTFB_new.n_coarse, dtype=complex)
        self.OTFB_new.DV_MOV_AVG[-self.OTFB_new.n_coarse:] = 1 + 1j * 0

        self.mod_phi = np.copy(self.OTFB_new.dphi_mod)
        self.OTFB_new.mod_to_frf()

        ref_DV_MOD_FRF = np.load("ref_DV_MOD_FRF.npy")

        np.testing.assert_allclose(self.OTFB_new.DV_MOD_FRF[-self.OTFB_new.n_coarse:], ref_DV_MOD_FRF)

        self.OTFB_new.DV_MOV_AVG = np.zeros(2 * self.OTFB_new.n_coarse, dtype=complex)
        self.OTFB_new.DV_MOV_AVG[-self.OTFB_new.n_coarse:] = 1 + 1j * 0

        self.OTFB_new.dphi_mod = 0
        self.OTFB_new.mod_to_frf()

        time_array = self.OTFB_new.rf_centers - 0.5*self.OTFB_new.T_s
        ref_sig = np.cos(-(self.OTFB_new.omega_c - self.OTFB_new.omega_r) * time_array[:self.OTFB_new.n_coarse]) + \
                  1j * np.sin(-(self.OTFB_new.omega_c - self.OTFB_new.omega_r) * time_array[:self.OTFB_new.n_coarse])

        np.testing.assert_allclose(self.OTFB_new.DV_MOD_FRF[-self.OTFB_new.n_coarse:], ref_sig)

        self.OTFB_new.dphi_mod = self.mod_phi

    def test_sum_and_gain(self):
        self.OTFB_new.V_SET[-self.OTFB_new.n_coarse:] = np.ones(self.OTFB_new.n_coarse, dtype=complex)
        self.OTFB_new.DV_MOD_FRF[-self.OTFB_new.n_coarse:] = np.ones(self.OTFB_new.n_coarse, dtype=complex)

        self.OTFB_new.sum_and_gain()

        sig = 2 * np.ones(self.OTFB_new.n_coarse) * self.OTFB_new.G_tx * self.OTFB_new.T_s / self.OTFB_new.TWC.R_gen

        np.testing.assert_allclose(self.OTFB_new.I_GEN[-self.OTFB_new.n_coarse:], sig)


    def test_gen_response(self):
        # Tests generator response at resonant frequency.
        self.OTFB_new.I_GEN = np.zeros(2 * self.OTFB_new.n_coarse, dtype=complex)
        self.OTFB_new.I_GEN[self.OTFB_new.n_coarse] = 1

        self.OTFB_new.TWC.impulse_response_gen(self.OTFB_new.TWC.omega_r, self.OTFB_new.rf_centers)
        self.OTFB_new.gen_response()

        sig = np.zeros(self.OTFB_new.n_coarse)
        sig[1:1 + self.OTFB_new.n_mov_av] = 2 * self.OTFB_new.TWC.R_gen / self.OTFB_new.TWC.tau
        sig[0] = self.OTFB_new.TWC.R_gen / self.OTFB_new.TWC.tau
        sig[self.OTFB_new.n_mov_av + 1] = self.OTFB_new.TWC.R_gen / self.OTFB_new.TWC.tau

        np.testing.assert_allclose(np.abs(self.OTFB_new.V_IND_COARSE_GEN[-self.OTFB_new.n_coarse:]), sig,
                                   atol=1e-5)

        # Tests generator response at carrier frequency.
        self.OTFB_new.TWC.impulse_response_gen(self.OTFB_new.omega_c, self.OTFB_new.rf_centers)

        self.OTFB_new.I_GEN = np.zeros(2 * self.OTFB_new.n_coarse, dtype=complex)
        self.OTFB_new.I_GEN[self.OTFB_new.n_coarse] = 1

        self.OTFB_new.gen_response()

        ref_V_IND_COARSE_GEN = np.load("ref_V_IND_COARSE_GEN.npy")
        np.testing.assert_allclose(self.OTFB_new.V_IND_COARSE_GEN[-self.OTFB_new.n_coarse:], ref_V_IND_COARSE_GEN)



if __name__ == '__main__':
    unittest.main()
