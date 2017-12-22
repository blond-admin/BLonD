# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:57:49 2017

@author: schwarz
"""

import unittest
import numpy as np

from input_parameters.ring import Ring
from input_parameters.rf_parameters import RFStation
from beam.beam import Beam, Proton
from beam.distributions import bigaussian
from beam.profile import Profile, CutOptions
from llrf.cavity_feedback import SPSCavityFeedback, CavityFeedbackCommissioning

class TestFeedback(unittest.TestCase):

    def test_FB_commissioning_v1(self):
        """Test OTFB commissioning with one slice per rf-bucket
        
        """
        C = 2*np.pi*1100.009        # Ring circumference [m]
        gamma_t = 18.0              # Gamma at transition
        alpha = 1/gamma_t**2        # Momentum compaction factor
        p_s = 25.92e9               # Synchronous momentum at injection [eV]
        h = [4620]                  # 200 MHz system harmonic
        V = [4.5e6]                 # 200 MHz RF voltage
        # With this setting, amplitude in the two four-section cavity 
        # must converge to 4.5 MV * 4/18 * 2 = 2.0 MV
        phi = [0.]                    # 200 MHz RF phase
        
        N_m = 1e5                   # Number of macro-particles for tracking
        N_b = 1.e11                 # Bunch intensity [ppb]
        N_t = 1                     # Number of turns to track

        # Set up machine parameters
        ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=N_t)
        rf = RFStation(ring, 1, h, V, phi)
        
        # Define beam and fill it
        beam = Beam(ring, N_m, N_b)
        bigaussian(ring, rf, beam, 3.2e-9/4, seed = 1234, reinsertion = True)
        n_shift = 5 # how many rf-buckets to shift beam
        beam.dt += n_shift * rf.t_rf[0]
        profile = Profile(beam, CutOptions = CutOptions(cut_left=0.e-9, 
            cut_right=rf.t_rev[0], n_slices=4620))
        profile.track()
        
        Commissioning = CavityFeedbackCommissioning(debug=False,
                        open_loop=False, open_FB=False, open_drive=False)
        
        OTFB = SPSCavityFeedback(rf, beam, profile, G_llrf=5, G_tx=0.5,
                                 a_comb=15/16, turns=25, 
                                 Commissioning=Commissioning)
        
        Vind_mean = np.around(
                np.mean(np.absolute(OTFB.OTFB_4.V_llrf_tot))/1e6, 12)
        Vind_std = np.around(
                np.std(np.absolute(OTFB.OTFB_4.V_llrf_tot))/1e6, 12)
        
        # Expected values from previous simulation
        Vind_mean_exp = np.around(1.9988140339387235, 12)
        Vind_std_exp = np.around(0.00019225190507084845, 12)
        
        self.assertEqual(Vind_mean, Vind_mean_exp,
            msg='In TestFeedback test_FB_commissioning_v1: mean value differs')
        self.assertEqual(Vind_std, Vind_std_exp,
            msg='In TestFeedback test_FB_commissioning_v1: '
                + 'standard deviation differs')
        
    def test_FB_commissioning_v2(self):
        
        digit_round = 3
        
        C = 2*np.pi*1100.009        # Ring circumference [m]
        gamma_t = 18.0              # Gamma at transition
        alpha = 1/gamma_t**2        # Momentum compaction factor
        p_s = 25.92e9               # Synchronous momentum at injection [eV]
        h = [4620]                  # 200 MHz system harmonic
        V = [4.5e6]                 # 200 MHz RF voltage
        # With this setting, amplitude in the two four-section, five-section
        # cavities must converge, respectively, to
        # 2.0 MV = 4.5 MV * 4/18 * 2
        # 2.5 MV = 4.5 MV * 5/18 * 2
        
        phi = [0.]                    # 200 MHz RF phase
        
        N_m = 1e5                   # Number of macro-particles for tracking
        N_b = 1.e11                 # Bunch intensity [ppb]
        N_t = 1                     # Number of turns to track
        
        ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=N_t)
        rf = RFStation(ring, 1, h, V, phi)
        
        beam = Beam(ring, N_m, N_b)
        bigaussian(ring, rf, beam, 3.2e-9/4, seed = 1234, reinsertion = True)
        n_shift = 5 # how many rf-buckets to shift beam
        beam.dt += n_shift * rf.t_rf[0]
        profile = Profile(beam, CutOptions =
                          CutOptions(cut_left=(n_shift-1.5)*rf.t_rf[0],
                                     cut_right=(n_shift+1.5)*rf.t_rf[0],
                                     n_slices = 140))
#        profile = Profile(beam, CutOptions = CutOptions(cut_left=0, 
#            cut_right=rf.t_rf[0], n_slices=256))
        profile.track()
        
        Commissioning = CavityFeedbackCommissioning(debug=False,
                        open_loop=False, open_FB=False, open_drive=False)
        
        OTFB = SPSCavityFeedback(rf, beam, profile, G_llrf=5, G_tx=0.5,
                                 a_comb=15/16, turns=50, 
                                 Commissioning=Commissioning)
        
        
        Vind4_mean = np.around(
                np.mean(np.absolute(OTFB.OTFB_4.V_llrf_tot))/1e6, digit_round)
        Vind4_std = np.around(
                np.std(np.absolute(OTFB.OTFB_4.V_llrf_tot))/1e6, digit_round)
        Vind4_mean_exp = np.around(1.99886351363, digit_round)
        Vind4_std_exp = np.around(2.148426e-6, digit_round)
        
        Vind5_mean = np.around(
                np.mean(np.absolute(OTFB.OTFB_5.V_llrf_tot))/1e6, digit_round)
        Vind5_std = np.around(
                np.std(np.absolute(OTFB.OTFB_5.V_llrf_tot))/1e6, digit_round)
        Vind5_mean_exp = np.around(2.49906605189, digit_round)
        Vind5_std_exp = np.around(2.221665e-6, digit_round)
        
        self.assertEqual(Vind4_mean, Vind4_mean_exp,
            msg='In TestFeedback test_FB_commissioning_v2: '
            +'mean value of four-section cavity differs')
        self.assertEqual(Vind4_std, Vind4_std_exp,
            msg='In TestFeedback test_FB_commissioning_v2: standard deviation'
            +' of four-section cavity differs')
        
        self.assertEqual(Vind5_mean, Vind5_mean_exp,
            msg='In TestFeedback test_FB_commissioning_v2: '
            +'mean value of five-section cavity differs')
        self.assertEqual(Vind5_std, Vind5_std_exp,
            msg='In TestFeedback test_FB_commissioning_v2: standard deviation'
            +' of five-section cavity differs')


if __name__ == '__main__':

    unittest.main()
