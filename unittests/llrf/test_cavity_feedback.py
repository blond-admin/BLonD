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
        h = [4620]                  # 200 MHz system harmonic
        phi = [0.]                  # 200 MHz RF phase
        
        # With this setting, amplitude in the two four-section, five-section
        # cavities must converge, respectively, to
        # 2.0 MV = 4.5 MV * 4/18 * 2
        # 2.5 MV = 4.5 MV * 5/18 * 2
        V = [4.5e6]                 # 200 MHz RF voltage
        
        N_t = 1                     # Number of turns to track
        
        self.ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=N_t)
        self.rf = RFStation(self.ring, 1, h, V, phi)
        
        N_m = 1e6                   # Number of macro-particles for tracking
        N_b = 72*1.0e11                 # Bunch intensity [ppb]
        
        # Gaussian beam profile
        self.beam = Beam(self.ring, N_m, N_b)
        sigma = 1.0e-9
        bigaussian(self.ring, self.rf, self.beam, sigma, seed = 1234,
                   reinsertion = False)


        n_shift = 1550 # how many rf-buckets to shift beam
        self.beam.dt += n_shift * self.rf.t_rf[0]
        
        self.profile = Profile(self.beam, CutOptions =\
                          CutOptions(cut_left=(n_shift-1.5)*self.rf.t_rf[0],
                                     cut_right=(n_shift+2.5)*self.rf.t_rf[0],
                                     n_slices = 4*64))
        self.profile.track()
        
        # Cavities
        l_cav = 43*0.374
        v_g = 0.0946
        tau = l_cav/(v_g*c)*(1 + v_g)
        f_cav = 200.222e6
        n_cav = 2 #factor 2 because of two four/five-sections cavities
        short_cavity = TravelingWaveCavity(l_cav**2 * n_cav * 27.1e3 / 8, f_cav,
                                           2*np.pi*tau)
        shortInducedVoltage = InducedVoltageTime(self.beam, self.profile,
                                                 [short_cavity])
        l_cav = 54*0.374
        tau = l_cav/(v_g*c)*(1 + v_g)
        long_cavity = TravelingWaveCavity(l_cav**2 * n_cav * 27.1e3 / 8, f_cav,
                                          2*np.pi*tau)
        longInducedVoltage = InducedVoltageTime(self.beam, self.profile,
                                                [long_cavity])
        self.induced_voltage = TotalInducedVoltage(self.beam, self.profile,
                                   [shortInducedVoltage, longInducedVoltage])
        self.induced_voltage.induced_voltage_sum()
        
        self.cavity_tracker = RingAndRFTracker(self.rf, self.beam,
                                   Profile=self.profile, interpolation=True,
                                   TotalInducedVoltage=self.induced_voltage)
        
        self.OTFB = SPSCavityFeedback(self.rf, self.beam, self.profile, G_llrf=5,
                                      G_tx=0.5, a_comb=15/16, turns=50,
                                      Commissioning \
                                      = CavityFeedbackCommissioning())

        self.OTFB_tracker = RingAndRFTracker(self.rf, self.beam,
                                             Profile=self.profile,
                                             TotalInducedVoltage=None,
                                             CavityFeedback=self.OTFB,
                                             interpolation=True)
        
    def test_FB_pre_tracking(self):
        
        digit_round = 3        
                
        Vind4_mean = np.around(
                np.mean(np.absolute(self.OTFB.OTFB_4.V_coarse_tot))/1e6, digit_round)
        Vind4_std = np.around(
                np.std(np.absolute(self.OTFB.OTFB_4.V_coarse_tot))/1e6, digit_round)
        Vind4_mean_exp = np.around(1.99886351363, digit_round)
        Vind4_std_exp = np.around(2.148426e-6, digit_round)
        
        Vind5_mean = np.around(
                np.mean(np.absolute(self.OTFB.OTFB_5.V_coarse_tot))/1e6, digit_round)
        Vind5_std = np.around(
                np.std(np.absolute(self.OTFB.OTFB_5.V_coarse_tot))/1e6, digit_round)
        Vind5_mean_exp = np.around(2.49906605189, digit_round)
        Vind5_std_exp = np.around(2.221665e-6, digit_round)
        
        self.assertEqual(Vind4_mean, Vind4_mean_exp,
            msg='In TestCavityFeedback test_FB_commissioning_v2: '
            +'mean value of four-section cavity differs')
        self.assertEqual(Vind4_std, Vind4_std_exp,
            msg='In TestCavityFeedback test_FB_commissioning_v2: standard deviation'
            +' of four-section cavity differs')
        
        self.assertEqual(Vind5_mean, Vind5_mean_exp,
            msg='In TestCavityFeedback test_FB_commissioning_v2: '
            +'mean value of five-section cavity differs')
        self.assertEqual(Vind5_std, Vind5_std_exp,
            msg='In TestCavityFeedback test_FB_commissioning_v2: standard deviation'
            +' of five-section cavity differs')
        
    def test_rf_voltage(self):
        
        # compute voltage
        self.cavity_tracker.rf_voltage_calculation()        
        
        # compute voltage after OTFB pre-tracking
        self.OTFB_tracker.rf_voltage_calculation()
        
        # Since there is a systematic offset between the voltages,
        # compare the maxium of the ratio
        max_ratio = np.max(self.cavity_tracker.rf_voltage \
                           / self.OTFB_tracker.rf_voltage)
        
        self.assertEqual(max_ratio, 1.0008217052569774, 
                         msg='In TestCavityFeedback test_rf_voltage: '
                         + 'RF-voltages differ')
        
    def test_beam_loading(self):
        
        # Compute voltage with beam loading
        self.cavity_tracker.rf_voltage_calculation()
        cavity_tracker_total_voltage = self.cavity_tracker.rf_voltage \
                    + self.cavity_tracker.totalInducedVoltage.induced_voltage

        self.OTFB.track()
        self.OTFB_tracker.rf_voltage_calculation()
        OTFB_tracker_total_voltage = self.OTFB_tracker.rf_voltage
        
        max_ratio = np.max(cavity_tracker_total_voltage \
                           / OTFB_tracker_total_voltage)
                
        self.assertEqual(max_ratio, 1.0051759770680779, 
                         msg='In TestCavityFeedback test_beam_loading: '
                         + 'total voltages differ')

if __name__ == '__main__':

    unittest.main()
