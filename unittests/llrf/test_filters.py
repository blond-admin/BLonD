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
from llrf.filters import moving_average, modulator, polar_to_cartesian, cartesian_to_polar


class TestMovingAverage(unittest.TestCase):
    
    # Run before every test
    def setUp(self, N = 3, center = False):
        self.x = np.array([0,3,6,3,0,3,6,3,0], dtype=float)
        self.y = moving_average(self.x, N, center)
        
    # Run after every test
    def tearDown(self):
         
        del self.x
        del self.y

    def test_1(self):
        
        self.setUp(N = 3, center = False)
        self.assertEqual(len(self.x), len(self.y) + 3 - 1, 
            msg="In TestMovingAverage, test_1: wrong array length")
        self.assertSequenceEqual(self.y.tolist(), 
            np.array([3, 4, 3, 2, 3, 4, 3], dtype=float).tolist(), 
            msg="In TestMovingAverage, test_1: arrays differ")

    def test_2(self):
        
        self.setUp(N = 4, center = False)
        self.assertEqual(len(self.x), len(self.y) + 4 - 1, 
            msg="In TestMovingAverage, test_2: wrong array length")
        self.assertSequenceEqual(self.y.tolist(), np.array([3, 3, 3, 3, 3, 3], 
            dtype=float).tolist(), 
            msg="In TestMovingAverage, test_2: arrays differ")

    def test_3(self):
        
        self.setUp(N = 3, center = True)
        self.assertEqual(len(self.x), len(self.y), 
            msg="In TestMovingAverage, test_3: wrong array length")
        self.assertSequenceEqual(self.y.tolist(), 
            np.array([1, 3, 4, 3, 2, 3, 4, 3, 1], dtype=float).tolist(), 
            msg="In TestMovingAverage, test_3: arrays differ")
    
    def test_4(self):
        
        self.setUp(N = 4, center = True)
        self.assertEqual(len(self.x), len(self.y), 
            msg="In TestMovingAverage, test_4: wrong array length")
        self.assertSequenceEqual(self.y.tolist(), 
            np.array([1.8, 2.4, 2.4, 3, 3.6, 3, 2.4, 2.4, 1.8], 
            dtype=float).tolist(), 
            msg="In TestMovingAverage, test_4: arrays differ")



class TestModulator(unittest.TestCase):
    
    def test(self):
        
        f_rf = 200.1e6 # initial frequency in Hz
        f_0 = 200.222e6 # final frequency in Hz
        T_s = 5e-10 # sampling time
        n = 1000 # number of points
        
        # Forwards and backwards transformation of a sine wave
        signal = np.cos(2*np.pi*np.arange(n)*f_rf*T_s) \
            + 1j*np.sin(2*np.pi*np.arange(n)*f_rf*T_s)
        signal_1 = modulator(signal, f_rf, f_0, T_s)
        signal_2 = modulator(signal_1, f_0, f_rf, T_s)
        
        # Drop some digits to avoid rounding errors
        signal = np.around(signal, 12)
        signal_2 = np.around(signal_2, 12)
        self.assertSequenceEqual(signal.tolist(), signal_2.tolist(),
            msg="In TestModulator, initial and final signals do not match")



class TestIQ(unittest.TestCase):
    
    def test_1(self):
        
        f_rf = 200.1e6 # initial frequency in Hz
        T_s = 5e-10 # sampling time
        n = 1000 # number of points
        
        # From IQ to polar
        phases = np.pi*(np.fmod(2*np.arange(n)*f_rf*T_s,2) - 1) # (-pi,pi)
        signal = np.cos(phases) + 1j*np.sin(phases)            
        amplitude, phase = cartesian_to_polar(signal)
        
        # Drop some digits to avoid rounding errors
        amplitude = np.around(amplitude, 12)
        phase = np.around(phase, 12)
        phases = np.around(phases, 12)
        self.assertSequenceEqual(amplitude.tolist(), np.ones(n).tolist(),
            msg="In TestIQ test_1, amplitude is not correct")
        self.assertSequenceEqual(phase.tolist(), phases.tolist(),
            msg="In TestIQ test_1, phase is not correct")
        #polar_to_cartesian(amplitude, phase)

        
if __name__ == '__main__':

    unittest.main()



