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
from llrf.signal_processing import moving_average, modulator, real_to_cartesian
from llrf.signal_processing import polar_to_cartesian, cartesian_to_polar


class TestRealToCartesian(unittest.TestCase):
    
    def test(self):
        
        signal = np.array([0, 3, 4, 5, 4, 3, 0], dtype=float)
        IQ_vector = real_to_cartesian(signal)
        I_comp = np.around(IQ_vector.real, 12)
        Q_comp = np.around(IQ_vector.imag, 12)

        self.assertSequenceEqual(signal.tolist(), I_comp.tolist(),
            msg="In TestRealToCartesian: real component differs")
        self.assertSequenceEqual(Q_comp.tolist(), 
            np.array([5, 4, 3, 0, 3, 4, 5], dtype=float).tolist(), 
            msg="In TestRealToCartesian: imaginary component differs")


        
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
    
    # Run before every test
    def setUp(self, f_rf = 200.1e6, T_s = 5e-10, n = 1000):
        
        self.f_rf = f_rf # initial frequency in Hz
        self.T_s = T_s # sampling time
        self.n = n # number of points
        
    # Run after every test
    def tearDown(self):
         
        del self.f_rf
        del self.T_s
        del self.n

    def test_1(self):
               
        # Define signal in range (-pi, pi)
        phases = np.pi*(np.fmod(2*np.arange(self.n)*self.f_rf*self.T_s,2) - 1)
        signal = np.cos(phases) + 1j*np.sin(phases)            
        # From IQ to polar
        amplitude, phase = cartesian_to_polar(signal)
        
        # Drop some digits to avoid rounding errors
        amplitude = np.around(amplitude, 12)
        phase = np.around(phase, 12)
        phases = np.around(phases, 12)
        self.assertSequenceEqual(amplitude.tolist(), np.ones(self.n).tolist(),
            msg="In TestIQ test_1, amplitude is not correct")
        self.assertSequenceEqual(phase.tolist(), phases.tolist(),
            msg="In TestIQ test_1, phase is not correct")

    def test_2(self):
               
        # Define signal in range (-pi, pi)
        phase = np.pi*(np.fmod(2*np.arange(self.n)*self.f_rf*self.T_s,2) - 1)
        amplitude = np.ones(self.n)             
        # From polar to IQ
        signal = polar_to_cartesian(amplitude, phase)
        
        # Drop some digits to avoid rounding errors
        signal_real = np.around(signal.real, 12)
        signal_imag = np.around(signal.imag, 12)
        theor_real = np.around(np.cos(phase), 12) # what it should be
        theor_imag = np.around(np.sin(phase), 12) # what it should be
        self.assertSequenceEqual(signal_real.tolist(), theor_real.tolist(),
            msg="In TestIQ test_2, real part is not correct")
        self.assertSequenceEqual(signal_imag.tolist(), theor_imag.tolist(),
            msg="In TestIQ test_2, imaginary part is not correct")

    def test_3(self):
               
        # Define signal in range (-pi, pi)
        phase = np.pi*(np.fmod(2*np.arange(self.n)*self.f_rf*self.T_s,2) - 1)
        amplitude = np.ones(self.n)             
        # Forwards and backwards transform
        signal = polar_to_cartesian(amplitude, phase)
        amplitude_new, phase_new = cartesian_to_polar(signal)
        
        # Drop some digits to avoid rounding errors
        phase = np.around(phase, 11)
        amplitude = np.around(amplitude, 11)
        amplitude_new = np.around(amplitude_new, 11) 
        phase_new = np.around(phase_new, 11)
        self.assertSequenceEqual(phase.tolist(), phase_new.tolist(),
            msg="In TestIQ test_3, phase is not correct")
        self.assertSequenceEqual(amplitude.tolist(), amplitude_new.tolist(),
            msg="In TestIQ test_3, amplitude is not correct")

    def test_4(self):
               
        # Define signal in range (-pi, pi)
        phase = np.pi*(np.fmod(2*np.arange(self.n)*self.f_rf*self.T_s,2) - 1)
        signal = np.cos(phase) + 1j*np.sin(phase)            
        # Forwards and backwards transform
        amplitude, phase = cartesian_to_polar(signal)
        signal_new = polar_to_cartesian(amplitude, phase)
        
        # Drop some digits to avoid rounding errors
        signal_real = np.around(signal.real, 11)
        signal_imag = np.around(signal.imag, 11)
        signal_real_2 = np.around(np.real(signal_new), 11)
        signal_imag_2 = np.around(np.imag(signal_new), 11)
        self.assertSequenceEqual(signal_real.tolist(), signal_real_2.tolist(),
            msg="In TestIQ test_4, real part is not correct")
        self.assertSequenceEqual(signal_imag.tolist(), signal_imag_2.tolist(),
            msg="In TestIQ test_4, imaginary part is not correct")


        
if __name__ == '__main__':

    unittest.main()



