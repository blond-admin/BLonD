# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for utils.bmath

:Authors: **Konstantinos Iliakis**
"""

import unittest
import numpy as np
# import inspect
from numpy import fft
from blond.utils import bmath as bm


class TestFFTS(unittest.TestCase):

    # Run before every test
    def setUp(self):
        np.random.seed(0)
        bm.use_fftw()
        pass

    # Run after every test
    def tearDown(self):
        pass

    def test_rfft_1(self):
        s = np.random.randn(10)
        try:
            res = bm.rfft(s)    
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.rfft(s), 8)


    def test_rfft_2(self):
        s = np.random.randn(100)
        try:
            res = bm.rfft(s)
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.rfft(s), 8)

    def test_rfft_3(self):
        s = np.random.randn(93)
        try:
            res = bm.rfft(s)
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.rfft(s), 8)


    def test_rfft_4(self):
        s = np.random.randn(17)
        try:
            res = bm.rfft(s)
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.rfft(s), 8)

    def test_rfft_5(self):
        s = np.random.randn(10000)
        try:
            res = bm.rfft(s)
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.rfft(s), 8)


    def test_rfft_6(self):
        s = np.random.randn(100)
        try:
            res = bm.rfft(s, n=50)
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.rfft(s, n=50), 8)

    def test_rfft_7(self):
        s = np.random.randn(100)
        try:
            res = bm.rfft(s, n=51)
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.rfft(s, n=51), 8)

    def test_rfft_8(self):
        s = np.random.randn(100)
        try:
            res = bm.rfft(s, n=151)
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.rfft(s, n=151), 8)

    def test_rfft_9(self):
        s = np.random.randn(100)
        try:
            res = bm.rfft(s, n=100)
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.rfft(s, n=100), 8)


    def test_rfft_10(self):
        s = np.random.randn(100)
        try:
            res = bm.rfft(s, n=1000)
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.rfft(s, n=1000), 8)

    def test_rfft_11(self):
        s = np.random.randn(100)
        try:
            res = bm.rfft(s, n=1)
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.rfft(s, n=1), 8)


    def test_irfft_1(self):
        s = np.random.randn(10)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o)    
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.irfft(o), 8)

    def test_irfft_2(self):
        s = np.random.randn(100)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o)    
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.irfft(o), 8)

    def test_irfft_3(self):
        s = np.random.randn(93)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o)    
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.irfft(o), 8)


    def test_irfft_4(self):
        s = np.random.randn(17)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o)    
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.irfft(o), 8)

    def test_irfft_5(self):
        s = np.random.randn(10000)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o)    
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.irfft(o), 8)

    def test_irfft_6(self):
        s = np.random.randn(100)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o, n=100)  
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.irfft(o, n=100), 8)

    def test_irfft_7(self):
        s = np.random.randn(100)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o, n=10)  
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.irfft(o, n=10), 8)


    def test_irfft_8(self):
        s = np.random.randn(100)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o, n=200)  
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.irfft(o, n=200), 8)


    def test_irfft_9(self):
        s = np.random.randn(100)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o, n=1)  
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.irfft(o, n=1), 8)


    def test_irfft_10(self):
        s = np.random.randn(100)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o, n=101)
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.irfft(o, n=101), 8)


    def test_front_back_1(self):
        s = np.random.randn(1000)
        try:
            res = bm.irfft(bm.rfft(s))    
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        
        np.testing.assert_almost_equal(res, fft.irfft(fft.rfft(s)), 8)

    def test_front_back_2(self):
        s = np.random.randn(101)
        try:
            res = bm.irfft(bm.rfft(s))    
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        np.testing.assert_almost_equal(res, fft.irfft(fft.rfft(s)), 8)

    def test_front_back_3(self):
        s = np.random.randn(100)
        try:
            res = bm.irfft(bm.rfft(s, n=50), n=100)    
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        np.testing.assert_almost_equal(res, fft.irfft(fft.rfft(s, n=50), n=100), 8)

    def test_front_back_4(self):
        s = np.random.randn(100)
        try:
            res = bm.irfft(bm.rfft(s, n=150), n=200)    
        except AttributeError as e:
            self.skipTest('Not compiled with FFTW')
        np.testing.assert_almost_equal(res, fft.irfft(fft.rfft(s, n=150), n=200), 8)


if __name__ == '__main__':

    unittest.main()
