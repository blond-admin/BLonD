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
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.rfft(s), 8)

    def test_rfft_2(self):
        s = np.random.randn(100)
        try:
            res = bm.rfft(s)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.rfft(s), 8)

    def test_rfft_3(self):
        s = np.random.randn(93)
        try:
            res = bm.rfft(s)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.rfft(s), 8)

    def test_rfft_4(self):
        s = np.random.randn(17)
        try:
            res = bm.rfft(s)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.rfft(s), 8)

    def test_rfft_5(self):
        s = np.random.randn(10000)
        try:
            res = bm.rfft(s)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.rfft(s), 8)

    def test_rfft_6(self):
        s = np.random.randn(100)
        try:
            res = bm.rfft(s, n=50)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.rfft(s, n=50), 8)

    def test_rfft_7(self):
        s = np.random.randn(100)
        try:
            res = bm.rfft(s, n=51)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.rfft(s, n=51), 8)

    def test_rfft_8(self):
        s = np.random.randn(100)
        try:
            res = bm.rfft(s, n=151)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.rfft(s, n=151), 8)

    def test_rfft_9(self):
        s = np.random.randn(100)
        try:
            res = bm.rfft(s, n=100)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.rfft(s, n=100), 8)

    def test_rfft_10(self):
        s = np.random.randn(100)
        try:
            res = bm.rfft(s, n=1000)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.rfft(s, n=1000), 8)

    def test_rfft_11(self):
        s = np.random.randn(100)
        try:
            res = bm.rfft(s, n=1)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.rfft(s, n=1), 8)

    def test_irfft_1(self):
        s = np.random.randn(10)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.irfft(o), 8)

    def test_irfft_2(self):
        s = np.random.randn(100)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.irfft(o), 8)

    def test_irfft_3(self):
        s = np.random.randn(93)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.irfft(o), 8)

    def test_irfft_4(self):
        s = np.random.randn(17)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.irfft(o), 8)

    def test_irfft_5(self):
        s = np.random.randn(10000)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.irfft(o), 8)

    def test_irfft_6(self):
        s = np.random.randn(100)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o, n=100)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.irfft(o, n=100), 8)

    def test_irfft_7(self):
        s = np.random.randn(100)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o, n=10)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.irfft(o, n=10), 8)

    def test_irfft_8(self):
        s = np.random.randn(100)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o, n=200)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.irfft(o, n=200), 8)

    def test_irfft_9(self):
        s = np.random.randn(100)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o, n=1)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.irfft(o, n=1), 8)

    def test_irfft_10(self):
        s = np.random.randn(100)
        o = fft.rfft(s)
        try:
            res = bm.irfft(o, n=101)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.irfft(o, n=101), 8)

    def test_front_back_1(self):
        s = np.random.randn(1000)
        try:
            res = bm.irfft(bm.rfft(s))
        except AttributeError:
            self.skipTest('Not compiled with FFTW')

        np.testing.assert_almost_equal(res, fft.irfft(fft.rfft(s)), 8)

    def test_front_back_2(self):
        s = np.random.randn(101)
        try:
            res = bm.irfft(bm.rfft(s))
        except AttributeError:
            self.skipTest('Not compiled with FFTW')
        np.testing.assert_almost_equal(res, fft.irfft(fft.rfft(s)), 8)

    def test_front_back_3(self):
        s = np.random.randn(100)
        try:
            res = bm.irfft(bm.rfft(s, n=50), n=100)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')
        np.testing.assert_almost_equal(
            res, fft.irfft(fft.rfft(s, n=50), n=100), 8)

    def test_front_back_4(self):
        s = np.random.randn(100)
        try:
            res = bm.irfft(bm.rfft(s, n=150), n=200)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')
        np.testing.assert_almost_equal(
            res, fft.irfft(fft.rfft(s, n=150), n=200), 8)

    def test_rfftfreq_1(self):
        delta_t = 1.0
        n_points = 10
        try:
            res = bm.rfftfreq(n_points, delta_t)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')
        np.testing.assert_almost_equal(res, fft.rfftfreq(n_points, delta_t), 8)

    def test_rfftfreq_2(self):
        n_points = 10000
        try:
            res = bm.rfftfreq(n_points)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')
        np.testing.assert_almost_equal(res, fft.rfftfreq(n_points), 8)

    def test_rfftfreq_3(self):
        n_points = 1000
        delta_t = 1.5
        try:
            res = bm.rfftfreq(n_points, delta_t)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')
        np.testing.assert_almost_equal(res, fft.rfftfreq(n_points, delta_t), 8)

    def test_rfftfreq_4(self):
        n_points = 1000
        delta_t = -0.1
        try:
            res = bm.rfftfreq(n_points, delta_t)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')
        np.testing.assert_almost_equal(res, fft.rfftfreq(n_points, delta_t), 8)

    def test_rfftfreq_5(self):
        n_points = 1000
        delta_t = -100
        try:
            res = bm.rfftfreq(n_points, delta_t)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')
        np.testing.assert_almost_equal(res, fft.rfftfreq(n_points, delta_t), 8)

    def test_rfftfreq_6(self):
        n_points = 1000
        delta_t = 0
        try:
            _ = bm.rfftfreq(n_points, delta_t)
        except AttributeError:
            self.skipTest('Not compiled with FFTW')
        except ZeroDivisionError:
            self.assertTrue(True, 'This testcase should raise a ZeroDivisionError')


if __name__ == '__main__':

    unittest.main()
