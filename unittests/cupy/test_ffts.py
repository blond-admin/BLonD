# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for the FFTs used in blond with CuPy and NumPy

:Authors: **Konstantinos Iliakis**
"""

import unittest
import pytest
import numpy as np
# import inspect
# from numpy import fft
from blond.utils import bmath as bm


class TestFFTS(unittest.TestCase):

    # Run before every test
    def setUp(self):
        try:
            import cupy as cp
            self.cp = cp
        except ModuleNotFoundError:
            raise self.skipTest('CuPy not found')

    # Run after every test
    def tearDown(self):
        pass

    def test_rfft_1(self):
        bm.use_cpu()
        s = np.random.randn(10)
        res = bm.rfft(s)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        res_gpu = bm.rfft(s_gpu)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_rfft_2(self):
        bm.use_cpu()
        s = np.random.randn(100)
        res = bm.rfft(s)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        res_gpu = bm.rfft(s_gpu)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_rfft_3(self):
        bm.use_cpu()
        s = np.random.randn(93)
        res = bm.rfft(s)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        res_gpu = bm.rfft(s_gpu)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_rfft_4(self):
        bm.use_cpu()
        s = np.random.randn(17)
        res = bm.rfft(s)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        res_gpu = bm.rfft(s_gpu)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)
        
    def test_rfft_5(self):
        bm.use_cpu()
        s = np.random.randn(10000)
        res = bm.rfft(s)
        
        bm.use_gpu()
        s_gpu = self.cp.array(s)
        res_gpu = bm.rfft(s_gpu)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)
        
    def test_rfft_6(self):
        bm.use_cpu()
        s = np.random.randn(100)
        res = bm.rfft(s, n=50)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        res_gpu = bm.rfft(s_gpu, n=50)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_rfft_7(self):
        bm.use_cpu()
        s = np.random.randn(100)
        res = bm.rfft(s, n=51)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        res_gpu = bm.rfft(s_gpu, n=51)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_rfft_8(self):
        bm.use_cpu()
        s = np.random.randn(100)
        res = bm.rfft(s, n=151)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        res_gpu = bm.rfft(s_gpu, n=151)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_rfft_9(self):
        bm.use_cpu()
        s = np.random.randn(100)
        res = bm.rfft(s, n=100)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        res_gpu = bm.rfft(s_gpu, n=100)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_rfft_10(self):
        bm.use_cpu()
        s = np.random.randn(100)
        res = bm.rfft(s, n=1000)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        res_gpu = bm.rfft(s_gpu, n=1000)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_rfft_11(self):
        bm.use_cpu()
        s = np.random.randn(100)
        res = bm.rfft(s, n=1)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        res_gpu = bm.rfft(s_gpu, n=1)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_irfft_1(self):
        bm.use_cpu()
        s = np.random.randn(10)
        o = bm.rfft(s)
        res = bm.irfft(o)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        o_gpu = bm.rfft(s_gpu)
        res_gpu = bm.irfft(o_gpu)

        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_irfft_2(self):
        bm.use_cpu()
        s = np.random.randn(100)
        o = bm.rfft(s)
        res = bm.irfft(o)
        
        bm.use_gpu()
        s_gpu = self.cp.array(s)
        o_gpu = bm.rfft(s_gpu)
        res_gpu = bm.irfft(o_gpu)

        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_irfft_3(self):
        bm.use_cpu()
        s = np.random.randn(93)
        o = bm.rfft(s)
        res = bm.irfft(o)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        o_gpu = bm.rfft(s_gpu)
        res_gpu = bm.irfft(o_gpu)

        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_irfft_4(self):
        bm.use_cpu()
        s = np.random.randn(17)
        o = bm.rfft(s)
        res = bm.irfft(o)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        o_gpu = bm.rfft(s_gpu)
        res_gpu = bm.irfft(o_gpu)

        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_irfft_5(self):
        bm.use_cpu()
        s = np.random.randn(10000)
        o = bm.rfft(s)
        res = bm.irfft(o)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        o_gpu = bm.rfft(s_gpu)
        res_gpu = bm.irfft(o_gpu)

        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)


    def test_irfft_6(self):
        bm.use_cpu()
        s = np.random.randn(100)
        o = bm.rfft(s)
        res = bm.irfft(o, n=100)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        o_gpu = bm.rfft(s_gpu)
        res_gpu = bm.irfft(o_gpu, n=100)

        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)


    def test_irfft_7(self):
        bm.use_cpu()
        s = np.random.randn(100)
        o = bm.rfft(s)
        res = bm.irfft(o, n=10)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        o_gpu = bm.rfft(s_gpu)
        res_gpu = bm.irfft(o_gpu, n=10)

        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)


    def test_irfft_8(self):
        bm.use_cpu()
        s = np.random.randn(100)
        o = bm.rfft(s)
        res = bm.irfft(o, n=200)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        o_gpu = bm.rfft(s_gpu)
        res_gpu = bm.irfft(o_gpu, n=200)

        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_irfft_9(self):
        bm.use_cpu()
        s = np.random.randn(100)
        o = bm.rfft(s)
        res = bm.irfft(o, n=1)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        o_gpu = bm.rfft(s_gpu)
        res_gpu = bm.irfft(o_gpu, n=1)

        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_irfft_10(self):
        bm.use_cpu()
        s = np.random.randn(100)
        o = bm.rfft(s)
        res = bm.irfft(o, n=101)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        o_gpu = bm.rfft(s_gpu)
        res_gpu = bm.irfft(o_gpu, n=101)

        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_front_back_1(self):
        bm.use_gpu()

        s = self.cp.random.randn(1000)
        res = bm.irfft(bm.rfft(s))

        self.cp.testing.assert_array_almost_equal(res, s, decimal=8)

    def test_front_back_2(self):
        bm.use_gpu()
        s = self.cp.random.randn(100)
        res = bm.irfft(bm.rfft(s))
        
        self.cp.testing.assert_array_almost_equal(res, s, decimal=8)

    def test_rfftfreq_1(self):
        bm.use_cpu()
        delta_t = 1.0
        n_points = 10
        res = bm.rfftfreq(n_points, delta_t)
        
        bm.use_gpu()
        res_gpu = bm.rfftfreq(n_points, delta_t)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_rfftfreq_2(self):
        bm.use_cpu()
        n_points = 10000
        res = bm.rfftfreq(n_points)

        bm.use_gpu()
        res_gpu = bm.rfftfreq(n_points)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)
        

    def test_rfftfreq_3(self):
        bm.use_cpu()
        n_points = 1000
        delta_t = 1.5
        res = bm.rfftfreq(n_points, delta_t)

        bm.use_gpu()
        res_gpu = bm.rfftfreq(n_points, delta_t)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)


    def test_rfftfreq_4(self):
        bm.use_cpu()
        n_points = 1000
        delta_t = -0.1
        res = bm.rfftfreq(n_points, delta_t)
        
        bm.use_gpu()
        res_gpu = bm.rfftfreq(n_points, delta_t)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)
        
    def test_rfftfreq_5(self):
        bm.use_cpu()
        n_points = 1000
        delta_t = -100
        res = bm.rfftfreq(n_points, delta_t)
        
        bm.use_gpu()
        res_gpu = bm.rfftfreq(n_points, delta_t)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_rfftfreq_6(self):
        bm.use_cpu()
        n_points = 1000
        delta_t = 0

        bm.use_gpu()
        try:
            res_gpu = bm.rfftfreq(n_points, delta_t)
        except ZeroDivisionError as e:
            self.assertTrue(
                True, 'This testcase should raise a ZeroDivisionError')


class TestCupyConvolve:

    # Run before every test
    def setup_method(self):
        try:
            import cupy as cp
            self.cp = cp
        except ModuleNotFoundError:
            raise self.skipTest('CuPy not found')

    # Run after every test
    def teardown_method(self):
        bm.use_cpu()

    @pytest.mark.parametrize('mode', ['full', 'same', 'valid'])
    def test_convolve_1(self, mode):
        s = np.random.randn(100)
        k = np.random.randn(100)
        res = np.convolve(s, k, mode=mode)

        bm.use_gpu()
        s = bm.array(s)
        k = bm.array(k)
        res_gpu = bm.convolve(s, k, mode=mode)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    @pytest.mark.parametrize('mode', ['full', 'same', 'valid'])
    def test_convolve_2(self, mode):
        s = np.random.randn(200)
        k = np.random.randn(200)
        res = np.convolve(s, k, mode=mode)

        bm.use_gpu()
        s = bm.array(s)
        k = bm.array(k)
        res_gpu = bm.convolve(s, k, mode=mode)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    @pytest.mark.parametrize('mode', ['full', 'same', 'valid'])
    def test_convolve_3(self, mode):
        s = np.random.randn(200)
        k = np.random.randn(300)
        res = np.convolve(s, k, mode=mode)

        bm.use_gpu()
        s = bm.array(s)
        k = bm.array(k)
        res_gpu = bm.convolve(s, k, mode=mode)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    @pytest.mark.parametrize('mode', ['full', 'same', 'valid'])
    def test_convolve_4(self, mode):
        s = np.random.randn(11)
        k = np.random.randn(13)
        res = np.convolve(s, k, mode=mode)

        bm.use_gpu()
        s = bm.array(s)
        k = bm.array(k)
        res_gpu = bm.convolve(s, k, mode=mode)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)


if __name__ == '__main__':

    unittest.main()
