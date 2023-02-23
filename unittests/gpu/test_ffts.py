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
from blond.utils import bmath as bm


class TestFFTS:

    # Run before every test
    def setup_method(self):
        # Try to import cupy, skip if not found
        self.cp = pytest.importorskip('cupy')

    # Run after every test
    def teardown_method(self):
        pass

    @pytest.mark.parametrize('size,n',
                             [(10, None), (100, None), (93, None),
                             (17, None), (10000, None), (100, 50),
                             (100, 51), (100, 151), (100, 100), 
                             (100, 1000), (100, 1)])
    def test_rfft(self, size, n):
        bm.use_cpu()
        s = np.random.randn(size)
        res = bm.rfft(s, n=n)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        res_gpu = bm.rfft(s_gpu, n=n)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    @pytest.mark.parametrize('size,n',
                             [(10, None), (100, None), (93, None),
                              (17, None), (10000, None), (100, 100),
                              (100, 10), (100, 200), (100, 1),
                              (100, 101)])
    def test_irfft(self, size, n):
        bm.use_cpu()
        s = np.random.randn(size)
        o = bm.rfft(s)
        res = bm.irfft(o, n=n)

        bm.use_gpu()
        s_gpu = self.cp.array(s)
        o_gpu = bm.rfft(s_gpu)
        res_gpu = bm.irfft(o_gpu, n=n)

        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    @pytest.mark.parametrize('size', [10, 18, 100, 1000])
    def test_front_back(self, size):
        bm.use_gpu()

        s = self.cp.random.randn(size)
        res = bm.irfft(bm.rfft(s))

        self.cp.testing.assert_array_almost_equal(res, s, decimal=8)

    @pytest.mark.parametrize('n_points,delta_t',
                             [(10, 1.0), (10000, 1.0), (1000, 1.5),
                              (1000, -0.1), (1000, -100)])
    def test_rfftfreq(self, n_points, delta_t):
        bm.use_cpu()
        res = bm.rfftfreq(n_points, delta_t)

        bm.use_gpu()
        res_gpu = bm.rfftfreq(n_points, delta_t)
        self.cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    def test_rfftfreq_zero(self):
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
        # Try to import cupy, skip if not found
        self.cp = pytest.importorskip('cupy')

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
