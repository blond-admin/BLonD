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

from utils import bmath as bm


class TestSin(unittest.TestCase):

    # Run before every test
    def setUp(self):
        pass
    # Run after every test

    def tearDown(self):
        pass

    def test_sin_scalar_1(self):
        a = np.random.rand()
        np.testing.assert_almost_equal(bm.sin(a), np.sin(a), decimal=8)

    def test_sin_scalar_2(self):
        np.testing.assert_almost_equal(
            bm.sin(-np.pi), np.sin(-np.pi), decimal=8)

    def test_sin_vector_1(self):
        a = np.random.randn(100)
        np.testing.assert_almost_equal(bm.sin(a), np.sin(a), decimal=8)


class TestCos(unittest.TestCase):

    # Run before every test
    def setUp(self):
        pass
    # Run after every test

    def tearDown(self):
        pass

    def test_cos_scalar_1(self):
        a = np.random.rand()
        np.testing.assert_almost_equal(bm.cos(a), np.cos(a), decimal=8)

    def test_cos_scalar_2(self):
        np.testing.assert_almost_equal(
            bm.cos(-2*np.pi), np.cos(-2*np.pi), decimal=8)

    def test_cos_vector_1(self):
        a = np.random.randn(100)
        np.testing.assert_almost_equal(bm.cos(a), np.cos(a), decimal=8)


class TestExp(unittest.TestCase):

    # Run before every test
    def setUp(self):
        pass
    # Run after every test

    def tearDown(self):
        pass

    def test_exp_scalar_1(self):
        a = np.random.rand()
        np.testing.assert_almost_equal(bm.exp(a), np.exp(a), decimal=8)

    def test_exp_vector_1(self):
        a = np.random.randn(100)
        np.testing.assert_almost_equal(bm.exp(a), np.exp(a), decimal=8)


class TestMean(unittest.TestCase):

    # Run before every test
    def setUp(self):
        pass
    # Run after every test

    def tearDown(self):
        pass

    def test_mean_1(self):
        a = np.random.randn(100)
        np.testing.assert_almost_equal(bm.mean(a), np.mean(a), decimal=8)

    def test_mean_2(self):
        a = np.random.randn(1)
        np.testing.assert_almost_equal(bm.mean(a), np.mean(a), decimal=8)


class TestStd(unittest.TestCase):

    # Run before every test
    def setUp(self):
        pass
    # Run after every test

    def tearDown(self):
        pass

    def test_std_1(self):
        a = np.random.randn(100)
        np.testing.assert_almost_equal(bm.std(a), np.std(a), decimal=8)

    def test_std_2(self):
        a = np.random.randn(1)
        np.testing.assert_almost_equal(bm.std(a), np.std(a), decimal=8)


class TestSum(unittest.TestCase):

    # Run before every test
    def setUp(self):
        pass
    # Run after every test

    def tearDown(self):
        pass

    def test_sum_1(self):
        a = np.random.randn(100)
        np.testing.assert_almost_equal(bm.sum(a), np.sum(a), decimal=8)

    def test_sum_2(self):
        a = np.random.randn(1)
        np.testing.assert_almost_equal(bm.sum(a), np.sum(a), decimal=8)


class TestLinspace(unittest.TestCase):

    # Run before every test
    def setUp(self):
        pass
    # Run after every test

    def tearDown(self):
        pass

    def test_linspace_1(self):
        start = 0.
        stop = 10.
        num = 33
        np.testing.assert_almost_equal(bm.linspace(start, stop, num),
                                       np.linspace(start, stop, num), decimal=8)

    def test_linspace_2(self):
        start = 0
        stop = 10
        num = 33
        np.testing.assert_almost_equal(bm.linspace(start, stop, num),
                                       np.linspace(start, stop, num), decimal=8)

    def test_linspace_3(self):
        start = 12.234
        stop = -10.456
        np.testing.assert_almost_equal(bm.linspace(start, stop),
                                       np.linspace(start, stop), decimal=8)

    def test_linspace_4(self):
        start = np.random.rand()
        stop = np.random.rand()
        num = int(np.random.rand())
        np.testing.assert_almost_equal(bm.linspace(start, stop, num),
                                       np.linspace(start, stop, num), decimal=8)


class TestArange(unittest.TestCase):

    # Run before every test
    def setUp(self):
        pass
    # Run after every test

    def tearDown(self):
        pass

    def test_arange_1(self):
        start = 0.
        stop = 1000.
        step = 33
        np.testing.assert_almost_equal(bm.arange(start, stop, step),
                                       np.arange(start, stop, step), decimal=8)

    def test_arange_2(self):
        start = 0
        stop = 1000
        step = 33
        np.testing.assert_almost_equal(bm.arange(start, stop, step),
                                       np.arange(start, stop, step), decimal=8)

    def test_arange_3(self):
        start = 12.234
        stop = -10.456
        step = -0.067
        np.testing.assert_almost_equal(bm.arange(start, stop, step),
                                       np.arange(start, stop, step), decimal=8)

    def test_arange_4(self):
        start = np.random.rand()
        stop = np.random.rand()
        start, stop = min(start, stop), max(start, stop)
        step = np.random.random() * (stop - start) / 60.
        np.testing.assert_almost_equal(bm.arange(start, stop, step),
                                       np.arange(start, stop, step), decimal=8)


class TestArgMin(unittest.TestCase):

    # Run before every test
    def setUp(self):
        pass
    # Run after every test

    def tearDown(self):
        pass

    def test_min_idx_1(self):
        a = np.random.randn(100)
        np.testing.assert_equal(bm.argmin(a), np.argmin(a))

    def test_min_idx_2(self):
        a = np.random.randn(1000)
        np.testing.assert_equal(bm.argmin(a), np.argmin(a))


class TestArgMax(unittest.TestCase):

    # Run before every test
    def setUp(self):
        pass
    # Run after every test

    def tearDown(self):
        pass

    def test_max_idx_1(self):
        a = np.random.randn(100)
        np.testing.assert_equal(bm.argmax(a), np.argmax(a))

    def test_max_idx_2(self):
        a = np.random.randn(1000)
        np.testing.assert_equal(bm.argmax(a), np.argmax(a))


class TestConvolve(unittest.TestCase):

    # Run before every test
    def setUp(self):
        pass
    # Run after every test

    def tearDown(self):
        pass

    def test_convolve_1(self):
        s = np.random.randn(100)
        k = np.random.randn(100)
        np.testing.assert_almost_equal(bm.convolve(s, k, mode='full'),
                                       np.convolve(s, k, mode='full'),
                                       decimal=8)

    def test_convolve_2(self):
        s = np.random.randn(200)
        k = np.random.randn(200)
        with self.assertRaises(RuntimeError):
            bm.convolve(s, k, mode='same', )
        with self.assertRaises(RuntimeError):
            bm.convolve(s, k, mode='valid')


class TestInterp(unittest.TestCase):

    # Run before every test
    def setUp(self):
        pass
    # Run after every test

    def tearDown(self):
        pass

    def test_interp_1(self):
        x = np.random.randn(100)
        xp = np.random.randn(100)
        xp.sort()
        yp = np.random.randn(100)
        np.testing.assert_almost_equal(bm.interp(x, xp, yp),
                                       np.interp(x, xp, yp), decimal=8)

    def test_interp_2(self):
        x = np.random.randn(200)
        x.sort()
        xp = np.random.randn(50)
        xp.sort()
        yp = np.random.randn(50)
        np.testing.assert_almost_equal(bm.interp(x, xp, yp),
                                       np.interp(x, xp, yp), decimal=8)

    def test_interp_3(self):
        x = np.random.randn(1)
        xp = np.random.randn(50)
        xp.sort()
        yp = np.random.randn(50)
        np.testing.assert_almost_equal(bm.interp(x, xp, yp),
                                       np.interp(x, xp, yp), decimal=8)

    def test_interp_4(self):
        x = np.random.randn(1)
        xp = np.random.randn(50)
        xp.sort()
        yp = np.random.randn(50)
        np.testing.assert_almost_equal(bm.interp(x, xp, yp, 0., 1.),
                                       np.interp(x, xp, yp, 0., 1.), decimal=8)


class TestTrapz(unittest.TestCase):

    # Run before every test
    def setUp(self):
        pass
    # Run after every test

    def tearDown(self):
        pass

    def test_trapz_1(self):
        y = np.random.randn(100)
        np.testing.assert_almost_equal(bm.trapz(y), np.trapz(y), decimal=8)

    def test_trapz_2(self):
        y = np.random.randn(100)
        x = np.random.rand(100)
        np.testing.assert_almost_equal(bm.trapz(y, x=x),
                                       np.trapz(y, x=x), decimal=8)

    def test_trapz_3(self):
        y = np.random.randn(100)
        np.testing.assert_almost_equal(bm.trapz(y, dx=0.1),
                                       np.trapz(y, dx=0.1), decimal=8)


class TestCumTrapz(unittest.TestCase):

    # Run before every test
    def setUp(self):
        pass
    # Run after every test

    def tearDown(self):
        pass

    def test_cumtrapz_1(self):
        import scipy.integrate
        y = np.random.randn(100)
        initial = np.random.rand()
        np.testing.assert_almost_equal(bm.cumtrapz(y, initial=initial),
                                       scipy.integrate.cumtrapz(
                                           y, initial=initial),
                                       decimal=8)

    def test_cumtrapz_2(self):
        import scipy.integrate
        y = np.random.randn(100)
        np.testing.assert_almost_equal(bm.cumtrapz(y),
                                       scipy.integrate.cumtrapz(y),
                                       decimal=8)

    def test_cumtrapz_3(self):
        import scipy.integrate
        y = np.random.randn(100)
        dx = np.random.rand()
        np.testing.assert_almost_equal(bm.cumtrapz(y, dx=dx),
                                       scipy.integrate.cumtrapz(y, dx=dx),
                                       decimal=8)

    def test_cumtrapz_4(self):
        import scipy.integrate
        y = np.random.randn(100)
        dx = np.random.rand()
        initial = np.random.rand()
        np.testing.assert_almost_equal(bm.cumtrapz(y, initial=initial, dx=dx),
                                       scipy.integrate.cumtrapz(
                                           y, initial=initial, dx=dx),
                                       decimal=8)


class TestSort(unittest.TestCase):

    # Run before every test
    def setUp(self):
        pass
    # Run after every test

    def tearDown(self):
        pass

    def test_sort_1(self):
        y = np.random.randn(100)
        y2 = np.copy(y)
        y2.sort()
        np.testing.assert_equal(bm.sort(y), y2)

    def test_sort_2(self):
        y = np.random.randn(200)
        y2 = np.copy(y)
        np.testing.assert_equal(bm.sort(y, reverse=True),
                                sorted(y2, reverse=True))

    def test_sort_3(self):
        y = np.random.randn(200)
        y2 = np.copy(y)
        bm.sort(y)
        y2.sort()
        np.testing.assert_equal(y, y2)
        bm.sort(y, reverse=True)
        y2 = sorted(y2, reverse=True)
        np.testing.assert_equal(y, y2)

    def test_sort_4(self):
        y = np.array([np.random.randint(100)
                      for i in range(100)], dtype=np.int32)
        y2 = np.copy(y)
        bm.sort(y)
        y2.sort()
        np.testing.assert_equal(y, y2)
        bm.sort(y, reverse=True)
        y2 = sorted(y2, reverse=True)
        np.testing.assert_equal(y, y2)

    def test_sort_5(self):
        y = np.array([np.random.randint(100)
                      for i in range(100)], dtype=int)
        y2 = np.copy(y)
        bm.sort(y)
        y2.sort()
        np.testing.assert_equal(y, y2)
        bm.sort(y, reverse=True)
        y2 = sorted(y2, reverse=True)
        np.testing.assert_equal(y, y2)


if __name__ == '__main__':

    unittest.main()
