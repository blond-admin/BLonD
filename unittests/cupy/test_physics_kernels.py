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

class TestBeamPhase:

    # Run before every test
    def setup_method(self):
        np.random.seed(0)

    # Run after every test
    def teardown_method(self):
        bm.use_cpu()

    @pytest.mark.parametrize('n_slices', [10, 33, 128, 1000])
    def test_beam_phase_uniform(self, n_slices):
        omega_rf = np.random.rand()
        phi_rf = np.random.rand()
        alpha = np.random.rand()
        bin_centers = np.linspace(0, 1e-6, n_slices)
        bin_size = bin_centers[1] - bin_centers[0]
        profile = np.random.randn(n_slices)

        res = bm.beam_phase(bin_centers, profile, alpha, omega_rf, phi_rf, bin_size)

        import cupy as cp
        bm.use_gpu()

        bin_centers = bm.array(bin_centers)
        profile = bm.array(profile)
        res_gpu = bm.beam_phase(bin_centers, profile,
                                alpha, omega_rf, phi_rf, bin_size)

        cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    @pytest.mark.parametrize('n_particles,n_slices', [(125421, 17), (100000, 100), (1000000, 100)])
    def test_beam_phase_normal(self, n_particles, n_slices):
        omega_rf = np.random.rand()
        phi_rf = np.random.rand()
        alpha = np.random.rand()
        dt = np.random.normal(1e-5, 1e-7, n_particles)
        profile, edges = np.histogram(dt, bins=n_slices)
        profile = profile.astype(dtype=float)
        bin_size = edges[1] - edges[0]
        bin_centers = edges[:-1] + bin_size/2

        res = bm.beam_phase(bin_centers, profile, alpha,
                            omega_rf, phi_rf, bin_size)

        import cupy as cp
        bm.use_gpu()
        bin_centers = bm.array(bin_centers)
        profile = bm.array(profile)
        res_gpu = bm.beam_phase(bin_centers, profile,
                                alpha, omega_rf, phi_rf, bin_size)

        cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)



if __name__ == '__main__':

    unittest.main()
