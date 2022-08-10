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

from cmath import tau
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

        res = bm.beam_phase(bin_centers, profile, alpha,
                            omega_rf, phi_rf, bin_size)

        import cupy as cp
        bm.use_gpu()

        bin_centers = bm.array(bin_centers)
        profile = bm.array(profile)
        res_gpu = bm.beam_phase(bin_centers, profile,
                                alpha, omega_rf, phi_rf, bin_size)

        cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    @pytest.mark.parametrize('n_particles,n_slices',
                             [(125421, 17), (100000, 100), (1000000, 100)])
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

    @pytest.mark.parametrize('n_slices', [10, 33, 128, 1000])
    def test_beam_phase_fast_uniform(self, n_slices):
        omega_rf = np.random.rand()
        phi_rf = np.random.rand()
        bin_centers = np.linspace(0, 1e-6, n_slices)
        bin_size = bin_centers[1] - bin_centers[0]
        profile = np.random.randn(n_slices)

        res = bm.beam_phase_fast(bin_centers, profile,
                                 omega_rf, phi_rf, bin_size)

        import cupy as cp
        bm.use_gpu()

        bin_centers = bm.array(bin_centers)
        profile = bm.array(profile)
        res_gpu = bm.beam_phase_fast(bin_centers, profile,
                                     omega_rf, phi_rf, bin_size)

        cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    @pytest.mark.parametrize('n_particles,n_slices',
                             [(125421, 17), (100000, 100), (1000000, 100)])
    def test_beam_phase_fast_normal(self, n_particles, n_slices):
        omega_rf = np.random.rand()
        phi_rf = np.random.rand()
        dt = np.random.normal(1e-5, 1e-7, n_particles)
        profile, edges = np.histogram(dt, bins=n_slices)
        profile = profile.astype(dtype=float)
        bin_size = edges[1] - edges[0]
        bin_centers = edges[:-1] + bin_size/2

        res = bm.beam_phase_fast(bin_centers, profile,
                                 omega_rf, phi_rf, bin_size)

        import cupy as cp
        bm.use_gpu()
        bin_centers = bm.array(bin_centers)
        profile = bm.array(profile)
        res_gpu = bm.beam_phase_fast(bin_centers, profile,
                                     omega_rf, phi_rf, bin_size)

        cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    @pytest.mark.parametrize('n_slices', [1, 8, 100, 100000])
    def test_rf_volt_comp(self, n_slices):
        voltages = np.random.randn(n_slices)
        omega_rf = np.random.randn(n_slices)
        phi_rf = np.random.randn(n_slices)
        bin_centers = np.linspace(1e-5, 1e-6, n_slices)

        res = bm.rf_volt_comp(voltages, omega_rf, phi_rf, bin_centers)

        import cupy as cp
        bm.use_gpu()
        voltages = bm.array(voltages)
        omega_rf = bm.array(omega_rf)
        phi_rf = bm.array(phi_rf)
        bin_centers = bm.array(bin_centers)
        res_gpu = bm.rf_volt_comp(voltages, omega_rf, phi_rf, bin_centers)

        cp.testing.assert_array_almost_equal(res_gpu, res, decimal=8)

    @pytest.mark.parametrize('n_particles,n_kicks', [(10, 1), (10000, 4), (1000000, 2)])
    def test_synch_rad(self, n_particles, n_kicks):
        import cupy as cp

        dE = np.random.uniform(-1e8, 1e8, n_particles)
        dE_gpu = cp.array(dE)

        U0 = np.random.rand()
        tau_z = np.random.rand()

        bm.synchrotron_radiation(dE, U0, n_kicks, tau_z)

        bm.use_gpu()
        bm.synchrotron_radiation(dE_gpu, U0, n_kicks, tau_z)

        cp.testing.assert_allclose(dE_gpu, dE, rtol=1e-8, atol=0)

    @pytest.mark.parametrize('n_particles,n_kicks', [(10, 1), (10000, 4), (1000000, 2)])
    def test_synch_rad_full(self, n_particles, n_kicks):
        import cupy as cp

        dE = np.random.uniform(-1e8, 1e8, n_particles)
        dE_gpu = cp.array(dE)

        U0 = np.random.rand()
        tau_z = np.random.rand()
        sigma_dE = dE.std()
        # Energy zero is required to mask off the random term
        energy = 0.0

        bm.synchrotron_radiation_full(dE, U0, n_kicks, tau_z, sigma_dE, energy)

        bm.use_gpu()
        bm.synchrotron_radiation_full(
            dE_gpu, U0, n_kicks, tau_z, sigma_dE, energy)

        cp.testing.assert_allclose(dE_gpu, dE, rtol=1e-8, atol=0)

    @pytest.mark.parametrize('n_particles,n_slices,cut_left,cut_right',
                             [(100, 5, 0.01, 0.01), (10000, 100, 0.5, 0.5), 
                              (1000000, 1000, 0.0, 0.0), (1000000, 10000, 0.05, 0.01),
                              (10000000, 100000, 0.01, 0.01)])
    def test_profile_slices(self, n_particles, n_slices, cut_left, cut_right):
        import cupy as cp

        dt = np.random.normal(loc=1e-5, scale=1e-7, size=n_particles)
        dt_gpu = cp.array(dt)

        max_dt = dt.max()
        min_dt = dt.min()
        cut_left = (1 + cut_left) * min_dt
        cut_right = (1 - cut_right) * max_dt
        profile = np.empty(n_slices, dtype=float)
        

        bm.slice(dt, profile, cut_left, cut_right)

        bm.use_gpu()
        profile_gpu = bm.empty(n_slices, dtype=float)
        bm.slice(dt_gpu, profile_gpu, cut_left, cut_right)

        cp.testing.assert_allclose(profile_gpu, profile, rtol=1e-8, atol=0)


if __name__ == '__main__':

    unittest.main()
