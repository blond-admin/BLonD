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

import numpy as np
import pytest

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, FitOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.trackers.tracker import RingAndRFTracker
from blond.utils import bmath as bm


class TestSyntheticData:
    # Run before every test
    def setup_method(self):
        # Try to import cupy, skip if not found
        pytest.importorskip("cupy")
        bm.use_gpu()
        bm.use_precision("double")

    # Run after every test
    def teardown_method(self):
        bm.use_precision("double")
        bm.use_cpu()

    @pytest.mark.parametrize("n_slices", [10, 33, 128, 1000])
    def test_beam_phase_uniform(self, n_slices):
        import cupy as cp

        cp.random.seed(0)

        omega_rf = cp.random.rand()
        phi_rf = cp.random.rand()
        alpha = cp.random.rand()
        bin_centers = cp.linspace(0, 1e-6, n_slices)
        bin_size = bin_centers[1] - bin_centers[0]
        profile = cp.random.randn(n_slices)

        res = bm.beam_phase(
            bin_centers, profile, alpha, omega_rf, phi_rf, bin_size
        )

        bm.use_precision("single")

        bin_centers = bm.array(bin_centers, dtype=cp.float32)
        profile = bm.array(profile, dtype=cp.float32)
        res_f32 = bm.beam_phase(
            bin_centers, profile, alpha, omega_rf, phi_rf, bin_size
        )

        cp.testing.assert_array_almost_equal(res_f32, res, decimal=6)

    @pytest.mark.parametrize(
        "n_particles,n_slices", [(125421, 17), (100000, 100), (1000000, 100)]
    )
    def test_beam_phase_normal(self, n_particles, n_slices):
        import cupy as cp

        cp.random.seed(0)

        omega_rf = cp.random.rand()
        phi_rf = cp.random.rand()
        alpha = cp.random.rand()
        dt = cp.random.normal(1e-5, 1e-7, n_particles)
        profile, edges = cp.histogram(dt, bins=n_slices)
        profile = profile.astype(dtype=float)
        bin_size = edges[1] - edges[0]
        bin_centers = edges[:-1] + bin_size / 2

        res = bm.beam_phase(
            bin_centers, profile, alpha, omega_rf, phi_rf, bin_size
        )

        bm.use_precision("single")
        bin_centers = bm.array(bin_centers, dtype=cp.float32)
        profile = bm.array(profile, dtype=cp.float32)
        res_f32 = bm.beam_phase(
            bin_centers, profile, alpha, omega_rf, phi_rf, bin_size
        )

        cp.testing.assert_array_almost_equal(res_f32, res, decimal=6)

    @pytest.mark.parametrize("n_slices", [10, 33, 128, 1000])
    def test_beam_phase_fast_uniform(self, n_slices):
        import cupy as cp

        cp.random.seed(0)

        omega_rf = cp.random.rand()
        phi_rf = cp.random.rand()
        bin_centers = cp.linspace(0, 1e-6, n_slices)
        bin_size = bin_centers[1] - bin_centers[0]
        profile = cp.random.randn(n_slices)

        res = bm.beam_phase_fast(
            bin_centers, profile, omega_rf, phi_rf, bin_size
        )

        bm.use_precision("single")

        bin_centers = bm.array(bin_centers, dtype=cp.float32)
        profile = bm.array(profile, dtype=cp.float32)
        res_f32 = bm.beam_phase_fast(
            bin_centers, profile, omega_rf, phi_rf, bin_size
        )

        cp.testing.assert_array_almost_equal(res_f32, res, decimal=6)

    @pytest.mark.parametrize(
        "n_particles,n_slices", [(125421, 17), (100000, 100), (1000000, 100)]
    )
    def test_beam_phase_fast_normal(self, n_particles, n_slices):
        import cupy as cp

        cp.random.seed(0)

        omega_rf = cp.random.rand()
        phi_rf = cp.random.rand()
        dt = cp.random.normal(1e-5, 1e-7, n_particles)
        profile, edges = cp.histogram(dt, bins=n_slices)
        profile = profile.astype(dtype=float)
        bin_size = edges[1] - edges[0]
        bin_centers = edges[:-1] + bin_size / 2

        res = bm.beam_phase_fast(
            bin_centers, profile, omega_rf, phi_rf, bin_size
        )
        bm.use_precision("single")

        bin_centers = bm.array(bin_centers, dtype=cp.float32)
        profile = bm.array(profile, dtype=cp.float32)
        res_f32 = bm.beam_phase_fast(
            bin_centers, profile, omega_rf, phi_rf, bin_size
        )

        cp.testing.assert_array_almost_equal(res_f32, res, decimal=6)

    @pytest.mark.parametrize("n_slices", [1, 8, 100, 10000])
    def test_rf_volt_comp(self, n_slices):
        import cupy as cp

        cp.random.seed(0)
        voltages = cp.random.randn(n_slices)
        omega_rf = cp.random.randn(n_slices)
        phi_rf = cp.random.randn(n_slices)
        bin_centers = cp.linspace(1e-5, 1e-6, n_slices)

        res = bm.rf_volt_comp(voltages, omega_rf, phi_rf, bin_centers)

        bm.use_precision("single")

        voltages = bm.array(voltages, dtype=cp.float32)
        omega_rf = bm.array(omega_rf, dtype=cp.float32)
        phi_rf = bm.array(phi_rf, dtype=cp.float32)
        bin_centers = bm.array(bin_centers, dtype=cp.float32)

        res_f32 = bm.rf_volt_comp(voltages, omega_rf, phi_rf, bin_centers)

        cp.testing.assert_allclose(res_f32, res, rtol=1e-5, atol=0)

    @pytest.mark.parametrize(
        "n_particles,n_kicks", [(10, 1), (10000, 4), (1000000, 2)]
    )
    def test_synch_rad(self, n_particles, n_kicks):
        import cupy as cp

        cp.random.seed(0)

        dE = cp.random.uniform(-1e8, 1e8, n_particles)
        dE_f32 = cp.array(dE, dtype=cp.float32)

        U0 = cp.random.rand().item()
        tau_z = cp.random.rand().item()

        bm.synchrotron_radiation(dE, U0, n_kicks, tau_z)

        bm.use_precision("single")

        bm.synchrotron_radiation(dE_f32, U0, n_kicks, tau_z)

        cp.testing.assert_allclose(dE_f32, dE, rtol=1e-6, atol=0)

    @pytest.mark.parametrize(
        "n_particles,n_kicks", [(10, 1), (10000, 4), (1000000, 2)]
    )
    def test_synch_rad_full(self, n_particles, n_kicks):
        import cupy as cp

        cp.random.seed(0)

        dE = cp.random.uniform(-1e8, 1e8, n_particles)
        dE_f32 = cp.array(dE, dtype=cp.float32)

        U0 = cp.random.rand().item()
        tau_z = cp.random.rand().item()
        sigma_dE = dE.std().item()
        # Energy zero is required to mask off the random term
        energy = 0.0

        bm.synchrotron_radiation_full(dE, U0, n_kicks, tau_z, sigma_dE, energy)

        bm.use_precision("single")
        bm.synchrotron_radiation_full(
            dE_f32, U0, n_kicks, tau_z, sigma_dE, energy
        )

        cp.testing.assert_allclose(dE_f32, dE, rtol=1e-6, atol=0)

    @pytest.mark.parametrize(
        "n_particles,n_slices,cut_left,cut_right",
        [
            (100, 5, 0.01, 0.01),
            (10000, 100, 0.05, 0.05),
            (1000000, 1000, 0.0, 0.0),
        ],
    )
    def test_profile_slices(self, n_particles, n_slices, cut_left, cut_right):
        import cupy as cp

        cp.random.seed(0)

        dt = cp.random.normal(loc=1e-5, scale=1e-6, size=n_particles)
        dt_f32 = cp.array(dt, cp.float32)

        max_dt = dt.max().item()
        min_dt = dt.min().item()
        cut_left = (1 + cut_left) * min_dt
        cut_right = (1 - cut_right) * max_dt
        profile = cp.empty(n_slices, dtype=cp.float64)

        bm.slice_beam(dt, profile, cut_left, cut_right)

        bm.use_precision("single")

        profile_f32 = bm.empty(n_slices, dtype=cp.float32)
        bm.slice_beam(dt_f32, profile_f32, cut_left, cut_right)

        cp.testing.assert_allclose(
            profile_f32.mean(), profile.mean(), rtol=1e-5, atol=0
        )

    @pytest.mark.parametrize(
        "n_particles,n_rf,n_iter",
        [(100, 1, 1), (100, 4, 10), (1000000, 1, 5), (1000000, 2, 5)],
    )
    def test_kick(self, n_particles, n_rf, n_iter):
        import cupy as cp

        cp.random.seed(0)

        dE = cp.random.normal(loc=0, scale=1e7, size=n_particles)
        dt = cp.random.normal(loc=1e-5, scale=1e-7, size=n_particles)
        dE_f32 = cp.array(dE, dtype=cp.float32)
        dt_f32 = cp.array(dt, dtype=cp.float32)

        charge = 1.0
        acceleration_kick = 1e3 * cp.random.rand().item()
        voltage = cp.random.randn(n_rf)
        omega_rf = cp.random.randn(n_rf)
        phi_rf = cp.random.randn(n_rf)

        for i in range(n_iter):
            bm.kick(
                dt,
                dE,
                voltage,
                omega_rf,
                phi_rf,
                charge,
                n_rf,
                acceleration_kick,
            )

        bm.use_precision("single")

        voltage = bm.array(voltage, dtype=cp.float32)
        omega_rf = bm.array(omega_rf, dtype=cp.float32)
        phi_rf = bm.array(phi_rf, dtype=cp.float32)

        for i in range(n_iter):
            bm.kick(
                dt_f32,
                dE_f32,
                voltage,
                omega_rf,
                phi_rf,
                charge,
                n_rf,
                acceleration_kick,
            )

        cp.testing.assert_allclose(dE_f32, dE, rtol=1e-5, atol=0)

    @pytest.mark.parametrize(
        "n_particles,solver,alpha_order,n_iter",
        [
            (100, "simple", 0, 1),
            (100, "legacy", 1, 10),
            (100, "exact", 2, 100),
            (10000, "simple", 1, 100),
            (10000, "legacy", 2, 100),
            (10000, "exact", 0, 100),
            (1000000, "simple", 2, 10),
            (1000000, "legacy", 0, 10),
            (1000000, "exact", 1, 10),
        ],
    )
    def test_drift(self, n_particles, solver, alpha_order, n_iter):
        import cupy as cp

        cp.random.seed(0)

        dE = cp.random.normal(loc=0, scale=1e7, size=n_particles)
        dt = cp.random.normal(loc=1e-5, scale=1e-7, size=n_particles)
        dt_f32 = cp.array(dt, dtype=cp.float32)
        dE_f32 = cp.array(dE, dtype=cp.float32)

        t_rev = cp.random.rand().item()
        length_ratio = cp.random.uniform().item()
        eta_0 = cp.random.rand().item()
        eta_1 = cp.random.rand().item()
        eta_2 = cp.random.rand().item()
        alpha_0 = cp.random.rand().item()
        alpha_1 = cp.random.rand().item()
        alpha_2 = cp.random.rand().item()
        beta = cp.random.rand().item()
        energy = cp.random.rand().item()

        for i in range(n_iter):
            bm.drift(
                dt,
                dE,
                solver,
                t_rev,
                length_ratio,
                alpha_order,
                eta_0,
                eta_1,
                eta_2,
                alpha_0,
                alpha_1,
                alpha_2,
                beta,
                energy,
            )

        bm.use_precision("single")

        for i in range(n_iter):
            bm.drift(
                dt_f32,
                dE_f32,
                solver,
                t_rev,
                length_ratio,
                alpha_order,
                eta_0,
                eta_1,
                eta_2,
                alpha_0,
                alpha_1,
                alpha_2,
                beta,
                energy,
            )

        cp.testing.assert_allclose(dE, dE_f32, rtol=1e-5, atol=0)
        cp.testing.assert_allclose(dt, dt_f32, rtol=1e-5, atol=0)

    @pytest.mark.parametrize("n_particles,n_iter", [(100, 1), (100, 2)])
    def test_kick_drift(self, n_particles, n_iter):
        import cupy as cp

        cp.random.seed(0)

        solver = "exact"
        alpha_order = 2
        n_rf = 1

        dE = cp.random.normal(loc=0, scale=1, size=n_particles)
        dt = cp.random.normal(loc=0, scale=1, size=n_particles)

        dt_f32 = cp.array(dt, dtype=cp.float32)
        dE_f32 = cp.array(dE, dtype=cp.float32)

        charge = 1.0
        acceleration_kick = 0.0
        voltage = cp.random.randn(n_rf)
        omega_rf = cp.random.randn(n_rf)
        phi_rf = cp.random.randn(n_rf)
        t_rev = cp.random.rand().item()
        length_ratio = cp.random.uniform().item()
        eta_0 = cp.random.rand().item()
        eta_1 = cp.random.rand().item()
        eta_2 = cp.random.rand().item()
        alpha_0 = cp.random.rand().item()
        alpha_1 = cp.random.rand().item()
        alpha_2 = cp.random.rand().item()
        beta = 1.0
        energy = 1.0

        for i in range(n_iter):
            bm.drift(
                dt,
                dE,
                solver,
                t_rev,
                length_ratio,
                alpha_order,
                eta_0,
                eta_1,
                eta_2,
                alpha_0,
                alpha_1,
                alpha_2,
                beta,
                energy,
            )
            bm.kick(
                dt,
                dE,
                voltage,
                omega_rf,
                phi_rf,
                charge,
                n_rf,
                acceleration_kick,
            )

        bm.use_precision("single")

        voltage = bm.array(voltage, dtype=cp.float32)
        omega_rf = bm.array(omega_rf, dtype=cp.float32)
        phi_rf = bm.array(phi_rf, dtype=cp.float32)

        for i in range(n_iter):
            bm.drift(
                dt_f32,
                dE_f32,
                solver,
                t_rev,
                length_ratio,
                alpha_order,
                eta_0,
                eta_1,
                eta_2,
                alpha_0,
                alpha_1,
                alpha_2,
                beta,
                energy,
            )
            bm.kick(
                dt_f32,
                dE_f32,
                voltage,
                omega_rf,
                phi_rf,
                charge,
                n_rf,
                acceleration_kick,
            )

        cp.testing.assert_allclose(dt_f32, dt, rtol=1e-5, atol=0)
        cp.testing.assert_allclose(dE_f32, dE, rtol=1e-5, atol=0)

    @pytest.mark.parametrize(
        "n_particles,n_slices,n_iter",
        [(100, 1, 10), (100, 10, 1), (10000, 256, 10), (100000, 100, 5)],
    )
    def test_interp_kick(self, n_particles, n_slices, n_iter):
        import cupy as cp

        cp.random.seed(0)

        dE = cp.random.normal(loc=0, scale=1e7, size=n_particles)
        dE_f32 = cp.array(dE, dtype=cp.float32)

        dt = cp.random.normal(loc=1e-5, scale=1e-7, size=n_particles)
        _, edges = cp.histogram(dt, bins=n_slices)
        bin_size = edges[1] - edges[0]
        bin_centers = edges[:-1] + bin_size / 2

        charge = 1.0
        acceleration_kick = 1e3 * cp.random.rand().item()
        voltage = cp.random.randn(n_slices)

        for i in range(n_iter):
            bm.linear_interp_kick(
                dt, dE, voltage, bin_centers, charge, acceleration_kick
            )

        bm.use_precision("single")

        dt_f32 = bm.array(dt, dtype=cp.float32)
        voltage = bm.array(voltage, dtype=cp.float32)
        bin_centers = bm.array(bin_centers, dtype=cp.float32)

        for i in range(n_iter):
            bm.linear_interp_kick(
                dt_f32, dE_f32, voltage, bin_centers, charge, acceleration_kick
            )

        cp.testing.assert_allclose(dE_f32, dE, rtol=1e-5, atol=0)


if __name__ == "__main__":
    unittest.main()
