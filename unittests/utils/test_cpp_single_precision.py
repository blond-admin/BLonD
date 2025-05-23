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

from blond.utils import bmath as bm


class TestSyntheticData:
    # Run before every test
    def setup_method(self):
        np.random.seed(0)
        bm.use_precision("double")

    # Run after every test
    def teardown_method(self):
        bm.use_precision("double")

    @pytest.mark.parametrize("n_slices", [10, 33, 128, 1000])
    def test_beam_phase_uniform(self, n_slices):
        omega_rf = np.random.rand()
        phi_rf = np.random.rand()
        alpha = np.random.rand()
        bin_centers = np.linspace(0, 1e-6, n_slices)
        bin_size = bin_centers[1] - bin_centers[0]
        profile = np.random.randn(n_slices)

        res = bm.beam_phase(
            bin_centers, profile, alpha, omega_rf, phi_rf, bin_size
        )

        bm.use_precision("single")

        bin_centers = bm.array(bin_centers, dtype=np.float32)
        profile = bm.array(profile, dtype=np.float32)
        res_f32 = bm.beam_phase(
            bin_centers, profile, alpha, omega_rf, phi_rf, bin_size
        )

        np.testing.assert_array_almost_equal(res_f32, res, decimal=4)

    @pytest.mark.parametrize(
        "n_particles,n_slices", [(125421, 17), (100000, 100), (1000000, 100)]
    )
    def test_beam_phase_normal(self, n_particles, n_slices):
        omega_rf = np.random.rand()
        phi_rf = np.random.rand()
        alpha = np.random.rand()
        dt = np.random.normal(1e-5, 1e-7, n_particles)
        profile, edges = np.histogram(dt, bins=n_slices)
        profile = profile.astype(dtype=float)
        bin_size = edges[1] - edges[0]
        bin_centers = edges[:-1] + bin_size / 2

        res = bm.beam_phase(
            bin_centers, profile, alpha, omega_rf, phi_rf, bin_size
        )

        bm.use_precision("single")
        bin_centers = bm.array(bin_centers, dtype=np.float32)
        profile = bm.array(profile, dtype=np.float32)
        res_f32 = bm.beam_phase(
            bin_centers, profile, alpha, omega_rf, phi_rf, bin_size
        )

        np.testing.assert_array_almost_equal(res_f32, res, decimal=4)

    @pytest.mark.parametrize("n_slices", [10, 33, 128, 1000])
    def test_beam_phase_fast_uniform(self, n_slices):
        omega_rf = np.random.rand()
        phi_rf = np.random.rand()
        bin_centers = np.linspace(0, 1e-6, n_slices)
        bin_size = bin_centers[1] - bin_centers[0]
        profile = np.random.randn(n_slices)

        res = bm.beam_phase_fast(
            bin_centers, profile, omega_rf, phi_rf, bin_size
        )

        bm.use_precision("single")

        bin_centers = bm.array(bin_centers, dtype=np.float32)
        profile = bm.array(profile, dtype=np.float32)
        res_f32 = bm.beam_phase_fast(
            bin_centers, profile, omega_rf, phi_rf, bin_size
        )

        np.testing.assert_array_almost_equal(res_f32, res, decimal=4)

    @pytest.mark.parametrize(
        "n_particles,n_slices", [(125421, 17), (100000, 100), (1000000, 100)]
    )
    def test_beam_phase_fast_normal(self, n_particles, n_slices):
        omega_rf = np.random.rand()
        phi_rf = np.random.rand()
        dt = np.random.normal(1e-5, 1e-7, n_particles)
        profile, edges = np.histogram(dt, bins=n_slices)
        profile = profile.astype(dtype=float)
        bin_size = edges[1] - edges[0]
        bin_centers = edges[:-1] + bin_size / 2

        res = bm.beam_phase_fast(
            bin_centers, profile, omega_rf, phi_rf, bin_size
        )
        bm.use_precision("single")

        bin_centers = bm.array(bin_centers, dtype=np.float32)
        profile = bm.array(profile, dtype=np.float32)
        res_f32 = bm.beam_phase_fast(
            bin_centers, profile, omega_rf, phi_rf, bin_size
        )

        np.testing.assert_array_almost_equal(res_f32, res, decimal=4)

    @pytest.mark.parametrize("n_slices", [1, 8, 100, 10000])
    def test_rf_volt_comp(self, n_slices):
        voltages = np.random.randn(n_slices)
        omega_rf = np.random.randn(n_slices)
        phi_rf = np.random.randn(n_slices)
        bin_centers = np.linspace(1e-5, 1e-6, n_slices)

        res = bm.rf_volt_comp(voltages, omega_rf, phi_rf, bin_centers)

        bm.use_precision("single")

        voltages = bm.array(voltages, dtype=np.float32)
        omega_rf = bm.array(omega_rf, dtype=np.float32)
        phi_rf = bm.array(phi_rf, dtype=np.float32)
        bin_centers = bm.array(bin_centers, dtype=np.float32)

        res_f32 = bm.rf_volt_comp(voltages, omega_rf, phi_rf, bin_centers)

        np.testing.assert_array_almost_equal(res_f32, res, decimal=4)

    @pytest.mark.parametrize(
        "n_particles,n_kicks", [(10, 1), (10000, 4), (1000000, 2)]
    )
    def test_synch_rad(self, n_particles, n_kicks):
        dE = np.random.uniform(-1e8, 1e8, n_particles)
        dE_f32 = np.array(dE, dtype=np.float32)

        U0 = np.random.rand()
        tau_z = np.random.rand()

        bm.synchrotron_radiation(dE, U0, n_kicks, tau_z)

        bm.use_precision("single")

        bm.synchrotron_radiation(dE_f32, U0, n_kicks, tau_z)

        np.testing.assert_allclose(dE_f32, dE, rtol=1e-4, atol=0)

    @pytest.mark.parametrize(
        "n_particles,n_kicks", [(10, 1), (10000, 4), (1000000, 2)]
    )
    def test_synch_rad_full(self, n_particles, n_kicks):
        dE = np.random.uniform(-1e8, 1e8, n_particles)
        dE_f32 = np.array(dE, dtype=np.float32)

        U0 = np.random.rand()
        tau_z = np.random.rand()
        sigma_dE = dE.std()
        # Energy zero is required to mask off the random term
        energy = 0.0

        bm.synchrotron_radiation_full(dE, U0, n_kicks, tau_z, sigma_dE, energy)

        bm.use_precision("single")
        bm.synchrotron_radiation_full(
            dE_f32, U0, n_kicks, tau_z, sigma_dE, energy
        )

        np.testing.assert_allclose(dE_f32, dE, rtol=1e-4, atol=0)

    @pytest.mark.parametrize(
        "n_particles,n_slices,cut_left,cut_right",
        [
            (100, 5, 0.01, 0.01),
            (10000, 100, 0.05, 0.05),
            (1000000, 1000, 0.0, 0.0),
        ],
    )
    def test_profile_slices(self, n_particles, n_slices, cut_left, cut_right):
        dt = np.random.normal(loc=1e-5, scale=1e-6, size=n_particles)
        dt_f32 = np.array(dt, np.float32)

        max_dt = dt.max()
        min_dt = dt.min()
        cut_left = (1 + cut_left) * min_dt
        cut_right = (1 - cut_right) * max_dt
        profile = np.empty(n_slices, dtype=np.float64)

        bm.slice_beam(dt, profile, cut_left, cut_right)

        bm.use_precision("single")

        profile_f32 = bm.empty(n_slices, dtype=np.float32)
        bm.slice_beam(dt_f32, profile_f32, cut_left, cut_right)

        np.testing.assert_allclose(
            profile_f32.mean(), profile.mean(), rtol=1e-4, atol=0
        )

    @pytest.mark.parametrize(
        "n_particles,n_rf,n_iter",
        [(100, 1, 1), (100, 4, 10), (1000000, 1, 5), (1000000, 2, 5)],
    )
    def test_kick(self, n_particles, n_rf, n_iter):
        dE = np.random.normal(loc=0, scale=1e7, size=n_particles)
        dt = np.random.normal(loc=1e-5, scale=1e-7, size=n_particles)
        dE_f32 = np.array(dE, dtype=np.float32)
        dt_f32 = np.array(dt, dtype=np.float32)

        charge = 1.0
        acceleration_kick = 1e3 * np.random.rand()
        voltage = np.random.randn(n_rf)
        omega_rf = np.random.randn(n_rf)
        phi_rf = np.random.randn(n_rf)

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

        voltage = bm.array(voltage, dtype=np.float32)
        omega_rf = bm.array(omega_rf, dtype=np.float32)
        phi_rf = bm.array(phi_rf, dtype=np.float32)

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

        np.testing.assert_allclose(dE_f32, dE, rtol=1e-4, atol=0)

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
        dE = np.random.normal(loc=0, scale=1e7, size=n_particles)
        dt = np.random.normal(loc=1e-5, scale=1e-7, size=n_particles)
        dt_f32 = np.array(dt, dtype=np.float32)
        dE_f32 = np.array(dE, dtype=np.float32)

        t_rev = np.random.rand()
        length_ratio = np.random.uniform()
        eta_0 = np.random.rand()
        eta_1 = np.random.rand()
        eta_2 = np.random.rand()
        alpha_0 = np.random.rand()
        alpha_1 = np.random.rand()
        alpha_2 = np.random.rand()
        beta = np.random.rand()
        energy = np.random.rand()

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

        np.testing.assert_allclose(dE, dE_f32, rtol=1e-4, atol=0)
        np.testing.assert_allclose(dt, dt_f32, rtol=1e-4, atol=0)

    @pytest.mark.parametrize("n_particles,n_iter", [(100, 1), (100, 2)])
    def test_kick_drift(self, n_particles, n_iter):
        solver = "exact"
        alpha_order = 2
        n_rf = 1

        dE = np.random.normal(loc=0, scale=1, size=n_particles)
        dt = np.random.normal(loc=0, scale=1, size=n_particles)

        dt_f32 = np.array(dt, dtype=np.float32)
        dE_f32 = np.array(dE, dtype=np.float32)

        charge = 1.0
        acceleration_kick = 0.0
        voltage = np.random.randn(n_rf)
        omega_rf = np.random.randn(n_rf)
        phi_rf = np.random.randn(n_rf)
        t_rev = np.random.rand()
        length_ratio = np.random.uniform()
        eta_0 = np.random.rand()
        eta_1 = np.random.rand()
        eta_2 = np.random.rand()
        alpha_0 = np.random.rand()
        alpha_1 = np.random.rand()
        alpha_2 = np.random.rand()
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

        voltage = bm.array(voltage, dtype=np.float32)
        omega_rf = bm.array(omega_rf, dtype=np.float32)
        phi_rf = bm.array(phi_rf, dtype=np.float32)

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

        np.testing.assert_allclose(dt_f32, dt, rtol=1e-4, atol=0)
        np.testing.assert_allclose(dE_f32, dE, rtol=1e-4, atol=0)

    @pytest.mark.parametrize(
        "n_particles,n_slices,n_iter",
        [(100, 1, 10), (100, 10, 1), (10000, 256, 10), (100000, 100, 5)],
    )
    def test_interp_kick(self, n_particles, n_slices, n_iter):
        dE = np.random.normal(loc=0, scale=1e7, size=n_particles)
        dE_f32 = np.array(dE, dtype=np.float32)

        dt = np.random.normal(loc=1e-5, scale=1e-7, size=n_particles)
        _, edges = np.histogram(dt, bins=n_slices)
        bin_size = edges[1] - edges[0]
        bin_centers = edges[:-1] + bin_size / 2

        charge = 1.0
        acceleration_kick = 1e3 * np.random.rand()
        voltage = np.random.randn(n_slices)

        for i in range(n_iter):
            bm.linear_interp_kick(
                dt, dE, voltage, bin_centers, charge, acceleration_kick
            )

        bm.use_precision("single")

        dt_f32 = bm.array(dt, dtype=np.float32)
        voltage = bm.array(voltage, dtype=np.float32)
        bin_centers = bm.array(bin_centers, dtype=np.float32)

        for i in range(n_iter):
            bm.linear_interp_kick(
                dt_f32, dE_f32, voltage, bin_centers, charge, acceleration_kick
            )

        np.testing.assert_allclose(dE_f32, dE, rtol=1e-4, atol=0)


if __name__ == "__main__":
    unittest.main()
