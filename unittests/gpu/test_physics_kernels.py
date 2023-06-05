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
        pytest.importorskip('cupy')
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
        bin_centers = edges[:-1] + bin_size / 2

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
        bin_centers = edges[:-1] + bin_size / 2

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
                              (1000000, 1000, 0.0, 0.0),
                              (1000000, 10000, 0.05, 0.01),
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

        bm.slice_beam(dt, profile, cut_left, cut_right)

        bm.use_gpu()
        profile_gpu = bm.empty(n_slices, dtype=float)
        bm.slice_beam(dt_gpu, profile_gpu, cut_left, cut_right)

        cp.testing.assert_allclose(profile_gpu, profile, rtol=1e-8, atol=0)

    @pytest.mark.parametrize('n_particles,n_rf,n_iter',
                             [(100, 1, 1), (100, 4, 10),
                              (1000000, 1, 100), (1000000, 10, 100)])
    def test_kick(self, n_particles, n_rf, n_iter):
        import cupy as cp

        dE = np.random.normal(loc=0, scale=1e7, size=n_particles)
        dt = np.random.normal(loc=1e-5, scale=1e-7, size=n_particles)
        dE_gpu = cp.array(dE)

        charge = 1.0
        acceleration_kick = 1e3 * np.random.rand()
        voltage = np.random.randn(n_rf)
        omega_rf = np.random.randn(n_rf)
        phi_rf = np.random.randn(n_rf)

        for i in range(n_iter):
            bm.kick(dt, dE, voltage, omega_rf, phi_rf, charge, n_rf,
                    acceleration_kick)

        bm.use_gpu()
        dt = bm.array(dt)
        voltage = bm.array(voltage)
        omega_rf = bm.array(omega_rf)
        phi_rf = bm.array(phi_rf)

        for i in range(n_iter):
            bm.kick(dt, dE_gpu, voltage, omega_rf, phi_rf, charge, n_rf,
                    acceleration_kick)

        cp.testing.assert_allclose(dE_gpu, dE, rtol=1e-8, atol=0)

    @pytest.mark.parametrize('n_particles,solver,alpha_order,n_iter',
                             [(100, 'simple', 0, 1), (100, 'legacy', 1, 10),
                              (100, 'exact', 2, 100), (10000, 'simple', 1, 100),
                              (10000, 'legacy', 2, 100), (10000, 'exact', 0, 100),
                              (1000000, 'simple', 2, 10), (1000000, 'legacy', 0, 10),
                              (1000000, 'exact', 1, 10)])
    def test_drift(self, n_particles, solver, alpha_order, n_iter):
        import cupy as cp

        solver = solver.encode(encoding='utf_8')
        dE = np.random.normal(loc=0, scale=1e7, size=n_particles)
        dt = np.random.normal(loc=1e-5, scale=1e-7, size=n_particles)
        dt_gpu = cp.array(dt)

        t_rev = np.random.rand()
        length_ratio = np.random.uniform()
        eta_0, eta_1, eta_2 = np.random.randn(3)
        alpha_0, alpha_1, alpha_2 = np.random.randn(3)
        beta = np.random.rand()
        energy = np.random.rand()

        for i in range(n_iter):
            bm.drift(dt, dE, solver, t_rev, length_ratio, alpha_order,
                     eta_0, eta_1, eta_2, alpha_0, alpha_1, alpha_2,
                     beta, energy)

        bm.use_gpu()
        dE = bm.array(dE)

        for i in range(n_iter):
            bm.drift(dt_gpu, dE, solver, t_rev, length_ratio, alpha_order,
                     eta_0, eta_1, eta_2, alpha_0, alpha_1, alpha_2,
                     beta, energy)

        cp.testing.assert_allclose(dt_gpu, dt, rtol=1e-8, atol=0)

    @pytest.mark.parametrize('n_particles,n_iter',
                             [(100, 1), (100, 10),
                              (10000000, 1), (10000000, 10)])
    def test_kick_drift(self, n_particles, n_iter):
        import cupy as cp

        solver = 'exact'
        solver = solver.encode(encoding='utf_8')
        alpha_order = 2
        n_rf = 1

        # dE = np.random.normal(loc=1e5, scale=1e2, size=n_particles)
        # dt = np.random.normal(loc=1e-5, scale=1e-7, size=n_particles)
        dE = np.random.normal(loc=0, scale=1, size=n_particles)
        dt = np.random.normal(loc=0, scale=1, size=n_particles)

        dt_gpu = cp.array(dt)
        dE_gpu = cp.array(dE)

        charge = 1.0
        acceleration_kick = 0.
        voltage = np.random.randn(n_rf)
        omega_rf = np.random.randn(n_rf)
        phi_rf = np.random.randn(n_rf)
        t_rev = np.random.rand()
        length_ratio = np.random.uniform()
        eta_0, eta_1, eta_2 = np.random.randn(3)
        alpha_0, alpha_1, alpha_2 = np.random.randn(3)

        beta = 1.
        energy = 1.

        for i in range(n_iter):
            bm.drift(dt, dE, solver, t_rev, length_ratio, alpha_order,
                     eta_0, eta_1, eta_2, alpha_0, alpha_1, alpha_2,
                     beta, energy)
            bm.kick(dt, dE, voltage, omega_rf, phi_rf, charge, n_rf,
                    acceleration_kick)

        bm.use_gpu()
        voltage = bm.array(voltage)
        omega_rf = bm.array(omega_rf)
        phi_rf = bm.array(phi_rf)

        for i in range(n_iter):
            bm.drift(dt_gpu, dE_gpu, solver, t_rev, length_ratio, alpha_order,
                     eta_0, eta_1, eta_2, alpha_0, alpha_1, alpha_2,
                     beta, energy)
            bm.kick(dt_gpu, dE_gpu, voltage, omega_rf, phi_rf, charge, n_rf,
                    acceleration_kick)

        cp.testing.assert_allclose(dt_gpu.mean(), dt.mean(), rtol=1e-6, atol=0)
        cp.testing.assert_allclose(dE_gpu.mean(), dE.mean(), rtol=1e-6, atol=0)

    @pytest.mark.parametrize('n_particles,n_slices,n_iter',
                             [(100, 1, 10), (100, 10, 1), (10000, 256, 100),
                              (1000000, 100, 100), (1000000, 1000, 100),
                              (1000000, 100000, 100)])
    def test_interp_kick(self, n_particles, n_slices, n_iter):
        import cupy as cp

        dE = np.random.normal(loc=0, scale=1e7, size=n_particles)
        dE_gpu = cp.array(dE)

        dt = np.random.normal(loc=1e-5, scale=1e-7, size=n_particles)
        _, edges = np.histogram(dt, bins=n_slices)
        bin_size = edges[1] - edges[0]
        bin_centers = edges[:-1] + bin_size / 2

        charge = 1.0
        acceleration_kick = 1e3 * np.random.rand()
        voltage = np.random.randn(n_slices)

        for i in range(n_iter):
            bm.linear_interp_kick(
                dt, dE, voltage, bin_centers, charge, acceleration_kick)

        bm.use_gpu()
        dt = bm.array(dt)
        voltage = bm.array(voltage)
        bin_centers = bm.array(bin_centers)

        for i in range(n_iter):
            bm.linear_interp_kick(
                dt, dE_gpu, voltage, bin_centers, charge, acceleration_kick)

        cp.testing.assert_allclose(dE_gpu, dE, rtol=1e-8, atol=0)


class TestBigaussianData:
    # Simulation parameters -------------------------------------------------------
    # Bunch parameters
    N_b = 1e9           # Intensity
    tau_0 = 0.4e-9          # Initial bunch length, 4 sigma [s]
    # Machine and RF parameters
    C = 26658.883        # Machine circumference [m]
    p_i = 450e9         # Synchronous momentum [eV/c]
    p_f = 460.005e9      # Synchronous momentum, final
    h = 35640            # Harmonic number
    V = 6e6                # RF voltage [V]
    dphi = 0             # Phase modulation/offset
    gamma_t = 55.759505  # Transition gamma
    alpha = 1. / gamma_t / gamma_t        # First order mom. comp. factor
    # Tracking details
    N_t = 2000           # Number of turns to track

    # Run before every test
    def setup_method(self):
        # Try to import cupy, skip if not found
        pytest.importorskip('cupy')
        self.ring = Ring(self.C, self.alpha,
                         np.linspace(self.p_i, self.p_f, self.N_t + 1),
                         Proton(), self.N_t)

        self.rf = RFStation(self.ring, [self.h],
                            self.V * np.linspace(1, 1.1, self.N_t + 1),
                            [self.dphi])

    # Run after every test
    def teardown_method(self):
        bm.use_cpu()

    @pytest.mark.parametrize('N_p,n_iter',
                             [(100, 10), (10000, 100), (100000, 100)])
    def test_kick(self, N_p, n_iter):
        import cupy as cp

        rf_gpu = RFStation(self.ring, [self.h],
                           self.V * np.linspace(1, 1.1, self.N_t + 1),
                           [self.dphi])

        beam = Beam(self.ring, N_p, self.N_b)
        beam_gpu = Beam(self.ring, N_p, self.N_b)

        bigaussian(self.ring, self.rf, beam,
                   self.tau_0 / 4, reinsertion=True, seed=1)

        beam_gpu.dt[:] = beam.dt[:]
        beam_gpu.dE[:] = beam.dE[:]

        long_tracker = RingAndRFTracker(self.rf, beam)
        long_tracker_gpu = RingAndRFTracker(rf_gpu, beam_gpu)

        cp.testing.assert_allclose(beam_gpu.dE, beam.dE, rtol=1e-8, atol=0,
                                   err_msg='Checking initial conditions')
        cp.testing.assert_allclose(beam_gpu.dt, beam.dt, rtol=1e-8, atol=0,
                                   err_msg='Checking initial conditions')

        for i in range(n_iter):
            long_tracker.kick(beam.dt, beam.dE, long_tracker.counter[0])
            long_tracker.drift(beam.dt, beam.dE, long_tracker.counter[0] + 1)

            long_tracker.counter[0] += 1

        bm.use_gpu()
        beam_gpu.to_gpu()
        long_tracker_gpu.to_gpu()
        for i in range(n_iter):

            long_tracker_gpu.kick(beam_gpu.dt, beam_gpu.dE,
                                  long_tracker_gpu.counter[0])
            long_tracker_gpu.drift(
                beam_gpu.dt, beam_gpu.dE, long_tracker_gpu.counter[0] + 1)

            long_tracker_gpu.counter[0] += 1

        cp.testing.assert_allclose(beam_gpu.dE, beam.dE, rtol=1e-8, atol=0)
        cp.testing.assert_allclose(beam_gpu.dt, beam.dt, rtol=1e-8, atol=0)

    @pytest.mark.parametrize('N_p,n_iter',
                             [(1 << 8, 10), (1 << 14, 100), (1 << 20, 100)])
    def test_simple_track(self, N_p, n_iter):
        import cupy as cp

        rf_gpu = RFStation(self.ring, [self.h],
                           self.V * np.linspace(1, 1.1, self.N_t + 1),
                           [self.dphi])

        beam = Beam(self.ring, N_p, self.N_b)
        beam_gpu = Beam(self.ring, N_p, self.N_b)

        bigaussian(self.ring, self.rf, beam,
                   self.tau_0 / 4, reinsertion=True, seed=1)

        beam_gpu.dt[:] = beam.dt[:]
        beam_gpu.dE[:] = beam.dE[:]

        long_tracker = RingAndRFTracker(self.rf, beam)
        long_tracker_gpu = RingAndRFTracker(rf_gpu, beam_gpu)

        cp.testing.assert_allclose(beam_gpu.dE, beam.dE, rtol=1e-8, atol=0,
                                   err_msg='Checking initial conditions')
        cp.testing.assert_allclose(beam_gpu.dt, beam.dt, rtol=1e-8, atol=0,
                                   err_msg='Checking initial conditions')

        for i in range(n_iter):
            long_tracker.track()

        bm.use_gpu()
        beam_gpu.to_gpu()
        long_tracker_gpu.to_gpu()
        for i in range(n_iter):
            long_tracker_gpu.track()

        cp.testing.assert_allclose(beam_gpu.dE, beam.dE, rtol=1e-8, atol=0)
        cp.testing.assert_allclose(beam_gpu.dt, beam.dt, rtol=1e-8, atol=0)

    @pytest.mark.parametrize('N_p,n_slices,n_iter',
                             [(100, 10, 10), (10000, 25, 100), (100000, 100, 100)])
    def test_profile_track(self, N_p, n_slices, n_iter):
        import cupy as cp

        rf_gpu = RFStation(self.ring, [self.h],
                           self.V * np.linspace(1, 1.1, self.N_t + 1),
                           [self.dphi])

        beam = Beam(self.ring, N_p, self.N_b)
        beam_gpu = Beam(self.ring, N_p, self.N_b)

        bigaussian(self.ring, self.rf, beam,
                   self.tau_0 / 4, reinsertion=True, seed=1)

        beam_gpu.dt[:] = beam.dt[:]
        beam_gpu.dE[:] = beam.dE[:]

        profile = Profile(beam, CutOptions(n_slices=n_slices, cut_left=0,
                                           cut_right=self.rf.t_rf[0, 0]),
                          FitOptions(fit_option='gaussian'))

        profile_gpu = Profile(beam_gpu, CutOptions(n_slices=n_slices, cut_left=0,
                                                   cut_right=self.rf.t_rf[0, 0]),
                              FitOptions(fit_option='gaussian'))

        long_tracker = RingAndRFTracker(self.rf, beam)
        long_tracker_gpu = RingAndRFTracker(rf_gpu, beam_gpu)

        cp.testing.assert_allclose(beam_gpu.dE, beam.dE, rtol=1e-8, atol=0,
                                   err_msg='Checking initial conditions')
        cp.testing.assert_allclose(beam_gpu.dt, beam.dt, rtol=1e-8, atol=0,
                                   err_msg='Checking initial conditions')

        for i in range(n_iter):
            long_tracker.track()
            profile.track()

        bm.use_gpu()
        beam_gpu.to_gpu()
        long_tracker_gpu.to_gpu()
        profile_gpu.to_gpu()
        for i in range(n_iter):
            long_tracker_gpu.track()
            profile_gpu.track()

        cp.testing.assert_allclose(beam_gpu.dE, beam.dE, rtol=1e-8, atol=0)
        cp.testing.assert_allclose(beam_gpu.dt, beam.dt, rtol=1e-8, atol=0)
        cp.testing.assert_allclose(profile_gpu.n_macroparticles,
                                   profile.n_macroparticles, rtol=1e-8, atol=0)

    @pytest.mark.parametrize('N_p,n_slices,n_iter',
                             [(100, 10, 10), (10000, 25, 100), (100000, 100, 100)])
    def test_rf_voltage_calc(self, N_p, n_slices, n_iter):
        import cupy as cp

        rf_gpu = RFStation(self.ring, [self.h],
                           self.V * np.linspace(1, 1.1, self.N_t + 1),
                           [self.dphi])

        beam = Beam(self.ring, N_p, self.N_b)
        beam_gpu = Beam(self.ring, N_p, self.N_b)

        bigaussian(self.ring, self.rf, beam,
                   self.tau_0 / 4, reinsertion=True, seed=1)

        beam_gpu.dt[:] = beam.dt[:]
        beam_gpu.dE[:] = beam.dE[:]

        profile = Profile(beam, CutOptions(n_slices=n_slices, cut_left=0,
                                           cut_right=self.rf.t_rf[0, 0]),
                          FitOptions(fit_option='gaussian'))

        profile_gpu = Profile(beam_gpu, CutOptions(n_slices=n_slices, cut_left=0,
                                                   cut_right=self.rf.t_rf[0, 0]),
                              FitOptions(fit_option='gaussian'))

        long_tracker = RingAndRFTracker(self.rf, beam, Profile=profile)
        long_tracker_gpu = RingAndRFTracker(
            rf_gpu, beam_gpu, Profile=profile_gpu)

        cp.testing.assert_allclose(beam_gpu.dE, beam.dE, rtol=1e-8, atol=0,
                                   err_msg='Checking initial conditions')
        cp.testing.assert_allclose(beam_gpu.dt, beam.dt, rtol=1e-8, atol=0,
                                   err_msg='Checking initial conditions')

        long_tracker.rf_voltage_calculation()

        bm.use_gpu()
        long_tracker_gpu.to_gpu()

        long_tracker_gpu.rf_voltage_calculation()

        cp.testing.assert_allclose(long_tracker_gpu.rf_voltage,
                                   long_tracker.rf_voltage, rtol=1e-8, atol=0)


if __name__ == '__main__':

    unittest.main()
