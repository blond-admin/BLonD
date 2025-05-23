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
from copy import deepcopy
import os

import numpy as np
import pytest
from scipy.constants import c

from blond.beam.beam import Beam, Proton
from blond.beam.sparse_slices import SparseSlices
from blond.impedances.music import Music
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.utils import bmath as bm
from blond.utils.butils_wrap_cpp import get_libblond
from blond.utils.butils_wrap_numba import \
    resonator_induced_voltage_1_turn as resonator_induced_voltage_1_turn_numba, \
    _resonator_induced_voltage_1_turn, distribution_from_tomoscope, \
    linear_interp_kick, set_random_seed, slice_beam, sparse_histogram, \
    synchrotron_radiation, synchrotron_radiation_full
from blond.utils.butils_wrap_python import \
    resonator_induced_voltage_1_turn as resonator_induced_voltage_1_turn_python

class Test:

    # Run before every test
    def setup_method(self):
        if get_libblond() is None:
            pytest.skip('C++ libblond not compiled')
        np.random.seed(0)

    # Run after every test
    def teardown_method(self):
        pass

    @pytest.mark.parametrize('n_particles,n_slices',
                             [(125421, 17), (100000, 100)])
    def test_beam_phase_normal(self, n_particles, n_slices):
        bm.use_cpp()
        np.testing.assert_equal(bm.device, 'CPU_CPP')

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

        bm.use_numba()
        np.testing.assert_equal(bm.device, 'CPU_NU')
        bin_centers = bm.array(bin_centers)
        profile = bm.array(profile)
        res_py = bm.beam_phase(bin_centers, profile,
                               alpha, omega_rf, phi_rf, bin_size)

        np.testing.assert_array_almost_equal(res_py, res, decimal=8)

    @pytest.mark.parametrize('n_particles,n_slices',
                             [(125421, 17), (100000, 100)])
    def test_beam_phase_fast_normal(self, n_particles, n_slices):
        bm.use_cpp()
        np.testing.assert_equal(bm.device, 'CPU_CPP')

        omega_rf = np.random.rand()
        phi_rf = np.random.rand()
        dt = np.random.normal(1e-5, 1e-7, n_particles)
        profile, edges = np.histogram(dt, bins=n_slices)
        profile = profile.astype(dtype=float)
        bin_size = edges[1] - edges[0]
        bin_centers = edges[:-1] + bin_size / 2

        res = bm.beam_phase_fast(bin_centers, profile,
                                 omega_rf, phi_rf, bin_size)

        bm.use_numba()
        np.testing.assert_equal(bm.device, 'CPU_NU')
        bin_centers = bm.array(bin_centers)
        profile = bm.array(profile)
        res_py = bm.beam_phase_fast(bin_centers, profile,
                                    omega_rf, phi_rf, bin_size)

        np.testing.assert_array_almost_equal(res_py, res, decimal=8)

    @pytest.mark.parametrize('n_slices', [16, 256, 1024])
    def test_rf_volt_comp(self, n_slices):
        bm.use_cpp()
        np.testing.assert_equal(bm.device, 'CPU_CPP')

        voltages = np.random.randn(n_slices)
        omega_rf = np.random.randn(n_slices)
        phi_rf = np.random.randn(n_slices)
        bin_centers = np.linspace(1e-5, 1e-6, n_slices)

        res = bm.rf_volt_comp(voltages, omega_rf, phi_rf, bin_centers)

        bm.use_numba()
        np.testing.assert_equal(bm.device, 'CPU_NU')
        voltages = bm.array(voltages)
        omega_rf = bm.array(omega_rf)
        phi_rf = bm.array(phi_rf)
        bin_centers = bm.array(bin_centers)
        res_py = bm.rf_volt_comp(voltages, omega_rf, phi_rf, bin_centers)

        np.testing.assert_array_almost_equal(res_py, res, decimal=8)

    @pytest.mark.parametrize('n_particles,n_kicks', [(10, 1), (10000, 4), (100000, 2)])
    def test_synch_rad(self, n_particles, n_kicks):
        bm.use_cpp()
        np.testing.assert_equal(bm.device, 'CPU_CPP')

        dE = np.random.uniform(-1e8, 1e8, n_particles)
        dE_py = dE.copy()

        U0 = np.random.rand()
        tau_z = np.random.rand()

        bm.synchrotron_radiation(dE, U0, n_kicks, tau_z)

        bm.use_numba()
        np.testing.assert_equal(bm.device, 'CPU_NU')
        bm.synchrotron_radiation(dE_py, U0, n_kicks, tau_z)

        np.testing.assert_allclose(dE_py, dE, rtol=1e-8, atol=0)

    @pytest.mark.parametrize('n_particles,n_kicks', [(10, 1), (10000, 4), (100000, 2)])
    def test_synch_rad_full(self, n_particles, n_kicks):
        bm.use_cpp()
        np.testing.assert_equal(bm.device, 'CPU_CPP')

        dE = np.random.uniform(-1e8, 1e8, n_particles)
        dE_py = dE.copy()

        U0 = np.random.rand()
        tau_z = np.random.rand()
        sigma_dE = dE.std()
        # Energy zero is required to mask off the random term
        energy = 0.0

        bm.synchrotron_radiation_full(dE, U0, n_kicks, tau_z, sigma_dE, energy)

        bm.use_numba()
        np.testing.assert_equal(bm.device, 'CPU_NU')
        bm.synchrotron_radiation_full(
            dE_py, U0, n_kicks, tau_z, sigma_dE, energy)

        np.testing.assert_allclose(dE_py, dE, rtol=1e-8, atol=0)

    @pytest.mark.parametrize('n_particles,n_slices,cut_left,cut_right',
                             [(100, 5, 0.01, 0.01),
                              (100000, 100, 0.0, 0.0),
                              (100000, 1000, 0.05, 0.01)])
    def test_profile_slices(self, n_particles, n_slices, cut_left, cut_right):
        bm.use_cpp()
        np.testing.assert_equal(bm.device, 'CPU_CPP')

        dt = np.random.normal(loc=1e-5, scale=1e-6, size=n_particles)
        dt_py = dt.copy()

        max_dt = dt.max()
        min_dt = dt.min()
        cut_left = (1 + cut_left) * min_dt
        cut_right = (1 - cut_right) * max_dt
        profile = np.empty(n_slices, dtype=float)

        bm.slice_beam(dt, profile, cut_left, cut_right)

        bm.use_numba()
        np.testing.assert_equal(bm.device, 'CPU_NU')
        profile_py = bm.empty(n_slices, dtype=float)
        bm.slice_beam(dt_py, profile_py, cut_left, cut_right)

        np.testing.assert_allclose(profile_py, profile, atol=1)

    def test_resonator_induced_voltage_1_turn(self):
        folder = os.path.abspath(os.path.dirname(__file__)) + "/_kwargs_resonator_induced_voltage_1_turn/"
        examples = {}
        for file in os.listdir(folder):
            print(f"{file=}")
            examples[file.replace(".npy", "")] = np.load(folder + file)

        kwargs_numba = dict(
            kappa1=examples['_kappa1'],
            n_macroparticles=examples['profile.n_macroparticles'],
            bin_centers=examples['profile.bin_centers'],
            bin_size=float(examples['profile.bin_size']),
            deltaT=np.empty((47594, 512)),
            tArray=examples['tArray'],
            reOmegaP=examples['_reOmegaP'],
            imOmegaP=examples['_imOmegaP'],
            Qtilde=examples['_Qtilde'],
            n_resonators=int(examples['n_resonators']),
            omega_r=examples['omega_r'],
            Q=examples['Q'],
            charge=float(examples['beam.Particle.charge']),
            beam_n_macroparticles=int(examples['beam.n_macroparticles']),
            ratio=float(examples['beam.ratio']),
            R=examples['R'],
            induced_voltage=examples['induced_voltage'],
            float_precision=np.float64,
            n_time=len(examples["tArray"]),
            tmp_matrix=np.empty(
                (
                    int(examples["n_resonators"]),
                    len(examples["induced_voltage"])
                )
            )
        )

        kwargs_py = deepcopy(kwargs_numba)

        induced_voltage_numba, deltaT_numba = resonator_induced_voltage_1_turn_numba(**kwargs_numba)
        induced_voltage_python, deltaT_python = resonator_induced_voltage_1_turn_python(**kwargs_py)
        assert np.allclose(deltaT_python, deltaT_numba), "Problem with calculation of deltaT"
        assert np.allclose(induced_voltage_python,
                           induced_voltage_numba), "Problem with calculation of induced_voltage"

    @pytest.mark.parametrize('n_particles,n_rf,n_iter',
                             [(100, 1, 1), (100, 4, 10),
                              (100000, 1, 10), (100000, 10, 10)])
    def test_kick(self, n_particles, n_rf, n_iter):
        bm.use_cpp()
        np.testing.assert_equal(bm.device, 'CPU_CPP')

        dE = np.random.normal(loc=0, scale=1e7, size=n_particles)
        dt = np.random.normal(loc=1e-5, scale=1e-7, size=n_particles)
        dE_py = np.array(dE)

        charge = 1.0
        acceleration_kick = 1e3 * np.random.rand()
        voltage = np.random.randn(n_rf)
        omega_rf = np.random.randn(n_rf)
        phi_rf = np.random.randn(n_rf)

        for i in range(n_iter):
            bm.kick(dt, dE, voltage, omega_rf, phi_rf, charge, n_rf,
                    acceleration_kick)

        bm.use_numba()
        np.testing.assert_equal(bm.device, 'CPU_NU')
        dt = bm.array(dt)
        voltage = bm.array(voltage)
        omega_rf = bm.array(omega_rf)
        phi_rf = bm.array(phi_rf)

        for i in range(n_iter):
            bm.kick(dt, dE_py, voltage, omega_rf, phi_rf, charge, n_rf,
                    acceleration_kick)

        np.testing.assert_allclose(dE_py, dE, rtol=1e-8, atol=0)

    @pytest.mark.parametrize('n_particles,solver,alpha_order,n_iter',
                             [(100, 'simple', 0, 1), (100, 'legacy', 1, 10),
                              (100, 'exact', 2, 10), (10000, 'simple', 1, 10),
                              (10000, 'legacy', 2, 10), (10000, 'exact', 0, 10),
                              (100000, 'simple', 2, 10), (100000, 'legacy', 0, 10),
                              (100000, 'exact', 1, 10)])
    def test_drift(self, n_particles, solver, alpha_order, n_iter):
        bm.use_cpp()
        np.testing.assert_equal(bm.device, 'CPU_CPP')

        dE = np.random.normal(loc=0, scale=1e7, size=n_particles)
        dt = np.random.normal(loc=1e-5, scale=1e-7, size=n_particles)
        dt_py = np.array(dt)

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

        bm.use_numba()
        np.testing.assert_equal(bm.device, 'CPU_NU')
        dE = bm.array(dE)

        for i in range(n_iter):
            bm.drift(dt_py, dE, solver, t_rev, length_ratio, alpha_order,
                     eta_0, eta_1, eta_2, alpha_0, alpha_1, alpha_2,
                     beta, energy)

        np.testing.assert_allclose(dt_py, dt, rtol=1e-8, atol=0)

    @pytest.mark.parametrize('n_particles,n_iter',
                             [(100, 1), (100, 10),
                              (100000, 1), (100000, 10)])
    def test_kick_drift(self, n_particles, n_iter):
        bm.use_cpp()
        np.testing.assert_equal(bm.device, 'CPU_CPP')

        solver = 'exact'
        alpha_order = 2
        n_rf = 1

        # dE = np.random.normal(loc=1e5, scale=1e2, size=n_particles)
        # dt = np.random.normal(loc=1e-5, scale=1e-7, size=n_particles)
        dE = np.random.normal(loc=0, scale=1, size=n_particles)
        dt = np.random.normal(loc=0, scale=1, size=n_particles)

        dt_py = np.array(dt)
        dE_py = np.array(dE)

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

        bm.use_numba()
        np.testing.assert_equal(bm.device, 'CPU_NU')
        voltage = bm.array(voltage)
        omega_rf = bm.array(omega_rf)
        phi_rf = bm.array(phi_rf)

        for i in range(n_iter):
            bm.drift(dt_py, dE_py, solver, t_rev, length_ratio, alpha_order,
                     eta_0, eta_1, eta_2, alpha_0, alpha_1, alpha_2,
                     beta, energy)
            bm.kick(dt_py, dE_py, voltage, omega_rf, phi_rf, charge, n_rf,
                    acceleration_kick)

        np.testing.assert_array_almost_equal(
            dt_py.mean(), dt.mean(), decimal=8)
        np.testing.assert_array_almost_equal(
            dE_py.mean(), dE.mean(), decimal=8)

    @pytest.mark.parametrize('n_particles,n_slices,n_iter',
                             [(100, 16, 10), (10000, 256, 10),
                              (100000, 100, 10)])
    def test_interp_kick(self, n_particles, n_slices, n_iter):
        bm.use_cpp()
        np.testing.assert_equal(bm.device, 'CPU_CPP')

        dE = np.random.normal(loc=0, scale=1e7, size=n_particles)
        dE_py = np.array(dE)

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

        bm.use_numba()
        np.testing.assert_equal(bm.device, 'CPU_NU')
        dt = bm.array(dt)
        voltage = bm.array(voltage)
        bin_centers = bm.array(bin_centers)

        for i in range(n_iter):
            bm.linear_interp_kick(
                dt, dE_py, voltage, bin_centers, charge, acceleration_kick)

        np.testing.assert_allclose(dE_py, dE, rtol=1e-8, atol=0)

    @pytest.mark.parametrize('n_particles,n_slices,cut_left,cut_right',
                             [(100, 5, 0.01, 0.01),
                              (100000, 100, 0.0, 0.0),
                              (100000, 1000, 0.05, 0.01)])
    def test_slice_smooth(self, n_particles, n_slices, cut_left, cut_right):
        bm.use_cpp()
        np.testing.assert_equal(bm.device, 'CPU_CPP')

        dt = np.random.normal(loc=1e-5, scale=1e-6, size=n_particles)
        dt_py = dt.copy()

        max_dt = dt.max()
        min_dt = dt.min()
        cut_left = (1 + cut_left) * min_dt
        cut_right = (1 - cut_right) * max_dt
        profile = np.empty(n_slices, dtype=float)

        bm.slice_smooth(dt, profile, cut_left, cut_right)

        bm.use_numba()
        np.testing.assert_equal(bm.device, 'CPU_NU')
        profile_py = bm.empty(n_slices, dtype=float)
        bm.slice_smooth(dt_py, profile_py, cut_left, cut_right)

        np.testing.assert_array_almost_equal(
            profile_py, profile, decimal=8)

    @pytest.mark.parametrize('size', [10, 17, 100, 256, 1000])
    def test_fast_resonator(self, size):
        bm.use_cpp()
        np.testing.assert_equal(bm.device, 'CPU_CPP')

        R_S = np.random.uniform(size=size)
        Q = np.random.uniform(low=0.5, high=1, size=size)
        frequency_array = np.linspace(1, 100, num=size)
        frequency_R = np.linspace(100, 1000, num=size)

        impedance = bm.fast_resonator(R_S, Q, frequency_array, frequency_R)

        bm.use_numba()
        np.testing.assert_equal(bm.device, 'CPU_NU')

        impedance_py = bm.fast_resonator(R_S, Q, frequency_array, frequency_R)

        np.testing.assert_array_almost_equal(impedance, impedance_py, decimal=8)


class TestWithObjects:
    # Simulation parameters -------------------------------------------------------
    # Bunch parameters
    N_p = 1000000  # Number of particles
    N_b = 1e9  # Intensity
    tau_0 = 0.4e-9  # Initial bunch length, 4 sigma [s]
    # Machine and RF parameters
    C = 26658.883  # Machine circumference [m]
    p_i = 450e9  # Synchronous momentum [eV/c]
    p_f = 460.005e9  # Synchronous momentum, final
    h = 35640  # Harmonic number
    V = 6e6  # RF voltage [V]
    dphi = 0  # Phase modulation/offset
    gamma_t = 55.759505  # Transition gamma
    alpha = 1. / gamma_t / gamma_t  # First order mom. comp. factor
    # Tracking details
    N_t = 100  # Number of turns to track

    # Run before every test
    def setup_method(self):
        if get_libblond() is None:
            pytest.skip('C++ libblond not compiled')

        self.ring = Ring(self.C, self.alpha,
                         np.linspace(self.p_i, self.p_f, self.N_t + 1),
                         Proton(), self.N_t)

        self.rf = RFStation(self.ring, [self.h],
                            self.V * np.linspace(1, 1.1, self.N_t + 1),
                            [self.dphi])
        self.beam = Beam(self.ring, self.N_p, self.N_b)
        self.beam.dE = np.random.uniform(-1e8, 1e8, self.N_p)
        self.beam.dt = np.random.normal(loc=1e-5, scale=1e-7, size=self.N_p)

        # bigaussian(self.ring, self.rf, self.beam,
        #           self.tau_0/4, reinsertion=True, seed=1)

    # Run after every test
    def teardown_method(self):
        pass

    @pytest.mark.parametrize('n_bunches,n_slices,bunch_spacing',
                             [(1, 16, 2),
                              (8, 64, 5),
                              (72, 128, 10)])
    def test_sparse_histo(self, n_bunches, n_slices, bunch_spacing):
        bm.use_cpp()
        np.testing.assert_equal(bm.device, 'CPU_CPP')

        filling_pattern = np.zeros(bunch_spacing * n_bunches)
        filling_pattern[::bunch_spacing] = 1
        bucket_length = self.C / c / self.h

        # Generate the multi-bunch beam
        indexes = np.arange(self.N_p)
        for i in range(int(np.sum(filling_pattern))):
            self.beam.dt[indexes[int(i * len(self.beam.dt) // np.sum(filling_pattern))]:
                         indexes[int((i + 1) * len(self.beam.dt) // np.sum(filling_pattern) - 1)]] += (
                    bucket_length * np.where(filling_pattern)[0][i])

        slice_beam_cpp = SparseSlices(
            self.rf, self.beam, n_slices, filling_pattern)
        slice_beam_cpp.track()

        bm.use_numba()
        np.testing.assert_equal(bm.device, 'CPU_NU')

        slice_beam_py = SparseSlices(
            self.rf, self.beam, n_slices, filling_pattern)
        slice_beam_py.track()

        np.testing.assert_array_almost_equal(
            slice_beam_py.n_macroparticles_array, slice_beam_cpp.n_macroparticles_array,
            decimal=8)

    @pytest.mark.parametrize('n_macroparticles', [100, 1000, 10000])
    def test_music_track(self, n_macroparticles):
        bm.use_cpp()
        np.testing.assert_equal(bm.device, 'CPU_CPP')

        beam = Beam(self.ring, n_macroparticles, self.N_b)
        beam.dt = np.random.normal(loc=1e-5, scale=1e-7, size=n_macroparticles)
        beam.dE = np.random.uniform(-1e8, 1e8, n_macroparticles)
        resonator = [2, 5, 0.95]

        music_cpp = Music(beam, resonator, n_macroparticles, self.N_b, self.ring.t_rev[0])

        music_cpp.track_cpp()

        bm.use_numba()
        np.testing.assert_equal(bm.device, 'CPU_NU')
        music_py = Music(beam, resonator, n_macroparticles, self.N_b, self.ring.t_rev[0])
        music_py.track_cpp()

        np.testing.assert_array_almost_equal(
            music_py.induced_voltage, music_cpp.induced_voltage, decimal=8)

    @pytest.mark.parametrize('n_macroparticles', [100, 1000, 10000])
    def test_music_track_multiturn(self, n_macroparticles):
        bm.use_cpp()
        np.testing.assert_equal(bm.device, 'CPU_CPP')

        np.testing.assert_equal(bm.device, 'CPU_CPP')
        beam = Beam(self.ring, n_macroparticles, self.N_b)
        beam.dt = np.random.normal(loc=1e-5, scale=1e-7, size=n_macroparticles)
        beam.dE = np.random.uniform(-1e8, 1e8, n_macroparticles)
        resonator = [2, 5, 0.95]

        music_cpp = Music(beam, resonator, n_macroparticles, self.N_b, self.ring.t_rev[0])

        music_cpp.track_cpp_multi_turn()

        bm.use_numba()
        np.testing.assert_equal(bm.device, 'CPU_NU')
        np.testing.assert_equal(bm.device, 'CPU_NU')

        music_py = Music(beam, resonator, n_macroparticles, self.N_b, self.ring.t_rev[0])
        music_py.track_cpp_multi_turn()

        np.testing.assert_array_almost_equal(
            music_py.induced_voltage, music_cpp.induced_voltage, decimal=8)




class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test__resonator_induced_voltage_1_turn(self):
        # TODO: implement test for `_resonator_induced_voltage_1_turn`
        _resonator_induced_voltage_1_turn(kappa1=None, n_macroparticles=None, bin_centers=None, bin_size=None, deltaT=None, tArray=None, reOmegaP=None, imOmegaP=None, Qtilde=None, n_resonators=None, omega_r=None, Q=None, charge=None, beam_n_macroparticles=None, ratio=None, R=None, induced_voltage=None)


    @unittest.skip
    def test_distribution_from_tomoscope(self):
        # TODO: implement test for `distribution_from_tomoscope`
        distribution_from_tomoscope(dt=None, dE=None, probDistr=None, seed=None, profLen=None, cutoff=None, x0=None, y0=None, dtBin=None, dEBin=None)

    @unittest.skip
    def test_linear_interp_kick(self):
        # TODO: implement test for `linear_interp_kick`
        linear_interp_kick(dt=None, dE=None, voltage=None, bin_centers=None, charge=None, acceleration_kick=None)

    @unittest.skip
    def test_set_random_seed(self):
        # TODO: implement test for `set_random_seed`
        set_random_seed(seed=None)

    @unittest.skip
    def test_slice_beam(self):
        # TODO: implement test for `slice_beam`
        slice_beam(dt=None, profile=None, cut_left=None, cut_right=None)

    @unittest.skip
    def test_sparse_histogram(self):
        # TODO: implement test for `sparse_histogram`
        sparse_histogram(dt=None, profile=None, cut_left=None, cut_right=None, bunch_indexes=None, n_slices_bucket=None)

    @unittest.skip
    def test_synchrotron_radiation(self):
        # TODO: implement test for `synchrotron_radiation`
        synchrotron_radiation(dE=None, U0=None, n_kicks=None, tau_z=None)

    @unittest.skip
    def test_synchrotron_radiation_full(self):
        # TODO: implement test for `synchrotron_radiation_full`
        synchrotron_radiation_full(dE=None, U0=None, n_kicks=None, tau_z=None, sigma_dE=None, energy=None)

if __name__ == '__main__':
    unittest.main()
