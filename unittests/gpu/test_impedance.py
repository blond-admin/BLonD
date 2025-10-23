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


import os
import unittest

import numpy as np
import pytest
from scipy.constants import c, e, m_p

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, FitOptions, Profile
from blond.impedances.impedance import (InducedVoltageFreq,
                                        InducedVoltageResonator,
                                        InducedVoltageTime,
                                        TotalInducedVoltage)
from blond.impedances.impedance_sources import Resonators
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.trackers.tracker import RingAndRFTracker
from blond.utils import bmath as bm

this_directory = os.path.dirname(os.path.realpath(__file__))


class TestImpedanceBigaussianData:

    # Simulation parameters -------------------------------------------------------
    # Beam parameters
    n_particles = 1e10
    n_macroparticles = 5 * 1e6
    tau_0 = 2e-9  # [s]

    # Machine and RF parameters
    gamma_transition = 1 / np.sqrt(0.00192)   # [1]
    C = 6911.56  # [m]

    # Tracking details
    n_turns = 1000

    # Derived parameters
    sync_momentum = 25.92e9  # [eV / c]
    momentum_compaction = 1 / gamma_transition**2  # [1]

    # Cavities parameters
    n_rf_systems = 1
    harmonic_number = 4620
    voltage_program = 0.9e6  # [V]
    phi_offset = 0.0

    # Run before every test
    def setup_method(self):
        # Try to import cupy, skip if not found
        pytest.importorskip('cupy')

        self.general_params = Ring(self.C, self.momentum_compaction,
                                   self.sync_momentum, Proton(), self.n_turns)

        self.rf = RFStation(self.general_params, [self.harmonic_number],
                            [self.voltage_program], [self.phi_offset], self.n_rf_systems)

        self.rf_gpu = RFStation(self.general_params, [self.harmonic_number],
                                [self.voltage_program], [self.phi_offset], self.n_rf_systems)

        self.beam = Beam(self.general_params,
                         self.n_macroparticles, self.n_particles)
        bigaussian(self.general_params, self.rf,
                   self.beam, self.tau_0 / 4, seed=1)

        self.beam_gpu = Beam(self.general_params,
                             self.n_macroparticles, self.n_particles)
        bigaussian(self.general_params, self.rf_gpu,
                   self.beam_gpu, self.tau_0 / 4, seed=1)

        self.rf_tracker = RingAndRFTracker(self.rf, self.beam)
        self.rf_tracker_gpu = RingAndRFTracker(self.rf_gpu, self.beam_gpu)

        # LOAD IMPEDANCE TABLE--------------------------------------------------------
        table = np.loadtxt(os.path.join(
            this_directory, '../../__EXAMPLES/input_files/EX_05_new_HQ_table.dat'), comments='!')

        R_shunt = table[:, 2] * 10**6
        f_res = table[:, 0] * 10**9
        Q_factor = table[:, 1]
        self.resonator = Resonators(R_shunt, f_res, Q_factor)

    # Run after every test
    def teardown_method(self):
        bm.use_cpu()

    @pytest.mark.parametrize('number_slices,mode',
                             [(256, 'time'), (1024, 'time'), (16384, 'time'),
                              (256, 'frequency'), (1024, 'frequency'),
                              (16384, 'frequency'), (256, 'resonator'),
                              (1024, 'resonator')])
    def test_ind_volt(self, number_slices, mode):
        import cupy as cp

        cut_options = CutOptions(cut_left=0, cut_right=2 * np.pi, n_slices=number_slices,
                                 RFSectionParameters=self.rf, cuts_unit='rad')
        slice_beam = Profile(self.beam, cut_options,
                             FitOptions(fit_option='gaussian'))

        cut_options_gpu = CutOptions(cut_left=0, cut_right=2 * np.pi, n_slices=number_slices,
                                     RFSectionParameters=self.rf_gpu, cuts_unit='rad')
        slice_beam_gpu = Profile(
            self.beam_gpu, cut_options_gpu, FitOptions(fit_option='gaussian'))

        slice_beam.track()
        slice_beam_gpu.track()

        if mode == 'time':
            ind_volt = InducedVoltageTime(
                self.beam, slice_beam, [self.resonator])
            ind_volt_gpu = InducedVoltageTime(
                self.beam_gpu, slice_beam_gpu, [self.resonator])

            cp.testing.assert_allclose(ind_volt_gpu.total_impedance,
                                       ind_volt.total_impedance, rtol=1e-8, atol=0,
                                       err_msg='Checking initial conditions')
        elif mode == 'frequency':
            ind_volt = InducedVoltageFreq(
                self.beam, slice_beam, [self.resonator], 1e5)
            ind_volt_gpu = InducedVoltageFreq(
                self.beam_gpu, slice_beam_gpu, [self.resonator], 1e5)

            cp.testing.assert_allclose(ind_volt_gpu.total_impedance,
                                       ind_volt.total_impedance, rtol=1e-8, atol=0,
                                       err_msg='Checking initial conditions')

        elif mode == 'resonator':
            ind_volt = InducedVoltageResonator(
                self.beam, slice_beam, self.resonator)
            ind_volt_gpu = InducedVoltageResonator(self.beam_gpu,
                                                   slice_beam_gpu, self.resonator)

        tot_volt = TotalInducedVoltage(self.beam, slice_beam, [ind_volt])
        tot_volt_gpu = TotalInducedVoltage(
            self.beam_gpu, slice_beam_gpu, [ind_volt_gpu])

        cp.testing.assert_allclose(self.beam_gpu.dE, self.beam.dE, rtol=1e-8, atol=0,
                                   err_msg='Checking initial conditions')
        cp.testing.assert_allclose(self.beam_gpu.dt, self.beam.dt, rtol=1e-8, atol=0,
                                   err_msg='Checking initial conditions')

        cp.testing.assert_allclose(slice_beam_gpu.n_macroparticles,
                                   slice_beam.n_macroparticles, rtol=1e-8, atol=0,
                                   err_msg='Checking initial conditions')

        tot_volt.induced_voltage_sum()

        bm.use_gpu()
        tot_volt_gpu.to_gpu()
        slice_beam_gpu.to_gpu()

        tot_volt_gpu.induced_voltage_sum()

        cp.testing.assert_allclose(tot_volt_gpu.induced_voltage,
                                   tot_volt.induced_voltage, rtol=1e-8, atol=1e-6)

    @pytest.mark.parametrize('number_slices,mode,n_iter',
                             [(256, 'time', 1), (1024, 'time', 100),
                              (256, 'frequency', 1), (1024, 'frequency', 100)])
    def test_ind_volt_track(self, number_slices, mode, n_iter):
        import cupy as cp

        cut_options = CutOptions(cut_left=0, cut_right=2 * np.pi, n_slices=number_slices,
                                 RFSectionParameters=self.rf, cuts_unit='rad')
        slice_beam = Profile(self.beam, cut_options,
                             FitOptions(fit_option='gaussian'))

        cut_options_gpu = CutOptions(cut_left=0, cut_right=2 * np.pi, n_slices=number_slices,
                                     RFSectionParameters=self.rf_gpu, cuts_unit='rad')
        slice_beam_gpu = Profile(
            self.beam_gpu, cut_options_gpu, FitOptions(fit_option='gaussian'))

        slice_beam.track()
        slice_beam_gpu.track()

        if mode == 'time':
            ind_volt = InducedVoltageTime(
                self.beam, slice_beam, [self.resonator])
            ind_volt_gpu = InducedVoltageTime(
                self.beam_gpu, slice_beam_gpu, [self.resonator])

            cp.testing.assert_allclose(ind_volt_gpu.total_impedance,
                                       ind_volt.total_impedance, rtol=1e-8, atol=0,
                                       err_msg='Checking initial conditions')
        elif mode == 'frequency':
            ind_volt = InducedVoltageFreq(
                self.beam, slice_beam, [self.resonator], 1e5)
            ind_volt_gpu = InducedVoltageFreq(
                self.beam_gpu, slice_beam_gpu, [self.resonator], 1e5)

            cp.testing.assert_allclose(ind_volt_gpu.total_impedance,
                                       ind_volt.total_impedance, rtol=1e-8, atol=0,
                                       err_msg='Checking initial conditions')

        elif mode == 'resonator':
            ind_volt = InducedVoltageResonator(
                self.beam, slice_beam, self.resonator)
            ind_volt_gpu = InducedVoltageResonator(self.beam_gpu,
                                                   slice_beam_gpu, self.resonator)

        tot_volt = TotalInducedVoltage(self.beam, slice_beam, [ind_volt])
        tot_volt_gpu = TotalInducedVoltage(
            self.beam_gpu, slice_beam_gpu, [ind_volt_gpu])

        cp.testing.assert_allclose(self.beam_gpu.dE, self.beam.dE, rtol=1e-8, atol=0,
                                   err_msg='Checking initial conditions')
        cp.testing.assert_allclose(self.beam_gpu.dt, self.beam.dt, rtol=1e-8, atol=0,
                                   err_msg='Checking initial conditions')

        cp.testing.assert_allclose(slice_beam_gpu.n_macroparticles,
                                   slice_beam.n_macroparticles, rtol=1e-8, atol=0,
                                   err_msg='Checking initial conditions')

        for i in range(n_iter):
            tot_volt.track()
            self.rf_tracker.track()
            slice_beam.track()

        bm.use_gpu()
        tot_volt_gpu.to_gpu()
        slice_beam_gpu.to_gpu()
        self.rf_tracker_gpu.to_gpu()

        for i in range(n_iter):
            tot_volt_gpu.track()
            self.rf_tracker_gpu.track()
            slice_beam_gpu.track()

        cp.testing.assert_allclose(slice_beam_gpu.n_macroparticles,
                                   slice_beam.n_macroparticles, rtol=1e-8, atol=0)

        cp.testing.assert_allclose(tot_volt_gpu.induced_voltage,
                                   tot_volt.induced_voltage, rtol=1e-8, atol=1e-6)

        cp.testing.assert_allclose(
            self.beam_gpu.dE, self.beam.dE, rtol=1e-7, atol=0)
        cp.testing.assert_allclose(
            self.beam_gpu.dt, self.beam.dt, rtol=1e-7, atol=0)


class TestImpedanceMTW:

    # Simulation parameters -------------------------------------------------------
    # Beam parameters
    n_particles = 1e11
    n_macroparticles = 5e5
    sigma_dt = 180e-9 / 4  # [s]
    kin_beam_energy = 1.4e9  # [eV]

    # Machine and RF parameters
    radius = 25.0
    gamma_transition = 4.4
    C = 2 * np.pi * radius  # [m]

    # Tracking details
    n_turns = 1000

    # Derived parameters
    E_0 = m_p * c**2 / e    # [eV]
    tot_beam_energy = E_0 + kin_beam_energy  # [eV]
    sync_momentum = np.sqrt(tot_beam_energy**2 - E_0**2)  # [eV/c]

    gamma = tot_beam_energy / E_0
    beta = np.sqrt(1.0 - 1.0 / gamma**2.0)

    momentum_compaction = 1 / gamma_transition**2

    # Cavities parameters
    n_rf_systems = 1
    harmonic_numbers = 1
    voltage_program = 8e3  # [V]
    phi_offset = np.pi

    # Run before every test

    def setup_method(self):
        # Try to import cupy, skip if not found
        pytest.importorskip('cupy')

        self.general_params = Ring(self.C, self.momentum_compaction,
                                   self.sync_momentum, Proton(), self.n_turns)

        self.rf = RFStation(self.general_params, [self.harmonic_numbers],
                            [self.voltage_program], [self.phi_offset], self.n_rf_systems)

        self.rf_gpu = RFStation(self.general_params, [self.harmonic_numbers],
                                [self.voltage_program], [self.phi_offset], self.n_rf_systems)

        self.bucket_length = 2.0 * np.pi / self.rf.omega_rf[0, 0]

        self.beam = Beam(self.general_params,
                         self.n_macroparticles, self.n_particles)
        bigaussian(self.general_params, self.rf,
                   self.beam, self.sigma_dt, seed=1)

        self.beam_gpu = Beam(self.general_params,
                             self.n_macroparticles, self.n_particles)
        bigaussian(self.general_params, self.rf_gpu,
                   self.beam_gpu, self.sigma_dt, seed=1)

        self.rf_tracker = RingAndRFTracker(self.rf, self.beam)
        self.rf_tracker_gpu = RingAndRFTracker(self.rf_gpu, self.beam_gpu)

        # LOAD IMPEDANCE TABLE--------------------------------------------------------
        R_S = 5e3
        frequency_R = 10e6
        Q = 10

        self.resonator = Resonators(R_S, frequency_R, Q)

    # Run after every test
    def teardown_method(self):
        bm.use_cpu()

    @pytest.mark.parametrize('number_slices,mode,mtw_mode,n_iter',
                             [(256, 'time', 'time', 1), (1024, 'time', 'time', 100),
                              #   (256, 'time', 'freq', 1), (1024, 'time', 'freq', 100),
                              (256, 'frequency', 'time', 1), (1024, 'frequency', 'time', 100),
                              #   (256, 'frequency', 'freq', 1), (1024, 'frequency', 'freq', 100)
                              ])
    def test_ind_volt_mtw(self, number_slices, mode, mtw_mode, n_iter):
        import cupy as cp

        cut_options = CutOptions(cut_left=0, cut_right=self.bucket_length,
                                 n_slices=number_slices)
        slice_beam = Profile(self.beam, cut_options)

        cut_options_gpu = CutOptions(cut_left=0, cut_right=self.bucket_length,
                                     n_slices=number_slices)
        slice_beam_gpu = Profile(self.beam_gpu, cut_options_gpu)

        slice_beam.track()
        slice_beam_gpu.track()

        if mode == 'time':
            ind_volt = InducedVoltageTime(
                self.beam, slice_beam, [self.resonator],
                RFParams=self.rf, multi_turn_wake=True, mtw_mode=mtw_mode,
                wake_length=self.n_turns * self.bucket_length)
            ind_volt_gpu = InducedVoltageTime(
                self.beam_gpu, slice_beam_gpu, [self.resonator],
                RFParams=self.rf_gpu, multi_turn_wake=True, mtw_mode=mtw_mode,
                wake_length=self.n_turns * self.bucket_length)

        elif mode == 'frequency':
            ind_volt = InducedVoltageFreq(
                self.beam, slice_beam, [self.resonator],
                RFParams=self.rf, frequency_resolution=1e3,
                multi_turn_wake=True, mtw_mode=mtw_mode)
            ind_volt_gpu = InducedVoltageFreq(
                self.beam_gpu, slice_beam_gpu, [self.resonator],
                RFParams=self.rf_gpu, frequency_resolution=1e3,
                multi_turn_wake=True, mtw_mode=mtw_mode)

        tot_volt = TotalInducedVoltage(self.beam, slice_beam, [ind_volt])
        tot_volt_gpu = TotalInducedVoltage(
            self.beam_gpu, slice_beam_gpu, [ind_volt_gpu])

        cp.testing.assert_allclose(self.beam_gpu.dE, self.beam.dE, rtol=1e-8, atol=0,
                                   err_msg='Checking initial conditions')
        cp.testing.assert_allclose(self.beam_gpu.dt, self.beam.dt, rtol=1e-8, atol=0,
                                   err_msg='Checking initial conditions')

        cp.testing.assert_allclose(slice_beam_gpu.n_macroparticles,
                                   slice_beam.n_macroparticles, rtol=1e-8, atol=0,
                                   err_msg='Checking initial conditions')

        cp.testing.assert_allclose(ind_volt_gpu.total_impedance,
                                   ind_volt.total_impedance, rtol=1e-8, atol=0,
                                   err_msg='Checking initial conditions')

        for i in range(n_iter):
            tot_volt.track()
            self.rf_tracker.track()
            slice_beam.track()

        bm.use_gpu()
        tot_volt_gpu.to_gpu()
        slice_beam_gpu.to_gpu()
        self.rf_tracker_gpu.to_gpu()

        for i in range(n_iter):
            tot_volt_gpu.track()
            self.rf_tracker_gpu.track()
            slice_beam_gpu.track()

        cp.testing.assert_allclose(slice_beam_gpu.n_macroparticles,
                                   slice_beam.n_macroparticles, rtol=1e-8, atol=0)

        cp.testing.assert_allclose(tot_volt_gpu.induced_voltage,
                                   tot_volt.induced_voltage, rtol=1e-8, atol=1e-6)

        cp.testing.assert_allclose(
            self.beam_gpu.dE, self.beam.dE, rtol=1e-7, atol=0)
        cp.testing.assert_allclose(
            self.beam_gpu.dt, self.beam.dt, rtol=1e-7, atol=0)


if __name__ == '__main__':
    unittest.main()
