# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for trackers.tracker.py

:Authors: **Konstantinos Iliakis**
"""

import unittest
import numpy as np
import os

from blond.utils import bmath as bm
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Electron
from blond.beam.distributions import matched_from_distribution_function
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.profile import Profile
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.synchrotron_radiation.synchrotron_radiation import SynchrotronRadiation
from scipy.constants import c, e, m_e
from blond.beam.profile import CutOptions


class TestSynchRad(unittest.TestCase):
    # SIMULATION PARAMETERS -------------------------------------------------------

    # Beam parameters
    particle_type = Electron()
    n_particles = int(1.7e11)
    n_macroparticles = int(1e5)
    sync_momentum = 175e9  # [eV]

    distribution_type = 'gaussian'
    emittance = 1.0
    distribution_variable = 'Action'

    # Machine and RF parameters
    radius = 15915.49
    gamma_transition = 377.96447
    C = 2 * np.pi * radius  # [m]

    # Tracking details
    n_turns = int(100)

    # Derived parameters
    E_0 = m_e * c**2 / e    # [eV]
    tot_beam_energy = np.sqrt(sync_momentum**2 + E_0**2)  # [eV]
    momentum_compaction = 1 / gamma_transition**2  # [1]

    # Cavities parameters
    n_rf_systems = 1
    harmonic_numbers = 133650
    voltage_program = 10e9
    phi_offset = np.pi

    bucket_length = C / c / harmonic_numbers
    n_sections = 2

    # Run before every testn_turns
    def setUp(self):
        self.general_params = Ring(np.ones(self.n_sections) * self.C/self.n_sections,
                                   np.tile(self.momentum_compaction,
                                           (1, self.n_sections)).T,
                                   np.tile(self.sync_momentum,
                                           (self.n_sections, self.n_turns+1)),
                                   self.particle_type, self.n_turns, n_sections=self.n_sections)
        self.RF_sct_par = []
        self.RF_sct_par_cpp = []
        for i in np.arange(self.n_sections)+1:
            self.RF_sct_par.append(RFStation(self.general_params,
                                             [self.harmonic_numbers], [
                                                 self.voltage_program/self.n_sections],
                                             [self.phi_offset], self.n_rf_systems, section_index=i))
            self.RF_sct_par_cpp.append(RFStation(self.general_params,
                                                 [self.harmonic_numbers], [
                                                     self.voltage_program/self.n_sections],
                                                 [self.phi_offset], self.n_rf_systems, section_index=i))

        # DEFINE BEAM------------------------------------------------------------------

        self.beam = Beam(self.general_params,
                         self.n_macroparticles, self.n_particles)
        self.beam_cpp = Beam(self.general_params,
                             self.n_macroparticles, self.n_particles)

        # DEFINE SLICES----------------------------------------------------------------

        number_slices = 500

        cut_options = CutOptions(
            cut_left=0., cut_right=self.bucket_length, n_slices=number_slices)
        self.slice_beam = Profile(self.beam, CutOptions=cut_options)

        self.slice_beam_cpp = Profile(self.beam_cpp, CutOptions=cut_options)

        # DEFINE TRACKER---------------------------------------------------------------
        self.longitudinal_tracker = []
        self.longitudinal_tracker_cpp = []
        for i in range(self.n_sections):
            self.longitudinal_tracker.append(RingAndRFTracker(
                self.RF_sct_par[i], self.beam, Profile=self.slice_beam))
            self.longitudinal_tracker_cpp.append(RingAndRFTracker(
                self.RF_sct_par_cpp[i], self.beam_cpp, Profile=self.slice_beam_cpp))

        full_tracker = FullRingAndRF(self.longitudinal_tracker)
        full_tracker_cpp = FullRingAndRF(self.longitudinal_tracker_cpp)

        # BEAM GENERATION--------------------------------------------------------------

        matched_from_distribution_function(self.beam, full_tracker, emittance=self.emittance,
                                           distribution_type=self.distribution_type,
                                           distribution_variable=self.distribution_variable, seed=1000)
        matched_from_distribution_function(self.beam_cpp, full_tracker_cpp, emittance=self.emittance,
                                           distribution_type=self.distribution_type,
                                           distribution_variable=self.distribution_variable, seed=1000)

        self.slice_beam.track()
        self.slice_beam_cpp.track()

    # Run after every test

    def tearDown(self):
        pass

    def test_no_quant_exc_10t(self):
        os.environ['OMP_NUM_THREADS'] = '1'
        turns = 10
        atol = 0
        rtol_avg = 1e-7
        rtol_std = 1e-7
        SR = []
        SR_cpp = []
        rho = 11e3

        for i in range(self.n_sections):
            SR.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                           self.beam, rho,
                                           quantum_excitation=False, python=True))

            SR_cpp.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                               self.beam_cpp, rho,
                                               quantum_excitation=False, python=False))
        map_ = []
        for i in range(self.n_sections):
            map_ += [self.longitudinal_tracker[i]] + [SR[i]]
        map_ += [self.slice_beam]

        map_cpp = []
        for i in range(self.n_sections):
            map_cpp += [self.longitudinal_tracker_cpp[i]] + [SR_cpp[i]]
        map_cpp += [self.slice_beam_cpp]

        avg_dt = np.zeros(turns)
        std_dt = np.zeros(turns)
        avg_dE = np.zeros(turns)
        std_dE = np.zeros(turns)

        avg_dt_cpp = np.zeros(turns)
        std_dt_cpp = np.zeros(turns)
        avg_dE_cpp = np.zeros(turns)
        std_dE_cpp = np.zeros(turns)

        for i in range(turns):
            for m in map_:
                m.track()
            for m in map_cpp:
                m.track()
            avg_dt[i] = np.mean(self.beam.dt)
            std_dt[i] = np.std(self.beam.dt)
            avg_dE[i] = np.mean(self.beam.dE)
            std_dE[i] = np.std(self.beam.dE)

            avg_dt_cpp[i] = np.mean(self.beam_cpp.dt)
            std_dt_cpp[i] = np.std(self.beam_cpp.dt)
            avg_dE_cpp[i] = np.mean(self.beam_cpp.dE)
            std_dE_cpp[i] = np.std(self.beam_cpp.dE)

        np.testing.assert_allclose(avg_dt, avg_dt_cpp, atol=atol, rtol=rtol_avg,
                                   err_msg="Pyhton and C++ avg beam dt arrays not close")
        np.testing.assert_allclose(std_dt, std_dt_cpp, atol=atol, rtol=rtol_std,
                                   err_msg="Pyhton and C++ std beam dt arrays not close")

        np.testing.assert_allclose(avg_dE, avg_dE_cpp, atol=atol, rtol=rtol_avg,
                                   err_msg="Pyhton and C++ avg beam dE arrays not close")
        np.testing.assert_allclose(std_dE, std_dE_cpp, atol=atol, rtol=rtol_std,
                                   err_msg="Pyhton and C++ std beam dE arrays not close")

    def test_with_quant_exc_10t(self):
        os.environ['OMP_NUM_THREADS'] = '1'
        turns = 10
        atol = 0
        rtol_avg = 1e-2
        rtol_std = 1e-1
        SR = []
        SR_cpp = []
        rho = 11e3

        for i in range(self.n_sections):
            SR.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                           self.beam, rho,
                                           quantum_excitation=False, python=True))

            SR_cpp.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                               self.beam_cpp, rho,
                                               quantum_excitation=True, python=False))
        map_ = []
        for i in range(self.n_sections):
            map_ += [self.longitudinal_tracker[i]] + [SR[i]]
        map_ += [self.slice_beam]

        map_cpp = []
        for i in range(self.n_sections):
            map_cpp += [self.longitudinal_tracker_cpp[i]] + [SR_cpp[i]]
        map_cpp += [self.slice_beam_cpp]

        avg_dt = np.zeros(turns)
        std_dt = np.zeros(turns)
        avg_dE = np.zeros(turns)
        std_dE = np.zeros(turns)

        avg_dt_cpp = np.zeros(turns)
        std_dt_cpp = np.zeros(turns)
        avg_dE_cpp = np.zeros(turns)
        std_dE_cpp = np.zeros(turns)

        for i in range(turns):
            for m in map_:
                m.track()
            for m in map_cpp:
                m.track()
            avg_dt[i] = np.mean(self.beam.dt)
            std_dt[i] = np.std(self.beam.dt)
            avg_dE[i] = np.mean(self.beam.dE)
            std_dE[i] = np.std(self.beam.dE)

            avg_dt_cpp[i] = np.mean(self.beam_cpp.dt)
            std_dt_cpp[i] = np.std(self.beam_cpp.dt)
            avg_dE_cpp[i] = np.mean(self.beam_cpp.dE)
            std_dE_cpp[i] = np.std(self.beam_cpp.dE)

        np.testing.assert_allclose(avg_dt, avg_dt_cpp, atol=atol, rtol=rtol_avg,
                                   err_msg="Pyhton and C++ avg beam dt arrays not close")
        np.testing.assert_allclose(std_dt, std_dt_cpp, atol=atol, rtol=rtol_std,
                                   err_msg="Pyhton and C++ std beam dt arrays not close")

        np.testing.assert_allclose(avg_dE, avg_dE_cpp, atol=atol, rtol=rtol_avg,
                                   err_msg="Pyhton and C++ avg beam dE arrays not close")
        np.testing.assert_allclose(std_dE, std_dE_cpp, atol=atol, rtol=rtol_std,
                                   err_msg="Pyhton and C++ std beam dE arrays not close")

    def test_no_quant_exc_100t(self):
        os.environ['OMP_NUM_THREADS'] = '1'
        turns = 100
        atol = 0
        rtol_avg = 1e-7
        rtol_std = 1e-7
        SR = []
        SR_cpp = []
        rho = 11e3

        for i in range(self.n_sections):
            SR.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                           self.beam, rho,
                                           quantum_excitation=False, python=True))

            SR_cpp.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                               self.beam_cpp, rho,
                                               quantum_excitation=False, python=False))
        map_ = []
        for i in range(self.n_sections):
            map_ += [self.longitudinal_tracker[i]] + [SR[i]]
        map_ += [self.slice_beam]

        map_cpp = []
        for i in range(self.n_sections):
            map_cpp += [self.longitudinal_tracker_cpp[i]] + [SR_cpp[i]]
        map_cpp += [self.slice_beam_cpp]

        avg_dt = np.zeros(turns)
        std_dt = np.zeros(turns)
        avg_dE = np.zeros(turns)
        std_dE = np.zeros(turns)

        avg_dt_cpp = np.zeros(turns)
        std_dt_cpp = np.zeros(turns)
        avg_dE_cpp = np.zeros(turns)
        std_dE_cpp = np.zeros(turns)

        for i in range(turns):
            for m in map_:
                m.track()
            for m in map_cpp:
                m.track()
            avg_dt[i] = np.mean(self.beam.dt)
            std_dt[i] = np.std(self.beam.dt)
            avg_dE[i] = np.mean(self.beam.dE)
            std_dE[i] = np.std(self.beam.dE)

            avg_dt_cpp[i] = np.mean(self.beam_cpp.dt)
            std_dt_cpp[i] = np.std(self.beam_cpp.dt)
            avg_dE_cpp[i] = np.mean(self.beam_cpp.dE)
            std_dE_cpp[i] = np.std(self.beam_cpp.dE)

        np.testing.assert_allclose(avg_dt, avg_dt_cpp, atol=atol, rtol=rtol_avg,
                                   err_msg="Pyhton and C++ avg beam dt arrays not close")
        np.testing.assert_allclose(std_dt, std_dt_cpp, atol=atol, rtol=rtol_std,
                                   err_msg="Pyhton and C++ std beam dt arrays not close")

        np.testing.assert_allclose(avg_dE, avg_dE_cpp, atol=atol, rtol=rtol_avg,
                                   err_msg="Pyhton and C++ avg beam dE arrays not close")
        np.testing.assert_allclose(std_dE, std_dE_cpp, atol=atol, rtol=rtol_std,
                                   err_msg="Pyhton and C++ std beam dE arrays not close")

    def test_with_quant_exc_100t(self):
        os.environ['OMP_NUM_THREADS'] = '1'
        turns = 100
        atol = 0
        rtol_avg = 1e-2
        rtol_std = 1e-0
        SR = []
        SR_cpp = []
        rho = 11e3

        for i in range(self.n_sections):
            SR.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                           self.beam, rho,
                                           quantum_excitation=False, python=True))

            SR_cpp.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                               self.beam_cpp, rho,
                                               quantum_excitation=True, python=False))
        map_ = []
        for i in range(self.n_sections):
            map_ += [self.longitudinal_tracker[i]] + [SR[i]]
        map_ += [self.slice_beam]

        map_cpp = []
        for i in range(self.n_sections):
            map_cpp += [self.longitudinal_tracker_cpp[i]] + [SR_cpp[i]]
        map_cpp += [self.slice_beam_cpp]

        avg_dt = np.zeros(turns)
        std_dt = np.zeros(turns)
        avg_dE = np.zeros(turns)
        std_dE = np.zeros(turns)

        avg_dt_cpp = np.zeros(turns)
        std_dt_cpp = np.zeros(turns)
        avg_dE_cpp = np.zeros(turns)
        std_dE_cpp = np.zeros(turns)

        for i in range(turns):
            for m in map_:
                m.track()
            for m in map_cpp:
                m.track()
            avg_dt[i] = np.mean(self.beam.dt)
            std_dt[i] = np.std(self.beam.dt)
            avg_dE[i] = np.mean(self.beam.dE)
            std_dE[i] = np.std(self.beam.dE)

            avg_dt_cpp[i] = np.mean(self.beam_cpp.dt)
            std_dt_cpp[i] = np.std(self.beam_cpp.dt)
            avg_dE_cpp[i] = np.mean(self.beam_cpp.dE)
            std_dE_cpp[i] = np.std(self.beam_cpp.dE)

        np.testing.assert_allclose(avg_dt, avg_dt_cpp, atol=atol, rtol=rtol_avg,
                                   err_msg="Pyhton and C++ avg beam dt arrays not close")
        np.testing.assert_allclose(std_dt, std_dt_cpp, atol=atol, rtol=rtol_std,
                                   err_msg="Pyhton and C++ std beam dt arrays not close")

        np.testing.assert_allclose(avg_dE, avg_dE_cpp, atol=atol, rtol=rtol_avg,
                                   err_msg="Pyhton and C++ avg beam dE arrays not close")
        np.testing.assert_allclose(std_dE, std_dE_cpp, atol=atol, rtol=rtol_std,
                                   err_msg="Pyhton and C++ std beam dE arrays not close")

    def test_no_quant_exc_10t_parallel(self):
        os.environ['OMP_NUM_THREADS'] = '2'
        turns = 10
        atol = 0
        rtol_avg = 1e-6
        rtol_std = 1e-7
        SR = []
        SR_cpp = []
        rho = 11e3

        for i in range(self.n_sections):
            SR.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                           self.beam, rho,
                                           quantum_excitation=False, python=True))

            SR_cpp.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                               self.beam_cpp, rho,
                                               quantum_excitation=False, python=False))
        map_ = []
        for i in range(self.n_sections):
            map_ += [self.longitudinal_tracker[i]] + [SR[i]]
        map_ += [self.slice_beam]

        map_cpp = []
        for i in range(self.n_sections):
            map_cpp += [self.longitudinal_tracker_cpp[i]] + [SR_cpp[i]]
        map_cpp += [self.slice_beam_cpp]

        avg_dt = np.zeros(turns)
        std_dt = np.zeros(turns)
        avg_dE = np.zeros(turns)
        std_dE = np.zeros(turns)

        avg_dt_cpp = np.zeros(turns)
        std_dt_cpp = np.zeros(turns)
        avg_dE_cpp = np.zeros(turns)
        std_dE_cpp = np.zeros(turns)

        for i in range(turns):
            for m in map_:
                m.track()
            for m in map_cpp:
                m.track()
            avg_dt[i] = np.mean(self.beam.dt)
            std_dt[i] = np.std(self.beam.dt)
            avg_dE[i] = np.mean(self.beam.dE)
            std_dE[i] = np.std(self.beam.dE)

            avg_dt_cpp[i] = np.mean(self.beam_cpp.dt)
            std_dt_cpp[i] = np.std(self.beam_cpp.dt)
            avg_dE_cpp[i] = np.mean(self.beam_cpp.dE)
            std_dE_cpp[i] = np.std(self.beam_cpp.dE)

        np.testing.assert_allclose(avg_dt, avg_dt_cpp, atol=atol, rtol=rtol_avg,
                                   err_msg="Pyhton and C++ avg beam dt arrays not close")
        np.testing.assert_allclose(std_dt, std_dt_cpp, atol=atol, rtol=rtol_std,
                                   err_msg="Pyhton and C++ std beam dt arrays not close")

        np.testing.assert_allclose(avg_dE, avg_dE_cpp, atol=atol, rtol=rtol_avg,
                                   err_msg="Pyhton and C++ avg beam dE arrays not close")
        np.testing.assert_allclose(std_dE, std_dE_cpp, atol=atol, rtol=rtol_std,
                                   err_msg="Pyhton and C++ std beam dE arrays not close")

    def test_with_quant_exc_10t_parallel(self):
        os.environ['OMP_NUM_THREADS'] = '2'
        turns = 10
        atol = 0
        rtol_avg = 1e-3
        rtol_std = 1e-1
        SR = []
        SR_cpp = []
        rho = 11e3

        for i in range(self.n_sections):
            SR.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                           self.beam, rho,
                                           quantum_excitation=False, python=True))

            SR_cpp.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                               self.beam_cpp, rho,
                                               quantum_excitation=True, python=False))
        map_ = []
        for i in range(self.n_sections):
            map_ += [self.longitudinal_tracker[i]] + [SR[i]]
        map_ += [self.slice_beam]

        map_cpp = []
        for i in range(self.n_sections):
            map_cpp += [self.longitudinal_tracker_cpp[i]] + [SR_cpp[i]]
        map_cpp += [self.slice_beam_cpp]

        avg_dt = np.zeros(turns)
        std_dt = np.zeros(turns)
        avg_dE = np.zeros(turns)
        std_dE = np.zeros(turns)

        avg_dt_cpp = np.zeros(turns)
        std_dt_cpp = np.zeros(turns)
        avg_dE_cpp = np.zeros(turns)
        std_dE_cpp = np.zeros(turns)

        for i in range(turns):
            for m in map_:
                m.track()
            for m in map_cpp:
                m.track()
            avg_dt[i] = np.mean(self.beam.dt)
            std_dt[i] = np.std(self.beam.dt)
            avg_dE[i] = np.mean(self.beam.dE)
            std_dE[i] = np.std(self.beam.dE)

            avg_dt_cpp[i] = np.mean(self.beam_cpp.dt)
            std_dt_cpp[i] = np.std(self.beam_cpp.dt)
            avg_dE_cpp[i] = np.mean(self.beam_cpp.dE)
            std_dE_cpp[i] = np.std(self.beam_cpp.dE)

        np.testing.assert_allclose(avg_dt, avg_dt_cpp, atol=atol, rtol=rtol_avg,
                                   err_msg="Pyhton and C++ avg beam dt arrays not close")
        np.testing.assert_allclose(std_dt, std_dt_cpp, atol=atol, rtol=rtol_std,
                                   err_msg="Pyhton and C++ std beam dt arrays not close")

        np.testing.assert_allclose(avg_dE, avg_dE_cpp, atol=atol, rtol=rtol_avg,
                                   err_msg="Pyhton and C++ avg beam dE arrays not close")
        np.testing.assert_allclose(std_dE, std_dE_cpp, atol=atol, rtol=rtol_std,
                                   err_msg="Pyhton and C++ std beam dE arrays not close")

    def test_no_quant_exc_100t_parallel(self):
        os.environ['OMP_NUM_THREADS'] = '2'
        turns = 100
        atol = 0
        rtol_avg = 1e-7
        rtol_std = 1e-7
        SR = []
        SR_cpp = []
        rho = 11e3

        for i in range(self.n_sections):
            SR.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                           self.beam, rho,
                                           quantum_excitation=False, python=True))

            SR_cpp.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                               self.beam_cpp, rho,
                                               quantum_excitation=False, python=False))
        map_ = []
        for i in range(self.n_sections):
            map_ += [self.longitudinal_tracker[i]] + [SR[i]]
        map_ += [self.slice_beam]

        map_cpp = []
        for i in range(self.n_sections):
            map_cpp += [self.longitudinal_tracker_cpp[i]] + [SR_cpp[i]]
        map_cpp += [self.slice_beam_cpp]

        avg_dt = np.zeros(turns)
        std_dt = np.zeros(turns)
        avg_dE = np.zeros(turns)
        std_dE = np.zeros(turns)

        avg_dt_cpp = np.zeros(turns)
        std_dt_cpp = np.zeros(turns)
        avg_dE_cpp = np.zeros(turns)
        std_dE_cpp = np.zeros(turns)

        for i in range(turns):
            for m in map_:
                m.track()
            for m in map_cpp:
                m.track()
            avg_dt[i] = np.mean(self.beam.dt)
            std_dt[i] = np.std(self.beam.dt)
            avg_dE[i] = np.mean(self.beam.dE)
            std_dE[i] = np.std(self.beam.dE)

            avg_dt_cpp[i] = np.mean(self.beam_cpp.dt)
            std_dt_cpp[i] = np.std(self.beam_cpp.dt)
            avg_dE_cpp[i] = np.mean(self.beam_cpp.dE)
            std_dE_cpp[i] = np.std(self.beam_cpp.dE)

        np.testing.assert_allclose(avg_dt, avg_dt_cpp, atol=atol, rtol=rtol_avg,
                                   err_msg="Pyhton and C++ avg beam dt arrays not close")
        np.testing.assert_allclose(std_dt, std_dt_cpp, atol=atol, rtol=rtol_std,
                                   err_msg="Pyhton and C++ std beam dt arrays not close")

        np.testing.assert_allclose(avg_dE, avg_dE_cpp, atol=atol, rtol=rtol_avg,
                                   err_msg="Pyhton and C++ avg beam dE arrays not close")
        np.testing.assert_allclose(std_dE, std_dE_cpp, atol=atol, rtol=rtol_std,
                                   err_msg="Pyhton and C++ std beam dE arrays not close")

    def test_with_quant_exc_100t_parallel(self):
        os.environ['OMP_NUM_THREADS'] = '2'
        turns = 100
        atol = 0
        rtol_avg = 1e-2
        rtol_std = 1e-0
        SR = []
        SR_cpp = []
        rho = 11e3

        for i in range(self.n_sections):
            SR.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                           self.beam, rho,
                                           quantum_excitation=False, python=True))

            SR_cpp.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                               self.beam_cpp, rho,
                                               quantum_excitation=True, python=False))
        map_ = []
        for i in range(self.n_sections):
            map_ += [self.longitudinal_tracker[i]] + [SR[i]]
        map_ += [self.slice_beam]

        map_cpp = []
        for i in range(self.n_sections):
            map_cpp += [self.longitudinal_tracker_cpp[i]] + [SR_cpp[i]]
        map_cpp += [self.slice_beam_cpp]

        avg_dt = np.zeros(turns)
        std_dt = np.zeros(turns)
        avg_dE = np.zeros(turns)
        std_dE = np.zeros(turns)

        avg_dt_cpp = np.zeros(turns)
        std_dt_cpp = np.zeros(turns)
        avg_dE_cpp = np.zeros(turns)
        std_dE_cpp = np.zeros(turns)

        for i in range(turns):
            for m in map_:
                m.track()
            for m in map_cpp:
                m.track()
            avg_dt[i] = np.mean(self.beam.dt)
            std_dt[i] = np.std(self.beam.dt)
            avg_dE[i] = np.mean(self.beam.dE)
            std_dE[i] = np.std(self.beam.dE)

            avg_dt_cpp[i] = np.mean(self.beam_cpp.dt)
            std_dt_cpp[i] = np.std(self.beam_cpp.dt)
            avg_dE_cpp[i] = np.mean(self.beam_cpp.dE)
            std_dE_cpp[i] = np.std(self.beam_cpp.dE)

        np.testing.assert_allclose(avg_dt, avg_dt_cpp, atol=atol, rtol=rtol_avg,
                                   err_msg="Pyhton and C++ avg beam dt arrays not close")
        np.testing.assert_allclose(std_dt, std_dt_cpp, atol=atol, rtol=rtol_std,
                                   err_msg="Pyhton and C++ std beam dt arrays not close")

        np.testing.assert_allclose(avg_dE, avg_dE_cpp, atol=atol, rtol=rtol_avg,
                                   err_msg="Pyhton and C++ avg beam dE arrays not close")
        np.testing.assert_allclose(std_dE, std_dE_cpp, atol=atol, rtol=rtol_std,
                                   err_msg="Pyhton and C++ std beam dE arrays not close")


if __name__ == '__main__':

    unittest.main()
