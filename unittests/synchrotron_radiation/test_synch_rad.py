# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for synchrotron_radiation.synchrotron_radiation.py

:Authors: **Markus Schwarz, Konstantinos Iliakis**
"""

import unittest
import numpy as np
import os

from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Electron, Positron
from blond.beam.distributions import bigaussian, matched_from_distribution_function
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.profile import Profile
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.synchrotron_radiation.synchrotron_radiation import SynchrotronRadiation
from scipy.constants import c, e, m_e
from blond.beam.profile import CutOptions


class TestSynchtrotronRadiation(unittest.TestCase):

    # Run before every test
    def setUp(self):
        circumference = 110.4  # [m]
        energy = 2.5e9  # [eV]
        alpha = 0.0082
        self.R_bend = 5.559  # bending radius [m]
        # C_gamma = e**2 / (3*epsilon_0 * (m_e*c**2)**4)  # [m J^3]
        # C_gamma *= e**3  # [m eV^3]

        harmonic_number = 184
        voltage = 800e3  # eV
        phi_offsets = 0

        self.seed = 1234
        self.intensity = 2.299e9
        self.n_macroparticles = int(1e2)
        self.sigma_dt = 10e-12  # RMS, [s]

        self.ring = Ring(circumference, alpha, energy, Positron(),
                         synchronous_data_type='total energy', n_turns=1)

        self.rf_station = RFStation(self.ring, harmonic_number, voltage,
                                    phi_offsets, n_rf=1)

        self.beam = Beam(self.ring, self.n_macroparticles, self.intensity)

        bigaussian(self.ring, self.rf_station, self.beam,
                   self.sigma_dt, seed=self.seed)

        # # energy loss per turn [eV]; assuming isomagnetic lattice
        # self.U0 = C_gamma * self.ring.beta[0,0]**3 * self.ring.energy[0,0]**4 / self.R_bend

    def test_initial_beam(self):
        atol = 0
        rtol = 1e-7
        np.testing.assert_allclose([np.mean(self.beam.dt)], [1.0019653703433631e-09],
                                   atol=atol, rtol=rtol,
                                   err_msg='Initial avg beam.dt wrong')

        np.testing.assert_allclose([np.std(self.beam.dt)], [1.1204522311616268e-11],
                                   atol=atol, rtol=rtol,
                                   err_msg='Initial std beam.dt wrong')

        np.testing.assert_allclose([np.mean(self.beam.dE)], [10748.973035173622],
                                   atol=atol, rtol=rtol,
                                   err_msg='Initial avg beam.dE wrong')
        np.testing.assert_allclose([np.std(self.beam.dE)], [447988.3517788615],
                                   atol=atol, rtol=rtol,
                                   err_msg='Initial std beam.dE wrong')

    def test_affect_only_dE(self):
        atol = 0
        rtol = 1e-7

        # incoherent synchrotron radiation, no displacement of beam
        iSR = SynchrotronRadiation(self.ring, self.rf_station, self.beam, self.R_bend,
                                   seed=self.seed, n_kicks=1, shift_beam=False,
                                   python=True, quantum_excitation=False)
        iSR.track()
        np.testing.assert_allclose([np.mean(self.beam.dt)], [1.0019653703433631e-09],
                                   atol=atol, rtol=rtol,
                                   err_msg='SR affected mean beam.dt')
        np.testing.assert_allclose([np.std(self.beam.dt)], [1.1204522311616268e-11],
                                   atol=atol, rtol=rtol,
                                   err_msg='SR affected std beam.dt')

        # np.testing.assert_almost_equal(
        #     self.beam.dt[0], 1.0054066581358374e-09, decimal=10,
        #     err_msg='SR affected beam.dt')

    def test_synchrotron_radiation_python_vs_C(self):
        atol = 0
        rtol = 1e-7

        iSR = SynchrotronRadiation(self.ring, self.rf_station, self.beam, self.R_bend,
                                   n_kicks=1, shift_beam=False,
                                   python=True, quantum_excitation=False, seed=self.seed)
        iSR.track()  # Python implementation

        beam_C = Beam(self.ring, self.n_macroparticles, self.intensity)
        bigaussian(self.ring, self.rf_station, beam_C,
                   self.sigma_dt, seed=self.seed)

        iSR = SynchrotronRadiation(self.ring, self.rf_station, beam_C, self.R_bend,
                                   n_kicks=1, shift_beam=False,
                                   python=False, quantum_excitation=False, seed=self.seed)
        iSR.track()  # C implementation

        np.testing.assert_allclose([np.mean(self.beam.dE)], [np.mean(beam_C.dE)],
                                   atol=atol, rtol=rtol,
                                   err_msg='Python anc C yield different avg beam.dE for single kick')
        np.testing.assert_allclose([np.std(self.beam.dE)], [np.std(beam_C.dE)],
                                   atol=atol, rtol=rtol,
                                   err_msg='Python anc C yield different std beam.dE for single kick')

    def test_synchrotron_radiation_python_vs_C_double_kick(self):
        atol = 0
        rtol = 1e-7

        iSR = SynchrotronRadiation(self.ring, self.rf_station, self.beam, self.R_bend,
                                   n_kicks=2, shift_beam=False,
                                   python=True, quantum_excitation=False, seed=self.seed)
        iSR.track()  # Python implementation

        beam_C = Beam(self.ring, self.n_macroparticles, self.intensity)
        bigaussian(self.ring, self.rf_station, beam_C,
                   self.sigma_dt, seed=self.seed)

        iSR = SynchrotronRadiation(self.ring, self.rf_station, beam_C, self.R_bend,
                                   n_kicks=2, shift_beam=False,
                                   python=False, quantum_excitation=False, seed=self.seed)
        iSR.track()  # C implementation

        np.testing.assert_allclose([np.mean(self.beam.dE)], [np.mean(beam_C.dE)],
                                   atol=atol, rtol=rtol,
                                   err_msg='Python anc C yield different avg beam.dE for two kicks')
        np.testing.assert_allclose([np.std(self.beam.dE)], [np.std(beam_C.dE)],
                                   atol=atol, rtol=rtol,
                                   err_msg='Python anc C yield different std beam.dE for two kicks')

    def test_U0(self):

        # LEP, values from S. Lee 2nd ed., table 4.2
        circumference = 26658.9  # [m]
        energy = 55e9  # [eV]
        R_bend = 3096.2  # bending radius [m]
        alpha = 1e-3  # dummy value

        ring = Ring(circumference, alpha, energy, Positron(),
                    synchronous_data_type='total energy', n_turns=1)

        rf_station_dummy = RFStation(ring, 42, 1e6, 0, n_rf=1)

        iSR = SynchrotronRadiation(ring, rf_station_dummy, None, R_bend, shift_beam=False,
                                   quantum_excitation=False)

        self.assertEqual(int(iSR.U0 / 1e6), 261, msg="Wrong U0")


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
    n_turns = int(200)

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
    rho = 11e3

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
                                                 [self.phi_offset], self.n_rf_systems,
                                                 section_index=i))

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
                                           distribution_variable=self.distribution_variable,
                                           seed=1000)

        matched_from_distribution_function(self.beam_cpp, full_tracker_cpp,
                                           emittance=self.emittance, seed=1000,
                                           distribution_type=self.distribution_type,
                                           distribution_variable=self.distribution_variable)

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

        for i in range(self.n_sections):
            SR.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                           self.beam, self.rho,
                                           quantum_excitation=False, python=True))

            SR_cpp.append(SynchrotronRadiation(self.general_params, self.RF_sct_par_cpp[i],
                                               self.beam_cpp, self.rho,
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
        rtol_std = 1e-2
        SR = []
        SR_cpp = []

        for i in range(self.n_sections):
            SR.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                           self.beam, self.rho,
                                           quantum_excitation=True, python=True))

            SR_cpp.append(SynchrotronRadiation(self.general_params, self.RF_sct_par_cpp[i],
                                               self.beam_cpp, self.rho,
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

        for i in range(self.n_sections):
            SR.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                           self.beam, self.rho,
                                           quantum_excitation=False, python=True))

            SR_cpp.append(SynchrotronRadiation(self.general_params, self.RF_sct_par_cpp[i],
                                               self.beam_cpp, self.rho,
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
        rtol_std = 1e-1
        SR = []
        SR_cpp = []

        for i in range(self.n_sections):
            SR.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                           self.beam, self.rho,
                                           quantum_excitation=True, python=True))

            SR_cpp.append(SynchrotronRadiation(self.general_params, self.RF_sct_par_cpp[i],
                                               self.beam_cpp, self.rho,
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
        rtol_avg = 1e-7
        rtol_std = 1e-7
        SR = []
        SR_cpp = []

        for i in range(self.n_sections):
            SR.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                           self.beam, self.rho,
                                           quantum_excitation=False, python=True))

            SR_cpp.append(SynchrotronRadiation(self.general_params, self.RF_sct_par_cpp[i],
                                               self.beam_cpp, self.rho,
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
        rtol_avg = 1e-2
        rtol_std = 1e-2
        SR = []
        SR_cpp = []

        for i in range(self.n_sections):
            SR.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                           self.beam, self.rho,
                                           quantum_excitation=True, python=True))

            SR_cpp.append(SynchrotronRadiation(self.general_params, self.RF_sct_par_cpp[i],
                                               self.beam_cpp, self.rho,
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

        for i in range(self.n_sections):
            SR.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                           self.beam, self.rho,
                                           quantum_excitation=False, python=True))

            SR_cpp.append(SynchrotronRadiation(self.general_params, self.RF_sct_par_cpp[i],
                                               self.beam_cpp, self.rho,
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
        rtol_std = 1e-1
        SR = []
        SR_cpp = []

        for i in range(self.n_sections):
            SR.append(SynchrotronRadiation(self.general_params, self.RF_sct_par[i],
                                           self.beam, self.rho,
                                           quantum_excitation=True, python=True))

            SR_cpp.append(SynchrotronRadiation(self.general_params, self.RF_sct_par_cpp[i],
                                               self.beam_cpp, self.rho,
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
