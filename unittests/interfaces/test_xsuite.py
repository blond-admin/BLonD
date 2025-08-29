# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for interfaces.xsuite

:Authors: **Birk Emil Karlsen-Bæck**, **Thom Arnoldus van Rijswijk**
"""

import os
import unittest

import numpy as np
from scipy.constants import c, e, m_p

import xpart as xp
import xtrack as xt

from blond.beam.beam import Beam, Proton
from blond.beam.profile import CutOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.monitors.monitors import BunchMonitor

from blond.interfaces.xsuite import (BlondElement, BlondObserver,
                                     EnergyUpdate,
                                     blond_to_xsuite_transform,
                                     xsuite_to_blond_transform)

# TODO: Make a test of the EnergyFrequencyUpdate and BlondObserver classes


class TestXsuiteBLonDTransforms(unittest.TestCase):

    def setUp(self):
        # Accelerator parameters
        C = 26658.8832
        p_s = 450e9
        h = 35640
        alpha = 0.00034849575112251314
        V = 5e6
        dphi = 0

        # BLonD ring and RF station
        self.ring = Ring(C, alpha, p_s, Proton(), n_turns=1)
        self.rfstation = RFStation(self.ring, [h], [V], [dphi])

    def testBlondToXsuiteTransform(self):
        # Arrays of time and energy offsets
        dts = np.linspace(
            self.rfstation.t_rf[0, 0] * 1e-3,
            self.rfstation.t_rf[0, 0], 10
        )
        dEs = np.linspace(-200e6, 200e6, 10)

        # Transform BLonD coordinates to Xsuite
        zeta, ptau = blond_to_xsuite_transform(
            dts, dEs, self.ring.beta[0, 0], self.ring.energy[0, 0],
            self.rfstation.omega_rf[0, 0], self.rfstation.phi_s[0]
        )

        # Correct values
        act_zeta = np.array(
            [0.37325428,  0.29022578,  0.20719727,  0.12416876,  0.04114025,
             -0.04188826, -0.12491676, -0.20794527, -0.29097378, -0.37400229]
        )
        act_ptau = np.array(
            [-4.44444444e-04, -3.45679012e-04, -2.46913580e-04, -1.48148148e-04, -4.93827160e-05,
             4.93827160e-05,  1.48148148e-04,  2.46913580e-04, 3.45679012e-04,  4.44444444e-04]
        )

        # Check if the coordinates are the same
        np.testing.assert_allclose(
            act_zeta, zeta,
            err_msg='In testBlondToXsuiteTransform the zeta-coordinate differs'
        )
        np.testing.assert_allclose(
            act_ptau, ptau,
            err_msg='In testBlondToXsuiteTransform the ptau-coordinate differs'
        )

    def testXsuiteToBlondTransform(self):
        # Arrays of z and ptau from xsuite
        zeta = np.linspace(
            -0.37, 0.37, 10
        )
        ptau = np.linspace(
            -4e-4, 4e-4, 10
        )

        # Transform BLonD coordinates to Xsuite
        dt, de = xsuite_to_blond_transform(
            zeta, ptau, self.ring.beta[0, 0], self.ring.energy[0, 0],
            self.rfstation.omega_rf[0, 0], self.rfstation.phi_s[0]
        )

        # Correct values
        act_dt = np.array(
            [2.48172990e-09, 2.20746549e-09, 1.93320108e-09, 1.65893668e-09, 1.38467227e-09,
             1.11040786e-09, 8.36143453e-10, 5.61879046e-10, 2.87614638e-10, 1.33502300e-11]
        )
        act_de = np.array(
            [-1.8e+08, -1.4e+08, -1.0e+08, -6.0e+07, -2.0e+07,
             2.0e+07,  6.0e+07,  1.0e+08, 1.4e+08,  1.8e+08]
        )

        # Check if the coordinates are the same
        np.testing.assert_allclose(
            act_dt, dt,
            err_msg='In testXsuiteToBlondTransform the time-coordinate differs'
        )
        np.testing.assert_allclose(
            act_de, de,
            err_msg='In testXsuiteToBlondTransform the energy-coordinate differs'
        )

    def testTransformInverse(self):
        # Arrays of time and energy offsets
        dts = np.linspace(
            self.rfstation.t_rf[0, 0] * 1e-3,
            self.rfstation.t_rf[0, 0], 20
        )
        dEs = np.linspace(-200e6, 200e6, 20)

        # Perform transforms to xsuite coordinates and back
        zeta, ptau = blond_to_xsuite_transform(
            dts, dEs, self.ring.beta[0, 0], self.ring.energy[0, 0],
            self.rfstation.omega_rf[0, 0], self.rfstation.phi_s[0]
        )

        _dts, _dEs = xsuite_to_blond_transform(
            zeta, ptau, self.ring.beta[0, 0], self.ring.energy[0, 0],
            self.rfstation.omega_rf[0, 0], self.rfstation.phi_s[0]
        )

        # Check if the coordinates are the same
        np.testing.assert_allclose(
            dts, _dts,
            err_msg='In testTransformInverse the time-coordinate differs'
        )
        np.testing.assert_allclose(
            dEs, _dEs,
            err_msg='In testTransformInverse the energy-coordinate differs'
        )

class TestXsuiteLHC(unittest.TestCase):

    def setUp(self):
        # Accelerator parameters
        self.C = 26658.8832                     # Machine circumference [m]
        self.p_s = 450e9                        # Synchronous momentum [eV/c]
        self.p_f = 450.1e9                      # Synchronous momentum, final
        self.h = 35640                          # Harmonic number [-]
        self.alpha = 0.00034849575112251314     # First order mom. comp. factor [-]
        self.V = 5e6                            # RF voltage [V]
        self.dphi = 0                           # Phase modulation/offset [rad]

    def testSingleParticle(self):
        r'''Test of a single particle simulation in the LHC at constant energy'''
        rtol = 4e-9                             # relative tolerance
        atol = 5e-9                             # absolute tolerance

        # ----- Interface Simulation -----
        # Initialize the BLonD cavity
        blond_tracker, beam = self.singleParticleBLonDSimulation()
        cavity = BlondElement(blond_tracker, beam)

        line = self.initializeXsuiteLine()

        # Insert the BLonD elements
        line.insert_element(index=0, element=cavity, name='blond_cavity')

        dt_inter, dE_inter = self.performSingleParticleInterfaceSimulation(line, blond_tracker, beam)
        dt_inter = dt_inter / np.max(dt_inter)
        dE_inter = dE_inter / np.max(dE_inter)

        # ----- Pure BLonD Simulation -----
        blond_tracker, beam = self.singleParticleBLonDSimulation()
        dt_blond, dE_blond = self.performSingleParticleBLonDSimulation(blond_tracker, beam)
        dt_blond = dt_blond / np.max(dt_blond)
        dE_blond = dE_blond / np.max(dE_blond)

        # ----- Perform test -----
        np.testing.assert_allclose(
            dt_inter, dt_blond, rtol=rtol, atol=atol,
            err_msg='In testSingleParticle the time-coordinate differs'
        )
        np.testing.assert_allclose(
            dE_inter, dE_blond, rtol=rtol, atol=atol,
            err_msg='In testSingleParticle the energy-coordinate differs'
        )

    def testSingleParticleRamp(self):
        r'''Test of acceleration of a single particle simulation in the LHC'''
        rtol = 4e-9                             # relative tolerance
        atol = 5e-9                             # absolute tolerance

        # Initialize the BLonD cavity
        blond_tracker, beam = self.singleParticleBLonDSimulation(ramp=True)
        cavity = BlondElement(blond_tracker, beam)

        line = self.initializeXsuiteLine()

        # Insert the BLonD elements
        line.insert_element(index=0, element=cavity, name='blond_cavity')

        # Insert energy ramp
        energy_update = EnergyUpdate(blond_tracker.rf_params.momentum)
        line.insert_element(index='blond_cavity', element=energy_update, name='energy_update')

        dt_inter, dE_inter = self.performSingleParticleInterfaceSimulation(line, blond_tracker, beam)
        dt_inter = dt_inter / np.max(dt_inter)
        dE_inter = dE_inter / np.max(dE_inter)
        line.get_table().show()

        # ----- Pure BLonD Simulation -----
        blond_tracker, beam = self.singleParticleBLonDSimulation(ramp=True, pure_blond=True)
        dt_blond, dE_blond = self.performSingleParticleBLonDSimulation(blond_tracker, beam)
        dt_blond = dt_blond / np.max(dt_blond)
        dE_blond = dE_blond / np.max(dE_blond)

        # ----- Perform test -----
        np.testing.assert_allclose(
            dt_inter, dt_blond, rtol=rtol, atol=atol,
            err_msg='In testSingleParticleRamp the time-coordinate differs'
        )
        np.testing.assert_allclose(
            dE_inter, dE_blond, rtol=rtol, atol=atol,
            err_msg='In testSingleParticleRamp the energy-coordinate differs'
        )

    def testSingleParticleTwoRFStations(self):
        r'''Test of a single particle simulation in the LHC at constant energy with two RF stations'''
        rtol = 4e-9                             # relative tolerance
        atol = 5e-9                             # absolute tolerance

        # Initialize the BLonD cavities
        blond_tracker_1, blond_tracker_2, beam = self.singleParticleBLonDSimulation(two_rfstations=True)
        cavity_1 = BlondElement(blond_tracker_1, beam)
        cavity_2 = BlondElement(blond_tracker_2, beam)

        line = self.initializeXsuiteLine(two_rfstations=True)

        # Insert the BLonD elements
        line.insert_element(index='matrix', element=cavity_1, name='blond_cavity_1')
        line.insert_element(index='matrix_2', element=cavity_2, name='blond_cavity_2')

        dt_inter, dE_inter = self.performSingleParticleInterfaceSimulation(line, blond_tracker_1, beam)
        dt_inter = dt_inter / np.max(dt_inter)
        dE_inter = dE_inter / np.max(dE_inter)

        # ----- Pure BLonD Simulation -----
        blond_tracker_1, blond_tracker_2, beam = self.singleParticleBLonDSimulation(two_rfstations=True)
        dt_blond, dE_blond = self.performSingleParticleBLonDSimulation(blond_tracker_1, beam, blond_tracker_2)
        dt_blond = dt_blond / np.max(dt_blond)
        dE_blond = dE_blond / np.max(dE_blond)

        # ----- Perform test -----
        np.testing.assert_allclose(
            dt_inter, dt_blond, rtol=rtol, atol=atol,
            err_msg='In testSingleParticleTwoRFStations the time-coordinate differs'
        )
        np.testing.assert_allclose(
            dE_inter, dE_blond, rtol=rtol, atol=atol,
            err_msg='In testSingleParticleTwoRFStations the energy-coordinate differs'
        )

    def singleParticleBLonDSimulation(self, ramp: bool = False, two_rfstations: bool = False,
                                      pure_blond: bool = False):
        r'''
        Method to generate a BLonD simulation in the LHC with a single particle with a time offset with respect to the
        RF bucket.
        '''
        # Bunch parameters
        N_m = 1                                 # Number of macroparticles [-]
        N_p = 1.15e11                           # Intensity

        # Simulation parameters
        N_t = 481                               # Number of (tracked) turns [-]
        dt_offset = 0.15                        # Input particles dt [s]
        v_part = 1

        # --- BLonD objects ---
        if ramp:
            cycle = np.linspace(self.p_s, self.p_f, N_t + 1)
            if pure_blond:
                cycle = np.concatenate(
                    (np.array([self.p_s]), np.linspace(self.p_s, self.p_f, N_t + 1)[:-1])
                )
            ring = Ring(self.C, self.alpha, cycle, Proton(), n_turns=N_t)
        else:
            if two_rfstations:
                # Additional input
                c_part = 0.3
                v_part = 0.5

                ring = Ring(
                    [self.C * c_part, self.C * (1 - c_part)],
                    [[self.alpha], [self.alpha]],
                    [self.p_s, self.p_s],
                    Proton(), n_turns=N_t, n_sections=2
                )
            else:
                ring = Ring(self.C, self.alpha, self.p_s, Proton(), n_turns=N_t)

        beam = Beam(ring, N_m, N_p)
        rfstation = RFStation(ring, [self.h], [self.V * v_part], [self.dphi])

        # --- Insert particle ---
        beam.dt = np.array([rfstation.t_rf[0, 0] * (1 - dt_offset)])
        beam.dE = np.array([0.0])

        if two_rfstations:
            second_rfstation = RFStation(ring, [self.h], [self.V * (1 - v_part)],
                                         [self.dphi], section_index=2)

            return (RingAndRFTracker(rfstation, beam),
                    RingAndRFTracker(second_rfstation, beam),
                    beam)

        return RingAndRFTracker(rfstation, beam), beam

    def initializeXsuiteLine(self, two_rfstations: bool = False):
        r'''
        Method to generate an Xsuite Line using the one-turn matrix.
        '''
        c_part = 1

        if two_rfstations:
            c_part = 0.3
            matrix_2 = xt.LineSegmentMap(
                longitudinal_mode='nonlinear',
                qx=1.1, qy=1.2,
                betx=1.,
                bety=1.,
                voltage_rf=0,
                frequency_rf=0,
                lag_rf=0,
                momentum_compaction_factor=self.alpha,
                length=self.C * (1 - c_part)
            )

        matrix = xt.LineSegmentMap(
            longitudinal_mode='nonlinear',
            qx=1.1, qy=1.2,
            betx=1.,
            bety=1.,
            voltage_rf=0,
            frequency_rf=0,
            lag_rf=0,
            momentum_compaction_factor=self.alpha,
            length=self.C * c_part
        )

        # Create line
        line = xt.Line(elements=[matrix], element_names={'matrix'})
        line['matrix'].length = self.C * c_part

        if two_rfstations:
            line.append_element(element=matrix_2, name='matrix_2')
            line['matrix_2'].length = (1 - c_part) * self.C

        return line

    def performSingleParticleInterfaceSimulation(self, line, blond_track, beam):
        r'''
        Perform simulation with the interface.
        '''
        # Add particles to line and build tracker
        line.particle_ref = xp.Particles(p0c=self.p_s, mass0=xp.PROTON_MASS_EV, q0=1.)
        line.build_tracker()

        # --- Convert the initial BLonD distribution to xsuite coordinates ---
        zeta, ptau = blond_to_xsuite_transform(
            beam.dt, beam.dE, line.particle_ref.beta0[0],
            line.particle_ref.energy0[0], phi_s=blond_track.rf_params.phi_s[0],
            omega_rf=blond_track.rf_params.omega_rf[0, 0]
        )

        # --- Track matrix ---
        N_t = len(blond_track.rf_params.phi_s) - 1
        particles = line.build_particles(x=0, y=0, px=0, py=0, zeta=np.copy(zeta), ptau=np.copy(ptau))
        line.track(particles, num_turns=N_t, turn_by_turn_monitor=True, with_progress=False)
        mon = line.record_last_track

        dt_array = np.zeros(N_t)
        dE_array = np.zeros(N_t)
        # Convert the xsuite particle coordinates back to BLonD
        for i in range(N_t):
            dt_array[i], dE_array[i] = xsuite_to_blond_transform(
                mon.zeta[:, i].T, mon.ptau[:, i].T, blond_track.rf_params.beta[i], blond_track.rf_params.energy[i],
                phi_s=blond_track.rf_params.phi_s[i] - blond_track.rf_params.phi_rf[0, 0],
                omega_rf=blond_track.rf_params.omega_rf[0, i]
            )

        return dt_array, dE_array

    @staticmethod
    def performSingleParticleBLonDSimulation(blond_track, beam, second_blond_tracker: RingAndRFTracker = None):
        r'''
        Perform simulation in pure BLonD.
        '''
        # The number of turns to track
        N_t = len(blond_track.rf_params.phi_s) - 1

        # Save particle coordinates
        dt_array = np.zeros(N_t)
        dE_array = np.zeros(N_t)

        if second_blond_tracker is not None:
            full_tracker = FullRingAndRF([blond_track, second_blond_tracker])
        else:
            full_tracker = blond_track

        for i in range(N_t):
            dt_array[i] = beam.dt[0]
            dE_array[i] = beam.dE[0]
            full_tracker.track()

        return dt_array, dE_array


class TestXsuitePSB(unittest.TestCase):

    def setUp(self):
        # Accelerator parameters
        self.C = 2 * np.pi * 25.00005794526065  # Machine circumference, radius 25 [m]
        gamma_t = 4.11635447373496              # Transition gamma [-]
        self.alpha = 1. / gamma_t / gamma_t     # First order mom. comp. factor [-]
        self.h = 1                              # Harmonic number [-]
        self.V = 8e3                            # RF voltage [V]
        self.dphi = np.pi                       # Phase modulation/offset [rad]

        # Derived parameters
        kin_energy = 1.4e9                      # Kinetic energy [eV]
        E_0 = m_p * c ** 2 / e                  # [eV]
        tot_energy = E_0 + kin_energy           # [eV]

        self.p_s = np.sqrt(tot_energy ** 2 - E_0 ** 2)
        self.p_f = self.p_s + 0.001e9

    def testSingleParticle(self):
        r'''Test of a single particle simulation in the PSB at constant energy'''
        rtol = 4e-4                             # relative tolerance
        atol = 4.1e-4                           # absolute tolerance

        # ----- Interface Simulation -----
        # Initialize the BLonD cavity
        blond_tracker, beam = self.singleParticleBLonDSimulation()
        cavity = BlondElement(blond_tracker, beam)

        line = self.initializeXsuiteLine()

        # Insert the BLonD elements
        line.insert_element(index=0, element=cavity, name='blond_cavity')

        dt_inter, dE_inter = self.performSingleParticleInterfaceSimulation(line, blond_tracker, beam)
        dt_inter = dt_inter / np.max(dt_inter)
        dE_inter = dE_inter / np.max(dE_inter)

        # ----- Pure BLonD Simulation -----
        blond_tracker, beam = self.singleParticleBLonDSimulation()
        dt_blond, dE_blond = self.performSingleParticleBLonDSimulation(blond_tracker, beam)
        dt_blond = dt_blond / np.max(dt_blond)
        dE_blond = dE_blond / np.max(dE_blond)

        # ----- Perform test -----
        np.testing.assert_allclose(
            dt_inter, dt_blond, rtol=rtol, atol=atol,
            err_msg='In testSingleParticle the time-coordinate differs'
        )
        np.testing.assert_allclose(
            dE_inter, dE_blond, rtol=rtol, atol=atol,
            err_msg='In testSingleParticle the energy-coordinate differs'
        )

    def testSingleParticleRamp(self):
        r'''Test of acceleration of a single particle simulation in the PSB'''
        rtol = 4e-4                             # relative tolerance
        atol = 4.1e-4                           # absolute tolerance

        # Initialize the BLonD cavity
        blond_tracker, beam = self.singleParticleBLonDSimulation(ramp=True)
        cavity = BlondElement(blond_tracker, beam)

        line = self.initializeXsuiteLine()

        # Insert the BLonD elements
        line.insert_element(index=0, element=cavity, name='blond_cavity')

        # Insert energy ramp
        energy_update = EnergyUpdate(blond_tracker.rf_params.momentum)
        line.insert_element(index='matrix', element=energy_update, name='energy_update')

        dt_inter, dE_inter = self.performSingleParticleInterfaceSimulation(line, blond_tracker, beam)
        dt_inter = dt_inter / np.max(dt_inter)
        dE_inter = dE_inter / np.max(dE_inter)

        # ----- Pure BLonD Simulation -----
        blond_tracker, beam = self.singleParticleBLonDSimulation(ramp=True, pure_blond=True)
        dt_blond, dE_blond = self.performSingleParticleBLonDSimulation(blond_tracker, beam)
        dt_blond = dt_blond / np.max(dt_blond)
        dE_blond = dE_blond / np.max(dE_blond)

        # ----- Perform test -----
        np.testing.assert_allclose(
            dt_inter, dt_blond, rtol=rtol, atol=atol,
            err_msg='In testSingleParticleRamp the time-coordinate differs'
        )
        np.testing.assert_allclose(
            dE_inter, dE_blond, rtol=rtol, atol=atol,
            err_msg='In testSingleParticleRamp the energy-coordinate differs'
        )

    def singleParticleBLonDSimulation(self, ramp: bool = False, pure_blond: bool = False):
        r'''
        Method to generate a BLonD simulation in the LHC with a single particle with a time offset with respect to the
        RF bucket.
        '''
        # Bunch parameters
        N_m = 1                                 # Number of macroparticles [-]
        N_p = 1e11                              # Intensity

        # Simulation parameters
        N_t = 6000                              # Number of (tracked) turns [-]
        dt_offset = 0.15                         # Input particles dt [s]

        # --- BLonD objects ---
        if ramp:
            cycle = np.linspace(self.p_s, self.p_f, N_t + 1)
            if pure_blond:
                cycle = np.concatenate(
                    (np.array([self.p_s]), np.linspace(self.p_s, self.p_f, N_t + 1)[:-1])
                )
            ring = Ring(self.C, self.alpha, cycle, Proton(), n_turns=N_t)
        else:
            ring = Ring(self.C, self.alpha, self.p_s, Proton(), n_turns=N_t)

        beam = Beam(ring, N_m, N_p)
        rfstation = RFStation(ring, [self.h], [self.V], [self.dphi])

        # --- Insert particle ---
        beam.dt = np.array([rfstation.t_rf[0, 0] * (1 - dt_offset)])
        beam.dE = np.array([0.0])

        return RingAndRFTracker(rfstation, beam), beam

    def initializeXsuiteLine(self):
        r'''
        Method to generate an Xsuite Line using the one-turn matrix.
        '''
        matrix = xt.LineSegmentMap(
            longitudinal_mode='nonlinear',
            qx=1.1, qy=1.2,
            betx=1.,
            bety=1.,
            voltage_rf=0,
            frequency_rf=0,
            lag_rf=0,
            momentum_compaction_factor=self.alpha,
            length=self.C
        )

        # Create line
        line = xt.Line(elements=[matrix], element_names={'matrix'})
        line['matrix'].length = self.C

        return line

    def performSingleParticleInterfaceSimulation(self, line, blond_track, beam):
        r'''
        Perform simulation with the interface.
        '''
        # Add particles to line and build tracker
        line.particle_ref = xp.Particles(p0c=self.p_s, mass0=xp.PROTON_MASS_EV, q0=1.)
        line.build_tracker()

        # --- Convert the initial BLonD distribution to xsuite coordinates ---
        zeta, ptau = blond_to_xsuite_transform(
            beam.dt, beam.dE, line.particle_ref.beta0[0], line.particle_ref.energy0[0],
            phi_s=blond_track.rf_params.phi_s[0] - blond_track.rf_params.phi_rf[0, 0],
            omega_rf=blond_track.rf_params.omega_rf[0, 0]
        )

        # --- Track matrix ---
        N_t = len(blond_track.rf_params.phi_s) - 1
        particles = line.build_particles(x=0, y=0, px=0, py=0, zeta=np.copy(zeta), ptau=np.copy(ptau))
        line.track(particles, num_turns=N_t, turn_by_turn_monitor=True, with_progress=False)
        mon = line.record_last_track

        dt_array = np.zeros(N_t)
        dE_array = np.zeros(N_t)
        # Convert the xsuite particle coordinates back to BLonD
        for i in range(N_t):
            dt_array[i], dE_array[i] = xsuite_to_blond_transform(
                mon.zeta[:, i].T, mon.ptau[:, i].T, blond_track.rf_params.beta[i], blond_track.rf_params.energy[i],
                phi_s=blond_track.rf_params.phi_s[i] - blond_track.rf_params.phi_rf[0, 0],
                omega_rf=blond_track.rf_params.omega_rf[0, i]
            )

        return dt_array, dE_array

    @staticmethod
    def performSingleParticleBLonDSimulation(blond_track, beam):
        r'''
        Perform simulation in pure BLonD.
        '''
        # The number of turns to track
        N_t = len(blond_track.rf_params.phi_s) - 1

        # Save particle coordinates
        dt_array = np.zeros(N_t)
        dE_array = np.zeros(N_t)

        full_tracker = blond_track

        for i in range(N_t):
            dt_array[i] = beam.dt[0]
            dE_array[i] = beam.dE[0]
            full_tracker.track()

        return dt_array, dE_array