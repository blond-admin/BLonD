
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Class to compute synchrotron radiation damping and quantum excitation**

:Authors: **Juan F. Esteban Mueller**
'''

from __future__ import division, print_function
from builtins import range, object
import numpy as np
from ..utils import bmath as bm


class SynchrotronRadiation(object):

    ''' Class to compute synchrotron radiation effects, including radiation
        damping and quantum excitation.
        For multiple RF section, instanciate one object per RF section an call
        the track() method after tracking each section.
    '''

    def __init__(self, Ring, RFParameters, Beam, bending_radius,
                 n_kicks=1, quantum_excitation=True, python=False, seed=None,
                 shift_beam=True):

        self.ring = Ring
        self.rf_params = RFParameters
        self.beam = Beam
        self.rho = bending_radius
        self.n_kicks = n_kicks  # To apply SR in several kicks
        np.random.seed(seed=seed)

        # Calculate static parameters
        self.C_gamma = self.ring.Particle.C_gamma
        self.C_q = self.ring.Particle.C_q

        self.I2 = 2.0 * np.pi / self.rho     # Assuming isomagnetic machine
        self.I3 = 2.0 * np.pi / self.rho**2.0
        self.I4 = self.ring.ring_circumference * self.ring.alpha_0[0, 0] / self.rho**2.0
        self.jz = 2.0 + self.I4 / self.I2

        # Calculate synchrotron radiation parameters
        self.calculate_SR_params()

        # Initialize the random number array if quantum excitation is included
        if quantum_excitation:
            self.random_array = np.zeros(self.beam.n_macroparticles)

        # Displace the beam in phase to account for the energy loss due to
        # synchrotron radiation (temporary until bunch generation is updated)
        if (shift_beam) and (self.rf_params.section_index == 0):
            self.beam_phase_to_compensate_SR = np.abs(np.arcsin(
                self.U0 / (self.ring.Particle.charge * self.rf_params.voltage[0][0]) ))
            self.beam_position_to_compensate_SR = self.beam_phase_to_compensate_SR \
                * self.rf_params.t_rf[0, 0] / (2.0*np.pi)

            self.beam.dt -= self.beam_position_to_compensate_SR

        # Select the right method for the tracker according to the selected
        # settings
        if python:
            if quantum_excitation:
                self.track = self.track_full_python
            else:
                self.track = self.track_SR_python
        else:
            if quantum_excitation:
                if seed is not None:
                    bm.set_random_seed(seed)
                self.track = self.track_full_C
            else:
                self.track = self.track_SR_C

    # Method to compute the SR parameters
    def calculate_SR_params(self):
        i_turn = self.rf_params.counter[0]

        # Energy loss per turn/RF section [eV]
        self.U0 = (self.C_gamma * self.ring.energy[0, i_turn]**4.0
                   * self.I2 / (2.0 * np. pi)
                   * self.rf_params.section_length
                   / self.ring.ring_circumference)

        # Damping time [turns]
        self.tau_z = (2.0 / self.jz * self.ring.energy[0, i_turn] /
                      self.U0)

        # Equilibrium energy spread
        self.sigma_dE = np.sqrt(self.C_q *
                                self.ring.gamma[0, i_turn]**2.0 *
                                self.I3 / (self.jz * self.I2))

    # Print SR parameters
    def print_SR_params(self):
        i_turn = self.rf_params.counter[0]

        print('------- Synchrotron radiation parameters -------')
        print(f'jz = {self.jz:1.8f}')
        if (self.rf_params.section_length
                == self.ring.ring_circumference):
            print(f'Energy loss per turn = {self.U0/1e9:1.4f} GeV/turn')
            print(f'Damping time = {self.tau_z:1.4f} turns')
        else:
            print('Energy loss per RF section = {0:1.4f} GeV/section'.format(
                self.U0*1e-9))
            print('Energy loss per turn = {0:1.4f} GeV/turn'.format(
                self.U0*1e-9 * self.ring.ring_circumference
                / self.rf_params.section_length))
            print('Damping time = {0:1.4f} turns'.format(self.tau_z
                                                         * self.rf_params.section_length
                                                         / self.ring.ring_circumference))
        print(f'Equilibrium energy spread = {self.sigma_dE * 100:1.4f}%'
              + f' ({self.sigma_dE * self.ring.energy[0, i_turn]*1e-6:1.4f}) MeV')
        print('------------------------------------------------')

    # Track particles with SR only (without quantum excitation)
    def track_SR_python(self):
        i_turn = self.rf_params.counter[0]
        # Recalculate SR parameters if energy changes
        if (i_turn != 0 and self.ring.energy[0, i_turn] !=
                self.ring.energy[0, i_turn-1]):
            self.calculate_SR_params()
        for i in range(self.n_kicks):
            self.beam.dE += -(2.0 / self.tau_z / self.n_kicks * self.beam.dE
                              + self.U0 / self.n_kicks)

    # Track particles with SR and quantum excitation
    def track_full_python(self):
        i_turn = self.rf_params.counter[0]
        # Recalculate SR parameters if energy changes
        if (i_turn != 0 and self.ring.energy[0, i_turn] !=
                self.ring.energy[0, i_turn-1]):
            self.calculate_SR_params()
        for i in range(self.n_kicks):
            self.beam.dE += -(2.0 / self.tau_z / self.n_kicks * self.beam.dE +
                              self.U0 / self.n_kicks - 2.0 * self.sigma_dE /
                              np.sqrt(self.tau_z * self.n_kicks) *
                              self.ring.energy[0, i_turn] *
                              np.random.randn(self.beam.n_macroparticles))

    # Track particles with SR only (without quantum excitation)
    # C implementation
    def track_SR_C(self):
        i_turn = self.rf_params.counter[0]
        # Recalculate SR parameters if energy changes
        if (i_turn != 0 and self.ring.energy[0, i_turn] !=
                self.ring.energy[0, i_turn-1]):
            self.calculate_SR_params()

        bm.synchrotron_radiation(self.beam.dE, self.U0,
                                 self.n_kicks, self.tau_z)

    # Track particles with SR and quantum excitation. C implementation
    def track_full_C(self):
        i_turn = self.rf_params.counter[0]
        # Recalculate SR parameters if energy changes
        if (i_turn != 0 and self.ring.energy[0, i_turn] !=
                self.ring.energy[0, i_turn-1]):
            self.calculate_SR_params()

        bm.synchrotron_radiation_full(self.beam.dE, self.U0, self.n_kicks,
                                      self.tau_z, self.sigma_dE,
                                      self.ring.energy[0, i_turn])

    def to_gpu(self, recursive=True):
        '''
        Transfer all necessary arrays to the GPU
        '''
        # Check if to_gpu has been invoked already
        if hasattr(self, '_device') and self._device == 'GPU':
            return

        assert bm.device == 'GPU'
        # No arrays need to be transfered

        # to make sure it will not be called again
        self._device = 'GPU'

    def to_cpu(self, recursive=True):
        '''
        Transfer all necessary arrays back to the CPU
        '''
        # Check if to_cpu has been invoked already
        if hasattr(self, '_device') and self._device == 'CPU':
            return

        assert bm.device == 'CPU'
        # No arrays need to be transfered

        # to make sure it will not be called again
        self._device = 'CPU'
