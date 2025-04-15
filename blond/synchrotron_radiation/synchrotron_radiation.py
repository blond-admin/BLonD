
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
import warnings
from builtins import range

import numpy as np
from blond.utils.exceptions import InputDataError
from blond.utils.exceptions import MissingParameterError
from blond.beam.beam import Electron, Positron, Beam
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from ..utils import bmath as bm


class SynchrotronRadiation:

    ''' Class to compute synchrotron radiation effects, including radiation
        damping and quantum excitation.
        For multiple RF section, instantiate one object per RF section and call
        the track() method after tracking each section.
    '''

    def __init__(self, Ring: Ring, RFParameters: RFStation, Beam: Beam, bending_radius:float = None, rad_int : (np.ndarray,list) = None,
                 n_kicks:int =1, quantum_excitation: bool=True, python: bool=True, seed: int=None,
                 shift_beam:bool=False):
        """
        Synchrotron radiation tracker
        Calculates and updates the energy losses per turn and longitudinal damping time according to the ring energy program, and implements the effect of
        synchrotron radiation damping and quantum excitation (if enabled) on the beam coordinates.
        :param Ring: a Ring-type class
        :param RFParameters: RF Station class
        :param Beam: Beam class
        :param bending_radius: to compute the radiation integral assuming an isomagnetic ring
        :param rad_int: to compute the damping and quantum excitation terms for a known ring. Expected list or numpy array of at least 5 components
        :param n_kicks: Number of kicks to distribute the SR and quantum excitation impact into
        :param quantum_excitation: Enable (DEFAULT) or disable the quantum excitation effect during the simulation
        :param python: Enable the use of the python functions (DEFAULT)
        :param seed: for tracking in C
        :param shift_beam: # Displace the beam in phase to account for the energy loss due to synchrotron radiation (temporary until bunch generation is updated)
        """

        self.ring = Ring
        self.rf_params = RFParameters
        self.beam = Beam

        # Input check
        if not isinstance(Ring.Particle, Electron) and not isinstance(Ring.Particle, Positron):
            raise TypeError('Particles not expected. Expected an electron or positron beam.')

        if rad_int is None:
            if bending_radius is None:
                if Ring.sr_flag:
                    self.I2 = Ring.I2
                    self.I3 = Ring.I3
                    self.I4 = Ring.I4
                    self.jz = 2.0 + self.I4 / self.I2
                else :
                    raise MissingParameterError("Synchrotron radiation damping and quantum excitation require either the bending radius "
                                            "for an isomagnetic ring, or the first five synchrotron radiation integrals.")
            if bending_radius is not None:
                self.rho = bending_radius
                self.I2 = 2.0 * np.pi / self.rho  # Assuming isomagnetic machine
                self.I3 = 2.0 * np.pi / self.rho ** 2.0
                self.I4 = self.ring.ring_circumference * self.ring.alpha_0[0, 0] / self.rho ** 2.0
                self.jz = 2.0 + self.I4 / self.I2

        if rad_int is not None:
            if type(rad_int) in {np.ndarray, list}:
                try :
                    integrals = np.array(rad_int)
                except ValueError as ve:
                    raise ValueError(ve)
                if integrals.__len__() >=5:
                    if bending_radius is not None:
                        warnings.warn('Synchrotron radiation integrals prevail. Bending radius input ignored.')
                    self.I2 = integrals[1]
                    self.I3 = integrals[2]
                    self.I4 = integrals[3]
                    self.jz = 2.0 + self.I4 / self.I2
                else :
                    raise ValueError("The first five synchrotron " +
                                     "radiation integrals are requires " +
                                     "Ignoring input.")
            else:
                raise TypeError(f"Expected a list or numpy.ndarray as an input. Received {type(rad_int)}.")

        self.n_kicks = n_kicks  # To apply SR in several kicks
        np.random.seed(seed=seed)

        # Calculate static parameters
        self.c_gamma = self.ring.Particle.c_gamma
        self.c_q = self.ring.Particle.c_q

        # Calculate synchrotron radiation parameters
        self.calculate_SR_params()

        # Initialize the random number array if quantum excitation is included
        if quantum_excitation:
            self.random_array = np.zeros(self.beam.n_macroparticles)

        # Displace the beam in phase to account for the energy loss due to
        # synchrotron radiation (temporary until bunch generation is updated)
        if (shift_beam) and (self.rf_params.section_index == 0):
            self.beam_phase_to_compensate_SR = np.abs(np.arcsin(
                self.U0 / (self.ring.Particle.charge * self.rf_params.voltage[0][0])))
            self.beam_position_to_compensate_SR = self.beam_phase_to_compensate_SR \
                * self.rf_params.t_rf[0, 0] / (2.0 * np.pi)

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
        self.U0 = (self.c_gamma * self.ring.energy[0, i_turn]**4.0
                   * self.I2 / (2.0 * np. pi)
                   * self.rf_params.section_length
                   / self.ring.ring_circumference)

        # Damping time [turns]
        self.tau_z = (2.0 / self.jz * self.ring.energy[0, i_turn] /
                      self.U0)

        # Equilibrium energy spread
        self.sigma_dE = np.sqrt(self.c_q *
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
                self.U0 * 1e-9))
            print('Energy loss per turn = {0:1.4f} GeV/turn'.format(
                self.U0 * 1e-9 * self.ring.ring_circumference
                / self.rf_params.section_length))
            print('Damping time = {0:1.4f} turns'.format(self.tau_z
                                                         * self.rf_params.section_length
                                                         / self.ring.ring_circumference))
        print(f'Equilibrium energy spread = {self.sigma_dE * 100:1.4f}%'
              + f' ({self.sigma_dE * self.ring.energy[0, i_turn]*1e-6:1.4f}) MeV')
        print('------------------------------------------------')

    # Track particles with SR only (without quantum excitation)
    def track_SR_python(self):
        "Adds the effect of synchrotron radiation damping on the beam coordinates"
        i_turn = self.rf_params.counter[0]
        # Recalculate SR parameters if energy changes
        if (i_turn != 0 and self.ring.energy[0, i_turn] !=
                self.ring.energy[0, i_turn - 1]):
            self.calculate_SR_params()
        for i in range(self.n_kicks):
            self.beam.dE += -(2.0 / self.tau_z / self.n_kicks * self.beam.dE
                              + self.U0 / self.n_kicks)

    # Track particles with SR and quantum excitation
    def track_full_python(self):
        "Adds the effect of synchrotron radiation damping and quantum excitation on the beam coordinates"
        i_turn = self.rf_params.counter[0]
        # Recalculate SR parameters if energy changes
        if (i_turn != 0 and self.ring.energy[0, i_turn] !=
                self.ring.energy[0, i_turn - 1]):
            self.calculate_SR_params()
        for i in range(self.n_kicks):
            self.beam.dE += -(2.0 / self.tau_z / self.n_kicks * self.beam.dE +  # damping
                              self.U0 / self.n_kicks # SR kick
                              - 2.0 * self.sigma_dE /  np.sqrt(self.tau_z * self.n_kicks) * # quantum excitation kick
                              self.beam.energy * np.random.normal(size=len(self.beam.dE)))



    # Track particles with SR only (without quantum excitation)
    # C implementation
    def track_SR_C(self):
        i_turn = self.rf_params.counter[0]
        # Recalculate SR parameters if energy changes
        if (i_turn != 0 and self.ring.energy[0, i_turn] !=
                self.ring.energy[0, i_turn - 1]):
            self.calculate_SR_params()

        bm.synchrotron_radiation(self.beam.dE, self.U0,
                                 self.n_kicks, self.tau_z)

    # Track particles with SR and quantum excitation. C implementation
    def track_full_C(self):
        i_turn = self.rf_params.counter[0]
        # Recalculate SR parameters if energy changes
        if (i_turn != 0 and self.ring.energy[0, i_turn] !=
                self.ring.energy[0, i_turn - 1]):
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

        # No arrays need to be transfered

        # to make sure it will not be called again
        self._device = 'CPU'
