# coding: utf8
# Copyright 2014-2023 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Abstract base class, used as a template to implement LLRF beam feedback loops,
such as phase, radial and synchro loops.**

:Authors: **Helga Timko**, **Alexandre Lasheen**, **Oleksandr Naumenko**
'''

from __future__ import division

debug = True
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..utils import bmath as bm

if TYPE_CHECKING or debug:
    from blond.input_parameters.rf_parameters import RFStation
    from blond.input_parameters.ring import Ring
    from blond.beam.profile import Profile


class BeamFeedback(ABC):
    '''
    Abstract base-class to implement one-turn beam phase loop for different machines with different hardware.
    Create a subclass of this to implement a custom beam feedback for a different machine.
    The phase loop acts directly on the RF frequency of all harmonics and affects the RF phase as well.
    '''

    def __init__(self,
                 Ring: Ring,
                 RFStation: RFStation,
                 Profile: Profile,
                 PhaseNoise: Optional[np.ndarray] = None,
                 sample_dE: int = 0,
                 time_offset: Optional[float] = None,
                 window_coefficient: float = 0,
                 radial_reference: float = 0,
                 delay: int = 0):
        """

        :param Ring: Ring Object
        :param RFStation: RF Station Object
        :param Profile: Profile Object
        :param PhaseNoise: array with length n_turns + 1
                Phase Noise injected into the machine. Will add the given phase at specified turn to the rf phase.
        :param sample_dE [int]: Determines which particles to sample for mean energy calculation.
                Every <sample_dE>. particle is sampled
        :param time_offset [float]: The beam phase is calculated in the dt interval [<time_offset>,<profile.cut_right>],
                i.e. from the time_offset value to the rightmost bucket
        :param window_coefficient [float]: Band-pass filter window coefficient for beam phase calculation
        :param radial_reference [float]: Reference orbit to which the radial loop tries to steer
        :param delay [int]: Turn from which the loops start to act
        """

        #: | *Import Ring*
        self.ring = Ring

        #: | *Import RFStation*
        self.rf_station = RFStation

        #: | *Import Profile*
        self.profile = Profile

        #: | *Band-pass filter window coefficient for beam phase calculation.*
        self.alpha = window_coefficient

        #: | Determines which particles to sample for mean energy calculation. Every <sample_dE>. particle is sampled
        self.sample_dE = sample_dE

        #: | determines from which RF-bucket the band-pass filter starts to act
        self.time_offset = time_offset

        #: | Delay in turns before the loops start to act. Phase Loop begins at turn = delay
        self.delay = delay

        #: | *Relative radial displacement [1], for radial loop.*
        self.drho = 0.

        #: | *Phase loop frequency correction of the main RF system.*
        self.domega_rf = 0.

        #: | *Beam phase measured at the main RF frequency.*
        self.phi_beam = 0.

        #: | *Phase difference between beam and RF.*
        self.dphi = 0.

        #: | *Reference orbit for the radial loop. The radial loop will steer to this value, if active.*
        self.radial_reference = radial_reference

        #: | *Optional import of RF PhaseNoise object*
        self.RFnoise = PhaseNoise
        if (self.RFnoise is not None
                and (len(self.RFnoise.dphi) != Ring.n_turns + 1)):
            # PhaseNoiseError
            raise RuntimeError(
                'Phase noise has to have a length of n_turns + 1')

    @abstractmethod
    def adjust_rf_frequency(self):
        """
        To be implemented in subclasses. Has to calculate an RF angular
        frequency adjustment based on all active loops and return it.

        :return: Angular frequency adjustment based on all active loops.
        """
        pass

    # TODO Decide on general template for loops
    """
    PID Control Loops based on first order backwards finite differences. Gains already take the
    time-spacing for the derivatives into account, so they are the same unit 
    (which depends on in- and output units)
    """

    def proportional_control(self, gain: float, difference: float):

        return gain * difference

    def integral_control(self, gain: float, previous_output: float, difference: float):
        return previous_output + self.ring.t_rev[0, self.rf_station.counter[0]] * gain * difference

    def derivative_control(self, gain: float, difference: float, previous_difference: float):
        return gain / self.ring.t_rev[0, self.rf_station.counter[0]] * (difference - previous_difference)

    def PI_control(self, P_gain: float, I_gain: float,
                   previous_output: float, difference: float, previous_difference: float):
        return previous_output + P_gain * (difference - previous_difference) \
            + I_gain * self.ring.t_rev[0, self.rf_station.counter[0]] * difference

    def PD_control(self, P_gain: float, D_gain: float,
                   difference: float, previous_difference: float):
        return P_gain * difference + D_gain / self.ring.t_rev[0, self.rf_station.counter[0]] * (
                    difference - previous_difference)

    def PID_control(self, P_gain: float, I_gain: float, D_gain: float, previous_output: float,
                    difference, previous_difference, prevprev_difference):

        return self.PI_control(P_gain, I_gain, previous_output, difference, previous_difference) \
            + D_gain / self.ring.t_rev[0, self.rf_station.counter[0]] * (
                        difference - 2 * previous_difference + prevprev_difference)

    def track(self):
        '''
        Calculate PL correction on main RF frequency depending on machine and
        propagate it to other RF systems.
        The update of the RF phase and frequency for the next turn,
        for all systems is done in the tracker.
        '''

        # Calculate PL correction on RF frequency
        counter = self.rf_station.counter[0]
        if counter >= self.delay:
            self.domega_rf = self.adjust_rf_frequency()

            # Update the RF frequency of all systems for the next turn
            counter = self.rf_station.counter[0] + 1
            self.rf_station.omega_rf[:, counter] += self.domega_rf * \
                                                    self.rf_station.harmonic[:, counter] / \
                                                    self.rf_station.harmonic[0, counter]

    def precalculate_time(self, Ring: Ring):
        '''
        *For machines like the PSB, where the PL acts only in certain time
        intervals, pre-calculate on which turns to act.*
        '''

        if self.dt > 0:
            n = self.delay + 1
            while n < Ring.t_rev.size:
                summa = 0
                while summa < self.dt:
                    try:
                        summa += Ring.t_rev[n]
                        n += 1
                    except Exception:
                        self.on_time = np.append(self.on_time, 0)
                        return
                self.on_time = np.append(self.on_time, n - 1)
        else:
            self.on_time = np.arange(Ring.t_rev.size)

    def beam_phase(self):
        '''
        *Beam phase measured at the main RF frequency and phase. The beam is
        convolved with the window function of the band-pass filter of the
        machine. The coefficients of sine and cosine components determine the
        beam phase, projected to the range Pi/2 to 3/2 Pi. Note that this beam
        phase is already w.r.t. the instantaneous RF phase.*
        '''

        # Main RF frequency at the present turn
        omega_rf = self.rf_station.omega_rf[0, self.rf_station.counter[0]]
        phi_rf = self.rf_station.phi_rf[0, self.rf_station.counter[0]]

        if self.time_offset is None:
            coeff = bm.beam_phase(self.profile.bin_centers,
                                  self.profile.n_macroparticles,
                                  self.alpha, omega_rf, phi_rf,
                                  self.profile.bin_size)
        else:
            indexes = self.profile.bin_centers >= self.time_offset
            time_offset = self.time_offset
            # Convolve with window function
            scoeff = np.trapz(np.exp(self.alpha * (self.profile.bin_centers[indexes] -
                                                   time_offset)) *
                              np.sin(omega_rf * self.profile.bin_centers[indexes] +
                                     phi_rf) *
                              self.profile.n_macroparticles[indexes],
                              dx=self.profile.bin_size)
            ccoeff = np.trapz(np.exp(self.alpha * (self.profile.bin_centers[indexes] -
                                                   time_offset)) *
                              np.cos(omega_rf * self.profile.bin_centers[indexes] +
                                     phi_rf) *
                              self.profile.n_macroparticles[indexes],
                              dx=self.profile.bin_size)
            coeff = scoeff / ccoeff

        # Project beam phase to (pi/2,3pi/2) range
        self.phi_beam = np.arctan(coeff) + np.pi
        return self.phi_beam

    def beam_phase_sharpWindow(self):
        '''
        *Beam phase measured at the main RF frequency and phase. The beam is
        averaged over a window. The coefficients of sine and cosine components
        determine the beam phase, projected to the range -Pi/2 to 3/2 Pi.
        Note that this beam phase is already w.r.t. the instantaneous RF phase.*
        '''

        # Main RF frequency at the present turn
        turn = self.rf_station.counter[0]
        omega_rf = self.rf_station.omega_rf[0, turn]
        phi_rf = self.rf_station.phi_rf[0, turn]

        if self.alpha != 0.0:
            indexes = bm.logical_and((self.time_offset - np.pi / omega_rf)
                                     <= self.profile.bin_centers,
                                     self.profile.bin_centers
                                     <= (-1 / self.alpha + self.time_offset -
                                         2 * np.pi / omega_rf))
        else:
            indexes = bm.ones(self.profile.n_slices, dtype=bool)

        # Convolve with window function
        scoeff = bm.trapz(bm.sin(omega_rf * self.profile.bin_centers[indexes]
                                 + phi_rf)
                          * self.profile.n_macroparticles[indexes],
                          dx=self.profile.bin_size)
        ccoeff = bm.trapz(bm.cos(omega_rf * self.profile.bin_centers[indexes]
                                 + phi_rf) *
                          self.profile.n_macroparticles[indexes],
                          dx=self.profile.bin_size)

        # Project beam phase to (pi/2,3pi/2) range
        self.phi_beam = np.arctan(scoeff / ccoeff) + np.pi
        return self.phi_beam

    def phase_difference(self, beam_phase_function=beam_phase):
        '''
        *Phase difference between beam and RF phase of the main RF system.
        Optional: add RF phase noise through dphi directly.*

        :param beam_phase_function [function]: The function used to calculate the beam phase.
               Default: beam_phase() implemented in this class
        '''

        beam_phase_function(self)
        # Correct for design stable phase
        counter = self.rf_station.counter[0]
        self.dphi = self.phi_beam

        self.phi_target = self.rf_station.phi_s[counter]
        # TODO: Should the target phase also be shifted into the [pi/2, 3/2 pi] interval?
        """
        # Force target phase to be in the range [pi/2, 3/2 pi], same as beam phase
        # effectively, while phi_target <np.pi /2:
        # phi_target = phi_target + np.pi
        # and analogous for the other side
        if self.phi_target < np.pi / 2:
            factor = np.ceil(np.abs((np.pi / 2 - self.phi_target) / np.pi))
            self.phi_target += np.pi * factor
        elif self.phi_target > 3 / 2 * np.pi:
            factor = np.ceil(np.abs((self.phi_target - 3 * np.pi / 2) / np.pi))
            self.phi_target -= np.pi * factor
        """
        self.dphi = self.phi_beam - self.phi_target
        if self.RFnoise is not None:
            self.dphi += self.RFnoise.dphi[counter]
        return self.dphi

    def radial_difference(self):
        '''
        *Radial difference between beam and design orbit.*
        '''

        counter = self.rf_station.counter[0]

        # Correct for design orbit
        #        self.average_dE = np.mean(self.profile.Beam.dE[(self.profile.Beam.dt >
        #            self.profile.bin_centers[0])*(self.profile.Beam.dt <
        #                                         self.profile.bin_centers[-1])])

        self.average_dE = bm.mean(self.profile.Beam.dE[::self.sample_dE])

        self.drho = self.ring.alpha_0[0, counter] * \
                    self.ring.ring_radius * self.average_dE / \
                    (self.ring.beta[0, counter] ** 2.
                     * self.ring.energy[0, counter])

        return self.drho

    def radial_steering_from_freq(self):
        '''
        *Frequency and phase change for the current turn due to the radial steering program.*
        '''

        counter = self.rf_station.counter[0]

        self.radial_steering_domega_rf = - self.rf_station.omega_rf_d[0, counter] * \
                                         self.rf_station.eta_0[counter] / self.ring.alpha_0[0, counter] * \
                                         self.radial_reference / self.ring.ring_radius

        return self.radial_steering_domega_rf

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