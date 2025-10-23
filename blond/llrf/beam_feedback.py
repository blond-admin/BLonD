# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Various beam phase loops with optional synchronisation/frequency/radial loops
for the CERN machines**

:Authors: **Helga Timko**, **Alexandre Lasheen**
'''

from __future__ import division


import numpy as np

from ..utils import bmath as bm


class BeamFeedback:
    '''
    One-turn beam phase loop for different machines with different hardware. 
    Use 'period' for a phase loop that is active only in certain turns. 
    The phase loop acts directly on the RF frequency of all harmonics and
    affects the RF phase as well.
    '''

    def __init__(self, Ring, RFStation, Profile,
                 configuration,
                 PhaseNoise=None,
                 LHCNoiseFB=None, delay=0):

        #: | *Import Ring*
        self.ring = Ring

        #: | *Import RFStation*
        self.rf_station = RFStation

        #: | *Import Profile*
        self.profile = Profile

        #: | *Machine-dependent configuration of LLRF system.*
        self.config = configuration

        self.delay = delay

        #: | *Machine name; see description of each machine.*
        if 'machine' not in self.config:
            self.machine = 'LHC'
        else:
            self.machine = self.config['machine']

        #: | *Band-pass filter window coefficient for beam phase calculation.*
        if 'window_coefficient' not in self.config:
            self.alpha = 0.
        else:
            self.alpha = self.config['window_coefficient']

        # determines from which RF-buckets the band-pass filter starts to acts
        if 'time_offset' not in self.config:
            self.time_offset = None
        else:
            self.time_offset = self.config['time_offset']

        #: | *Phase loop gain. Implementation depends on machine.*
        try:
            self.gain = self.config['PL_gain']
        except Exception:
            # PhaseLoopError
            raise RuntimeError(
                "You need to specify the Phase Loop gain! Aborting")

        # LHC CONFIGURATION
        if self.machine == 'LHC':

            #: | *Synchronisation loop gain.*
            if 'SL_gain' not in self.config:
                self.gain2 = 0.
            else:
                self.gain2 = self.config['SL_gain']

            #: | *LHC Synchroronisation loop recursion variable*
            self.lhc_y = 0

            if self.gain2 != 0:

                #: | *LHC Synchronisation loop coefficient [1]*
                self.lhc_a = 5.25 - self.rf_station.omega_s0 / (np.pi * 40.)
                #: | *LHC Synchronisation loop time constant [turns]*
                self.lhc_t = (2 * np.pi * self.rf_station.Q_s * np.sqrt(self.lhc_a)) / \
                    np.sqrt(1 + self.gain / self.gain2 *
                            np.sqrt((1 + 1 / self.lhc_a) / (1 + self.lhc_a)))

            else:

                self.lhc_a = np.zeros(self.rf_station.n_turns + 1)
                self.lhc_t = np.zeros(self.rf_station.n_turns + 1)

        # LHC_F CONFIGURATION
        elif self.machine == 'LHC_F':

            #: | *Frequency loop gain.*
            if 'FL_gain' not in self.config:
                self.gain2 = 0.
            else:
                self.gain2 = self.config['FL_gain']

        # SPS_RL CONFIGURATION
        elif self.machine == 'SPS_RL':

            #: | *Frequency loop gain.*
            if 'RL_gain' not in self.config:
                self.gain2 = 0.
            else:
                self.gain2 = self.config['RL_gain']

            #: | *Number of particles to sample from dE for orbit calculation*
            if 'sample_dE' not in self.config:
                self.sample_dE = 1
            else:
                self.sample_dE = self.config['sample_dE']

        elif self.machine == 'SPS_F':
            #: | *Frequency loop gain.*
            if 'FL_gain' not in self.config:
                self.gain2 = 0.
            else:
                self.gain2 = self.config['FL_gain']

        # PSB CONFIGURATION
        elif self.machine == 'PSB':

            self.gain = self.gain * np.ones(Ring.n_turns + 1)

            #: | *Radial loop gain, proportional [1] and integral [1/s].*
            if 'RL_gain' not in self.config:
                self.gain2 = [0., 0.]
            else:
                self.gain2 = self.config['RL_gain']

            self.gain2[0] = self.gain2[0] * np.ones(Ring.n_turns + 1)
            self.gain2[1] = self.gain2[1] * np.ones(Ring.n_turns + 1)

            #: | *Optional: PL & RL acting only in certain time intervals/turns.*
            self.dt = 0
            # | *Phase Loop sampling period [s]*
            if 'period' not in self.config:
                self.dt = 10.e-6  # [s]
            else:
                self.dt = self.config['period']

            # Counter of turns passed since last time the PL was active
            self.PL_counter = 0
            self.on_time = np.array([])

            self.precalculate_time(Ring)

            #: | *Array of transfer function coefficients.*
            if 'coefficients' not in self.config:
                self.coefficients = [0.999019, -0.999019, 0., 1., -0.998038, 0.]
            else:
                self.coefficients = self.config['coefficients']

            #: | *Memory of previous phase correction, for phase loop.*
            self.dphi_sum = 0.
            self.dphi_av = 0.
            self.dphi_av_prev = 0.

            #: | *Memory of previous relative radial correction, for rad loop.*
            self.dR_over_R_prev = 0.

            #: | *Phase loop frequency correction [1/s]*
            self.domega_PL = 0.

            #: | *Radial loop frequency correction [1/s]*
            self.domega_RL = 0.

            self.dR_over_R = 0

        #: | *Relative radial displacement [1], for radial loop.*
        self.drho = 0.

        #: | *Phase loop frequency correction of the main RF system.*
        self.domega_rf = 0.

        #: | *Beam phase measured at the main RF frequency.*
        self.phi_beam = 0.

        #: | *Phase difference between beam and RF.*
        self.dphi = 0.

        #: | *Reference signal for secondary loop to test step response.*
        self.reference = 0.

        #: | *Optional import of RF PhaseNoise object*
        self.RFnoise = PhaseNoise
        if (self.RFnoise is not None
                and (len(self.RFnoise.dphi) != Ring.n_turns + 1)):
            # PhaseNoiseError
            raise RuntimeError(
                'Phase noise has to have a length of n_turns + 1')

        #: | *Optional import of amplitude-scaling feedback object LHCNoiseFB*
        self.noiseFB = LHCNoiseFB

    def track(self):
        '''
        Calculate PL correction on main RF frequency depending on machine and
        propagate it to other RF systems.
        The update of the RF phase and frequency for the next turn,
        for all systems is done in the tracker.
        '''

        # Calculate PL correction on RF frequency
        getattr(self, self.machine)()

        # Update the RF frequency of all systems for the next turn
        counter = self.rf_station.counter[0] + 1
        self.rf_station.omega_rf[:, counter] += self.domega_rf * \
            self.rf_station.harmonic[:, counter] / \
            self.rf_station.harmonic[0, counter]

    def precalculate_time(self, Ring):
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
        beam phase, projected to the range -Pi/2 to 3/2 Pi. Note that this beam
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
            coeff = bm.beam_phase(self.profile.bin_centers[indexes],
                                  self.profile.n_macroparticles[indexes],
                                  self.alpha, omega_rf, phi_rf,
                                  self.profile.bin_size)

        # Project beam phase to (pi/2,3pi/2) range
        self.phi_beam = np.arctan(coeff) + np.pi

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

    def phase_difference(self):
        '''
        *Phase difference between beam and RF phase of the main RF system.
        Optional: add RF phase noise through dphi directly.*
        '''

        # Correct for design stable phase
        counter = self.rf_station.counter[0]
        self.dphi = self.phi_beam - self.rf_station.phi_s[counter]

        # Possibility to add RF phase noise through the PL
        if self.RFnoise is not None:
            if self.noiseFB is not None:
                self.dphi += self.noiseFB.x * self.RFnoise.dphi[counter]
            else:
                if self.machine == 'PSB':
                    self.dphi = self.dphi
                else:
                    self.dphi += self.RFnoise.dphi[counter]

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
            (self.ring.beta[0, counter]**2.
             * self.ring.energy[0, counter])

    def radial_steering_from_freq(self):
        '''
        *Frequency and phase change for the current turn due to the radial steering program.*
        '''

        counter = self.rf_station.counter[0]

        self.radial_steering_domega_rf = - self.rf_station.omega_rf_d[0, counter] * \
            self.rf_station.eta_0[counter] / self.ring.alpha_0[0, counter] * \
            self.reference / self.ring.ring_radius

        self.rf_station.omega_rf[:, counter] += self.radial_steering_domega_rf * \
            self.rf_station.harmonic[:, counter] / \
            self.rf_station.harmonic[0, counter]

        # Update the RF phase of all systems for the next turn
        # Accumulated phase offset due to PL in each RF system
        self.rf_station.dphi_rf_steering += 2. * np.pi * self.rf_station.harmonic[:, counter] * \
            (self.rf_station.omega_rf[:, counter] -
             self.rf_station.omega_rf_d[:, counter]) / \
            self.rf_station.omega_rf_d[:, counter]

        # Total phase offset
        self.rf_station.phi_rf[:, counter] += self.rf_station.dphi_rf_steering

    def LHC_F(self):
        '''
        Calculation of the LHC RF frequency correction from the phase difference
        between beam and RF (actual synchronous phase). The transfer function is

        .. math::
            \\Delta \\omega_{rf}^{PL} = - g_{PL} (\\Delta\\varphi_{PL} + \\phi_{N}) 

        where the phase noise for the controlled blow-up can be optionally 
        activated.  
        Using 'gain2', a frequency loop can be activated in addition to remove
        long-term frequency drifts:

        .. math::
            \\Delta \\omega_{rf}^{FL} = - g_{FL} (\\omega_{rf} - h \\omega_{0})    
        '''

        counter = self.rf_station.counter[0]

        self.beam_phase()
        self.phase_difference()

        # Frequency correction from phase loop and frequency loop
        self.domega_rf = - self.gain * self.dphi \
            - self.gain2 * (self.rf_station.omega_rf[0, counter] -
                            self.rf_station.omega_rf_d[0, counter] +
                            self.reference)

    def SPS_F(self):
        '''
        Calculation of the SPS RF frequency correction from the phase
        difference between beam and RF (actual synchronous phase). Same as 
        LHC_F, except the calculation of the beam phase.
        '''

        counter = self.rf_station.counter[0]

        self.beam_phase_sharpWindow()
        self.phase_difference()

        # Frequency correction from phase loop and frequency loop
        self.domega_dphi = - self.gain * self.dphi
        self.domega_df = - self.gain2 * (self.rf_station.omega_rf[0, counter] -
                                         self.rf_station.omega_rf_d[0, counter])

        self.domega_rf = self.domega_dphi + self.domega_df

    def SPS_RL(self):
        '''
        Calculation of the SPS RF frequency correction from the phase difference
        between beam and RF (actual synchronous phase). The transfer function is

        .. math::
            \\Delta \\omega_{rf}^{PL} = - g_{PL} (\\Delta\\varphi_{PL} + \\phi_{N}) 

        where the phase noise for the controlled blow-up can be optionally 
        activated.  
        Using 'gain2', a radial loop can be activated in addition to remove
        long-term frequency drifts
        '''

        counter = self.rf_station.counter[0]

        if self.reference != 0:
            self.radial_steering_from_freq()

        self.beam_phase()
        self.phase_difference()
        self.radial_difference()

        # Frequency correction from phase loop and radial loop
        self.domega_dphi = - self.gain * self.dphi
        self.domega_dR = - bm.sign(self.rf_station.eta_0[counter]) * self.gain2 * \
            (self.reference - self.drho) / self.ring.ring_radius

        self.domega_rf = self.domega_dphi + self.domega_dR

    def LHC(self):
        '''
        Calculation of the LHC RF frequency correction from the phase difference
        between beam and RF (actual synchronous phase). The transfer function is

        .. math::
            \\Delta \\omega_{rf}^{PL} = - g_{PL} (\\Delta\\varphi_{PL} + \\phi_{N}) 

        where the phase noise for the controlled blow-up can be optionally 
        activated.  
        Using 'gain2', a synchro loop can be activated in addition to remove
        long-term frequency drifts:     

        .. math::
            \\Delta \\omega_{rf}^{SL} = - g_{SL} (y + a \\Delta\\varphi_{rf}) ,

        where we use the recursion

        .. math::
            y_{n+1} = (1 - \\tau) y_n + (1 - a) \\tau \\Delta\\varphi_{rf} ,

        with a and \tau being defined through the synchrotron frequency f_s and
        the synchrotron tune Q_s as

        .. math::
            a (f_s) \\equiv 5.25 - \\frac{f_s}{\\pi 40~\\text{Hz}} ,

        .. math::
            \\tau(f_s) \\equiv 2 \\pi Q_s \\sqrt{ \\frac{a}{1 + \\frac{g_{PL}}{g_{SL}} \\sqrt{\\frac{1 + 1/a}{1 + a}} }}
        '''

        counter = self.rf_station.counter[0]
        dphi_rf = self.rf_station.dphi_rf[0]

        self.beam_phase()
        self.phase_difference()

        # Frequency correction from phase loop and synchro loop
        self.domega_rf = - self.gain * self.dphi \
                         - self.gain2 * (self.lhc_y + self.lhc_a[counter]
                                         * (dphi_rf + self.reference))

        # Update recursion variable
        self.lhc_y = (1 - self.lhc_t[counter]) * self.lhc_y + \
                     (1 - self.lhc_a[counter]) * self.lhc_t[counter] * \
                     (dphi_rf + self.reference)

    def PSB(self):
        '''
        Phase and radial loops for PSB. See documentation on-line for details.
        '''

        # Average phase error while frequency is updated
        counter = self.rf_station.counter[0]
        self.beam_phase()
        self.phase_difference()

        self.dphi_sum += self.dphi

        # Phase and radial loop active on certain turns
        if counter == self.on_time[self.PL_counter] and counter >= self.delay:
            # Phase loop
            self.dphi_av = self.dphi_sum / (self.on_time[self.PL_counter]
                                            - self.on_time[self.PL_counter - 1])

            if self.RFnoise is not None:
                self.dphi_av += self.RFnoise.dphi[counter]

            self.domega_PL = 0.99803799 * self.domega_PL \
                + self.gain[counter] * (0.99901903 * self.dphi_av -
                                        0.99901003 * self.dphi_av_prev)

            self.dphi_av_prev = self.dphi_av
            self.dphi_sum = 0.

            # Radial loop
            self.dR_over_R = (self.rf_station.omega_rf[0, counter] -
                              self.rf_station.omega_rf_d[0, counter]) / (
                self.rf_station.omega_rf_d[0, counter] *
                (1. / (self.ring.alpha_0[0, counter] *
                       self.rf_station.gamma[counter]**2) - 1.))

            self.domega_RL = self.domega_RL + self.gain2[0][counter] * (self.dR_over_R
                                                                        - self.dR_over_R_prev) + self.gain2[1][counter] * self.dR_over_R

            self.dR_over_R_prev = self.dR_over_R

            # Counter to pick the next time step when the PL & RL will be active
            self.PL_counter += 1

        # Apply frequency correction
        self.domega_rf = - self.domega_PL - self.domega_RL

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
