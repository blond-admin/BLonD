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
from builtins import object
import numpy as np
from ..utils import bmath as bm
import matplotlib.pyplot as plt


class BeamFeedback(object):
    '''
    One-turn beam phase loop for different machines with different hardware. 
    Use 'period' for a phase loop that is active only in certain turns. 
    The phase loop acts directly on the RF frequency of all harmonics and
    affects the RF phase as well.
    '''

    def __init__(self, Ring, RFStation, Profile,
                 configuration,
                 PhaseNoise=None,
                 dphi_delay=None,
                 LHCNoiseFB=None, delay=0, delay_turns=1, n_bunches = 0,
                 bunch_spacing_buckets=0, slices_per_bucket=0, left_limit_first_bunch = 0):

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
        except:
            pass

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
                self.lhc_a = 5.25 - self.rf_station.omega_s0/(np.pi*40.)
                #: | *LHC Synchronisation loop time constant [turns]*
                self.lhc_t = (2*np.pi*self.rf_station.Q_s*np.sqrt(self.lhc_a)) / \
                    np.sqrt(1 + self.gain/self.gain2 *
                            np.sqrt((1 + 1/self.lhc_a)/(1 + self.lhc_a)))

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

        
        # SPS PL SL FL CONFIGURATION
        elif self.machine == 'SPS_PL_SL_FL':
            
            self.K_phi_n = self.config['K_phi_n']
            self.K_phi_nm1 = self.config['K_phi_nm1']
            self.K_eps_n = self.config['K_eps_n']
            self.K_Z_n = self.config['K_Z_n']
            self.K_a_n = self.config['K_a_n']
            self.K_b_n = self.config['K_b_n']
            self.phi_sync = self.config['phi_sync']
            self.global_gain = self.config['global_gain']
            self.dphi_prev = 0
            self.epsilon_prev = 0
            self.Zeta = 0
            self.Alpha = 0
            self.Alpha_prev = 0
        
            
        # SPS PL RL CONFIGURATION
        elif self.machine == 'SPS_PL_RL':
            
            self.K_phi = self.config['K_phi']
            self.K_R = self.config['K_R']
            self.K_Gamma = self.config['K_Gamma']
            self.S_radSteer = self.config['S_radSteer']
            self.global_gain = self.config['global_gain']
            self.dR_mS_norm_prev = 0
            self.Gamma = 0
            self.mean_dE_method = 'particles' # particles, slices
            self.dE_min = None
            self.dE_max = None

        # PSB CONFIGURATION
        elif self.machine == 'PSB':

            self.gain = self.gain * np.ones(Ring.n_turns+1)

            #: | *Radial loop gain, proportional [1] and integral [1/s].*
            if 'RL_gain' not in self.config:
                self.gain2 = [0., 0.]
            else:
                self.gain2 = self.config['RL_gain']

            self.gain2[0] = self.gain2[0] * np.ones(Ring.n_turns+1)
            self.gain2[1] = self.gain2[1] * np.ones(Ring.n_turns+1)

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
        # self.RFnoise = PhaseNoise
        # if (self.RFnoise != None
        #         and (len(self.RFnoise.dphi) != Ring.n_turns + 1)):
        #     # PhaseNoiseError
        #     raise RuntimeError(
        #         'Phase noise has to have a length of n_turns + 1')

        #: | *Optional import of amplitude-scaling feedback object LHCNoiseFB*
        self.noiseFB = LHCNoiseFB

        self.PhaseNoise = PhaseNoise
        self.dphi_delay = dphi_delay
        self.delay_turns = delay_turns
        self.n_bunches = n_bunches
        self.bunch_spacing_buckets = bunch_spacing_buckets
        self.slices_per_bucket = slices_per_bucket
        self.left_limit_first_bunch = left_limit_first_bunch


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
        counter = self.rf_station.counter[0] + self.delay_turns
        if counter < self.ring.n_turns+1:
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
                    except:
                        self.on_time = np.append(self.on_time, 0)
                        return
                self.on_time = np.append(self.on_time, n-1)
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
            # indexes = np.ones(self.profile.n_slices, dtype=bool)
            # time_offset = 0.0
            coeff = bm.beam_phase(self.profile.bin_centers,
                                  self.profile.n_macroparticles,
                                  self.alpha, omega_rf, phi_rf,
                                  self.profile.bin_size)
        else:
            indexes = self.profile.bin_centers >= self.time_offset
            time_offset = self.time_offset
            # Convolve with window function
            scoeff = np.trapz(np.exp(self.alpha*(self.profile.bin_centers[indexes] -
                                                 time_offset)) *
                              np.sin(omega_rf*self.profile.bin_centers[indexes] +
                                     phi_rf) *
                              self.profile.n_macroparticles[indexes],
                              dx=self.profile.bin_size)
            ccoeff = np.trapz(np.exp(self.alpha*(self.profile.bin_centers[indexes] -
                                                 time_offset)) *
                              np.cos(omega_rf*self.profile.bin_centers[indexes] +
                                     phi_rf) *
                              self.profile.n_macroparticles[indexes],
                              dx=self.profile.bin_size)
            coeff = scoeff/ccoeff

        # Project beam phase to (pi/2,3pi/2) range
        self.phi_beam = np.arctan(coeff) + np.pi
    
    
    def beam_phase_multibunch(self, debug = False):
        
        counter = self.rf_station.counter[0]
        # Fetch RF parameters
        omega_rf = self.rf_station.omega_rf[0, counter]
        phi_rf = self.rf_station.phi_rf[0, counter]
        trf0 = self.ring.t_rev[0]/self.rf_station.harmonic[0,0]
        # Compute the slice shift due to phi_rf != 0
        actual_trf = 2*np.pi/omega_rf
        if self.rf_station.eta_0[counter] < 0:
            shift_time = (phi_rf-np.pi)/omega_rf # [s]
        else:
            shift_time = phi_rf/omega_rf # [s]
        shift_time_slice = int(shift_time/self.profile.bin_size)
        margin_slices = int(actual_trf*self.slices_per_bucket/trf0/4)
        # Sum of the phases of the different bunches (between -pi/2 and pi/2)
        phi_sum = 0
        self.phibc_array = np.zeros(self.n_bunches)
        for i in range(self.n_bunches):
            delta_left_limit_time = i*self.bunch_spacing_buckets*actual_trf
            delta_right_limit_time = delta_left_limit_time + actual_trf
            n_slices_left = int(delta_left_limit_time*self.slices_per_bucket/trf0)
            n_slices_right = int(delta_right_limit_time*self.slices_per_bucket/trf0)
            left_limit_slicing = self.left_limit_first_bunch - shift_time_slice + n_slices_left - margin_slices
            right_limit_slicing = self.left_limit_first_bunch - shift_time_slice + n_slices_right + margin_slices
            slice_bin_centers = self.profile.bin_centers[left_limit_slicing:right_limit_slicing]
            slice_n_macroparticles = self.profile.n_macroparticles[left_limit_slicing:right_limit_slicing]
            scoeff = np.trapz(np.sin(omega_rf*slice_bin_centers + phi_rf) * slice_n_macroparticles,
                              dx=self.profile.bin_size)
            ccoeff = np.trapz(np.cos(omega_rf*slice_bin_centers + phi_rf) * slice_n_macroparticles,
                              dx=self.profile.bin_size)
            # FOR DEBUG: weirdly, the C++ equivalent routine is slower (almost factor 2) 
            # coeff = bm.beam_phase_fast(slice_bin_centers,
            #                       slice_n_macroparticles,
            #                       omega_rf, phi_rf,
            #                       self.profile.bin_size)
            self.phibc_array[i] = np.arctan(scoeff/ccoeff)
        phi_sum = np.sum(self.phibc_array)
        # Divide by # bunches to obtain the average
        self.phi_beam = phi_sum/self.n_bunches + np.pi
        
        if debug:
            plt.figure('anim')
            plt.ion()
            plt.plot(slice_bin_centers*omega_rf, slice_n_macroparticles)
            plt.axvline(self.phi_beam, color ='r')
            plt.axvline(self.phi_beam-(phi_rf-np.pi), color ='g')
            plt.pause(0.0000001)
            plt.clf()
    
    
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
            indexes = np.logical_and((self.time_offset - np.pi / omega_rf)
                                     <= self.profile.bin_centers,
                                     self.profile.bin_centers
                                     <= (-1/self.alpha + self.time_offset -
                                         2 * np.pi / omega_rf))
        else:
            indexes = np.ones(self.profile.n_slices, dtype=bool)

        # Convolve with window function
        scoeff = np.trapz(np.sin(omega_rf*self.profile.bin_centers[indexes]
                                 + phi_rf)
                          * self.profile.n_macroparticles[indexes],
                          dx=self.profile.bin_size)
        ccoeff = np.trapz(np.cos(omega_rf*self.profile.bin_centers[indexes]
                                 + phi_rf) *
                          self.profile.n_macroparticles[indexes],
                          dx=self.profile.bin_size)

        # Project beam phase to (pi/2,3pi/2) range
        self.phi_beam = np.arctan(scoeff/ccoeff) + np.pi


    def phase_difference(self):
        '''
        *Phase difference between beam and RF phase of the main RF system.
        Optional: add RF phase noise through dphi directly.*
        '''

        # Correct for design stable phase
        counter = self.rf_station.counter[0]
        self.dphi = self.phi_beam - self.rf_station.phi_s[counter]
        if self.dphi_delay != None:
            self.dphi += self.dphi_delay

        # Possibility to add RF phase noise through the PL
        if isinstance(self.PhaseNoise, np.ndarray):
            if self.noiseFB != None:
                self.dphi += self.noiseFB.x*self.PhaseNoise[counter]
            else:
                if self.machine == 'PSB':
                    self.dphi = self.dphi
                else:
                    self.dphi += self.PhaseNoise[counter]
            

    def radial_difference(self):
        '''
        *Radial difference between beam and design orbit.*
        '''

        counter = self.rf_station.counter[0]

        # Correct for design orbit
#        self.average_dE = np.mean(self.profile.Beam.dE[(self.profile.Beam.dt >
#            self.profile.bin_centers[0])*(self.profile.Beam.dt <
#                                         self.profile.bin_centers[-1])])
        self.average_dE = np.mean(self.profile.Beam.dE[::self.sample_dE])   

        self.drho = self.ring.alpha_0[0, counter] * \
            self.ring.ring_radius*self.average_dE / \
            (self.ring.beta[0, counter]**2.
             * self.ring.energy[0, counter])

    def radial_steering_from_freq(self):
        '''
        *Frequency and phase change for the current turn due to the radial steering program.*
        '''

        counter = self.rf_station.counter[0]

        self.radial_steering_domega_rf = - self.rf_station.omega_rf_d[0, counter] * \
            self.rf_station.eta_0[counter]/self.ring.alpha_0[0, counter] * \
            self.reference/self.ring.ring_radius

        self.rf_station.omega_rf[:, counter] += self.radial_steering_domega_rf * \
            self.rf_station.harmonic[:, counter] / \
            self.rf_station.harmonic[0, counter]

        # Update the RF phase of all systems for the next turn
        # Accumulated phase offset due to PL in each RF system
        self.rf_station.dphi_rf_steering += 2.*np.pi*self.rf_station.harmonic[:, counter] * \
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
        self.domega_rf = - self.gain*self.dphi \
            - self.gain2*(self.rf_station.omega_rf[0, counter] -
               self.rf_station.omega_rf_d[0, counter] +
                          self.reference)

    def SPS_PL_SL_FL(self):
        '''
        Calculation of the SPS RF frequency correction from the phase
        difference between beam and RF (actual synchronous phase). Same as 
        LHC_F, except the calculation of the beam phase.
        '''

        counter = self.rf_station.counter[0]
        
        # Phase loop
        self.beam_phase_multibunch()
        self.phase_difference()
        self.domega_dphi = -self.K_phi_n[counter]*self.dphi - self.K_phi_nm1[counter]*self.dphi_prev
        
        # Synchro Loop
        self.epsilon= self.rf_station.phi_rf[0, counter] - self.phi_sync[counter]
        self.Zeta += self.epsilon_prev
        self.domega_sync = -self.K_eps_n[counter]*self.epsilon - self.K_Z_n[counter]*self.Zeta
                            
        # Frequency Loop
        self.domega_freq = - self.K_a_n[counter]*self.Alpha - self.K_b_n[counter]*self.Alpha_prev

        # Total frequency correction
        self.domega_rf = self.domega_dphi + self.domega_sync + self.domega_freq
        
        # Update some parameters for the next turn
        self.Alpha_prev = self.Alpha
        self.Alpha = self.domega_rf*self.rf_station.t_rev[counter]
        self.epsilon_prev = self.epsilon
        self.dphi_prev = self.dphi
        
        # Apply global gain
        self.domega_rf *= self.global_gain[counter]

    def SPS_PL_RL(self):
        
        counter = self.rf_station.counter[0]
        
        # Phase loop
        self.beam_phase_multibunch()
        self.phase_difference()
        self.domega_dphi = -self.K_phi[counter]*self.dphi
        
        # Radial Loop
        if self.mean_dE_method == 'particles':
            indices_dE = bm.where(self.profile.Beam.dE, more_than=self.dE_min, less_than=self.dE_max)
            self.average_dE = bm.mean(self.profile.Beam.dE[indices_dE])
        self.dR = self.ring.alpha_0[0, counter]*self.ring.ring_radius*self.average_dE / \
                  (self.ring.beta[0, counter]**2 * self.ring.energy[0, counter])
        self.domega_rad = -self.K_R[counter]*self.dR/self.ring.ring_radius -self.K_Gamma[counter]*self.Gamma

        # Total frequency correction
        self.domega_rf = self.domega_dphi + self.domega_rad
        
        # Update some parameters for the next turn
        self.Gamma += (self.dR-self.S_radSteer[counter])/self.ring.ring_radius
        
        # Apply global gain
        self.domega_rf *= self.global_gain[counter]

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
        self.domega_rf = - self.gain*self.dphi \
                         - self.gain2*(self.lhc_y + self.lhc_a[counter]
                                       * (dphi_rf + self.reference))

        # Update recursion variable
        self.lhc_y = (1 - self.lhc_t[counter])*self.lhc_y + \
                     (1 - self.lhc_a[counter])*self.lhc_t[counter] * \
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
                                            - self.on_time[self.PL_counter-1])

            if self.PhaseNoise:
                self.dphi_av += self.PhaseNoise[counter]

            self.domega_PL = 0.99803799*self.domega_PL \
                + self.gain[counter]*(0.99901903*self.dphi_av -
                                      0.99901003*self.dphi_av_prev)

            self.dphi_av_prev = self.dphi_av
            self.dphi_sum = 0.

            # Radial loop
            self.dR_over_R = (self.rf_station.omega_rf[0, counter] -
                              self.rf_station.omega_rf_d[0, counter])/(
                self.rf_station.omega_rf_d[0, counter] *
                         (1./(self.ring.alpha_0[0, counter] *
                              self.rf_station.gamma[counter]**2) - 1.))

            self.domega_RL = self.domega_RL + self.gain2[0][counter]*(self.dR_over_R
                                                                      - self.dR_over_R_prev) + self.gain2[1][counter]*self.dR_over_R

            self.dR_over_R_prev = self.dR_over_R

            # Counter to pick the next time step when the PL & RL will be active
            self.PL_counter += 1

        # Apply frequency correction
        self.domega_rf = - self.domega_PL - self.domega_RL


    def to_gpu(self):
        '''
        Transfer all necessary arrays to the GPU
        '''
        # Check if to_gpu has been invoked already
        if hasattr(self, '__device') and self.__device == 'GPU':
            return

        assert bm.device == 'GPU'
        # No arrays need to be transfered

        # to make sure it will not be called again
        self.__device = 'GPU'

    def to_cpu(self):
        '''
        Transfer all necessary arrays back to the CPU
        '''
        # Check if to_cpu has been invoked already
        if hasattr(self, '__device') and self.__device == 'CPU':
            return

        assert bm.device == 'CPU'
        # No arrays need to be transfered

        # to make sure it will not be called again
        self.__device = 'CPU'
