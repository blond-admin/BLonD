# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Methods to generate RF phase noise from noise spectrum and feedback noise
amplitude as a function of bunch length**

:Authors: **Helga Timko**
'''

from __future__ import division, print_function
from builtins import range, object
import numpy as np
import numpy.random as rnd
from scipy.constants import c
from ..plots.plot import *
from ..plots.plot_llrf import *
from ..toolbox.next_regular import next_regular

# from input_parameters.rf_parameters import calculate_phi_s
cfwhm = np.sqrt(2. / np.log(2.))
import matplotlib.pyplot as plt
import sys


class FlatSpectrum(object):

    def __init__(self, Ring, RFStation, ideal_delta_f=1,
                 corr_time=10000, fmin_s0=0.8571, fmax_s0=1.1,
                 mode_amplitude='constant_rms', mode_amplitude_value=2.7219e-03,
                 seed1=1234, seed2=7564, initial_amplitude=1.e-6,
                 predistortion=None, continuous_phase=False,
                 print_option=True, initial_final_times=[0, 1], cumulative_time=None,
                 RF_mode='single', fs0_double=None, parent_folder=None,
                 folder_plots='fig_noise'):

        '''
        Generate phase noise from a band-limited spectrum.
        Input frequency band using 'fmin' and 'fmax' w.r.t. the synchrotron 
        frequency. Input double-sided spectrum amplitude [rad^2/Hz] using 
        'initial_amplitude'. Fix seeds to obtain reproducible phase noise.
        Select 'time_points' suitably to resolve the spectrum in frequency 
        domain. After 'corr_time' turns, the seed is changed to cut numerical
        correlated sequences of the random number generator.
        '''
        self.total_n_turns = Ring.n_turns
        self.initial_final_times = initial_final_times
        self.cumulative_time = cumulative_time
        if cumulative_time is not None:
            self.n_start_noise = np.where(cumulative_time>=self.initial_final_times[0])[0][0]
            self.n_end_noise = np.where(cumulative_time<=self.initial_final_times[1])[0][-1]
        else:
            self.n_start_noise = 0
            self.n_end_noise = -1

        if self.n_end_noise == -1:
            self.n_end_noise = self.total_n_turns + 1
        self.f0 = Ring.f_rev[self.n_start_noise:self.n_end_noise]
        self.delta_f = ideal_delta_f  # frequency resolution [Hz]
        self.corr = corr_time  # adjust noise every 'corr' time steps
        self.fmin_s0 = fmin_s0  # spectrum lower bound in synchr. freq.
        self.fmax_s0 = fmax_s0  # spectrum upper bound in synchr. freq.
        self.A_i = initial_amplitude  # initial spectrum amplitude [rad^2/Hz]
        self.seed1 = seed1
        self.seed2 = seed2
        self.predistortion = predistortion
        if self.predistortion == 'weightfunction':
            # Overwrite frequencies
            self.fmin_s0 = 0.8571
            self.fmax_s0 = 1.001
        if RF_mode == 'single':
            self.fs = RFStation.omega_s0[self.n_start_noise:self.n_end_noise] / (2 * np.pi)
        elif RF_mode == 'double':
            self.fs = fs0_double[self.n_start_noise:self.n_end_noise]
        self.n_turns = len(self.fs) - 1
        self.dphi = np.zeros(self.n_turns + 1)
        self.continuous_phase = continuous_phase
        if self.continuous_phase:
            self.dphi2 = np.zeros(int(self.n_turns + 1 + self.corr / 4))
        self.folder_plots = folder_plots
        self.print_option = print_option
        self.parent_folder = parent_folder
        self.mode_amplitude = mode_amplitude
        self.mode_amplitude_value = mode_amplitude_value

    def spectrum_to_phase_noise(self, freq, spectrum, transform=None):

        nf = len(spectrum)
        fmax = freq[nf - 1]

        # Resolution in time domain
        if transform == None or transform == 'r':
            nt = 2 * (nf - 1)
            dt = 1 / (2 * fmax)  # in [s]
        elif transform == 'c':
            nt = nf
            dt = 1. / fmax  # in [s]
        else:
            # NoiseError
            raise RuntimeError('ERROR: The choice of Fourier transform for the\
             RF noise generation could not be recognized. Use "r" or "c".')

        # Generate white noise in time domain
        rnd.seed(self.seed1)
        r1 = rnd.random_sample(nt)
        rnd.seed(self.seed2)
        r2 = rnd.random_sample(nt)
        if transform == None or transform == 'r':
            Gt = np.cos(2 * np.pi * r1) * np.sqrt(-2 * np.log(r2))
        elif transform == 'c':
            Gt = np.exp(2 * np.pi * 1j * r1) * np.sqrt(-2 * np.log(r2))

            # FFT to frequency domain
        if transform == None or transform == 'r':
            Gf = np.fft.rfft(Gt)
        elif transform == 'c':
            Gf = np.fft.fft(Gt)

            # Multiply by desired noise probability density
        if transform == None or transform == 'r':
            s = np.sqrt(2 * fmax * spectrum)  # in [rad]
        elif transform == 'c':
            s = np.sqrt(fmax * spectrum)  # in [rad]
        dPf = s * Gf.real + 1j * s * Gf.imag  # in [rad]

        # FFT back to time domain to get final phase shift
        if transform == None or transform == 'r':
            dPt = np.fft.irfft(dPf)  # in [rad]
        elif transform == 'c':
            dPt = np.fft.ifft(dPf)  # in [rad]

        # Use only real part for the phase shift and normalize
        self.t = np.linspace(0, float(nt * dt), nt)
        self.dphi_output = dPt.real

    def generate(self):

        for i in range(np.int(np.ceil(self.n_turns / self.corr))):

            # Scale amplitude to keep area (phase noise amplitude) constant
            k = i * self.corr  # current time step
            ampl = self.A_i * self.fs[0] / self.fs[k]

            # Calculate the frequency step
            f_max = self.f0[k] / 2
            n_points_pos_f_incl_zero = int(np.ceil(f_max / self.delta_f) + 1)
            nt = 2 * (n_points_pos_f_incl_zero - 1)
            nt_regular = next_regular(int(nt))
            if nt_regular % 2 != 0 or nt_regular < self.corr:
                # NoiseError
                raise RuntimeError('Error in noise generation!')
            n_points_pos_f_incl_zero = int(nt_regular / 2 + 1)
            freq = np.linspace(0, float(f_max), n_points_pos_f_incl_zero)
            delta_f = f_max / (n_points_pos_f_incl_zero - 1)

            # Construct spectrum
            f_down = self.fmin_s0[i]*self.fs[k]
            f_up = self.fmax_s0[i]*self.fs[k]
            nmin = np.int(np.floor(f_down/delta_f))  
            nmax = np.int(np.ceil(f_up/delta_f))
            if self.mode_amplitude == 'constant_rms':
                ampl = self.mode_amplitude_value**2/(f_up-f_down)/2
            
            # To compensate the notch due to PL at central frequency
            if self.predistortion == 'exponential':
                
                spectrum = np.concatenate((np.zeros(nmin), ampl*np.exp(
                    np.log(100.)*np.arange(0,nmax-nmin+1)/(nmax-nmin) ), 
                                           np.zeros(n_points_pos_f_incl_zero-nmax-1) ))
             
            elif self.predistortion == 'linear':

                spectrum = np.concatenate((np.zeros(nmin),
                                           np.linspace(0, float(ampl), nmax - nmin + 1),
                                           np.zeros(n_points_pos_f_incl_zero - nmax - 1)))

            elif self.predistortion == 'hyperbolic':

                spectrum = np.concatenate((np.zeros(nmin),
                                           ampl * np.ones(nmax - nmin + 1) * \
                                           1 / (1 + 0.99 * (nmin - np.arange(nmin, nmax + 1))
                                                / (nmax - nmin)), np.zeros(n_points_pos_f_incl_zero - nmax - 1)))

            elif self.predistortion == 'weightfunction':

                frel = freq[nmin:nmax + 1] / self.fs[k]  # frequency relative to fs0
                frel[np.where(frel > 0.999)[0]] = 0.999  # truncate center freqs
                sigma = 0.754  # rms bunch length in rad corresponding to 1.2 ns
                gamma = 0.577216
                weight = (4. * np.pi * frel / sigma ** 2) ** 2 * \
                         np.exp(-16. * (1. - frel) / sigma ** 2) + \
                         0.25 * (1 + 8. * frel / sigma ** 2 *
                                 np.exp(-8. * (1. - frel) / sigma ** 2) *
                                 (gamma + np.log(8. * (1. - frel) / sigma ** 2) +
                                  8. * (1. - frel) / sigma ** 2)) ** 2
                weight /= weight[0]  # normalise to have 1 at fmin
                spectrum = np.concatenate((np.zeros(nmin), ampl * weight,
                                           np.zeros(n_points_pos_f_incl_zero - nmax - 1)))

            else:
                spectrum = np.concatenate((np.zeros(nmin),
                                           ampl * np.ones(nmax - nmin + 1),
                                           np.zeros(n_points_pos_f_incl_zero - nmax - 1)))

                # Fill phase noise array
            if i < int(self.n_turns / self.corr) - 1:
                kmax = (i + 1) * self.corr
            else:
                kmax = self.n_turns + 1

            self.spectrum_to_phase_noise(freq, spectrum)
            self.seed1 += 239
            self.seed2 += 158
            self.dphi[k:kmax] = self.dphi_output[0:(kmax - k)]

            if self.continuous_phase:
                if i == 0:
                    self.spectrum_to_phase_noise(freq, spectrum)
                    self.seed1 += 239
                    self.seed2 += 158
                    self.dphi2[:int(self.corr / 4)] = self.dphi_output[:int(self.corr / 4)]

                self.spectrum_to_phase_noise(freq, spectrum)
                self.seed1 += 239
                self.seed2 += 158
                self.dphi2[int(k + self.corr / 4):int(kmax + self.corr / 4)] = self.dphi_output[0:(kmax - k)]

            if self.folder_plots != None:
                fig_folder(self.folder_plots)
                plot_noise_spectrum(freq, spectrum, sampling=1, figno=i,
                                    dirname=self.folder_plots)
                plot_phase_noise(self.t[0:(kmax - k)], self.dphi_output[0:(kmax - k)],
                                 sampling=1, figno=i, dirname=self.folder_plots)

            # Plot
            fft_phase_full_freq = np.fft.rfftfreq(nt_regular, 1 / (2 * f_max))
            fft_phase_full = np.fft.rfft(self.dphi_output)
            fft_phase_reduced_freq = np.fft.rfftfreq(nt_regular, 1 / (2 * f_max))
            fft_phase_reduced = np.fft.rfft(self.dphi[k:kmax], nt_regular)
            ax = plt.gca()
            ax2 = ax.twinx()
            ax.plot(freq / self.fs[k], spectrum, color='b')
            ax2.plot(fft_phase_full_freq / self.fs[k], np.abs(fft_phase_full), color='k')
            ax2.plot(fft_phase_reduced_freq / self.fs[k], np.abs(fft_phase_reduced), color='m')
            ax.set_xlabel(r"Frequency/$f_{s0}$ [1]")
            ax.set_ylabel(r"Noise spectral density [$\frac{rad^2}{Hz}$]")
            ax2.set_ylabel(r"|DFT($\phi_{noise}$)| [rad]", color='m')
            ax.set_xlim(0.7 * f_down / self.fs[k], 1.2 * f_up / self.fs[k])
            ax.axvline(f_down / self.fs[k], ls='--', color='r')
            ax.axvline(f_up / self.fs[k], ls='--', color='r')
            ax.axvline(self.fs[k] / self.fs[k], ls='--', color='r')
            ax.axhline(ampl, ls='--', color='r')
            plt.title(r'Chunk ' + str(i + 1), fontsize=20, fontweight='bold')
            plt.grid()
            plt.savefig(self.parent_folder + 'fig_noise/noise_spectrum_' "%d" % i + '.png', bbox_inches='tight')
            plt.clf()

            # Compute rms
            rms_noise = np.std(self.dphi_output)
            if self.print_option:
                print("RF noise for time step %.4e s (iter %d) has r.m.s. phase %.4e rad (%.3e deg)" \
                      % (self.t[1], i, rms_noise, rms_noise * 180 / np.pi))

        if self.continuous_phase:
            psi = np.arange(0, self.n_turns + 1) * 2 * np.pi / self.corr
            self.dphi = self.dphi * np.sin(psi[:self.n_turns + 1]) + self.dphi2[:(self.n_turns + 1)] * np.cos(
                psi[:self.n_turns + 1])

        if self.initial_final_turns[0] > 0 or self.initial_final_turns[1] < self.total_n_turns + 1:
            self.dphi = np.concatenate((np.zeros(self.initial_final_turns[0]), self.dphi,
                                        np.zeros(1 + self.total_n_turns - self.initial_final_turns[1])))

        ax = plt.gca()
        ax.plot(self.cumulative_time, self.dphi, color='m')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(r"Phase [rad]")
        for i in range(np.int(np.ceil(self.n_turns / self.corr))):
            k = i * self.corr
            ax.axvline(self.cumulative_time[self.n_start_noise + k], ls='--', color='b')
        plt.grid()
        plt.savefig(self.parent_folder + 'noise_time_domain.png', bbox_inches='tight')
        plt.clf()


class LHCNoiseFB(object):
    '''
    *Feedback on phase noise amplitude for LHC controlled longitudinal emittance
    blow-up using noise injection through cavity controller or phase loop.
    The feedback compares the FWHM bunch length of the bunch to a target value 
    and scales the phase noise to keep the targeted value.
    Activate the feedback either by passing it in RFStation or in
    the PhaseLoop object.
    Update the noise amplitude scaling using track().
    Pass the bunch pattern (occupied bucket numbers from 0...h-1) in buckets 
    for multi-bunch simulations; the feedback uses the average bunch length.*
    '''

    def __init__(self, RFStation, Profile, bl_target, gain=0.1e9,
                 factor=0.93, update_frequency=22500, variable_gain=True,
                 bunch_pattern=None):

        #: | *Import RFStation*
        self.rf_params = RFStation

        #: | *Import Profile*
        self.profile = Profile

        #: | *Phase noise scaling factor. Initially 0.*
        self.x = 0.

        #: | *Target bunch length [s], 4-sigma value.*        
        self.bl_targ = bl_target

        #: | *Measured bunch length [s], FWHM.*          
        self.bl_meas = bl_target

        #: | *Feedback recursion scaling factor.*  
        self.a = factor

        #: | *Update feedback every n_update turns.*
        self.n_update = update_frequency

        #: | *Switch to use constant or variable gain*
        self.variable_gain = variable_gain

        #: | *Feedback gain [1/s].*  
        if self.variable_gain == True:
            self.g = gain * (self.rf_params.omega_s0[0] /
                             self.rf_params.omega_s0) ** 2
        else:
            self.g = gain * np.ones(self.rf_params.n_turns + 1)

            #: | *Bunch pattern for multi-bunch simulations*
        self.bunch_pattern = bunch_pattern

        #: | *Function dictionary to calculate FWHM bunch length*
        fwhm_functions = {'single': self.fwhm_single_bunch,
                          'multi': self.fwhm_multi_bunch}
        if self.bunch_pattern == None:
            self.fwhm = fwhm_functions['single']
            self.bl_meas_bbb = None
        else:
            self.bunch_pattern = np.ascontiguousarray(self.bunch_pattern)
            self.bl_meas_bbb = np.zeros(len(self.bunch_pattern))
            self.fwhm = fwhm_functions['multi']

    def track(self):
        '''
        *Calculate PhaseNoise Feedback scaling factor as a function of measured
        FWHM bunch length.*
        '''

        # Track only in certain turns
        if (self.rf_params.counter[0] % self.n_update) == 0:

            # Update bunch length, every x turns determined in main file
            self.fwhm()

            # Update noise amplitude-scaling factor
            self.x = self.a * self.x + self.g[self.rf_params.counter[0]] * \
                     (self.bl_targ - self.bl_meas)

            # Limit to range [0,1]
            if self.x < 0:
                self.x = 0
            if self.x > 1:
                self.x = 1

    def fwhm_interpolation(self, index, half_height):

        time_resolution = self.profile.bin_centers[1] - self.profile.bin_centers[0]

        left = self.profile.bin_centers[index[0]] - (self.profile.n_macroparticles[index[0]] -
                                                     half_height) / (self.profile.n_macroparticles[index[0]] -
                                                                     self.profile.n_macroparticles[
                                                                         index[0] - 1]) * time_resolution

        right = self.profile.bin_centers[index[-1]] + (self.profile.n_macroparticles[index[-1]]
                                                       - half_height) / (self.profile.n_macroparticles[index[-1]] -
                                                                         self.profile.n_macroparticles[
                                                                             index[-1] + 1]) * time_resolution

        return cfwhm * (right - left)

    def fwhm_single_bunch(self):
        '''
        *Single-bunch FWHM bunch length calculation with interpolation.*
        '''

        half_height = np.max(self.profile.n_macroparticles) / 2.
        index = np.where(self.profile.n_macroparticles > half_height)[0]

        self.bl_meas = self.fwhm_interpolation(index, half_height)

    def fwhm_multi_bunch(self):
        '''
        *Multi-bunch FWHM bunch length calculation with interpolation.*
        '''

        # Find correct RF buckets
        phi_RF = self.rf_params.phi_RF[0, self.rf_params.counter[0]]
        omega_RF = self.rf_params.omega_RF[0, self.rf_params.counter[0]]
        bucket_min = (phi_RF + 2. * np.pi * self.bunch_pattern) / omega_RF
        bucket_max = bucket_min + 2. * np.pi / omega_RF

        # Bunch-by-bunch FWHM bunch length
        for i in range(len(self.bunch_pattern)):
            bind = np.where((self.profile.bin_centers - bucket_min[i]) *
                            (self.profile.bin_centers - bucket_max[i]) < 0)[0]
            hheight = np.max(self.profile.n_macroparticles[bind]) / 2.
            index = np.where(self.profile.n_macroparticles[bind] > hheight)[0]
            self.bl_meas_bbb[i] = self.fwhm_interpolation(bind[index], hheight)

        # Average FWHM bunch length            
        self.bl_meas = np.mean(self.bl_meas_bbb)


def noise_for_PSB(frequencies, times, delta_t, shape, phi_rms, 
                  growth_rate=None, seed = 1111, plot=False,
                  directory_plot_and_save = None, show_only=False):
        
        time_array_output = np.array([])
        phi_array_output = np.array([])
        
        for i in range(len(frequencies)):
            
            # Determination of nt and nt_regular for Fourier speed-up
            times[i][1] -= 1e-6
            duration = times[i][1]-times[i][0]
            number_intervals = int(duration/delta_t[i])+1
            if number_intervals % 2 == 0:
                number_intervals += 1
            delta_t_current = duration/number_intervals 
            nt = number_intervals + 1
            ### NEXT_REGULAR TO BE CHECKED, IT GIVES MEAN DIFFERENT FROM 0 FOR phi_output, nt_regular = next_regular(nt)    
            nt_regular = nt
            duration_regular = (nt_regular-1)*delta_t_current
            
            # White noise generation in time and frequency domain
            rnd.seed(seed)
            seed += 34324
            Gt = np.random.normal(loc=0, scale=1, size=nt_regular)
            Gf = np.fft.rfft(Gt)  
            
            # Spectrum calculation
            delta_f = 1/duration_regular
            f_max = 1/(2*delta_t_current)
            n_points_pos_f_incl_zero = int(nt_regular/2) + 1    
            nmin = int(np.floor(frequencies[i][0]/delta_f))  
            nmax = int(np.ceil(frequencies[i][1]/delta_f))
             
            if shape[i] == 'exponential':
                
                spectrum = np.concatenate((np.zeros(int(nmin)), np.exp(
                    np.log(growth_rate[i])*np.arange(0,int(nmax-nmin+1))/(nmax-nmin) ), 
                                           np.zeros(int(n_points_pos_f_incl_zero-nmax-1)) ))
             
            elif shape[i] == 'linear':
                
                spectrum = np.concatenate((np.zeros(nmin), 
                    np.linspace(0, growth_rate[i], nmax-nmin+1), np.zeros(n_points_pos_f_incl_zero-nmax-1)))   
                
            elif shape[i] == 'flat':
                
                spectrum = np.concatenate((np.zeros(nmin), 
                            np.ones(nmax-nmin+1), np.zeros(n_points_pos_f_incl_zero-nmax-1))) 
            
            # Multiplication of spectrum and white noise
            dPf = spectrum*Gf       
            dPt = np.fft.irfft(dPf)
            
            # Return real part 
            phi_output = dPt.real[:nt]
            rescaling_coefficient = phi_rms[i]/np.std(phi_output)
            phi_output = phi_output*rescaling_coefficient
            time_output = np.linspace(times[i][0]+1e-12, times[i][1]+1e-12,nt)
            
            index_min = np.argmin(np.abs(phi_output))
            phi_output = np.append(phi_output[index_min:], phi_output[:index_min])
            
            if plot:
                freq = np.linspace(0, f_max, n_points_pos_f_incl_zero)
                plt.figure('spectrum')
                plt.plot(freq, spectrum)
                
                plt.figure('product of white noise and spectrum, fft of phase noise')
                plt.plot(freq, np.abs(dPf)*rescaling_coefficient, color='r')
                plt.plot(np.fft.rfftfreq(len(phi_output),delta_t_current), np.abs(np.fft.rfft(phi_output)), color='k')
                
                plt.figure('time domain phase noise')
                plt.plot(time_output, phi_output, '.-')
                
            time_array_output = np.append(time_array_output, time_output)
            phi_array_output = np.append(phi_array_output, phi_output)
        
        if plot:
            if not show_only:
                np.save(directory_plot_and_save+'phase_program_blow_up', np.array([time_array_output, phi_array_output]))
            plt.figure('spectrum')
            plt.xlabel('frequency [Hz]')
            plt.ylabel(r'spectrum [$rad^2$/Hz]')
            if not show_only:
                plt.savefig(directory_plot_and_save+'spectrum.png')
            plt.figure('product of white noise and spectrum, fft of phase noise')
            plt.xlabel('frequency [Hz]')
            plt.ylabel(r'noise spectrum [rad]')
            if not show_only:
                plt.savefig(directory_plot_and_save+'noise_x_spectrum.png')
            plt.figure('time domain phase noise')
            plt.xlabel('time [s]')
            plt.ylabel('phase noise [rad]')
            if not show_only:
                plt.savefig(directory_plot_and_save+'phase_noise.png')
            if show_only:
                plt.show()
            plt.close('all')
        
        return time_array_output, phi_array_output
