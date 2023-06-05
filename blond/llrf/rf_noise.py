
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

from builtins import range

import numpy as np
import numpy.random as rnd

from ..plots.plot import fig_folder
from ..plots.plot_llrf import plot_phase_noise, plot_noise_spectrum
from ..toolbox.next_regular import next_regular

cfwhm = np.sqrt(2. / np.log(2.))


class FlatSpectrum:

    def __init__(self, Ring, RFStation, delta_f=1,
                 corr_time=10000, fmin_s0=0.8571, fmax_s0=1.1,
                 initial_amplitude=1.e-6, seed1=1234, seed2=7564,
                 predistortion=None, continuous_phase=False, folder_plots='fig_noise', print_option=True, initial_final_turns=[0, -1]):
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
        self.initial_final_turns = initial_final_turns
        if self.initial_final_turns[1] == -1:
            self.initial_final_turns[1] = self.total_n_turns + 1

        self.f0 = Ring.f_rev[self.initial_final_turns[0]:self.initial_final_turns[1]]  # revolution frequency in Hz
        self.delta_f = delta_f           # frequency resolution [Hz]
        self.corr = corr_time           # adjust noise every 'corr' time steps
        self.fmin_s0 = fmin_s0                # spectrum lower bound in synchr. freq.
        self.fmax_s0 = fmax_s0                # spectrum upper bound in synchr. freq.
        self.A_i = initial_amplitude    # initial spectrum amplitude [rad^2/Hz]
        self.seed1 = seed1
        self.seed2 = seed2
        self.predistortion = predistortion
        if self.predistortion == 'weightfunction':
            # Overwrite frequencies
            self.fmin_s0 = 0.8571
            self.fmax_s0 = 1.001
        self.fs = RFStation.omega_s0[self.initial_final_turns[0]:self.initial_final_turns[1]] / (2 * np.pi)  # synchrotron frequency in Hz
        self.n_turns = len(self.fs) - 1
        self.dphi = np.zeros(self.n_turns + 1)
        self.continuous_phase = continuous_phase
        if self.continuous_phase:
            self.dphi2 = np.zeros(self.n_turns + 1 + self.corr / 4)
        self.folder_plots = folder_plots
        self.print_option = print_option

    def spectrum_to_phase_noise(self, freq, spectrum, transform=None):

        nf = len(spectrum)
        fmax = freq[nf - 1]

        # Resolution in time domain
        if transform is None or transform == 'r':
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
        if transform is None or transform == 'r':
            Gt = np.cos(2 * np.pi * r1) * np.sqrt(-2 * np.log(r2))
        elif transform == 'c':
            Gt = np.exp(2 * np.pi * 1j * r1) * np.sqrt(-2 * np.log(r2))

        # FFT to frequency domain
        if transform is None or transform == 'r':
            Gf = np.fft.rfft(Gt)
        elif transform == 'c':
            Gf = np.fft.fft(Gt)

        # Multiply by desired noise probability density
        if transform is None or transform == 'r':
            s = np.sqrt(2 * fmax * spectrum)  # in [rad]
        elif transform == 'c':
            s = np.sqrt(fmax * spectrum)  # in [rad]
        dPf = s * Gf.real + 1j * s * Gf.imag  # in [rad]

        # FFT back to time domain to get final phase shift
        if transform is None or transform == 'r':
            dPt = np.fft.irfft(dPf)  # in [rad]
        elif transform == 'c':
            dPt = np.fft.ifft(dPf)  # in [rad]

        # Use only real part for the phase shift and normalize
        self.t = np.linspace(0, float(nt * dt), nt)
        self.dphi_output = dPt.real

    def generate(self):

        for i in range(0, int(np.ceil(self.n_turns / self.corr))):

            # Scale amplitude to keep area (phase noise amplitude) constant
            k = i * self.corr       # current time step
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
            nmin = int(np.floor(self.fmin_s0 * self.fs[k] / delta_f))
            nmax = int(np.ceil(self.fmax_s0 * self.fs[k] / delta_f))

            # To compensate the notch due to PL at central frequency
            if self.predistortion == 'exponential':

                spectrum = np.concatenate((np.zeros(nmin), ampl * np.exp(
                    np.log(100.) * np.arange(0, nmax - nmin + 1) / (nmax - nmin)),
                    np.zeros(n_points_pos_f_incl_zero - nmax - 1)))

            elif self.predistortion == 'linear':

                spectrum = np.concatenate((np.zeros(nmin),
                                           np.linspace(0, float(ampl), nmax - nmin + 1), np.zeros(n_points_pos_f_incl_zero - nmax - 1)))

            elif self.predistortion == 'hyperbolic':

                spectrum = np.concatenate((np.zeros(nmin),
                                           ampl * np.ones(nmax - nmin + 1) *
                                           1 / (1 + 0.99 * (nmin - np.arange(nmin, nmax + 1))
                                                / (nmax - nmin)), np.zeros(n_points_pos_f_incl_zero - nmax - 1)))

            elif self.predistortion == 'weightfunction':

                frel = freq[nmin:nmax + 1] / self.fs[k]  # frequency relative to fs0
                frel[np.where(frel > 0.999)[0]] = 0.999  # truncate center freqs
                sigma = 0.754  # rms bunch length in rad corresponding to 1.2 ns
                gamma = 0.577216
                weight = (4. * np.pi * frel / sigma**2)**2 * \
                    np.exp(-16. * (1. - frel) / sigma**2) + \
                    0.25 * (1 + 8. * frel / sigma**2 *
                            np.exp(-8. * (1. - frel) / sigma**2) *
                            (gamma + np.log(8. * (1. - frel) / sigma**2) +
                             8. * (1. - frel) / sigma**2))**2
                weight /= weight[0]  # normalise to have 1 at fmin
                spectrum = np.concatenate((np.zeros(nmin), ampl * weight,
                                           np.zeros(n_points_pos_f_incl_zero - nmax - 1)))

            else:
                spectrum = np.concatenate((np.zeros(nmin),
                                           ampl * np.ones(nmax - nmin + 1), np.zeros(n_points_pos_f_incl_zero - nmax - 1)))

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
                    self.dphi2[:self.corr / 4] = self.dphi_output[:self.corr / 4]

                self.spectrum_to_phase_noise(freq, spectrum)
                self.seed1 += 239
                self.seed2 += 158
                self.dphi2[(k + self.corr / 4):(kmax + self.corr / 4)] = self.dphi_output[0:(kmax - k)]

            if self.folder_plots is not None:
                fig_folder(self.folder_plots)
                plot_noise_spectrum(freq, spectrum, sampling=1, figno=i,
                                    dirname=self.folder_plots)
                plot_phase_noise(self.t[0:(kmax - k)], self.dphi_output[0:(kmax - k)],
                                 sampling=1, figno=i, dirname=self.folder_plots)

            rms_noise = np.std(self.dphi_output)
            if self.print_option:
                print("RF noise for time step %.4e s (iter %d) has r.m.s. phase %.4e rad (%.3e deg)"
                      % (self.t[1], i, rms_noise, rms_noise * 180 / np.pi))

        if self.continuous_phase:
            psi = np.arange(0, self.n_turns + 1) * 2 * np.pi / self.corr
            self.dphi = self.dphi * np.sin(psi[:self.n_turns + 1]) + self.dphi2[:(self.n_turns + 1)] * np.cos(psi[:self.n_turns + 1])

        if self.initial_final_turns[0] > 0 or self.initial_final_turns[1] < self.total_n_turns + 1:
            self.dphi = np.concatenate((np.zeros(self.initial_final_turns[0]), self.dphi, np.zeros(1 + self.total_n_turns - self.initial_final_turns[1])))


class LHCNoiseFB:
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
        if self.variable_gain:
            self.g = gain * (self.rf_params.omega_s0[0] /
                             self.rf_params.omega_s0)**2
        else:
            self.g = gain * np.ones(self.rf_params.n_turns + 1)

        #: | *Bunch pattern for multi-bunch simulations*
        self.bunch_pattern = bunch_pattern

        #: | *Function dictionary to calculate FWHM bunch length*
        fwhm_functions = {'single': self.fwhm_single_bunch,
                          'multi': self.fwhm_multi_bunch}
        if self.bunch_pattern is None:
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
                                                                     self.profile.n_macroparticles[index[0] - 1]) * time_resolution

        right = self.profile.bin_centers[index[-1]] + (self.profile.n_macroparticles[index[-1]]
                                                       - half_height) / (self.profile.n_macroparticles[index[-1]] -
                                                                         self.profile.n_macroparticles[index[-1] + 1]) * time_resolution

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
