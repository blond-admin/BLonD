
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to compute intensity effects**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**
'''

from __future__ import division
import numpy as np
from numpy.fft import irfft


class TotalInducedVoltage(object):
    '''
    *Object gathering all the induced voltage contributions. The input is a 
    list of objects able to compute induced voltages (InducedVoltageTime, 
    InducedVoltageFreq, InductiveImpedance). All the induced voltages will
    be summed in order to reduce the computing time. All the induced
    voltages should have the same slicing resolution.*
    '''
    
    def __init__(self, Slices, induced_voltage_list):
        
        
        #: *Copy of the Slices object in order to access the profile info.*
        self.slices = Slices
        
        # The slicing has to be done in 'tau' in order to use intensity effects
        if self.slices.slicing_coord is not 'tau':
            raise RuntimeError('The slicing has to be done in tau (slicing_coord option) in order to use intensity effects !')
        
        #: *Induced voltage list.*
        self.induced_voltage_list = induced_voltage_list
        
        #: *Induced voltage from the sum of the wake sources in [V]*
        self.induced_voltage = 0
        
        #: *Time array of the wake in [s]*
        self.time_array = self.slices.bins_centers
        
        
    def reprocess(self, new_slicing):
        '''
        *Reprocess the impedance contributions with respect to the new_slicing.*
        '''
        
        self.slices = new_slicing
        
        for induced_voltage_object in self.induced_voltage_list:
            induced_voltage_object.reprocess(self.slices)
                    

    def induced_voltage_sum(self, Beam, length = 'slice_frame'):
        '''
        *Method to sum all the induced voltages in one single array.*
        '''
        
        temp_induced_voltage = 0
        extended_induced_voltage = 0
        for induced_voltage_object in self.induced_voltage_list:
            if isinstance(length, int):
                extended_induced_voltage += induced_voltage_object.induced_voltage_generation(Beam, length)
            else:
                induced_voltage_object.induced_voltage_generation(Beam, length)
            temp_induced_voltage += induced_voltage_object.induced_voltage
            
        self.induced_voltage = temp_induced_voltage
        
        if isinstance(length, int):
            return extended_induced_voltage
    
    
    def track(self, Beam, no_update_induced_voltage = False):
        '''
        *Track method to apply the induced voltage kick on the beam.*
        '''
        
        if not no_update_induced_voltage:
            self.induced_voltage_sum(Beam)
        
        induced_voltage_kick = np.interp(Beam.tau, self.slices.bins_centers, self.induced_voltage)
        Beam.dE += induced_voltage_kick



class InducedVoltageTime(object):
    '''
    *Induced voltage derived from the sum of several wake fields (time domain).*
    '''
    
    def __init__(self, Slices, wake_source_list):       
        
        #: *Copy of the Slices object in order to access the profile info.*
        self.slices = Slices
        
        # The slicing has to be done in 'tau' in order to use intensity effects
        if self.slices.slicing_coord is not 'tau':
            raise RuntimeError('The slicing has to be done in tau (slicing_coord option) in order to use intensity effects !')
        
        #: *Wake sources inputed as a list (eg: list of BBResonators objects)*
        self.wake_source_list = wake_source_list
        
        #: *Time array of the wake in [s]*
        self.time_array = 0
        
        #: *Total wake array of all sources in* [:math:`\Omega / s`]
        self.total_wake = 0
        
        #: *Induced voltage from the sum of the wake sources in [V]*
        self.induced_voltage = 0
        
        # Pre-processing the wakes
        self.time_array = self.slices.bins_centers - self.slices.bins_centers[0]
        self.sum_wakes(self.time_array)
            
            
    def reprocess(self, new_slicing):
        '''
        *Reprocess the wake contributions with respect to the new_slicing.*
        '''
        
        self.slices = new_slicing
        
        self.time_array = self.slices.bins_centers - self.slices.bins_centers[0]
        self.sum_wakes(self.time_array)
    
    
    def sum_wakes(self, time_array):
        '''
        *Summing all the wake contributions in one total wake.*
        '''
        
        self.total_wake = np.zeros(time_array.shape)
        for wake_object in self.wake_source_list:
            wake_object.wake_calc(time_array)
            self.total_wake += wake_object.wake
        
    
    def induced_voltage_generation(self, Beam, length = 'slice_frame'): 
        '''
        *Method to calculate the induced voltage from wakes with convolution.*
        '''
        
        induced_voltage = - Beam.charge * Beam.intensity / Beam.n_macroparticles * np.convolve(self.total_wake, self.slices.n_macroparticles)
        
        self.induced_voltage = induced_voltage[0:self.slices.n_slices]
        
        if isinstance(length, int):
            max_length = len(induced_voltage)
            if length > max_length:
                induced_voltage = np.lib.pad(induced_voltage, (0,length - max_length), 'constant', constant_values=(0,0))
            return induced_voltage[0:length]
        
            
    def track(self, Beam):
        '''
        *Tracking method.*
        '''
        
        self.induced_voltage_generation(Beam)
        induced_voltage_kick = np.interp(Beam.tau, self.slices.bins_centers, self.induced_voltage)
        Beam.dE += induced_voltage_kick
    
    
    
class InducedVoltageFreq(object):
    '''
    *Induced voltage derived from the sum of several impedances.
    frequency_resolution is equal to 1/(dist_centers * n) where dist_centers is
    the distance between the centers of two consecutive slides and (n/2 + 1)
    is the number of sampling points for the frequency array; see the 
    frequency_array method.
    Sum_slopes_from_induc_imp is the sum of all the inductances derived from
    all the inductive impedance, included space charge; see in addition the
    ind_vol_derivative method.
    The frequency resolution is defined by your input, but this value will
    be adapted in order to optimize the FFT. The number of points used in the
    FFT should be a power of 2, to be faster, but this number of points also
    changes the frequency resolution. The frequency is then set to be the
    closest power of two to have the closest resolution wrt your input. The
    way the code chooses the power is set by the freq_res_option. If this is set
    to 'round' (default), the closest (higher or lower) resolution that also 
    fulfills optimisation will be used. If set to 'best', the frequency resolution
    will be at least your input, so you always have a better resolution.*
    '''
        
    def __init__(self, Slices, impedance_source_list, frequency_resolution_input, 
                 freq_res_option = 'round'):
    

        #: *Copy of the Slices object in order to access the profile info.*
        self.slices = Slices
        
        # The slicing has to be done in 'tau' in order to use intensity effects
        if self.slices.slicing_coord is not 'tau':
            raise RuntimeError('The slicing has to be done in tau (slicing_coord option) in order to use intensity effects !')
        
        #: *Impedance sources inputed as a list (eg: list of BBResonators objects)*
        self.impedance_source_list = impedance_source_list
        
        time_resolution = (self.slices.bins_centers[1] - self.slices.bins_centers[0])
        
        #: *Frequency resolution calculation option*
        self.freq_res_option = freq_res_option
        
        #: *Input frequency resolution in [Hz], the beam profile sampling for the spectrum
        #: will be adapted according to the freq_res_option.*
        self.frequency_resolution_input = frequency_resolution_input
        
        if self.freq_res_option is 'round':
            #: *Number of points used to FFT the beam profile (by padding zeros), 
            #: this is calculated in order to have at least the input 
            #: frequency_resolution.*
            self.n_fft_sampling = 2**int(np.round(np.log(1 / (self.frequency_resolution_input * time_resolution)) / np.log(2)))
        elif self.freq_res_option is 'best':
            self.n_fft_sampling = 2**int(np.ceil(np.log(1 / (self.frequency_resolution_input * time_resolution)) / np.log(2)))
        else:
            raise RuntimeError('The input freq_res_option is not recognized')
        
        if self.n_fft_sampling < self.slices.n_slices:
            print 'The input frequency resolution step is too big, and the whole \
                   bunch is not sliced... The number of sampling points for the \
                   FFT is corrected in order to sample the whole bunch (and \
                   you might consider changing the input in order to have \
                   a finer resolution).'
            frequency_resolution = 1 / (self.slices.bins_centers[-1] - self.slices.bins_centers[0])
            self.n_fft_sampling = 2**int(np.ceil(np.log(1 / (frequency_resolution * time_resolution)) / np.log(2)))
        
        #: *Real frequency resolution in [Hz], according to the obtained n_fft_sampling.*
        self.frequency_resolution = 1 / (self.n_fft_sampling * time_resolution)

        self.slices.beam_spectrum_generation(self.n_fft_sampling)
        #: *Frequency array of the impedance in [Hz]*
        self.frequency_array = self.slices.beam_spectrum_freq
        
        #: *Total impedance array of all sources in* [:math:`\Omega`]
        self.total_impedance = 0
        self.sum_impedances(self.frequency_array)
        
        #: *Induced voltage from the sum of the wake sources in [V]*
        self.induced_voltage = 0

        
    def reprocess(self, new_slicing):
        '''
        *Reprocess the impedance contributions with respect to the new_slicing.*
        '''
        
        self.slices = new_slicing
        
        time_resolution = (self.slices.bins_centers[1] - self.slices.bins_centers[0])
        if self.freq_res_option is 'round':
            #: *Number of points used to FFT the beam profile (by padding zeros), 
            #: this is calculated in order to have at least the input 
            #: frequency_resolution.*
            self.n_fft_sampling = 2**int(np.round(np.log(1 / (self.frequency_resolution_input * time_resolution)) / np.log(2)))
        elif self.freq_res_option is 'best':
            self.n_fft_sampling = 2**int(np.ceil(np.log(1 / (self.frequency_resolution_input * time_resolution)) / np.log(2)))
        else:
            raise RuntimeError('The input freq_res_option is not recognized')
        
        if self.n_fft_sampling < self.slices.n_slices:
            print 'The input frequency resolution step is too big, and the whole \
                   bunch is not sliced... The number of sampling points for the \
                   FFT is corrected in order to sample the whole bunch (and \
                   you might consider changing the input in order to have \
                   a finer resolution).'
            frequency_resolution = 1 / (self.slices.bins_centers[-1] - self.slices.bins_centers[0])
            self.n_fft_sampling = 2**int(np.ceil(np.log(1 / (frequency_resolution * time_resolution)) / np.log(2)))
        
        #: *Frequency resolution in [Hz], the beam profile sampling for the spectrum
        #: will be adapted accordingly.*
        self.frequency_resolution = 1 / (self.n_fft_sampling * time_resolution)

        self.slices.beam_spectrum_generation(self.n_fft_sampling, only_rfft = True)
        #: *Frequency array of the impedance in [Hz]*
        self.frequency_array = self.slices.beam_spectrum_freq
        
        #: *Total impedance array of all sources in* [:math:`\Omega`]
        self.total_impedance = 0
        self.sum_impedances(self.frequency_array)
            
    
    def sum_impedances(self, frequency_array):
        '''
        *Summing all the wake contributions in one total impedance.*
        '''
        
        self.total_impedance = np.zeros(frequency_array.shape) + 0j
        for imped_object in self.impedance_source_list:
            imped_object.imped_calc(frequency_array)
            self.total_impedance += imped_object.impedance

        
    def induced_voltage_generation(self, Beam, length = 'slice_frame'):
        '''
        *Method to calculate the induced voltage from the inverse FFT of the
        impedance times the spectrum (fourier convolution).*
        '''
        
        self.slices.beam_spectrum_generation(self.n_fft_sampling, filter_option = None)
        induced_voltage = - Beam.charge * Beam.intensity / Beam.n_macroparticles * irfft(self.total_impedance * self.slices.beam_spectrum) * self.slices.beam_spectrum_freq[1] * 2*(len(self.slices.beam_spectrum)-1) 
        
        self.induced_voltage = induced_voltage[0:self.slices.n_slices]
        
        if isinstance(length, int):
            max_length = len(induced_voltage)
            if length > max_length:
                induced_voltage = np.lib.pad(induced_voltage, (0, length-max_length), 'constant', constant_values=(0,0))
            return induced_voltage[0:length]
    
    
    def track(self, Beam):
        '''
        *Tracking method.*
        '''
        
        self.induced_voltage_generation(Beam)
        
        induced_voltage_kick = np.interp(Beam.tau, self.slices.bins_centers, self.induced_voltage)
        Beam.dE += induced_voltage_kick        
        
        
class InductiveImpedance(object):
    '''
    *Constant imaginary Z/n impedance. This needs to be extended to the
    cases where there is acceleration as the revolution frequency f0 used
    in the calculation of n=f/f0 is changing (general_params as input ?).*
    '''
    
    def __init__(self, Slices, Z_over_n, revolution_frequency, current_turn, deriv_mode = 'gradient'):
        
        #: *Copy of the Slices object in order to access the profile info.*
        self.slices = Slices
        
        # The slicing has to be done in 'tau' in order to use intensity effects
        if self.slices.slicing_coord is not 'tau':
            raise RuntimeError('The slicing has to be done in tau (slicing_coord option) in order to use intensity effects !')
        
        #: *Constant imaginary Z/n in* [:math:`\Omega / Hz`]
        self.Z_over_n = Z_over_n
        
        #: *Revolution frequency in [Hz]*
        self.revolution_frequency = revolution_frequency
        
        #: *Frequency array of the impedance in [Hz]*
        self.freq_array = 0
        
        #: *Impedance array in* [:math:`\Omega`]
        self.impedance = 0
        
        #: *Induced voltage from the sum of the wake sources in [V]*
        self.induced_voltage = 0
        
        #: *Derivation method to compute induced voltage*
        self.deriv_mode = deriv_mode
        
        #: *Current turn taken from RFSectionParameters*
        self.current_turn = current_turn 
        
        
    def reprocess(self, new_slicing):
        '''
        *Reprocess the impedance contributions with respect to the new_slicing.*
        '''
        
        self.slices = new_slicing
        
        
    def imped_calc(self, freq_array):
        '''
        *Impedance calculation method as a function of frequency.*
        '''    
        
        self.freq_array = freq_array
        self.impedance = (self.freq_array / self.revolution_frequency) * \
                         self.Z_over_n * 1j
                         

    def induced_voltage_generation(self, Beam, length = 'slice_frame'):
        '''
        *Method to calculate the induced voltage through the derivative of the
        profile; the impedance must be of inductive type.*
        '''
        index = self.current_turn[0] + 1
        
        induced_voltage = - Beam.charge / (2 * np.pi) * Beam.intensity / Beam.n_macroparticles * \
            self.Z_over_n[0][index] / self.revolution_frequency[index] * \
            self.slices.beam_profile_derivative(self.deriv_mode)[1] / \
            (self.slices.bins_centers[1] - self.slices.bins_centers[0])
            
        self.induced_voltage = induced_voltage[0:self.slices.n_slices]
        
        if isinstance(length, int):
            max_length = len(induced_voltage)
            if length > max_length:
                induced_voltage = np.lib.pad(self.induced_voltage, (0,length - max_length), 'constant', constant_values=(0,0))
            return induced_voltage[0:length]
                            
                            
    def track(self, Beam):
        '''
        *Track method.*
        '''
        
        self.induced_voltage_generation(Beam)
        
        induced_voltage_kick = np.interp(Beam.tau, self.slices.bins_centers, self.induced_voltage)
        Beam.dE += induced_voltage_kick
    
    
    
class InputTable(object):
    '''
    *Intensity effects from impedance and wake tables.
    If this constructor takes just two arguments, then a wake table is passed;
    if it takes three arguments, then an impedance table is passed. Be careful
    that if you input a wake, the input wake for W(t=0) should be already 
    divided by two (beam loading theorem) ; and that if you input impedance, 
    only the positive  frequencies of the impedance is needed (the impedance
    will be assumed to be Hermitian (Real part symmetric and Imaginary part
    antisymmetric).Note that we add the point (f, Z(f)) = (0, 0) to the 
    frequency and impedance arrays derived from the table.*
    '''
    
    def __init__(self, input_1, input_2, input_3 = None):       
        
        if input_3 is None:
            #: *Time array of the wake in [s]*
            self.time_array = input_1
            #: *Wake array in* [:math:`\Omega / s`]
            self.wake_array = input_2
        else:
            #: *Frequency array of the impedance in [Hz]*
            self.frequency_array_loaded = input_1
            #: *Real part of impedance in* [:math:`\Omega`]
            self.Re_Z_array_loaded = input_2
            #: *Imaginary part of impedance in* [:math:`\Omega`]
            self.Im_Z_array_loaded = input_3
            #: *Impedance array in* [:math:`\Omega`]
            self.impedance_loaded = self.Re_Z_array_loaded + 1j * self.Im_Z_array_loaded
            
            if self.frequency_array_loaded[0] != 0:
                self.frequency_array_loaded = np.hstack((0, self.frequency_array_loaded))
                self.Re_Z_array_loaded = np.hstack((0, self.Re_Z_array_loaded))
                self.Im_Z_array_loaded = np.hstack((0, self.Im_Z_array_loaded))
    
    
    def wake_calc(self, new_time_array):
        '''
        *The wake is interpolated in order to scale with the new time array.*
        '''
        
        wake = np.interp(new_time_array, self.time_array, self.wake_array, 
                           right = 0)
        self.wake_array = wake
        self.time_array = new_time_array
        
    
    def imped_calc(self, new_frequency_array):
        '''
        *The impedance is interpolated in order to scale with the new frequency
        array.*
        '''

        Re_Z = np.interp(new_frequency_array, self.frequency_array_loaded, self.Re_Z_array_loaded, 
                      right = 0)
        Im_Z = np.interp(new_frequency_array, self.frequency_array_loaded, self.Im_Z_array_loaded, 
                      right = 0)
        self.frequency_array = new_frequency_array
        self.Re_Z_array = Re_Z
        self.Im_Z_array = Im_Z
        self.impedance = Re_Z + 1j * Im_Z
        
    
    
class Resonators(object):
    '''
    *Impedance contribution from resonators, analytic formulas for both wake 
    and impedance. The resonant modes (and the corresponding R and Q) 
    can be inputed as a list in case of several modes.*
    
    *The model is the following:*
    
    .. math::
    
        Z(f) = \\frac{R}{1 + j Q \\left(\\frac{f}{f_r}-\\frac{f_r}{f}\\right)}
        
    .. math::
        
        W(t>0) = 2\\alpha R e^{-\\alpha t}\\left(\\cos{\\bar{\\omega}t} - \\frac{\\alpha}{\\bar{\\omega}}\\sin{\\bar{\\omega}t}\\right)

        W(0) = \\alpha R
        
    .. math::
        
        \\omega_r = 2 \\pi f_r
        
        \\alpha = \\frac{\\omega_r}{2Q}
        
        \\bar{\\omega} = \\sqrt{\\omega_r^2 - \\alpha^2}
        
    '''
    
    def __init__(self, R_S, frequency_R, Q):
        
        #: *Shunt impepdance in* [:math:`\Omega`]
        self.R_S = np.array([R_S]).flatten()
        
        #: *Resonant frequency in [Hz]*
        self.frequency_R = np.array([frequency_R]).flatten()
        
        #: *Resonant angular frequency in [rad/s]*
        self.omega_R = 2 *np.pi * self.frequency_R
        
        #: *Quality factor*
        self.Q = np.array([Q]).flatten()
        
        #: *Number of resonant modes*
        self.n_resonators = len(self.R_S)
        
        #: *Time array of the wake in [s]*
        self.time_array = 0
        
        #: *Wake array in* [:math:`\Omega / s`]
        self.wake = 0
        
        #: *Frequency array of the impedance in [Hz]*
        self.freq_array = 0
        
        #: *Impedance array in* [:math:`\Omega`]
        self.impedance = 0


    def wake_calc(self, time_array):
        '''
        *Wake calculation method as a function of time.*
        '''
        
        self.time_array = time_array
        self.wake = np.zeros(self.time_array.shape)
        
        for i in range(0, self.n_resonators):
       
            alpha = self.omega_R[i] / (2 * self.Q[i])
            omega_bar = np.sqrt(self.omega_R[i] ** 2 - alpha ** 2)
            
            self.wake += (np.sign(self.time_array) + 1) * self.R_S[i] * alpha * \
                         np.exp(-alpha * self.time_array) * \
                         (np.cos(omega_bar * self.time_array) - 
                          alpha / omega_bar * np.sin(omega_bar * self.time_array))
    
    
    def imped_calc(self, freq_array):
        '''
        *Impedance calculation method as a function of frequency.*
        '''
        
        self.freq_array = freq_array
        self.impedance = np.zeros(len(self.freq_array)) + 0j
        
        for i in range(0, self.n_resonators):
            
            self.impedance[1:] += self.R_S[i] / (1 + 1j * self.Q[i] * 
                                                 ((self.freq_array[1:] / self.frequency_R[i]) - 
                                                  (self.frequency_R[i] / self.freq_array[1:])))
 
 

class TravelingWaveCavity(object):
    '''
    *Impedance contribution from traveling wave cavities, analytic formulas for 
    both wake and impedance. The resonance modes (and the corresponding R and a) 
    can be inputed as a list in case of several modes.*
    
    *The model is the following:*
    
    .. math::
    
        Z_+(f) = R \\left[\\left(\\frac{\\sin{\\frac{a\\left(f-f_r\\right)}{2}}}{\\frac{a\\left(f-f_r\\right)}{2}}\\right)^2 - 2i \\frac{a\\left(f-f_r\\right) - \\sin{a\\left(f-f_r\\right)}}{\\left(a\\left(f-f_r\\right)\\right)^2}\\right]
        
        Z_-(f) = R \\left[\\left(\\frac{\\sin{\\frac{a\\left(f+f_r\\right)}{2}}}{\\frac{a\\left(f+f_r\\right)}{2}}\\right)^2 - 2i \\frac{a\\left(f+f_r\\right) - \\sin{a\\left(f+f_r\\right)}}{\\left(a\\left(f+f_r\\right)\\right)^2}\\right]
        
        Z = Z_+ + Z_-
        
    .. math::
        
        W(0<t<\\tilde{a}) = \\frac{4R}{\\tilde{a}}\\left(1-\\frac{t}{\\tilde{a}}\\right)\\cos{\\omega_r t} 

        W(0) = \\frac{2R}{\\tilde{a}}
        
    .. math::
        
        a = 2 \\pi \\tilde{a}
        
    '''
    
    def __init__(self, R_S, frequency_R, a_factor):
        
        #: *Shunt impepdance in* [:math:`\Omega`]
        self.R_S = np.array([R_S]).flatten()
        
        #: *Resonant frequency in [Hz]*
        self.frequency_R = np.array([frequency_R]).flatten()
        
        #: *Damping time a in [s]*
        self.a_factor = np.array([a_factor]).flatten()
        
        #: *Number of resonant modes*
        self.n_twc = len(self.R_S)
        
        #: *Time array of the wake in [s]*
        self.time_array = 0
        
        #: *Wake array in* [:math:`\Omega / s`]
        self.wake = 0
        
        #: *Frequency array of the impedance in [Hz]*
        self.freq_array = 0
        
        #: *Impedance array in* [:math:`\Omega`]
        self.impedance = 0
        
    
    def wake_calc(self, time_array):
        '''
        *Wake calculation method as a function of time.*
        '''
        
        self.time_array = time_array
        self.wake = np.zeros(self.time_array.shape)
        
        for i in range(0, self.n_twc):
            a_tilde = self.a_factor[i] / (2 * np.pi)
            indexes = np.where(self.time_array <= a_tilde)
            self.wake[indexes] += (np.sign(self.time_array[indexes]) + 1) * 2 * self.R_S[i] / a_tilde * \
                                  (1 - self.time_array[indexes] / a_tilde) * \
                                  np.cos(2 * np.pi * self.frequency_R[i] * self.time_array[indexes])
    
    
    def imped_calc(self, freq_array):
        '''
        *Impedance calculation method as a function of frequency.*
        '''
        
        self.freq_array = freq_array
        self.impedance = np.zeros(len(self.freq_array)) + 0j
        
        for i in range(0, self.n_twc):
            
            Zplus = self.R_S[i] * ((np.sin(self.a_factor[i] / 2 * (self.freq_array - self.frequency_R[i])) / 
                                    (self.a_factor[i] / 2 * (self.freq_array - self.frequency_R[i])))**2 - 
                                   2j*(self.a_factor[i] * (self.freq_array - self.frequency_R[i]) - 
                                       np.sin(self.a_factor[i] * (self.freq_array - self.frequency_R[i]))) / \
                                    (self.a_factor[i] * (self.freq_array - self.frequency_R[i]))**2)
            
            Zminus = self.R_S[i] * ((np.sin(self.a_factor[i] / 2 * (self.freq_array + self.frequency_R[i])) / 
                                    (self.a_factor[i] / 2 * (self.freq_array + self.frequency_R[i])))**2 - 
                                   2j*(self.a_factor[i] * (self.freq_array + self.frequency_R[i]) - 
                                       np.sin(self.a_factor[i] * (self.freq_array + self.frequency_R[i]))) / \
                                    (self.a_factor[i] * (self.freq_array + self.frequency_R[i]))**2)
            
            self.impedance += Zplus + Zminus   
    



 
