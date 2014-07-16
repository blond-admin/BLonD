'''

@author: Hannes Bartosik, Danilo Quartullo, Alexandre Lasheen

'''

from __future__ import division
import numpy as np
from numpy import convolve, interp
from scipy.constants import c, e
from scipy.constants import physical_constants
import time
from numpy.fft import irfft, fftfreq
import matplotlib.pyplot as plt


class Induced_voltage_from_wake(object):
    '''
    Induced voltage derived from the sum of several wake fields.
    Apart from further optimisation, these are the important results obtained
    after the benchmarking which are applied to the following code:
    1) in general update_with_interpolation is faster than update_without_interpolation
    2) in general induced_voltage_with_convolv is faster than induced_voltage_with_matrix
    3) if slices.mode == const_charge, you are obliged to use 
        induced_voltage_with_matrix, then you should use update_with_interpolation
    4) if slices.mode == const_space, i.e. you want to calculate slices statistics
        (otherwise slices.mode == const_space_hist is faster), you should use 
        induced_voltage_with_convolv and then update_with_interpolation
    5) if slices.mode == const_space_hist, use induced_voltage_with_convolv and
        update_with_interpolation
    If there is no acceleration then precalc == 'on', except for the const_charge method.
    If you have acceleration and slices.coord == z or theta, then precalc == 'off';
    if slices.coord == tau then precalc == 'on'
    '''
    
    def __init__(self, slices, acceleration, wake_sum, bunch):       
        '''
        Constructor
        '''
        self.slices = slices
        self.acceleration = acceleration
        self.wake_sum = wake_sum
        
        if self.slices.mode != 'const_charge':
            
            if self.acceleration == 'off' or self.slices.coord == 'tau':
                self.precalc = 'on'
                if self.slices.coord == 'tau':
                    dtau = self.slices.bins_centers - self.slices.bins_centers[0]
                elif self.slices.coord == 'theta':
                    dtau = (self.slices.bins_centers - self.slices.bins_centers[0])\
                       * bunch.ring_radius / (bunch.beta_rel * c)
                elif self.slices.coord == 'z':
                    dtau = (self.slices.bins_centers - self.slices.bins_centers[0])\
                       / (bunch.beta_rel * c)
                self.wake_array = self.sum_wakes(dtau, self.wake_sum)
            else:
                self.precalc = 'off' 
    
    
    def sum_wakes(self, translation, wake_object_sum):
        
        total_wake = np.zeros(len(translation))
        for wake_object in self.wake_sum:
            total_wake += wake_object.wake_calc(translation)
            
        return total_wake
    
    
    def track(self, bunch):
        
        if self.slices.mode == 'const_charge':
            ind_vol = self.induced_voltage_with_matrix(bunch)
        else:
            ind_vol = self.induced_voltage_with_convolv(bunch)
        
        update_with_interpolation(bunch, ind_vol, self.slices)
        
           
    def induced_voltage_with_matrix(self, bunch):
        
        if self.slices.coord == 'tau':
            dtau_matrix = self.slices.bins_centers - \
                            np.transpose([self.slices.bins_centers])
            self.wake_matrix = self.sum_wakes(dtau_matrix, self.wake_object_sum)
        elif self.slices.coord == 'z':
            dtau_matrix = (np.transpose([self.slices.bins_centers]) - \
                           self.slices.bins_centers) / (bunch.beta_rel * c)
            self.wake_matrix = self.sum_wakes(dtau_matrix, self.wake_object_sum)
        elif self.slices.coord == 'theta':
            dtau_matrix = bunch.ring_radius / (bunch.beta_rel * c) * \
            (self.slices.bins_centers - np.transpose([self.slices.bins_centers])) 
            self.wake_matrix = self.sum_wakes(dtau_matrix, self.wake_object_sum)
        
        return - bunch.charge * bunch.intensity / bunch.n_macroparticles * \
                np.dot(self.slices.n_macroparticles, self.wake_matrix)
    
    
    def induced_voltage_with_convolv(self, bunch): 
    
        if self.precalc == 'off':
            
            if self.slices.coord == 'tau':
                dtau = self.slices.bins_centers - self.slices.bins_centers[0]
                self.wake_array = self.sum_wakes(dtau, self.wake_sum)
            
            elif self.slices.coord == 'z':
                dtau = (self.slices.bins_centers - self.slices.bins_centers[0])\
                       /(bunch.beta_rel * c)
                self.wake_array = self.sum_wakes(dtau, self.wake_sum)
                reversed_array = self.wake_array[::-1]
                return - bunch.charge * bunch.intensity / bunch.n_macroparticles * \
                    convolve(reversed_array, self.slices.n_macroparticles)[(len(reversed_array) - 1):] 
            
            elif self.slices.coord == 'theta':
                dtau = (self.slices.bins_centers - self.slices.bins_centers[0]) \
                       * bunch.ring_radius / (bunch.beta_rel * c)
                self.wake_array = self.sum_wakes(dtau, self.wake_sum)
        
        if self.precalc == 'on' and self.slices.coord == 'z':
                reversed_array = self.wake_array[::-1]
                return - bunch.charge * bunch.intensity / bunch.n_macroparticles * \
                    convolve(reversed_array, self.slices.n_macroparticles)[(len(reversed_array) - 1):]  
        
        return - bunch.charge * bunch.intensity / bunch.n_macroparticles * \
            convolve(self.wake_array, self.slices.n_macroparticles)[0:len(self.wake_array)] 
    
    
class Induced_voltage_from_impedance(object):
    '''
    Induced voltage derived from the sum of several impedances.
    '''
    
    def __init__(self, slices, acceleration, impedance_sum, frequency_step, 
                 bunch):       
        '''
        Constructor
        '''
        self.slices = slices
        self.acceleration = acceleration
        self.impedance_sum = impedance_sum
        
        self.frequency_step = frequency_step
        
        if self.acceleration == 'off' or self.slices.coord == 'tau':
                self.precalc = 'on'
                
                self.frequency_fft, self.n_sampling_fft = self.frequency_array(slices, bunch)
                
                self.impedance_array = self.sum_impedances(self.frequency_fft, self.impedance_sum)
        else:
            self.precalc = 'off' 
    
    
    def sum_impedances(self, frequency, imped_object_sum):
        
        total_impedance = np.zeros(len(frequency)) + 0j
        for imped_object in self.impedance_sum:
            imp = imped_object.imped_calc(frequency)
            total_impedance += imp
       
        return total_impedance
    
    
    def frequency_array(self, slices, bunch):
        
        if self.slices.coord == 'tau':
                    dtau = self.slices.bins_centers[1] - self.slices.bins_centers[0]
        elif self.slices.coord == 'theta':
                    dtau = (self.slices.bins_centers[1] - self.slices.bins_centers[0]) \
                       * bunch.ring_radius / (bunch.beta_rel * c)
        elif self.slices.coord == 'z':
                    dtau = (self.slices.bins_centers[1] - self.slices.bins_centers[0])\
                       /(bunch.beta_rel * c)
        
        power = int(np.floor(np.log2(1 / (self.frequency_step * dtau)))) + 1
        
        rfft_freq = fftfreq(2 ** power, dtau)[0:2**(power-1)+1]
        rfft_freq[-1] = - rfft_freq[-1]
        
        return rfft_freq, 2 ** power
        
    
    def track(self, bunch):
        
        if self.precalc == 'off':
            self.frequency_fft, self.n_sampling_fft = self.frequency_array(self.slices, bunch)
            self.impedance_array = self.sum_impedances(self.frequency_fft, self.imped_sum)
          
        self.spectrum = self.slices.beam_spectrum(self.n_sampling_fft)
        
        self.ind_vol = - bunch.charge * bunch.intensity / bunch.n_macroparticles \
            * irfft(self.impedance_array * self.spectrum) * self.frequency_fft[1] \
            * 2*(len(self.frequency_fft)-1)
        self.ind_vol = self.ind_vol[0:self.slices.n_slices]

        update_with_interpolation(bunch, self.ind_vol, self.slices)
    

def update_without_interpolation(bunch, induced_voltage, slices):
    '''
    This method can be used only if one has not used the histogram Python method 
    for the slicing.
    '''
    
    for i in range(0, slices.n_slices):
            
            bunch.dE[slices.first_index_in_bin[i]:
              slices.first_index_in_bin[i+1]] += induced_voltage[i]
    
    
def update_with_interpolation(bunch, induced_voltage, slices):
    
    temp1 = slices.bins_centers[0]
    temp2 = slices.bins_centers[-1]
    slices.bins_centers[0] = slices.edges[0]
    slices.bins_centers[-1] = slices.edges[-1]
    
    if slices.coord == 'tau':
        
        induced_voltage_interpolated = interp(bunch.tau, 
                            slices.bins_centers, induced_voltage, 0, 0)
        
    elif slices.coord == 'z':
        
        induced_voltage_interpolated = interp(bunch.z, 
                            slices.bins_centers, induced_voltage, 0, 0)
        
    elif slices.coord == 'theta':
        
        induced_voltage_interpolated = interp(bunch.theta, 
                            slices.bins_centers, induced_voltage, 0, 0)
        
    slices.bins_centers[0] = temp1
    slices.bins_centers[-1] = temp2
    bunch.dE += induced_voltage_interpolated
    
    
class Longitudinal_table(object):
    '''
    classdocs
    '''
    
    def __init__(self, a, b, c = None):       
        '''
        Constructor
        '''
        if c == None:
            self.dtau_array = a
            self.wake_array = b
        else:
            self.frequency_array = a
            self.Re_Z_array = b
            self.Im_Z_array = c
    
    
    def wake_calc(self, dtau):
        
        self.wake = interp(dtau, self.dtau_array - self.dtau_array[0], 
                      self.wake_array, left = 0, right = 0)
        
        return self.wake
    
    
    def imped_calc(self, frequency):
        
        Re_Z = interp(frequency, self.frequency_array, self.Re_Z_array, 
                      left=self.Re_Z_array[0], right = self.Re_Z_array[-1])
        Im_Z = interp(frequency, self.frequency_array, self.Im_Z_array, 
                      left=self.Im_Z_array[0], right = self.Re_Z_array[-1])
        self.impedance = Re_Z + 1j * Im_Z
        
        return self.impedance
    
    
class Longitudinal_resonators(object):
    '''
    
    '''
    def __init__(self, R_S, frequency_R, Q):
        '''
        Constructor
        '''
        self.R_S = np.array([R_S]).flatten()
        self.omega_R = 2 *np.pi * np.array([frequency_R]).flatten()
        self.Q = np.array([Q]).flatten()
        self.n_resonators = len(self.R_S)
        
    
    def wake_calc(self, dtau):
        
        self.wake = np.zeros(len(dtau))
        
        for i in range(0, self.n_resonators):
       
            alpha = self.omega_R[i] / (2 * self.Q[i])
            omega_bar = np.sqrt(self.omega_R[i] ** 2 - alpha ** 2)
            
            self.wake += (np.sign(dtau) + 1) * self.R_S[i] * alpha * np.exp(-alpha * 
                    dtau) * (np.cos(omega_bar * dtau) - alpha / omega_bar * 
                    np.sin(omega_bar * dtau))
       
        return self.wake
    
    
    def imped_calc(self, frequency):
        
        self.impedance = np.zeros(len(frequency)) + 0j
  
        for i in range(0, len(self.R_S)):
            
            self.impedance[1:] +=  self.R_S[i] / (1 + 1j * self.Q[i] * \
                    (-self.omega_R[i] / (2 * np.pi *frequency[1:]) + (2 * np.pi *frequency[1:]) / self.omega_R[i]))
        
        return self.impedance
 

class Longitudinal_travelling_waves(object):
    '''
    
    '''
    def __init__(self, R_S, frequency_R, a_factor):
        '''
        Constructor
        '''
        self.R_S = np.array([R_S]).flatten()
        self.frequency_R = np.array([frequency_R]).flatten()
        self.a_factor = np.array([a_factor]).flatten()
        self.n_twc = len(self.R_S)
        
    
    def wake_calc(self, dtau):
        
        self.wake = np.zeros(len(dtau))
        
        for i in range(0, self.n_twc):
       
            a_tilde = self.a_factor[i] / (2 * np.pi)
            indexes = np.where(dtau <= a_tilde)
            self.wake[indexes] += (np.sign(dtau[indexes]) + 1) * 2 * self.R_S[i] \
                / a_tilde * (1 - dtau[indexes] / a_tilde) * np.cos(2 * np.pi * 
                self.frequency_R[i] * dtau[indexes])
                 
        return self.wake
    
    
    def imped_calc(self, frequency):
        
        self.impedance = np.zeros(len(frequency)) + 0j
        
        for i in range(0, self.n_twc):
            
            self.impedance +=  self.R_S[i] * ((np.sin(self.a_factor[i] / 2 * 
                (frequency - self.frequency_R[i])) / (self.a_factor[i] / 2 * (frequency - 
                self.frequency_R[i])))**2 - 2j*(self.a_factor[i] * (frequency - 
                self.frequency_R[i]) - np.sin(self.a_factor[i] * (frequency - 
                self.frequency_R[i]))) / (self.a_factor[i] * (frequency - 
                self.frequency_R[i]))**2) + self.R_S[i] * ((np.sin(self.a_factor[i] 
                / 2 * (frequency + self.frequency_R[i])) / (self.a_factor[i] / 2 * (
                frequency + self.frequency_R[i])))**2 - 2j*(self.a_factor[i] * (frequency 
                + self.frequency_R[i]) - np.sin(self.a_factor[i] * (frequency + 
                self.frequency_R[i]))) / (self.a_factor[i] * (frequency + 
                self.frequency_R[i]))**2)
            
        return self.impedance       
    

class Longitudinal_inductive_impedance(object):
    '''
    
    '''
    def __init__(self, Z_over_n, general_param):
        '''
        Constructor
        '''
        self.Z_over_n = Z_over_n
        self.counter = 0
        self.T0 = general_param.T0
        
    def imped_calc(self, frequency):    
        
        self.impedance = self.T0[0][self.counter] * frequency * self.Z_over_n * 1j
        self.counter += 1
        
        return self.impedance 
 
