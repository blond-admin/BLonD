'''
**Module to compute longitudinal intensity effects**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**, **Hannes Bartosik**
'''

from __future__ import division
import numpy as np
from numpy import convolve, interp
from scipy.constants import c
from numpy.fft import irfft, fftfreq
import math


class Induced_voltage_from_wake(object):
    '''
    *Induced voltage derived from the sum of several wake fields.
    Apart from further optimisations, note that:*\n
    *1) in general update_with_interpolation is faster than update_without_interpolation*\n
    *2) in general induced_voltage_with_convolv is faster than induced_voltage_with_matrix*\n
    *3) if slices.mode == const_charge, you are obliged to use* 
        *induced_voltage_with_matrix, then you should use update_with_interpolation*\n
    *4) if slices.mode == const_space, i.e. you want to calculate slices statistics*
        *(otherwise slices.mode == const_space_hist is faster), you should use*
        *induced_voltage_with_convolv and then update_with_interpolation*\n
    *5) if slices.mode == const_space_hist, use induced_voltage_with_convolv and*
        *update_with_interpolation*
    '''
    
    def __init__(self, slices, acceleration, wake_sum, bunch):       
        '''
        Constructor
        
        If there is no acceleration then obviously precalc == 'on', 
        except for the const_charge method where the distances between the 
        slide centers change from turn to turn.
        If there is acceleration and slices.coord == z or theta, then precalc == 'off';
        If slices.coord == tau then precalc == 'on since the wake, at least for
        the analytic formulas presented in the code, doesn't depend on the energy
        of the beam.
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
                       * bunch.ring_radius / (bunch.beta_r * c)
                elif self.slices.coord == 'z':
                    dtau = (self.slices.bins_centers - self.slices.bins_centers[0])\
                       / (bunch.beta_r * c)
                self.wake_array = self.sum_wakes(dtau, self.wake_sum)
            else:
                self.precalc = 'off' 
    
    
    def sum_wakes(self, translation, wake_object_sum):
        
        
        total_wake = np.zeros(len(translation))
        for wake_object in self.wake_sum:
            total_wake += wake_object.wake_calc(translation)
            
        return total_wake
    
    
    def track(self, bunch):
        '''
        Note that if slices.mode = 'const_charge' one is obliged to use the
        matrix method for the calculation of the induced voltage; otherwise
        update_with_interpolation is faster.
        '''
        
        if self.slices.mode == 'const_charge':
            self.ind_vol = self.induced_voltage_with_matrix(bunch)
        else:
            self.ind_vol = self.induced_voltage_with_convolv(bunch)
        
        update_with_interpolation(bunch, self.ind_vol, self.slices)
        
           
    def induced_voltage_with_matrix(self, bunch):
        '''
        Method to calculate the induced voltage from wakes with the matrix method;
        note that if slices.coord = z one has to "fix" the usual method, since
        the head and the tail of the bunch are inverted.
        '''
        
        if self.slices.coord == 'tau':
            dtau_matrix = self.slices.bins_centers - \
                            np.transpose([self.slices.bins_centers])
            self.wake_matrix = self.sum_wakes(dtau_matrix, self.wake_object_sum)
        elif self.slices.coord == 'z':
            dtau_matrix = (np.transpose([self.slices.bins_centers]) - \
                           self.slices.bins_centers) / (bunch.beta_r * c)
            self.wake_matrix = self.sum_wakes(dtau_matrix, self.wake_object_sum)
        elif self.slices.coord == 'theta':
            dtau_matrix = bunch.ring_radius / (bunch.beta_r * c) * \
            (self.slices.bins_centers - np.transpose([self.slices.bins_centers])) 
            self.wake_matrix = self.sum_wakes(dtau_matrix, self.wake_object_sum)
        
        return - bunch.charge * bunch.intensity / bunch.n_macroparticles * \
                np.dot(self.slices.n_macroparticles, self.wake_matrix)
    
    
    def induced_voltage_with_convolv(self, bunch): 
        '''
        Method to calculate the induced voltage from wakes with convolution;
        note that if slices.coord = z one has to "fix" the usual method, since
        the head and the tail of the bunch are inverted.
        '''
        
        if self.precalc == 'off':
            
            if self.slices.coord == 'tau':
                dtau = self.slices.bins_centers - self.slices.bins_centers[0]
                self.wake_array = self.sum_wakes(dtau, self.wake_sum)
            
            elif self.slices.coord == 'z':
                dtau = (self.slices.bins_centers - self.slices.bins_centers[0])\
                       /(bunch.beta_r * c)
                self.wake_array = self.sum_wakes(dtau, self.wake_sum)
                reversed_array = self.wake_array[::-1]
                return - bunch.charge * bunch.intensity / bunch.n_macroparticles * \
                    convolve(reversed_array, self.slices.n_macroparticles)[(len(reversed_array) - 1):] 
            
            elif self.slices.coord == 'theta':
                dtau = (self.slices.bins_centers - self.slices.bins_centers[0]) \
                       * bunch.ring_radius / (bunch.beta_r * c)
                self.wake_array = self.sum_wakes(dtau, self.wake_sum)
        
        if self.precalc == 'on' and self.slices.coord == 'z':
                reversed_array = self.wake_array[::-1]
                return - bunch.charge * bunch.intensity / bunch.n_macroparticles * \
                    convolve(reversed_array, self.slices.n_macroparticles)[(len(reversed_array) - 1):]  
        
        return - bunch.charge * bunch.intensity / bunch.n_macroparticles * \
            convolve(self.wake_array, self.slices.n_macroparticles)[0:len(self.wake_array)] 
    
    
class Induced_voltage_from_impedance(object):
    '''
    *Induced voltage derived from the sum of several impedances.*
    '''
    
    def __init__(self, slices, impedance_sum, frequency_step, 
                 sum_slopes_from_induc_imp = None, deriv_mode = 2, mode = 'only_spectrum'):       
        '''
        Constructor
        
        Frequency_step is equal to 1/(dist_centers * n) where dist_centers is
        the distance between the centers of two consecutive slides and (n/2 + 1)
        is the number of sampling points for the frequency array; see the 
        frequency_array method.
        Sum_slopes_from_induc_imp is the sum of all the inductances derived from
        all the inductive impedance, included space charge; see in addition the
        ind_vol_derivative method.
        '''
        self.slices = slices
        self.impedance_sum = impedance_sum
        self.frequency_step = frequency_step
        self.sum_slopes_from_induc_imp = sum_slopes_from_induc_imp
        self.deriv_mode = deriv_mode
        self.mode = mode
        
        if self.mode != 'only_derivative':
            
            self.frequency_fft, self.n_sampling_fft = self.frequency_array(slices)
            self.impedance_array = self.sum_impedances(self.frequency_fft, \
                                           self.impedance_sum)
            
    
    def sum_impedances(self, frequency, imped_object_sum):
        
        total_impedance = np.zeros(len(frequency)) + 0j
        for imped_object in self.impedance_sum:
            imp = imped_object.imped_calc(frequency)
            total_impedance += imp
       
        return total_impedance
    
    
    def frequency_array(self, slices):
        '''
        Method to calculate the sampling frequency array through the rfftfreq
        Python command. Since this command is not included in older version
        of Numpy, the similar fftfreq has been used with some fixes to obtain
        the same result as rfftfreq.
        '''
        
        dcenters = self.slices.bins_centers[1] - self.slices.bins_centers[0]
        
        n = int(math.ceil(1 / (self.frequency_step * dcenters) ))
        
        if n/2 + 1 >= slices.n_slices:
            pass
        else:
            print 'Warning! Resolution in frequency domain is too small, \
                you can get an error or a truncated bunch profile'
            
        if n%2 == 1:
            n += 1
        
        rfftfreq = fftfreq(n, dcenters)[0:int(n/2+1)]
        rfftfreq[-1] = - rfftfreq[-1]
        
        return rfftfreq, n
        
    
    def track(self, bunch):
        '''
        Method to calculate the induced voltage through the bunch spectrum, or
        the derivative of profile, or both; these three choices are represented
        by the mode 'only_spectrum', 'only_derivative', 'spectrum + derivative'
        respectively.
        '''
        if self.mode != 'only_derivative':
        
            self.spectrum = self.slices.beam_spectrum(self.n_sampling_fft)
            
            self.ind_vol = - bunch.charge * bunch.intensity / bunch.n_macroparticles \
                * irfft(self.impedance_array * self.spectrum) * self.frequency_fft[1] \
                * 2*(len(self.frequency_fft)-1) 
            
            self.ind_vol = self.ind_vol[0:self.slices.n_slices]
            
            if self.slices.coord == 'tau':
                pass
            elif self.slices.coord == 'theta':
                self.ind_vol *= (bunch.beta_r * c / bunch.ring_radius) ** 2
            elif self.slices.coord == 'z':
                self.ind_vol *= (bunch.beta_r * c) ** 2
                self.ind_vol = self.ind_vol[::-1]
            if self.mode == 'spectrum + derivative':
                self.ind_vol += self.ind_vol_derivative(bunch)
               
        elif self.mode == 'only_derivative':
            
            self.ind_vol = self.ind_vol_derivative(bunch)
            
        update_with_interpolation(bunch, self.ind_vol, self.slices)
        
        
    def ind_vol_derivative(self, bunch):
        '''
        Method to calculate the induced voltage through the derivative of the
        profile; the impedance must be of inductive type.
        '''
        
        ind_vol_deriv = bunch.charge / (2 * np.pi) * bunch.intensity / bunch.n_macroparticles * \
                            self.sum_slopes_from_induc_imp * \
                            self.slices.beam_profile_derivative(self.deriv_mode)[1] / \
                            (self.slices.bins_centers[1] - self.slices.bins_centers[0]) 
        
        if self.slices.coord == 'tau':
            pass
        elif self.slices.coord == 'theta':
            ind_vol_deriv *= (bunch.beta_r * c /  bunch.ring_radius) ** 2
        elif self.slices.coord == 'z':
            ind_vol_deriv *= (bunch.beta_r * c) ** 2
        
        return ind_vol_deriv


def update_without_interpolation(bunch, induced_voltage, slices):
    '''
    *Other method to update the energy of the particles; this method can be used
     only if one has not used slices.mode == const_space_hist for the slicing.
     Maybe this method could be optimised through Cython or trying to avoid
     the for loop.*
    '''
    
    for i in range(0, slices.n_slices):
            
            bunch.dE[slices.first_index_in_bin[i]:
              slices.first_index_in_bin[i+1]] += induced_voltage[i]
    
    
def update_with_interpolation(bunch, induced_voltage, slices):
    '''
    *Method to update the energy of the particles through interpolation of
     the induced voltage. Note that there is a fix to prevent that one neglects
     all the particles situated between the first edge and the first slice center
     and between the last edge and the last slice center*
    '''
    
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
    *Intensity effects from impedance and wake tables.*
    '''
    
    def __init__(self, a, b, c = None):       
        '''
        Constructor
        
        If this constructor takes just two arguments, then a wake table is passed;
        if it takes three arguments, then an impedance table is passed.
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
        '''
        Note that we add the point (f, Z(f)) = (0, 0) to the frequency and 
        impedance arrays derived from the table.
        '''
        self.frequency_array = np.hstack((0, self.frequency_array))
        self.Re_Z_array = np.hstack((0, self.Re_Z_array))
        self.Im_Z_array = np.hstack((0, self.Im_Z_array))
        Re_Z = interp(frequency, self.frequency_array, 
                      self.Re_Z_array, right = self.Re_Z_array[-1])
        Im_Z = interp(frequency, self.frequency_array, 
                      self.Im_Z_array, right = self.Im_Z_array[-1])
        self.impedance = Re_Z + 1j * Im_Z
        
        return self.impedance
    
    
class Longitudinal_resonators(object):
    '''
    *Intensity effects from resonators, analytic formulas for both wake
    and impedance..*
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
    *Intensity effects from travelling waves, analytic formulas for both wake
    and impedance.*
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
    *Longitudinal inductive impedance.*
    '''
    def __init__(self, Z_over_frequency):
        '''
        Constructor
        '''
        self.Z_over_frequency = Z_over_frequency
        
        
    def imped_calc(self, frequency):    
        
        self.impedance = frequency * self.Z_over_frequency * 1j
        
        return self.impedance



 
