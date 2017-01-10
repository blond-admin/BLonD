
# Copyright 2016 CERN. This software is distributed under the
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

from __future__ import division, print_function
from builtins import range, object
import numpy as np
from toolbox.next_regular import next_regular
from numpy.fft import  rfft, irfft, rfftfreq
import ctypes, time
from scipy.constants import e
from scipy.signal import filtfilt
import scipy.ndimage as ndimage
from toolbox.convolution import convolution
from setup_cpp import libblond
import matplotlib.pyplot as plt


class TotalInducedVoltage(object):
    '''
    *Object gathering all the induced voltage contributions. The input is a 
    list of objects able to compute induced voltages (InducedVoltageTime, 
    InducedVoltageFreq, InductiveImpedance). All the induced voltages will
    be summed in order to reduce the computing time. All the induced
    voltages should have the same slicing resolution.*
    '''
    
    def __init__(self, Beam, Slices, induced_voltage_list, n_turns_memory=0, rev_time_array=None, mode_mtw='fourth_method'):
        '''
        *Constructor.*
        '''
        #: *Copy of the Beam object in order to access the beam info.*
        self.beam = Beam
        
        #: *Copy of the Slices object in order to access the profile info.*
        self.slices = Slices
        
        #: *Induced voltage list.*
        self.induced_voltage_list = induced_voltage_list
        
        #: *Induced voltage from the sum of the wake sources in [V]*
        self.induced_voltage = np.zeros(self.slices.n_slices)
        
        #: *Time array of the wake in [s]*
        self.time_array = self.slices.bin_centers
        
        self.mode_mtw = mode_mtw
        
        #: *Creation of fundamental objects/parameters in case of multi-turn wake.*
        if n_turns_memory > 0:
            
            self.n_turns_memory = n_turns_memory
            self.rev_time_array = rev_time_array
            self.counter_turn = 0
            self.inductive_impedance_on = False
            i = 0
            for induced_voltage_object in self.induced_voltage_list:
                
                if type(induced_voltage_object) is InducedVoltageFreq:
                    self.n_windows_before = induced_voltage_object.n_windows_before
                    self.sum_impedances_memory = induced_voltage_object.total_impedance_memory
                    self.len_array_memory = induced_voltage_object.len_array_memory
                    self.n_fft_sampling = induced_voltage_object.n_fft_sampling
                    self.frequency_array_memory = induced_voltage_object.frequency_array_memory
                    self.omegaj_array_memory = 2.0j * np.pi * self.frequency_array_memory
                    self.coefficient = - self.beam.charge * e * self.beam.ratio / (self.slices.bin_centers[1]-self.slices.bin_centers[0])
                    self.main_harmonic_number = induced_voltage_object.main_harmonic_number
                    self.time_array_memory = induced_voltage_object.time_array_memory
                    self.index_save_individual_voltage = induced_voltage_object.index_save_individual_voltage
                    if self.index_save_individual_voltage != -1:
                        self.individual_impedance = induced_voltage_object.impedance_source_list[self.index_save_individual_voltage].impedance
                    i += 1
                    
                elif type(induced_voltage_object) is InducedVoltageTime:
                    self.n_points_wake = induced_voltage_object.n_points_wake
                    self.total_wake = induced_voltage_object.total_wake
                    self.main_harmonic_number = induced_voltage_object.main_harmonic_number
                    self.coefficient_time = - self.beam.charge * e * self.beam.ratio
                    i += 1
                    
                elif type(induced_voltage_object) is InductiveImpedance:
                    self.inductive_impedance_on = True    
                    self.index_inductive_impedance = i
                    i += 1
                    
                else:
                    raise RuntimeError('Error! Aborting...')
            
            if self.mode_mtw=='first_method':
                self.induced_voltage_extended = np.zeros(self.n_fft_sampling)       
            
            elif self.mode_mtw=='second_method':
                self.ind_volt_freq_table_memory = np.zeros((n_turns_memory+1, len(self.frequency_array_memory)), complex)
                self.pointer_current_turn = 0
                if self.index_save_individual_voltage != -1:
                    self.ind_volt_freq_table_memory_ind_volt = np.zeros((n_turns_memory+1, len(self.frequency_array_memory)), complex)
                    
            elif self.mode_mtw=='third_method':
                self.induced_voltage_extended = np.zeros(self.n_fft_sampling)
            
            elif self.mode_mtw=='fourth_method':
                self.induced_voltage_extended = np.zeros(self.n_points_wake)
                
            elif self.mode_mtw=='fifth_method':
                self.points_before = self.slices.n_slices*self.n_windows_before
                self.points_ext_ind_volt = self.n_fft_sampling-self.points_before
                self.induced_voltage_extended = np.zeros(self.points_ext_ind_volt)
                t_max = self.points_ext_ind_volt * (self.slices.bin_centers[1]-self.slices.bin_centers[0])
                self.time_array_interp = np.linspace(self.slices.bin_centers[0], t_max-(self.slices.bin_centers[1]-self.slices.bin_centers[0])/2, self.points_ext_ind_volt)
                
            else:
                raise RuntimeError('Error! Aborting...')
    
    
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
    
    
    def track(self):
        '''
        *Track method to apply the induced voltage kick on the beam.*
        '''
        
        self.induced_voltage_sum(self.beam)
        
        induced_energy = self.beam.charge * self.induced_voltage
        libblond.linear_interp_kick(self.beam.dt.ctypes.data_as(ctypes.c_void_p),
                                  self.beam.dE.ctypes.data_as(ctypes.c_void_p), 
                                  induced_energy.ctypes.data_as(ctypes.c_void_p), 
                                  self.slices.bin_centers.ctypes.data_as(ctypes.c_void_p), 
                                  ctypes.c_uint(self.slices.n_slices),
                                  ctypes.c_uint(self.beam.n_macroparticles))
        
    
    def track_memory(self):
        '''
        Calculates the induced voltage energy kick to particles taking into
        account multi-turn induced voltage plus inductive impedance contribution.
        '''
        # Four different methods. It should be reminded that self.n_fft_sampling
        # is the next regular of n_slices*(n_turns+1); the first, second and
        # third methods assume the induced voltage zero for all the points
        # between n_slices*(n_turns+1) and its next regular. For the fourth
        # method instead, self.n_points_wake is precisely n_slices*(n_turns+1).
        # The parameter self.main_harmonic_number indicates that all the ring
        # is sliced for the different methods.
        # To avoid undesirable contributions of the induced voltage caused by
        # circular convolution when front wake is present, the profile can
        # be padded with zeros in front: self.n_windows_before represents
        # the number of buckets that should be taken into account if self.slices.n_slices
        # is the number of slices per bucket.
        
        if self.mode_mtw=='first_method':
            # At each turn the induced voltage is calculated for the current
            # turn and the future, then this voltage is summed to the voltage
            # deriving from the past; the rotation technique is used, that is
            # the voltage from the past is multiplied by a complex
            # exponential after imposing zero values in appropriate cells to
            # avoid overlapping.
            self.induced_voltage_extended[:(self.slices.n_slices*self.n_windows_before+self.slices.n_slices*self.main_harmonic_number)]=0
            self.induced_voltage_extended[(self.slices.n_slices*self.main_harmonic_number*(self.n_turns_memory+1)+self.slices.n_slices*self.n_windows_before):]=0
            self.array_memory = rfft(self.induced_voltage_extended, self.n_fft_sampling)
            self.array_memory *= np.exp(self.omegaj_array_memory * self.rev_time_array[self.counter_turn])
            padded_before_profile = np.lib.pad(self.slices.n_macroparticles, (self.slices.n_slices*self.n_windows_before,0), 'constant', constant_values=(0,0))
            self.fourier_transf_profile = rfft(padded_before_profile, self.n_fft_sampling)
            self.induced_voltage_extended =  irfft(self.array_memory + self.coefficient * \
                    self.fourier_transf_profile * self.sum_impedances_memory, self.n_fft_sampling)
            self.induced_voltage = self.induced_voltage_extended[(self.slices.n_slices*self.n_windows_before):((self.slices.n_slices*self.n_windows_before)+self.slices.n_slices*self.main_harmonic_number)]
            
        elif self.mode_mtw=='second_method':
            # Here the rotation technique is used, as in the first method.
            # The main difference is that a matrix m x n is used, where m is the
            # number of memory turns +1, and n is the number of fft points.
            # Each turn, scanning from top to bottom, the current induced voltage 
            # is calculated in frequency domain and saved into a row of the matrix;
            # all the other raws are rotated of T_rev to have the suitable
            # induced voltage from the past.
            self.ind_volt_freq_table_memory *= np.exp(self.omegaj_array_memory * self.rev_time_array[self.counter_turn])
            padded_before_profile = np.lib.pad(self.slices.n_macroparticles, (self.slices.n_slices*self.n_windows_before,0), 'constant', constant_values=(0,0))
            self.fourier_transf_profile = rfft(padded_before_profile, self.n_fft_sampling)
            self.ind_volt_freq_table_memory[self.pointer_current_turn,:] = self.fourier_transf_profile * self.sum_impedances_memory
            ind_volt_total_f = np.sum(self.ind_volt_freq_table_memory, axis=0)
            self.induced_voltage_extended = self.coefficient * irfft(ind_volt_total_f, self.n_fft_sampling)
            self.induced_voltage = self.induced_voltage_extended[(self.slices.n_slices*self.n_windows_before):((self.slices.n_slices*self.n_windows_before)+self.slices.n_slices*self.main_harmonic_number)]
            
            if self.index_save_individual_voltage != -1:
                self.ind_volt_freq_table_memory_ind_volt *= np.exp(self.omegaj_array_memory * self.rev_time_array[self.counter_turn])
                self.ind_volt_freq_table_memory_ind_volt[self.pointer_current_turn,:] = self.fourier_transf_profile * self.individual_impedance
                ind_volt_total_f_individual = np.sum(self.ind_volt_freq_table_memory_ind_volt,axis=0)
                self.voltage_saved = self.coefficient * irfft(ind_volt_total_f_individual, self.n_fft_sampling)[(self.slices.n_slices*self.n_windows_before):((self.slices.n_slices*self.n_windows_before)+self.slices.n_slices)]
                
            self.pointer_current_turn += 1
            self.pointer_current_turn = np.mod(self.pointer_current_turn, self.n_turns_memory+1)
            
        elif self.mode_mtw=='third_method':
            # At each turn the induced voltage is calculated for the current
            # turn and the future, then this voltage is summed to the voltage
            # deriving from the past; the substitution technique is used, that is
            # at every turn the calculated induced voltage for the current and 
            # following turns will be used as past voltage in the following turn.
            padded_before_profile = np.lib.pad(self.slices.n_macroparticles, (self.slices.n_slices*self.n_windows_before,0), 'constant', constant_values=(0,0))
            self.fourier_transf_profile = rfft(padded_before_profile, self.n_fft_sampling)
            induced_voltage_current = self.coefficient * \
                irfft(self.fourier_transf_profile*
                      self.sum_impedances_memory, self.n_fft_sampling)
            self.induced_voltage_extended[(self.slices.n_slices*self.n_windows_before):(self.slices.n_slices*self.main_harmonic_number*self.n_turns_memory+self.slices.n_slices*self.n_windows_before)] = \
                induced_voltage_current[(self.slices.n_slices*self.n_windows_before):(self.slices.n_slices*self.main_harmonic_number*self.n_turns_memory+self.slices.n_slices*self.n_windows_before)] + \
                self.induced_voltage_extended[(self.slices.n_slices*self.main_harmonic_number+self.slices.n_slices*self.n_windows_before):
                                              (self.slices.n_slices*self.main_harmonic_number*(self.n_turns_memory+1)+self.slices.n_slices*self.n_windows_before)]
            self.induced_voltage_extended[(self.slices.n_slices*self.main_harmonic_number*self.n_turns_memory+self.slices.n_slices*self.n_windows_before):] = \
                induced_voltage_current[(self.slices.n_slices*self.main_harmonic_number*self.n_turns_memory+self.slices.n_slices*self.n_windows_before):]
            self.induced_voltage = self.induced_voltage_extended[(self.slices.n_slices*self.n_windows_before):((self.slices.n_slices*self.n_windows_before)+self.slices.n_slices*self.main_harmonic_number)]
  
        elif self.mode_mtw=='fourth_method':
            # The only difference between this and the third method is that here
            # the voltage is calculated in time domain with a convolution.
            induced_voltage_current = self.coefficient_time * \
                convolution(self.slices.n_macroparticles, self.total_wake)[:self.n_points_wake]
            self.induced_voltage_extended[:(self.slices.n_slices*self.main_harmonic_number*self.n_turns_memory)] = \
                induced_voltage_current[:(self.slices.n_slices*self.main_harmonic_number*self.n_turns_memory)] + \
                self.induced_voltage_extended[(self.slices.n_slices*self.main_harmonic_number):
                                              (self.slices.n_slices*self.main_harmonic_number*(self.n_turns_memory+1))]
            self.induced_voltage_extended[(self.slices.n_slices*self.main_harmonic_number*self.n_turns_memory):] = \
                induced_voltage_current[(self.slices.n_slices*self.main_harmonic_number*self.n_turns_memory):]
            self.induced_voltage = self.induced_voltage_extended[:self.slices.n_slices]
        
        elif self.mode_mtw=='fifth_method':
            # To be used in presence of acceleration. Interpolation in time domain.
            padded_before_profile = np.lib.pad(self.slices.n_macroparticles, (self.points_before,0), 'constant', constant_values=(0,0))
            self.fourier_transf_profile = rfft(padded_before_profile, self.n_fft_sampling)
            time_array_shifted = self.time_array_interp + self.rev_time_array[self.counter_turn]
            interpolation2 = np.zeros(self.points_ext_ind_volt)
            libblond.linear_interp_time_translation(self.time_array_interp.ctypes.data_as(ctypes.c_void_p),
                                  self.induced_voltage_extended.ctypes.data_as(ctypes.c_void_p), 
                                  time_array_shifted.ctypes.data_as(ctypes.c_void_p), 
                                  interpolation2.ctypes.data_as(ctypes.c_void_p), 
                                  ctypes.c_uint(self.points_ext_ind_volt))
            self.induced_voltage_extended = irfft(self.coefficient * self.fourier_transf_profile * self.sum_impedances_memory, self.n_fft_sampling)[self.points_before:] + \
                                                + interpolation2
            self.induced_voltage = self.induced_voltage_extended[:self.slices.n_slices]
            
                                                
        # Contribution from inductive impedance
        if self.inductive_impedance_on:  
            self.induced_voltage_list[self.index_inductive_impedance].induced_voltage_generation(self.beam, 'slice_frame')
            self.induced_voltage += self.induced_voltage_list[self.index_inductive_impedance].induced_voltage

        
        # Induced voltage energy kick to particles through linear interpolation
        induced_energy = self.beam.charge * self.induced_voltage
        libblond.linear_interp_kick(self.beam.dt.ctypes.data_as(ctypes.c_void_p),
                                  self.beam.dE.ctypes.data_as(ctypes.c_void_p), 
                                  induced_energy.ctypes.data_as(ctypes.c_void_p), 
                                  self.slices.bin_centers.ctypes.data_as(ctypes.c_void_p), 
                                  ctypes.c_uint(self.slices.n_slices),
                                  ctypes.c_uint(self.beam.n_macroparticles))
            
        # Counter update
        self.counter_turn += 1
    
    
    def track_ghosts_particles(self, ghostBeam):
        
        induced_energy = self.beam.charge * self.induced_voltage
        libblond.linear_interp_kick(ghostBeam.dt.ctypes.data_as(ctypes.c_void_p),
                                  ghostBeam.dE.ctypes.data_as(ctypes.c_void_p), 
                                  induced_energy.ctypes.data_as(ctypes.c_void_p), 
                                  self.slices.bin_centers.ctypes.data_as(ctypes.c_void_p), 
                                  ctypes.c_uint(self.slices.n_slices),
                                  ctypes.c_uint(ghostBeam.n_macroparticles))



class InducedVoltageTime(object):
    '''
    *Induced voltage derived from the sum of several wake fields (time domain).*
    '''
    
    def __init__(self, Slices, wake_source_list, n_turns_memory=0, time_or_freq = 'freq', main_harmonic_number = 1):       
        
        #: *Copy of the Slices object in order to access the profile info.*
        self.slices = Slices
              
        #: *Wake sources inputed as a list (eg: list of BBResonators objects)*
        self.wake_source_list = wake_source_list
        
        #: *Time array of the wake in [s]*
        self.time_array = 0
        
        #: *Total wake array of all sources in* [:math:`\Omega / s`]
        self.total_wake = 0
        
        #: *Induced voltage from the sum of the wake sources in [V]*
        self.induced_voltage = 0
        
        # Pre-processing the wakes
        self.time_array = self.slices.bin_centers - self.slices.bin_centers[0]
        self.sum_wakes(self.time_array)
        
        self.cut = len(self.time_array) + len(self.slices.n_macroparticles) - 1
        self.fshape = next_regular(self.cut)
        
        self.time_or_freq = time_or_freq
        
        if n_turns_memory > 0:
            self.main_harmonic_number = main_harmonic_number
            delta_t = self.slices.bin_centers[1]-self.slices.bin_centers[0]
            self.n_points_wake = self.slices.n_slices * main_harmonic_number * (n_turns_memory+1)
            self.time_array_wake_extended = np.linspace(0, delta_t*(self.n_points_wake-1), self.n_points_wake)
            self.sum_wakes(self.time_array_wake_extended)
            self.time_array_memory = self.time_array_wake_extended+delta_t/2
    
    def reprocess(self, new_slicing):
        '''
        *Reprocess the wake contributions with respect to the new_slicing.*
        '''
        
        self.slices = new_slicing
        
        self.time_array = self.slices.bin_centers - self.slices.bin_centers[0]
        self.sum_wakes(self.time_array)
        
        self.cut = len(self.time_array) + len(self.slices.n_macroparticles) - 1
        self.fshape = next_regular(self.cut)
    
    
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
        if self.time_or_freq == 'freq':
            induced_voltage = - Beam.charge * e * Beam.intensity / Beam.n_macroparticles * irfft(rfft(self.slices.n_macroparticles, self.fshape) * rfft(self.total_wake, self.fshape), self.fshape)
        elif self.time_or_freq == 'time':
            induced_voltage = - Beam.charge * e * Beam.intensity / Beam.n_macroparticles * convolution(self.total_wake, self.slices.n_macroparticles)
        else:
            raise RuntimeError('Error: just freq ot time are allowed!')
        
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
        induced_energy = self.beam.charge * self.induced_voltage
        libblond.linear_interp_kick(Beam.dt.ctypes.data_as(ctypes.c_void_p),
                                  Beam.dE.ctypes.data_as(ctypes.c_void_p), 
                                  induced_energy.ctypes.data_as(ctypes.c_void_p), 
                                  self.slices.bin_centers.ctypes.data_as(ctypes.c_void_p), 
                                  ctypes.c_uint(self.slices.n_slices),
                                  ctypes.c_uint(Beam.n_macroparticles))

    
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
        
    def __init__(self, Slices, impedance_source_list, frequency_resolution_input = None, 
                 freq_res_option = 'round', n_turns_memory = 0, recalculation_impedance = False, 
                 index_save_individual_voltage = -1, n_windows_before = 0, main_harmonic_number = 1, overwrite_n_fft_sampling = None):
    

        #: *Copy of the Slices object in order to access the profile info.*
        self.slices = Slices
        
        #: *Impedance sources inputed as a list (eg: list of BBResonators objects)*
        self.impedance_source_list = impedance_source_list
        
        #: *Input frequency resolution in [Hz], the beam profile sampling for the spectrum
        #: will be adapted according to the freq_res_option.*
        self.frequency_resolution_input = frequency_resolution_input
        
        #: *Number of turns to be considered as memory for induced voltage calculation.*
        self.n_turns_memory = n_turns_memory
        
        #: *Length of one slice.*
        time_resolution = (self.slices.bin_centers[1] - self.slices.bin_centers[0])
        
        self.recalculation_impedance = recalculation_impedance
        
        self.index_save_individual_voltage = index_save_individual_voltage
        
        if n_turns_memory==0:
            
            if self.frequency_resolution_input == None:
                self.n_fft_sampling = self.slices.n_slices
            else:    
                self.freq_res_option = freq_res_option
                if self.freq_res_option is 'round':
                    self.n_fft_sampling = next_regular(int(np.round(1/(self.frequency_resolution_input * time_resolution))))
                elif self.freq_res_option is 'ceil':
                    self.n_fft_sampling = next_regular(int(np.ceil(1/(self.frequency_resolution_input * time_resolution))))
                elif self.freq_res_option is 'floor':
                    self.n_fft_sampling = next_regular(int(np.floor(1/(self.frequency_resolution_input * time_resolution))))
                else:
                    raise RuntimeError('The input freq_res_option is not recognized')
                
                if self.n_fft_sampling < self.slices.n_slices:
                    print('The input frequency resolution step is too big, and the whole \
                           bunch is not sliced... The number of sampling points for the \
                           FFT is corrected in order to sample the whole bunch (and \
                           you might consider changing the input in order to have \
                           a finer resolution).')
                    self.n_fft_sampling = next_regular(self.slices.n_slices)
                    
            if overwrite_n_fft_sampling != None:
                self.n_fft_sampling = overwrite_n_fft_sampling    
                
            #: *Real frequency resolution in [Hz], according to the obtained n_fft_sampling.*
            self.frequency_resolution = 1 / (self.n_fft_sampling * time_resolution)
            
            #: *Frequency array of the impedance in [Hz]*
            self.frequency_array = rfftfreq(self.n_fft_sampling, self.slices.bin_centers[1] - self.slices.bin_centers[0])
                                           
            #: *Total impedance array of all sources in* [:math:`\Omega`]
            self.total_impedance = 0
            
            self.sum_impedances(self.frequency_array)
            
            #: *Induced voltage from the sum of the wake sources in [V]*
            self.induced_voltage = 0
        
        else:
            
            self.n_windows_before = n_windows_before
            self.n_turns_memory = n_turns_memory
            self.main_harmonic_number = main_harmonic_number
            self.len_array_memory = (self.n_turns_memory+1+n_windows_before) * self.slices.n_slices * main_harmonic_number
            self.n_fft_sampling = next_regular(self.len_array_memory)
            if overwrite_n_fft_sampling != None:
                self.n_fft_sampling = overwrite_n_fft_sampling    
            self.frequency_array_memory = rfftfreq(self.n_fft_sampling, time_resolution)
            self.total_impedance_memory = np.zeros(self.frequency_array_memory.shape) + 0j
            
            #Costruction of time array for plotting
            self.time_array_memory = np.linspace(time_resolution/2, time_resolution*self.n_fft_sampling-time_resolution/2, self.n_fft_sampling)
                
            for imped_object in self.impedance_source_list:
                imped_object.imped_calc(self.frequency_array_memory)
                self.total_impedance_memory += imped_object.impedance
         
                
    def reprocess(self, new_slicing):
        '''
        *Reprocess the impedance contributions with respect to the new_slicing.*
        '''
        
        self.slices = new_slicing
        
        time_resolution = (self.slices.bin_centers[1] - self.slices.bin_centers[0])
        
        if self.frequency_resolution_input == None:
                self.n_fft_sampling = self.slices.n_slices
        else:    
            if self.freq_res_option is 'round':
                self.n_fft_sampling = next_regular(int(np.round(1/(self.frequency_resolution_input * time_resolution))))
            elif self.freq_res_option is 'ceil':
                self.n_fft_sampling = next_regular(int(np.ceil(1/(self.frequency_resolution_input * time_resolution))))
            elif self.freq_res_option is 'floor':
                self.n_fft_sampling = next_regular(int(np.floor(1/(self.frequency_resolution_input * time_resolution))))
            else:
                raise RuntimeError('The input freq_res_option is not recognized')
            
            if self.n_fft_sampling < self.slices.n_slices:
                print('The input frequency resolution step is too big, and the whole \
                       bunch is not sliced... The number of sampling points for the \
                       FFT is corrected in order to sample the whole bunch (and \
                       you might consider changing the input in order to have \
                       a finer resolution).')
                self.n_fft_sampling = next_regular(self.slices.n_slices)
                
        #: *Real frequency resolution in [Hz], according to the obtained n_fft_sampling.*
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
       
        for i in range(len(self.impedance_source_list)):
            self.impedance_source_list[i].imped_calc(frequency_array)
            self.total_impedance += self.impedance_source_list[i].impedance
    
    
    def sum_impedances_memory(self, frequency_array_memory):
        '''
        *Summing all the wake contributions in one total impedance.*
        '''
        
        self.total_impedance_memory = np.zeros(frequency_array_memory.shape) + 0j
       
        for i in range(len(self.impedance_source_list)):
            self.impedance_source_list[i].imped_calc(frequency_array_memory)
            self.total_impedance_memory += self.impedance_source_list[i].impedance

        
    def induced_voltage_generation(self, Beam, length = 'slice_frame'):
        '''
        *Method to calculate the induced voltage from the inverse FFT of the
        impedance times the spectrum (fourier convolution).*
        '''
        if self.recalculation_impedance:
            self.sum_impedances(self.frequency_array)
        
        self.slices.beam_spectrum_generation(self.n_fft_sampling)

        if self.index_save_individual_voltage != -1:
            self.voltage_saved = - Beam.charge * e * Beam.ratio * irfft(self.impedance_source_list[self.index_save_individual_voltage].impedance * self.slices.beam_spectrum)[:self.slices.n_slices] * self.slices.beam_spectrum_freq[1] * 2*(len(self.slices.beam_spectrum)-1)
        
        
        induced_voltage = - Beam.charge * e * Beam.ratio * irfft(self.total_impedance * self.slices.beam_spectrum) * self.slices.beam_spectrum_freq[1] * 2*(len(self.slices.beam_spectrum)-1) 
        self.induced_voltage_ext = induced_voltage
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
        induced_energy = self.beam.charge * self.induced_voltage
        libblond.linear_interp_kick(Beam.dt.ctypes.data_as(ctypes.c_void_p),
                                  Beam.dE.ctypes.data_as(ctypes.c_void_p), 
                                  induced_energy.ctypes.data_as(ctypes.c_void_p), 
                                  self.slices.bin_centers.ctypes.data_as(ctypes.c_void_p), 
                                  ctypes.c_uint(self.slices.n_slices),
                                  ctypes.c_uint(Beam.n_macroparticles))
        
        
class InductiveImpedance(object):
    '''
    *Constant imaginary Z/n impedance. This needs to be extended to the
    cases where there is acceleration as the revolution frequency f0 used
    in the calculation of n=f/f0 is changing (general_params as input ?).*
    '''
    
    def __init__(self, Slices, Z_over_n, revolution_frequency, current_turn, 
                 deriv_mode = 'gradient', periodicity = False, 
                 smooth_before_after = [False, False], filter_ind_imp = None, filter_options = None, t_rev = None):
        
        #: *Copy of the Slices object in order to access the profile info.*
        self.slices = Slices
        
        #: *Constant imaginary Z/n program in* [:math:`\Omega / Hz`].*
        self.Z_over_n = Z_over_n
        
        #: *Revolution frequency program in [Hz]*
        self.revolution_frequency = revolution_frequency
        
        #: *Frequency array of the impedance in [Hz]*
        self.frequency_array = 0
        
        #: *Impedance array in* [:math:`\Omega`]
        self.impedance = 0
        
        #: *Induced voltage from the sum of the wake sources in [V]*
        self.induced_voltage = 0
        
        #: *Derivation method to compute induced voltage*
        self.deriv_mode = deriv_mode
        
        #: *Current turn taken from RFSectionParameters*
        self.current_turn = current_turn 
        
        self.periodicity = periodicity

        self.smooth_before_after = smooth_before_after
        self.filter_ind_imp = filter_ind_imp
        self.filter_options = filter_options
        
        self.t_rev = t_rev
        
    
    def reprocess(self, new_slicing):
        '''
        *Reprocess the impedance contributions with respect to the new_slicing.*
        '''
        
        self.slices = new_slicing
        
        
    def imped_calc(self, frequency_array):
        '''
        *Impedance calculation method as a function of frequency.*
        '''    
        index = self.current_turn[0]
        self.frequency_array = frequency_array
        self.impedance = (self.frequency_array / self.revolution_frequency[index]) * \
                         self.Z_over_n[index] * 1j
        
                         

    def induced_voltage_generation(self, Beam, length = 'slice_frame'):
        '''
        *Method to calculate the induced voltage through the derivative of the
        profile; the impedance must be of inductive type.*
        '''
        
        index = self.current_turn[0]
            
        if self.periodicity:
            self.derivative_line_density_not_filtered = np.zeros(self.slices.n_slices)
            find_index_slice = np.searchsorted(self.slices.edges, self.t_rev[index])
            if self.smooth_before_after[0]:
                if self.filter_ind_imp == 'gaussian':
                    self.slices.n_macroparticles = ndimage.gaussian_filter1d(self.slices.n_macroparticles, sigma=self.filter_options, mode='wrap')
                elif self.filter_ind_imp == 'chebyshev':
                    nCoefficients, b, a = self.slices.beam_profile_filter_chebyshev(self.filter_options)
                else:
                    raise RuntimeError('filter method not recognised')
            temp = np.concatenate((np.array([self.slices.n_macroparticles[find_index_slice-1]]), self.slices.n_macroparticles[:find_index_slice], np.array([self.slices.n_macroparticles[0]])))
            self.derivative_line_density_not_filtered[:find_index_slice] = np.gradient(temp, self.slices.bin_centers[1]-self.slices.bin_centers[0])[1:-1] / (self.slices.bin_centers[1] - self.slices.bin_centers[0])
            if self.smooth_before_after[1]:
                if self.filter_ind_imp == 'gaussian':
                    self.derivative_line_density_filtered = ndimage.gaussian_filter1d(self.derivative_line_density_not_filtered, sigma=self.filter_options, mode='wrap')
                elif self.filter_ind_imp == 'chebyshev':
                    self.derivative_line_density_filtered = filtfilt(b, a, self.derivative_line_density_not_filtered)
                    self.derivative_line_density_filtered = np.ascontiguousarray(self.derivative_line_density_filtered)
                else:
                    raise RuntimeError('filter method not recognised')
                induced_voltage = - Beam.charge * e * Beam.ratio * \
                self.Z_over_n[index] * \
                self.derivative_line_density_filtered / (2 * np.pi * self.revolution_frequency[index])   
            else:
                induced_voltage = - Beam.charge * e * Beam.ratio * \
                self.Z_over_n[index] * \
                self.derivative_line_density_not_filtered / (2 * np.pi * self.revolution_frequency[index])
        else:
            induced_voltage = - Beam.charge * e / (2 * np.pi) * Beam.ratio * \
                self.Z_over_n[index] / self.revolution_frequency[index] * \
                self.slices.beam_profile_derivative(self.deriv_mode)[1] / \
                (self.slices.bin_centers[1] - self.slices.bin_centers[0])    
        
        self.induced_voltage = induced_voltage[0:self.slices.n_slices]
        
        if isinstance(length, int):
            max_length = len(induced_voltage)
            if length > max_length:
                induced_voltage = np.lib.pad(self.induced_voltage, (0, length - max_length), 'constant', constant_values=(0,0))
            return induced_voltage[0:length]
                            
                            
    def track(self, Beam):
        '''
        *Track method.*
        '''
        
        self.induced_voltage_generation(Beam)
        induced_energy = self.beam.charge * self.induced_voltage
        libblond.linear_interp_kick(self.beam.dt.ctypes.data_as(ctypes.c_void_p),
                                  self.beam.dE.ctypes.data_as(ctypes.c_void_p), 
                                  induced_energy.ctypes.data_as(ctypes.c_void_p), 
                                  self.slices.bin_centers.ctypes.data_as(ctypes.c_void_p), 
                                  ctypes.c_uint(self.slices.n_slices),
                                  ctypes.c_uint(self.beam.n_macroparticles))

    
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
        
        self.wake = np.interp(new_time_array, self.time_array, self.wake_array, 
                           right = 0)
        
    
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
        self.frequency_array = 0
        
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
    
    
    def imped_calc(self, frequency_array):
        '''
        *Impedance calculation method as a function of frequency.*
        '''
        
        self.frequency_array = frequency_array
        self.impedance = np.zeros(len(self.frequency_array)) + 0j
        
        for i in range(0, self.n_resonators):
            
            self.impedance[1:] += self.R_S[i] / (1 + 1j * self.Q[i] * 
                                                 (self.frequency_array[1:] / self.frequency_R[i] - 
                                                  self.frequency_R[i] / self.frequency_array[1:]))
 
 

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
        self.frequency_array = 0
        
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
    
    
    def imped_calc(self, frequency_array):
        '''
        *Impedance calculation method as a function of frequency.*
        '''
        
        self.frequency_array = frequency_array
        self.impedance = np.zeros(len(self.frequency_array)) + 0j
        
        for i in range(0, self.n_twc):
            
            Zplus = self.R_S[i] * ((np.sin(self.a_factor[i] / 2 * (self.frequency_array - self.frequency_R[i])) / 
                                    (self.a_factor[i] / 2 * (self.frequency_array - self.frequency_R[i])))**2 - 
                                   2j*(self.a_factor[i] * (self.frequency_array - self.frequency_R[i]) - 
                                       np.sin(self.a_factor[i] * (self.frequency_array - self.frequency_R[i]))) / \
                                    (self.a_factor[i] * (self.frequency_array - self.frequency_R[i]))**2)
            
            Zminus = self.R_S[i] * ((np.sin(self.a_factor[i] / 2 * (self.frequency_array + self.frequency_R[i])) / 
                                    (self.a_factor[i] / 2 * (self.frequency_array + self.frequency_R[i])))**2 - 
                                   2j*(self.a_factor[i] * (self.frequency_array + self.frequency_R[i]) - 
                                       np.sin(self.a_factor[i] * (self.frequency_array + self.frequency_R[i]))) / \
                                    (self.a_factor[i] * (self.frequency_array + self.frequency_R[i]))**2)
            
            self.impedance += Zplus + Zminus   
