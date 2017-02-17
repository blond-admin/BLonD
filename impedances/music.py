
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Implementation of the MuSiC algorithm to calculate the exact induced voltage 
generated by resonant modes without using slices, O(n). The method track_classic,
which calculates the same thing with the O(n^2) algorithm used in the classical
definition, is kept just as a reference for benchmark. The commented block of 
code is the not-used Python version of the MuSiC code, again kept for reference.

WARNING: 

Reference: 

M. Migliorati, L. Palumbo, 'Multibunch and multiparticle simulation code with 
an alternative approach to wakefield effects', Phys. Rev. ST Accel. Beams 18, 2015** 
 
:Authors: **Danilo Quartullo, Konstantinos Iliakis**
'''

from __future__ import division
from builtins import range, object
import numpy as np
from scipy.constants import e
import ctypes
from setup_cpp import libblond
import time


class Music(object):

    
    def __init__(self, Beam, resonator, n_macroparticles, n_particles, t_rev):

        self.beam = Beam
        self.R_S = resonator[0]
        self.omega_R = resonator[1]
        self.Q = resonator[2]
        self.n_macroparticles = n_macroparticles
        self.n_particles = n_particles
        self.alpha = self.omega_R / (2*self.Q)
        self.omega_bar = np.sqrt(self.omega_R ** 2 - self.alpha ** 2)
        self.const = -e*self.R_S*self.omega_R * \
            self.n_particles/(self.n_macroparticles*self.Q)
        self.induced_voltage = np.zeros(len(self.beam.dt))
        self.induced_voltage[0] = self.const/2
        self.coeff1 = -self.alpha/self.omega_bar
        self.coeff2 = -self.R_S*self.omega_R/(self.Q*self.omega_bar)
        self.coeff3 = self.omega_R*self.Q/(self.R_S*self.omega_bar)
        self.coeff4 = self.alpha/self.omega_bar
        
        self.input_first_component = 1
        self.input_second_component = 0
        self.t_rev = t_rev
        self.last_dt = self.beam.dt[-1]
        self.array_parameters = np.array([self.input_first_component, self.input_second_component, self.t_rev, self.last_dt])
    
    
    def track_cpp(self):

        libblond.music_track(self.beam.dt.ctypes.data_as(ctypes.c_void_p),
                             self.beam.dE.ctypes.data_as(ctypes.c_void_p),
                             self.induced_voltage.ctypes.data_as(ctypes.c_void_p),
                             self.array_parameters.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(len(self.beam.dt)),
                             ctypes.c_double(self.alpha),
                             ctypes.c_double(self.omega_bar),
                             ctypes.c_double(self.const),
                             ctypes.c_double(self.coeff1),
                             ctypes.c_double(self.coeff2),
                             ctypes.c_double(self.coeff3),
                             ctypes.c_double(self.coeff4))
    
    
    def track_py(self):
        
        indices_sorted = np.argsort(self.beam.dt)
        self.beam.dt = self.beam.dt[indices_sorted]
        self.beam.dE = self.beam.dE[indices_sorted]
        self.beam.dE[0] += self.induced_voltage[0]
        self.input_first_component = 1
        self.input_second_component = 0
    
        for i in range(len(self.beam.dt)-1):
    
            time_difference = self.beam.dt[i+1]-self.beam.dt[i]
              
            exp_term = np.exp(-self.alpha * time_difference)
            cos_term = np.cos(self.omega_bar * time_difference)
            sin_term = np.sin(self.omega_bar * time_difference)
            
            product_first_component = exp_term * \
                ((cos_term+self.coeff1*sin_term)*self.input_first_component +
                 self.coeff2*sin_term*self.input_second_component)
            product_second_component = exp_term * \
                (self.coeff3*sin_term*self.input_first_component +
                 (cos_term+self.coeff4*sin_term)*self.input_second_component)
                
            self.induced_voltage[i+1] = self.const * \
                (0.5+product_first_component)
            self.beam.dE[i+1] += self.induced_voltage[i+1]
            
            self.input_first_component = product_first_component+1.0
            self.input_second_component = product_second_component
            self.last_dt = self.beam.dt[-1]
    
    
    def track_classic(self):
        
        indices_sorted = np.argsort(self.beam.dt)
        self.beam.dt = self.beam.dt[indices_sorted]
        self.beam.dE = self.beam.dE[indices_sorted]
        self.beam.dE[0] += self.induced_voltage[0]
        self.induced_voltage[1:] = 0
        
        for i in range(len(self.beam.dt)-1):
            
            for j in range(i+1):
                
                time_difference = self.beam.dt[i+1]-self.beam.dt[j]
                exp_term = np.exp(-self.alpha * time_difference)
                cos_term = np.cos(self.omega_bar * time_difference)
                sin_term = np.sin(self.omega_bar * time_difference)
                self.induced_voltage[i+1] += exp_term*(cos_term+self.coeff1*sin_term)
                
            self.induced_voltage[i+1] = self.const*(0.5+self.induced_voltage[i+1])    
            self.beam.dE[i+1] += self.induced_voltage[i+1]

    
    
    def track_py_multi_turn(self):
        
        indices_sorted = np.argsort(self.beam.dt)
        self.beam.dt = self.beam.dt[indices_sorted]
        self.beam.dE = self.beam.dE[indices_sorted]
        time_difference_0 = self.beam.dt[0] + self.t_rev - self.last_dt
        exp_term = np.exp(-self.alpha * time_difference_0)
        cos_term = np.cos(self.omega_bar * time_difference_0)
        sin_term = np.sin(self.omega_bar * time_difference_0)
        product_first_component = exp_term * \
            ((cos_term+self.coeff1*sin_term)*self.input_first_component +
             self.coeff2*sin_term*self.input_second_component)
        product_second_component = exp_term * \
            (self.coeff3*sin_term*self.input_first_component +
             (cos_term+self.coeff4*sin_term)*self.input_second_component)
        self.induced_voltage[0] = self.const * \
            (0.5+product_first_component)
        self.beam.dE[0] += self.induced_voltage[0]
        self.input_first_component = product_first_component+1.0
        self.input_second_component = product_second_component
    
        for i in range(len(self.beam.dt)-1):
    
            time_difference = self.beam.dt[i+1]-self.beam.dt[i]
              
            exp_term = np.exp(-self.alpha * time_difference)
            cos_term = np.cos(self.omega_bar * time_difference)
            sin_term = np.sin(self.omega_bar * time_difference)
            
            product_first_component = exp_term * \
                ((cos_term+self.coeff1*sin_term)*self.input_first_component +
                 self.coeff2*sin_term*self.input_second_component)
            product_second_component = exp_term * \
                (self.coeff3*sin_term*self.input_first_component +
                 (cos_term+self.coeff4*sin_term)*self.input_second_component)
                
            self.induced_voltage[i+1] = self.const * \
                (0.5+product_first_component)
            self.beam.dE[i+1] += self.induced_voltage[i+1]
            
            self.input_first_component = product_first_component+1.0
            self.input_second_component = product_second_component 
        
        self.last_dt = self.beam.dt[-1]    
        
        
    def track_cpp_multi_turn(self):
        
        libblond.music_track_multiturn(self.beam.dt.ctypes.data_as(ctypes.c_void_p),
                             self.beam.dE.ctypes.data_as(ctypes.c_void_p),
                             self.induced_voltage.ctypes.data_as(ctypes.c_void_p),
                             self.array_parameters.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(len(self.beam.dt)),
                             ctypes.c_double(self.alpha),
                             ctypes.c_double(self.omega_bar),
                             ctypes.c_double(self.const),
                             ctypes.c_double(self.coeff1),
                             ctypes.c_double(self.coeff2),
                             ctypes.c_double(self.coeff3),
                             ctypes.c_double(self.coeff4))