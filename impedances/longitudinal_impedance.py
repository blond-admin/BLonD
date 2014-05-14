from __future__ import division
'''
@class Wakefields
@author Hannes Bartosik & Kevin Li & Giovanni Rumolo
@date March 2014
@Class for creation and management of wakefields from impedance sources
@copyright CERN
'''


import numpy as np


from scipy.constants import c, e
from scipy.constants import physical_constants


sin = np.sin
cos = np.cos


class Wakefields(object):
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
        pass        


    def wake_factor(self, bunch):
        
        particles_per_macroparticle = bunch.n_particles / bunch.n_macroparticles
        return -(bunch.charge) ** 2 / (bunch.mass * bunch.gamma * (bunch.beta * c) ** 2) * particles_per_macroparticle


    #~ @profile         
    def longitudinal_wakefield_kicks(self, bunch, slices): 
        
        wake = self.wake_longitudinal 
        
        # matrix with distances to target slice
        dz_to_target_slice = [bunch.slices.dz_centers[1:-2]] - np.transpose([bunch.slices.dz_centers[1:-2]])

        # compute kicks
        self.longitudinal_kick = np.zeros(bunch.slices.n_slices+4)
        self.longitudinal_kick[1:-3] = np.dot(bunch.slices.n_macroparticles[1:-3], wake(bunch, dz_to_target_slice)) * self.wake_factor(bunch)
        
        bunch.dp += self.longitudinal_kick[bunch.in_slice]           
             

class Wake_table_longitudinal(Wakefields):
    '''
    classdocs
    '''
    def __init__(self):       
        '''
        Constructor
        '''
        self.wake_table = {}

    
    @classmethod
    def from_ASCII(cls, wake_file, keys):
        self = cls()
        table = np.loadtxt(wake_file, delimiter="\t")
        self.wake_table = dict(zip(keys, np.array(zip(*table))))
        self.unit_conversion()
        return self

    def unit_conversion(self):
        longitudinal_wakefield_keys = ['longitudinal']
        self.wake_field_keys = []
        print 'Converting wake table to correct units ... '
        self.wake_table['time'] *= 1e-9 # unit convention [ns]
        print '\t converted time from [ns] to [s]'        
        
        for wake in longitudinal_wakefield_keys:
            try: 
                self.wake_table[wake] *= - 1.e12 # unit convention [V/pC] and sign convention !!
                print '\t converted "' + wake + '" wake from [V/pC/mm] to [V/C/m]'
                self.wake_field_keys += [wake]
            except:
                print '\t "' + wake + '" wake not provided'

    def wake_longitudinal(self, bunch, z):
        time = np.array(self.wake_table['time'])
        wake = np.array(self.wake_table['longitudinal'])
        wake_interpolated = np.interp(- z / c / bunch.beta, time, wake, left=0, right=0)
        if time[0] < 0:
            return wake_interpolated
        elif time[0] == 0:
            # beam loading theorem: half value of wake at z=0; 
            return (np.sign(-z) + 1) / 2 * wake_interpolated
    
    
    def track(self, bunch):
        
        if 'longitudinal' in self.wake_field_keys:
            self.longitudinal_wakefield_kicks(bunch)


class BB_Resonator_longitudinal(Wakefields):
    '''
    classdocs
    '''
    def __init__(self, R_shunt, frequency, Q):
        '''
        Constructor
        '''
        self.R_shunt = np.array([R_shunt]).flatten()
        self.frequency = np.array([frequency]).flatten()
        self.Q = np.array([Q]).flatten()
        assert(len(self.R_shunt) == len(self.frequency) == len(self.Q))


    def wake_longitudinal(self, bunch, z):
        return reduce(lambda x,y: x+y, [self.wake_BB_resonator(self.R_shunt[i], self.frequency[i], self.Q[i], bunch, z) for i in np.arange(len(self.Q))])

    
    def wake_BB_resonator(self, R_shunt, frequency, Q, bunch, z):        
        # Taken from Alex Chao's resonator model (2.82)
        omega = 2 * np.pi * frequency
        alpha = omega / (2 * Q)
        omegabar = np.sqrt(np.abs(omega ** 2 - alpha ** 2))

        if Q > 0.5:
            wake =  - (np.sign(z) - 1) * R_shunt * alpha * np.exp(alpha * z.clip(max=0) / c / bunch.beta) * \
                    (cos(omegabar * z.clip(max=0) / c / bunch.beta) + alpha / omegabar * sin(omegabar * z.clip(max=0) / c / bunch.beta))
        elif Q == 0.5:
            wake =  - (np.sign(z) - 1) * R_shunt * alpha * np.exp(alpha * z.clip(max=0) / c / bunch.beta) * \
                    (1. + alpha * z.clip(max=0) / c / bunch.beta)
        elif Q < 0.5:
            wake =  - (np.sign(z) - 1) * R_shunt * alpha * np.exp(alpha * z.clip(max=0) / c / bunch.beta) * \
                    (np.cosh(omegabar * z.clip(max=0) / c / bunch.beta) + alpha / omegabar * np.sinh(omegabar * z.clip(max=0) / c / bunch.beta))
        return wake
        
        
    def track(self, bunch):
        
        self.longitudinal_wakefield_kicks(bunch)
        
  
 
