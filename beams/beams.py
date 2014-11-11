
# Copyright 2014 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module containing the fundamental beam class with methods to compute beam statistics**

:Authors: **Danilo Quartullo**, **Helga Timko**, **ALexandre Lasheen**
'''

from __future__ import division
import numpy as np
from scipy.constants import c
from trackers.utilities import is_in_separatrix



class Beam(object):
    
    def __init__(self, General_parameters, n_macroparticles, intensity):
        
        # Beam and ring-dependent properties
  
        self.mass = General_parameters.mass
        self.charge = General_parameters.charge
        self.ring_radius = General_parameters.ring_radius
        self.intensity = intensity # total no of particles
        
        #: Relativistic beta of the synchronous particle
        self.beta_r = General_parameters.beta_r[0][0]
        #: Relativistic gamma of the synchronous particle
        self.gamma_r = General_parameters.gamma_r[0][0]
        #: Energy of the synchronous particle [eV]
        self.energy = General_parameters.energy[0][0]
        #: Momentum of the synchronous particle [eV/c]
        self.momentum = General_parameters.momentum[0][0] 

        # Beam coordinates
        self.theta = np.zeros([n_macroparticles])
        self.dE = np.zeros([n_macroparticles])
     
        # Properties and statistics       
        self.mean_theta = 0
        self.mean_dE = 0
        self.sigma_theta = 0
        self.sigma_dE = 0
        
        # Particle/loss counts
        self.n_macroparticles = int(n_macroparticles)
        self.n_macroparticles_lost = 0
        self.n_macroparticles_alive = self.n_macroparticles - self.n_macroparticles_lost
        self.id = np.arange(1, self.n_macroparticles + 1, dtype=int)


    # Coordinate conversions
    @property
    def z(self):
        return - self.theta * self.ring_radius
     
    @z.setter
    def z(self, value):
        self.theta = - value / self.ring_radius
    
    @property
    def delta(self):
        return self.dE / (self.beta_r**2 * self.energy)

    @delta.setter
    def delta(self, value):
        self.dE = value * self.beta_r**2 * self.energy

    @property
    def tau(self):
        return  self.theta * self.ring_radius / (self.beta_r * c)
     
    @tau.setter
    def tau(self, value):
        self.theta = value * self.beta_r * c / self.ring_radius

    # Statistics
    @property    
    def mean_z(self):
        return - self.mean_theta * self.ring_radius
    @mean_z.setter
    def mean_z(self, value):
        self.mean_theta = - value / self.ring_radius
    
    @property
    def mean_delta(self):
        return self.mean_dE / (self.beta_r**2 * self.energy)
    @mean_delta.setter
    def mean_delta(self, value):
        self.mean_dE = value * self.beta_r**2 * self.energy
    
    @property    
    def mean_tau(self):
        return self.mean_theta * self.ring_radius / (self.beta_r * c)
    @mean_tau.setter
    def mean_tau(self, value):
        self.mean_theta = value * self.beta_r * c / self.ring_radius

    @property    
    def sigma_z(self):
        return - self.sigma_theta * self.ring_radius
    @sigma_z.setter
    def sigma_z(self, value):
        self.sigma_theta = - value / self.ring_radius
    
    @property
    def sigma_delta(self):
        return self.sigma_dE / (self.beta_r**2 * self.energy)
    @sigma_delta.setter
    def sigma_delta(self, value):
        self.sigma_dE = value * self.beta_r**2 * self.energy
    
    @property
    def sigma_tau(self):
        return self.sigma_theta * self.ring_radius / (self.beta_r * c)
    @sigma_tau.setter
    def sigma_tau(self, value):
        self.sigma_theta = value * self.beta_r * c / self.ring_radius
    
    # Gaussian fit conversion   
    @property
    def bl_gauss_tau(self):
        '''*Gaussian bunch length converted to the tau coordinate in [s]*'''
        return self.bl_gauss * self.ring_radius / (self.beta_r * c)
    
    @property
    def bl_gauss_z(self):
        '''*Gaussian bunch length to the z coordinate in [m]*'''
        return self.bl_gauss * self.ring_radius 

    @property
    def bp_gauss_tau(self):
        '''*Gaussian bunch position converted to the tau coordinate in [s]*'''
        return self.bp_gauss * self.ring_radius / (self.beta_r * c)
    
    @property
    def bp_gauss_z(self):
        '''*Gaussian bunch position converted to the z coordinate in [m]*'''
        return - self.bp_gauss * self.ring_radius      

    
    def statistics(self):
        
        # Statistics only for particles that are not flagged as lost
        itemindex = np.where(self.id != 0)[0]
        self.mean_theta = np.mean(self.theta[itemindex])
        self.mean_dE = np.mean(self.dE[itemindex])
        self.sigma_theta = np.std(self.theta[itemindex])
        self.sigma_dE = np.std(self.dE[itemindex])
       
        ##### R.m.s. emittance in Gaussian approximation, other emittances to be defined
        self.epsn_rms_l = np.pi * self.sigma_dE * self.sigma_theta \
                        * self.ring_radius / (self.beta_r * c) # in eVs

        
    def losses_separatrix(self, GeneralParameters, RFSectionParameters):
        
        itemindex = np.where(is_in_separatrix(GeneralParameters, RFSectionParameters,
                                 self.theta, self.dE, self.delta) == False)[0]

        if itemindex.size != 0:    
            self.id[itemindex] = 0
    
    
    def losses_cut(self, theta_min, theta_max): 
    
        itemindex = np.where( (self.theta - theta_min)*(theta_max - self.theta) < 0 )[0]
        
        if itemindex.size != 0:          
            self.id[itemindex] = 0       
        
        

                



