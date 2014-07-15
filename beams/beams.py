'''
Created on 12.06.2014

@author: Kevin Li, Danilo Quartullo, Helga Timko, ALexandre Lasheen
'''

from __future__ import division
import numpy as np
import sys
from scipy.constants import c, e, m_p
import cython_functions.stats as cp
from scipy.optimize import curve_fit
from trackers.longitudinal_tracker import is_in_separatrix
from scipy import ndimage


class Beam(object):
    
    def __init__(self, General_parameters, n_macroparticles, intensity):
        
        # Beam and ring-dependent properties
        
        self.mass = General_parameters.mass
        self.charge = General_parameters.charge
        self.ring_radius = General_parameters.ring_radius
        self.intensity = intensity # total no of particles
        
        #: Relativistic beta of the synchronous particle
        self.beta_rel = General_parameters.beta_rel_program[0][0]
        #: Relativistic gamma of the synchronous particle
        self.gamma_rel = General_parameters.gamma_rel_program[0][0]
        #: Energy of the synchronous particle [eV]
        self.energy = General_parameters.energy_program[0][0]
        #: Momentum of the synchronous particle [eV/c]
        self.momentum = General_parameters.momentum_program[0][0] 

        # Beam coordinates
        self.x = np.empty([n_macroparticles])
        self.xp = np.empty([n_macroparticles])
        self.y = np.empty([n_macroparticles])
        self.yp = np.empty([n_macroparticles])
        self.theta = np.empty([n_macroparticles])
        self.dE = np.empty([n_macroparticles])
     
        # Transverse and longitudinal properties, statistics       
        
        self.alpha_x = 0
        self.beta_x = 0
        self.epsn_x = 0
        self.alpha_y = 0
        self.beta_y = 0
        self.epsn_y = 0
        self.sigma_theta = 0
        self.sigma_dE = 0
        
        # Particle/loss counts
        self.n_macroparticles = n_macroparticles
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
        return self.dE / (self.beta_rel**2 * self.energy)

    @delta.setter
    def delta(self, value):
        self.dE = value * self.beta_rel**2 * self.energy

    @property
    def tau(self):
        return  self.theta * self.ring_radius / (self.beta_rel * c)
     
    @tau.setter
    def tau(self, value):
        self.theta = value * self.beta_rel * c / self.ring_radius

    # Statistics
    
    @property    
    def mean_z(self):
        return - self.mean_theta * self.ring_radius
    @mean_z.setter
    def mean_z(self, value):
        self.mean_theta = - value / self.ring_radius
    
    @property
    def mean_delta(self):
        return self.mean_dE / (self.beta_rel**2 * self.energy)
    @mean_delta.setter
    def mean_delta(self, value):
        self.mean_dE = value * self.beta_rel**2 * self.energy
    
    @property    
    def mean_tau(self):
        return self.mean_theta * self.ring_radius / (self.beta_rel * c)
    @mean_tau.setter
    def mean_tau(self, value):
        self.mean_theta = value * self.beta_rel * c / self.ring_radius

    @property    
    def sigma_z(self):
        return - self.sigma_theta * self.ring_radius
    @sigma_z.setter
    def sigma_z(self, value):
        self.sigma_theta = - value / self.ring_radius
    
    @property
    def sigma_delta(self):
        return self.sigma_dE / (self.beta_rel**2 * self.energy)
    @sigma_delta.setter
    def sigma_delta(self, value):
        self.sigma_dE = value * self.beta_rel**2 * self.energy
    
    @property
    def sigma_tau(self):
        return self.sigma_theta * self.ring_radius / (self.beta_rel * c)
    @sigma_tau.setter
    def sigma_tau(self, value):
        self.sigma_theta = value * self.beta_rel * c / self.ring_radius

    
    def longit_statistics(self, gaussian_fit="Off", slices=None):
        
        # Statistics only for particles that are not flagged as lost
        itemindex = np.where(self.id != 0)[0]
        self.mean_theta = np.mean(self.theta[itemindex])
        self.mean_dE = np.mean(self.dE[itemindex])
        self.sigma_theta = np.std(self.theta[itemindex])
        self.sigma_dE = np.std(self.dE[itemindex])
       
        ##### R.m.s. emittance in Gaussian approximation, other emittances to be defined
        self.epsn_rms_l = np.pi * self.sigma_dE * self.sigma_theta \
                        * self.ring_radius / (self.beta_rel * c) # in eVs

        ##### Gaussian fit to theta-profile
        if gaussian_fit == "On":

            p0 = [max(slices.n_macroparticles), self.mean_theta, self.sigma_theta]
        
            def gauss(x, *p):
                A, x0, sx = p
                return A*np.exp(-(x-x0)**2/2./sx**2) 
            
            pfit = curve_fit(gauss, slices.bins_centers, 
                                   slices.n_macroparticles, p0)[0]
            
            self.bl_gauss = 4 * abs(pfit[2]) 

                                
    def transv_statistics(self):
        
        self.mean_x = np.mean(self.x)
        self.mean_xp = np.mean(self.xp)
        self.mean_y = np.mean(self.y)
        self.mean_yp = np.mean(self.yp)
        self.sigma_x = np.std(self.x)
        self.sigma_y = np.std(self.y)
        self.epsn_x_xp = cp.emittance(self.x, self.xp) * self.gamma_rel \
                        * self.beta_rel * 1e6
        self.epsn_y_yp = cp.emittance(self.y, self.yp) * self.gamma_rel \
                        * self.beta_rel * 1e6
    
    def losses_separatrix(self, GeneralParameters, RingAndRFSection):
        
        itemindex = np.where(is_in_separatrix(GeneralParameters, RingAndRFSection,
                                 self.theta, self.dE, self.delta) == False)[0]

        if itemindex.size != 0:    
            self.id[itemindex] = 0
    
    def losses_longitudinal_cut(self, theta_min, theta_max): 
    
        itemindex = np.where( (self.theta < theta_min or self.theta > theta_max) )[0]
        
        if itemindex.size != 0:          
            self.id[itemindex] = 0       
        
        



