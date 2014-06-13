'''
Created on 12.06.2014

@author: Kevin Li, Danilo Quartullo, Helga Timko
'''

import numpy as np
import copy, h5py, sys
from scipy.constants import c, e, epsilon_0, m_e, m_p, pi
import cobra_functions.stats as cp


class Beam(object):
    
    def __init__(self, ring_and_RF, mass, n_macroparticles, charge, intensity):
        
        # Beam properties
        self.mass = mass # in kg
        self.charge = charge # in C
        self.intensity = intensity # total no of particles
        self.gamma = ring_and_RF.gamma_i(self)
        self.beta = ring_and_RF.beta_i(self)
        self.p0 = ring_and_RF.p0_i # in eV
        self.energy = ring_and_RF.energy_i(self) # in eV
        self.radius = ring_and_RF.radius # in m
        self.harmonic = ring_and_RF.harmonic # in eV
                
        # Beam coordinates
        self.x = np.empty([n_macroparticles])
        self.xp = np.empty([n_macroparticles])
        self.y = np.empty([n_macroparticles])
        self.yp = np.empty([n_macroparticles])
        self.theta = np.empty([n_macroparticles])
        self.dE = np.empty([n_macroparticles])
        
        # Initial coordinates (e.g. for ecloud)
        self.x0 = np.empty([n_macroparticles])
        self.xp0 = np.empty([n_macroparticles])
        self.y0 = np.empty([n_macroparticles])
        self.yp0 = np.empty([n_macroparticles])
        self.theta0 = np.empty([n_macroparticles])
        self.dE0 = np.empty([n_macroparticles])
        
        # Transverse and longitudinal properties, statistics       
        self.alpha_x = 0
        self.beta_x = 0
        self.epsn_x = 0
        self.alpha_y = 0
        self.beta_y = 0
        self.epsn_y = 0
        self.sigma_theta = 0
        self.sigma_dE = 0
        #self.epsn_z = 0
        
        # Particle/loss counts
        self.n_macroparticles = n_macroparticles
        self.n_macroparticles_lost = 0
        self.id = np.arange(1, self.n_macroparticles + 1, dtype=int)
    


    @property
    def z(self):
        return - self.theta * self.radius * self.harmonic[0] 
    
    @z.setter
    def z(self, value):
        self.theta = - value / self.radius / self.harmonic[0]
    
    @property
    def delta(self):
        return self.dE / (self.beta**2 * self.energy)

    @delta.setter
    def delta(self, value):
        self.dE = value * self.beta**2 * self.energy

    @property
    def z0(self):
        return - self.theta0 * self.radius * self.harmonic[0]
    @z0.setter
    def z0(self, value):
        self.theta0 = - value / self.radius / self.harmonic[0]
    
    @property
    def delta0(self):
        return self.dE0 / (self.beta**2 * self.energy)
    @delta0.setter
    def delta0(self, value):
        self.dE0 = value * self.beta**2 * self.energy

    def reinit(self):

        np.copyto(self.x, self.x0)
        np.copyto(self.xp, self.xp0)
        np.copyto(self.y, self.y0)
        np.copyto(self.yp, self.yp0)
        np.copyto(self.z, self.z0)
        np.copyto(self.delta, self.delta0)
        
    # Statistics
    @property    
    def mean_z(self):
        return - self.mean_theta * self.radius * self.harmonic[0]
    @mean_z.setter
    def mean_z(self, value):
        self.mean_theta = - value / self.radius / self.harmonic[0]
    
    @property
    def mean_delta(self):
        return self.mean_dE / (self.beta**2 * self.energy)
    @mean_delta.setter
    def mean_delta(self, value):
        self.mean_dE = value * self.beta**2 * self.energy

    @property    
    def sigma_z(self):
        return - self.sigma_theta * self.radius * self.harmonic[0]
    @sigma_z.setter
    def sigma_z(self, value):
        self.sigma_theta = - value / self.radius / self.harmonic[0]
    
    @property
    def sigma_delta(self):
        return self.sigma_dE / (self.beta**2 * self.energy)
    @sigma_delta.setter
    def sigma_delta(self, value):
        self.sigma_dE = value * self.beta**2 * self.energy

    def longit_statistics(self, ring):
        
        self.mean_theta = cp.mean(self.theta)
        self.mean_dE = cp.mean(self.dE)
        self.sigma_theta = cp.std(self.theta)
        self.sigma_dE = cp.std(self.dE)
        #self.epsn_l = 4 * np.pi * self.sigma_theta * self.sigma_dE * self.mass * ring.gamma_f * ring.beta_f * c / e
        # R.m.s. emittance in Gaussian approximation, other emittances to be defined
        self.eps_rms_l = np.pi * self.sigma_dE * self.sigma_theta \
                         * self.radius / (ring.beta_f(self) * c) # in eVs
    
    def transv_statistics(self, ring):
        
        self.mean_x = cp.mean(self.x)
        self.mean_xp = cp.mean(self.xp)
        self.mean_y = cp.mean(self.y)
        self.mean_yp = cp.mean(self.yp)
        self.sigma_x = cp.std(self.x)
        self.sigma_y = cp.std(self.y)
        self.epsn_x_xp = cp.emittance(x, xp) * ring.gamma_f * ring.beta_f * 1e6
        self.epsn_y_yp = cp.emittance(y, yp) * ring.gamma_f * ring.beta_f * 1e6
    
    def losses(self, ring):
         
        for i in xrange(self.n_macroparticles):
            if not ring.is_in_separatrix(ring, self, self.theta[i], self.dE[i], self.delta[i]):
                # Set ID to zero
                self.id[i] = 0
         



