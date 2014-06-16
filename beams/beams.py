'''
Created on 12.06.2014

@author: Kevin Li, Danilo Quartullo, Helga Timko
'''

import numpy as np
#import copy, h5py, sys
import sys
from scipy.constants import c, e
import cobra_functions.stats as cp


class Beam(object):
    
    def __init__(self, radius, mass, n_macroparticles, charge, intensity, momentum_or_gamma_program, mom_var=None):
        
        # Beam properties
        self.radius = radius
        self.mass = mass # in kg
        self.charge = charge # in C
        self.intensity = intensity # total no of particles
        self.counter = 0 # To step in the momentum program      

        # Beam energy
        if mom_var == None or mom_var == 'p':
            self.momentum_program = momentum_or_gamma_program # in eV
        elif mom_var == 'gamma':
            self.momentum_program = np.sqrt( momentum_or_gamma_program**2 - 1 ) \
                                    * mass * c**2 / e  # in eV
        else:
            print "ERROR: Unit of momentum program not recognized! Aborting..."
            sys.exit()

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

    # Derived energy-related properties 
    # Energy and momentum in units of eV   
    def p0_i(self):
        return self.momentum_program[self.counter]
    
    def p0_f(self):
        return self.momentum_program[self.counter + 1]

    def p0(self):
        return (self.p0_i() + self.p0_f()) / 2   
        
    def beta_i(self):
        return np.sqrt( 1 / (1 + (self.mass * c**2)**2 / (self.p0_i() * e)**2) )
        
    def beta_f(self):
        return np.sqrt( 1 / (1 + (self.mass * c**2)**2 / (self.p0_f() * e)**2) )
 
    def beta(self):
        return (self.beta_i() + self.beta_f()) / 2
        
    def gamma_i(self):
        return np.sqrt( 1 + (self.p0_i() * e)**2 / (self.mass * c**2)**2 )
    
    def gamma_f(self):
        return np.sqrt( 1 + (self.p0_f() * e)**2 / (self.mass * c**2)**2 )
    
    def gamma(self):
        return (self.gamma_i() + self.gamma_f()) / 2
    
    def energy_i(self):
        return np.sqrt( self.p0_i()**2 + (self.mass * c**2 / e)**2 )
    
    def energy_f(self):
        return np.sqrt( self.p0_f()**2 + (self.mass * c**2 / e)**2 )

    def energy(self):
        return (self.energy_i() + self.energy_f()) / 2    
 
            
    # Coordinate conversions
    @property
    def z(self):
        return - self.theta * self.radius 
     
    @z.setter
    def z(self, value):
        self.theta = - value / self.radius
    
    @property
    def delta(self):
        return self.dE / (self.beta_i()**2 * self.energy_i())

    @delta.setter
    def delta(self, value):
        self.dE = value * self.beta_i()**2 * self.energy_i()

    @property
    def z0(self):
        return - self.theta0 * self.radius 
    @z0.setter
    def z0(self, value):
        self.theta0 = - value / self.radius 
    
    @property
    def delta0(self):
        return self.dE0 / (self.beta_i()**2 * self.energy_i())
    @delta0.setter
    def delta0(self, value):
        self.dE0 = value * self.beta_i()**2 * self.energy_i()

    def reinit(self):

        np.copyto(self.x, self.x0)
        np.copyto(self.xp, self.xp0)
        np.copyto(self.y, self.y0)
        np.copyto(self.yp, self.yp0)
        np.copyto(self.theta, self.theta0)
        np.copyto(self.dE, self.dE0)
        np.copyto(self.z, self.z0)
        np.copyto(self.delta, self.delta0)
        
    # Statistics
    @property    
    def mean_z(self):
        return - self.mean_theta * self.radius 
    @mean_z.setter
    def mean_z(self, value):
        self.mean_theta = - value / self.radius 
    
    @property
    def mean_delta(self):
        return self.mean_dE / (self.beta_i()**2 * self.energy_i())
    @mean_delta.setter
    def mean_delta(self, value):
        self.mean_dE = value * self.beta_i()**2 * self.energy_i()

    @property    
    def sigma_z(self):
        return - self.sigma_theta * self.radius 
    @sigma_z.setter
    def sigma_z(self, value):
        self.sigma_theta = - value / self.radius 
    
    @property
    def sigma_delta(self):
        return self.sigma_dE / (self.beta_i()**2 * self.energy_i())
    @sigma_delta.setter
    def sigma_delta(self, value):
        self.sigma_dE = value * self.beta_i()**2 * self.energy_i()

    def longit_statistics(self):
        
        self.mean_theta = cp.mean(self.theta)
        self.mean_dE = cp.mean(self.dE)
        self.sigma_theta = cp.std(self.theta)
        self.sigma_dE = cp.std(self.dE)
        #self.epsn_l = 4 * np.pi * self.sigma_theta * self.sigma_dE * self.mass * ring.gamma_f * ring.beta_f * c / e
        # R.m.s. emittance in Gaussian approximation, other emittances to be defined
        self.eps_rms_l = np.pi * self.sigma_dE * self.sigma_theta \
                        * self.radius / (self.beta_i() * c) # in eVs
                                
    def transv_statistics(self):
        
        self.mean_x = cp.mean(self.x)
        self.mean_xp = cp.mean(self.xp)
        self.mean_y = cp.mean(self.y)
        self.mean_yp = cp.mean(self.yp)
        self.sigma_x = cp.std(self.x)
        self.sigma_y = cp.std(self.y)
        self.epsn_x_xp = cp.emittance(x, xp) * self.gamma_i() * self.beta_f() * 1e6
        self.epsn_y_yp = cp.emittance(y, yp) * self.gamma_f() * self.beta_f() * 1e6
    
    def losses(self, ring):
         
        for i in xrange(self.n_macroparticles):
            if not ring.is_in_separatrix(ring, self, self.theta[i], self.dE[i], self.delta[i]):
                # Set ID to zero
                self.id[i] = 0



