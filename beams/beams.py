
# Copyright 2015 CERN. This software is distributed under the
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
from trackers.utilities import is_in_separatrix




class Beam(object):
    '''
    *Object containing the beam coordinates and beam properties such as mass,
    charge, synchronous energy, momentum, etc.
    The beam coordinate 'dt' is defined as the particle arrival time to the RF 
    station w.r.t. the reference time that is the sum of turns.
    The beam coordiate 'dE' is defined as the particle energy offset w.r.t. the
    energy of the synchronous particle.* 
    '''
    
    def __init__(self, GeneralParameters, n_macroparticles, intensity):
        
        #: *Import particle mass [eV] (from GeneralParameters)*
        self.mass = GeneralParameters.mass
        
        #: *Import particle charge [e] (from GeneralParameters)*
        self.charge = GeneralParameters.charge
        
        #: *Import synchronous relativistic beta [1] (from GeneralParameters)*
        self.beta = GeneralParameters.beta[0][0]
        
        #: *Import synchronous relativistic gamma [1] (from GeneralParameters)*
        self.gamma = GeneralParameters.gamma[0][0]
        
        #: *Import synchronous total energy [eV] (from GeneralParameters)*
        self.energy = GeneralParameters.energy[0][0]
        
        #: *Import synchronous momentum [eV] (from GeneralParameters)*
        self.momentum = GeneralParameters.momentum[0][0] 
        
        #: *Import ring radius [m] (from GeneralParameters)*
        self.ring_radius = GeneralParameters.ring_radius

        #: | *Beam arrival time with respect to reference time [s]*
        self.dt = np.zeros([n_macroparticles])
        
        #: | *Beam energy offset with respect to synchronous energy [eV]*
        self.dE = np.zeros([n_macroparticles])
     
        #: | *Average beam arrival time [s]*
        self.mean_dt = 0
        
        #: | *Average beam energy offset [eV]*
        self.mean_dE = 0
        
        #: | *Standard deviation of beam arrival time [s]*
        self.sigma_dt = 0
        
        #: | *Standard deviation of beam energy offset [eV]*
        self.sigma_dE = 0
        
        #: | *Total beam intensity [1]*
        self.intensity = intensity 
        
        #: | *Total number of macro-particles in the beam [1]*
        self.n_macroparticles = int(n_macroparticles)
        
        #: | *Number of macro-particles marked as 'lost' [1]*
        #: | *Losses defined via loss mechanisms chosen by user*
        self.n_macroparticles_lost = 0
        
        #: | *Number of transmitted macro-particles (= total - lost) [1]*        
        self.n_macroparticles_alive = self.n_macroparticles - self.n_macroparticles_lost
        
        #: | *Unique macro-particle ID number; zero if particle is 'lost'*                
        self.id = np.arange(1, self.n_macroparticles + 1, dtype=int)

 
    def statistics(self):
        '''
        *Calculation of the mean and standard deviation of beam coordinates,
        as well as beam emittance using different definitions.*
        '''
        
        # Statistics only for particles that are not flagged as lost
        itemindex = np.where(self.id != 0)[0]
        self.mean_dt = np.mean(self.dt[itemindex])
        self.mean_dE = np.mean(self.dE[itemindex])
        self.sigma_dt = np.std(self.dt[itemindex])
        self.sigma_dE = np.std(self.dE[itemindex])
       
        # R.m.s. emittance in Gaussian approximation
        self.epsn_rms_l = 1.e3*np.pi*self.sigma_dE*self.sigma_dt # in eVs

        
    def losses_separatrix(self, GeneralParameters, RFSectionParameters, Beam):
        '''
        *Beam losses based on separatrix.*
        '''
        
        itemindex = np.where(is_in_separatrix(GeneralParameters, 
                                              RFSectionParameters, 
                                              Beam, self.dt, self.dE) 
                             == False)[0]

        if itemindex.size != 0:    
            self.id[itemindex] = 0
    
    
    def losses_cut(self, dt_min, dt_max): 
        '''
        *Beam losses based on longitudinal cuts.*
        '''
    
        itemindex = np.where( (self.dt - dt_min)*(dt_max - self.dt) < 0 )[0]
        
        if itemindex.size != 0:          
            self.id[itemindex] = 0       
        
        
           
