
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module gathering all general input parameters used for the simulation.**

:Authors: **Alexandre Lasheen**, **Danilo Quartullo**, **Helga Timko**
'''

from __future__ import division
from builtins import str, range, object
import numpy as np
import warnings
from scipy.constants import m_p, m_e, e, c



class GeneralParameters(object):
    '''
    *Object containing all the general input parameters used for the simulation.*
    '''

    def __init__(self, n_turns, ring_length, alpha, momentum, 
                 particle_type, user_mass = None, user_charge = None, 
                 particle_type_2 = None, user_mass_2 = None, 
                 user_charge_2 = None, number_of_sections = 1, force_beta_equal_one = False):
        
        #: | *Number of RF sections over the ring; default is one.*
        self.n_sections = number_of_sections
        
        #: | *Particle type. Recognized types are: 'proton' and 'electron'.*
        #: | *Use 'user_input' to input mass and charge manually.*
        #: | *Input particle mass in [eV]* :math:`: \quad m`
        #: | *Input particle charge in [e]* :math:`: \quad q`
        self.particle_type = particle_type
        
        #: *Second particle type: optional; does not affect the momentum, 
        #: energy, beta, and gamma.*
        self.particle_type_2 = particle_type_2
        
        # Attribution of mass and charge with respect to particle_type
        if self.particle_type is 'proton':
            self.mass =  m_p*c**2/e # [eV]
            self.charge = 1. # [e]
        elif self.particle_type is 'electron':
            self.mass =  m_e*c**2/e # [eV]
            self.charge = -1. # [e]
        elif self.particle_type is 'user_input':
            self.mass = user_mass # [eV]
            self.charge = user_charge # [e]
        else:
            raise RuntimeError('ERROR: Particle type not recognized!')
        
        if self.particle_type_2 == None:
            pass
        elif self.particle_type_2 is 'proton':
            self.mass2 =  m_p*c**2/e # [eV]
            self.charge2 = 1. # [e]
        elif self.particle_type_2 is 'electron':
            self.mass2 =  m_e*c**2/e # [eV]
            self.charge2 = -1. # [e]
        elif self.particle_type_2 is 'user_input':
            self.mass2 = user_mass_2 # [eV]
            self.charge2 = user_charge_2 # [e]
        else:
            raise RuntimeError('ERROR: Second particle type not recognized!')
        
        #: *Number of turns of the simulation*
        self.n_turns = n_turns 
        
        if type(momentum)==tuple:
            self.momentum = np.array(momentum[1], ndmin = 2)
            self.cumulative_times = momentum[0]
        else:
            #: | *Synchronous momentum (program) in [eV] for each RF section* :math:`: \quad p_{s,k}^n`
            #: | *Can be given as a single constant value, or as a program of (n_turns + 1) turns.*
            #: | *In case of several sections without acceleration, input: [[momentum_section_1], [momentum_section_2]]*
            #: | *In case of several sections with acceleration, input: [momentum_program_section_1, momentum_program_section_2]*
            self.momentum = np.array(momentum, ndmin = 2)

        #: | *Momentum compaction factor (up to 2nd order) for each RF section* :math:`: \quad \alpha_{k,i}`
        #: | *Should be given as a list for multiple RF stations (each element of the list should be a list of alpha factors up to 2nd order)*
        self.alpha = np.array(alpha, ndmin = 2) 
        
        #: *Number of orders for the momentum compaction*
        self.alpha_order = self.alpha.shape[1]

        #: | *Ring length contains the length of the RF sections, in [m]*
        #: | *Should be given as a list for multiple RF stations*
        self.ring_length = ring_length
        if isinstance(self.ring_length, float) or isinstance(self.ring_length, int):
            self.ring_length = [self.ring_length]
        
        #: | *Ring circumference is the sum of section lengths* :math:`: \quad C = \sum_k L_k`
        self.ring_circumference = np.sum(self.ring_length)
        
        #: *Ring radius in [m]* :math:`: \quad R`
        self.ring_radius = self.ring_circumference/(2*np.pi)         
        
        # Check consistency of input data; raise error if not consistent
        
        if self.n_sections != len(self.ring_length) or \
           self.n_sections != self.alpha.shape[0] or \
           self.n_sections != self.momentum.shape[0]:
            raise RuntimeError('ERROR: Number of sections, ring length, alpha,'+
                               ' and/or momentum data do not match!')    
        
        if self.n_sections > 1:
            if self.momentum.shape[1] == 1:
                self.momentum = self.momentum*np.ones(self.n_turns + 1)
        else:
            if self.momentum.size == 1:
                self.momentum = self.momentum*np.ones(self.n_turns + 1)

        if not self.momentum.shape[1] == self.n_turns + 1:
                raise RuntimeError('The input momentum program does not match'+ 
                ' the proper length (n_turns+1)')
            
        
        #: *Synchronous relativistic beta (program)* :math:`: \quad \beta_{s,k}^n`
        #:
        #: .. math:: \beta_s = \frac{1}{\sqrt{1 + \left(\frac{m}{p_s}\right)^2} }
        
        self.beta = np.sqrt(1/(1 + (self.mass/self.momentum)**2))
        if force_beta_equal_one:
            self.beta = np.array([np.ones(self.n_turns + 1)])
        
        #: *Synchronous relativistic gamma (program)* :math:`: \quad \gamma_{s,k}^n`
        #:
        #: .. math:: \gamma_s = \sqrt{ 1 + \left(\frac{p_s}{m}\right)^2 }
        self.gamma = np.sqrt(1 + (self.momentum/self.mass)**2) 
        
        #: *Synchronous total energy (program) in [eV]* :math:`: \quad E_{s,k}^n`
        #:
        #: .. math:: E_s = \sqrt{ p_s^2 + m^2 }
        self.energy = np.sqrt(self.momentum**2 + self.mass**2)
        
        #: *Synchronous kinetic energy (program) in [eV]
        #: .. math:: E_s^kin = \sqrt{ p_s^2 + m^2 } - m
        self.kin_energy = np.sqrt(self.momentum**2 + self.mass**2) - self.mass
        
        # Be careful that self.cycle_time in the else statement starts always with 0.
        if type(momentum)==tuple:
            #: *Cumulative times [s] taken from preprocess ramp method*
            self.cycle_time = self.cumulative_times
            #: *Revolution period [s]* :math:`: \quad T_0 = \frac{C}{\beta_s c}`
            self.t_rev = np.append(np.diff(self.cycle_time),self.ring_circumference/(self.beta[0][-1]*c))
        else:    
            self.t_rev = np.dot(self.ring_length, 1/(self.beta*c))
            self.cycle_time = np.cumsum(self.t_rev)
            
        #: *Revolution frequency [Hz]* :math:`: \quad f_0 = \frac{1}{T_0}`
        self.f_rev = 1/self.t_rev
         
        
        #: *Revolution angular frequency [1/s]* :math:`: \quad \omega_0 = 2\pi f_0`
        self.omega_rev = 2*np.pi*self.f_rev
        
        #: *Slippage factor (0th order)* :math:`: \quad \eta_{k,0}`
        #:
        #: .. math:: \eta_0 = \alpha_0 - \frac{1}{\gamma_s^2}
        self.eta_0 = 0
        
        #: *Slippage factor (1st order)* :math:`: \quad \eta_{k,1}`
        #:
        #: .. math:: \eta_1 = \frac{3\beta_s^2}{2\gamma_s^2} + \alpha_1 - \alpha_0\eta_0
        self.eta_1 = 0
        
        #: *Slippage factor (2nd order)* :math:`: \quad \eta_{k,2}`
        #:
        #: .. math:: \eta_2 = -\frac{\beta_s^2\left(5\beta_s^2-1\right)}{2\gamma_s^2} + \alpha_2 - 2\alpha_0\alpha_1 + \frac{\alpha_1}{\gamma_s^2} + \alpha_0^2\eta_0 - \frac{3\beta_s^2\alpha_0}{2\gamma_s^2}
        self.eta_2 = 0
        
        # Warning that higher orders for alpha will not be used
        if self.alpha_order > 3:
            warnings.filterwarnings("once")
            warnings.warn("WARNING: Momentum compaction factor is implemented \
                          only up to 2nd order")
            self.alpha_order = 3         
                 
        # Processing the slippage factor
        self.eta_generation()
                
                
    def eta_generation(self):
        '''
        | *Pre-processing of the slippage factor parameters with respect to the input momentum compaction factor (up to 2nd order) and the momentum program.*
        | *For eta coefficients, see Lee: Accelerator Physics (Wiley).*
        '''
        
        for i in range(self.alpha_order):
            getattr(self, '_eta' + str(i))()

    
    def _eta0(self):
        '''
        *Calculation of the slippage factor (0th order) with respect to the
        momentum program and momentum compaction factor.*
        ''' 

        self.eta_0 = np.empty([self.n_sections, self.n_turns+1])        
        for i in range(0, self.n_sections):
            self.eta_0[i] = self.alpha[i,0] - self.gamma[i]**-2 
   
    
    def _eta1(self):
        '''
        *Calculation of the slippage factor (1st order) with respect to the
        momentum program and momentum compaction factor.*
        ''' 
                
        self.eta_1 = np.empty([self.n_sections, self.n_turns+1])        
        for i in range(0, self.n_sections):
            self.eta_1[i] = 3*self.beta[i]**2/(2*self.gamma[i]**2) + \
                           self.alpha[i,1] - self.alpha[i,0]*self.eta_0[i]
        
        
    def _eta2(self):
        '''
        *Calculation of the slippage factor (2nd order) with respect to the
        momentum program and momentum compaction factor.*
        ''' 
                
        self.eta_2 = np.empty([self.n_sections, self.n_turns+1])        
        for i in range(0, self.n_sections):
            self.eta_2[i] = - self.beta[i]**2*(5*self.beta[i]**2 - 1)/ \
                           (2*self.gamma[i]**2) + self.alpha[i,2] - \
                           2*self.alpha[i,0]*self.alpha[i,1] + self.alpha[i,1]/ \
                           self.gamma[i]**2 + self.alpha[i,0]**2*self.eta_0[i] - \
                           3*self.beta[i]**2*self.alpha[i,0]/(2*self.gamma[i]**2)



    def parameters_at_time(cycle_time):
        '''
        *Function to return various cycle parameters at a specific point in time.*
        '''

        parameters = {}
        parameters['momentum'] = np.interp(cycle_time, self.cumulative_times, self.momentum[0])
        parameters['beta'] = np.interp(cycle_time, self.cumulative_times, self.beta[0])
        parameters['gamma'] = np.interp(cycle_time, self.cumulative_times, self.gamma[0])
        parameters['energy'] = np.interp(cycle_time, self.cumulative_times, self.energy[0])
        parameters['kin_energy'] = np.interp(cycle_time, self.cumulative_times, self.kin_energy[0])
        parameters['f_rev'] = np.interp(cycle_time, self.cumulative_times, self.f_rev)
	parameters['t_rev'] = np.interp(cycle_time, self.cumulative_times, self.t_rev)
	parameters['omega_rev'] = np.interp(cycle_time, self.cumulative_times, self.omega_rev)
	parameters['eta_0'] = np.interp(cycle_time, self.cumulative_times, self.eta_0[0])

        return parameters





