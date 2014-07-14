'''
**Module gathering all the general input parameters used for the simulation**

:Authors: **Alexandre Lasheen**, **Danilo Quartullo**

'''

from __future__ import division
from scipy.constants import m_p, m_e, e, c
import numpy as np


class General_parameters(object):
    '''
    *Object containing all the general input parameters used for the simulation*
    '''

    def __init__(self, n_turns, ring_length_list, alpha, 
                 momentum_program, number_of_sections = 1, particle_type, user_mass = None, 
                 user_charge = None, particle_type_2 = None, user_mass_2 = None, 
                 user_charge_2 = None):

        #: | *Number of sections defines how many longitudinal maps are done per turn.*
        #: | *Default is one.*
        self.n_sections = number_of_sections
        
        #: | *Particle type*
        #: | *Recognized types: 'proton' and 'user_input' to input mass and charge manually.*
        #: | *Particle mass in [kg]* :math:`: \quad m` *
        #: | *Particle charge in [C]* :math:`: \quad q` *
        #: | *Second particle type: optional; does not affect the momentum, energy, beta, and gamma.*
        self.particle_type = particle_type
        self.particle_type_2 = particle_type_2
        
        # Attribution of mass and charge with respect to particle_type
        if self.particle_type is 'proton':
            self.mass =  m_p # [Kg]
            self.charge = e
        elif self.particle_type is 'electron':
            self.mass =  m_e # [Kg]
            self.charge = -e
        elif self.particle_type is 'user_input':
            self.mass = user_mass # [Kg]
            self.charge = user_charge
        else:
            raise RuntimeError('ERROR: Particle type not recognized!')
        
        if self.particle_type_2 == None:
            pass
        elif self.particle_type_2 is 'proton':
            self.mass2 =  m_p # [Kg]
            self.charge2 = e
        elif self.particle_type_2 is 'electron':
            self.mass2 =  m_e # [Kg]
            self.charge2 = -e
        elif self.particle_type_2 is 'user_input':
            self.mass2 = user_mass_2 # [Kg]
            self.charge2 = user_charge_2
        else:
            raise RuntimeError('ERROR: Second particle type not recognized!')
        
        #: *Number of turns of the simulation*
        self.n_turns = n_turns 

        #: | *Momentum (program) in [eV/c] for each RF section* :math:`: \quad p_n`
        #: | *Can be given as a single value to be assumed constant, or as a program of (n_turns + 1) terms in case of acceleration.*
        self.momentum_program = np.array(momentum_program)
        
        #: *Momentum compation factor (up to 2nd order) for each RF section* :math:`: \quad \alpha_i`
        self.alpha = np.array(alpha) 
        
        #: | *Ring length array contains the length of the RF sections, in [m]*
        self.ring_length_list = ring_length_list
        
        #: | *Ring circumference is the sum of lengths* :math:`: \quad C = \sum_k L_k`
        self.ring_circumference = np.sum(self.ring_length_list)
        
        #: *Ring radius in [m]* :math:`: \quad R`
        self.ring_radius = self.ring_circumference / (2 * np.pi)         
        
        #: *Check consistency of input data; raise error if not consistent*        
        if (self.n_sections == 1 and isinstance(self.ring_length_list, float) == False) or \
            (self.n_sections > 1 and self.n_sections != len(self.ring_length_list)) or \
            self.n_sections != self.alpha.shape[0] or \
            self.n_sections != self.momentum_program.shape[0]:
            raise RuntimeError('ERROR: Number of sections, ring length, alpha, and/or momentum data do not match!')    

        #: *Relativistic beta (program)* :math:`: \quad \beta_n`
        #:
        #: .. math:: \beta = \sqrt{ 1 + \frac{1}{1 + \left(\frac{mc^2}{ep}\right)^2} }
        self.beta_rel_program = np.sqrt(1 / (1 + (self.mass * c**2)**2 / 
                                             (self.momentum_program * self.charge)**2))
        
        #: *Relativistic gamma (program)* :math:`: \quad \gamma_n`
        #:
        #: .. math:: \gamma = \sqrt{ 1 + \left(\frac{ep}{mc^2}\right)^2 }
        self.gamma_rel_program = np.sqrt(1 + (self.momentum_program * self.charge)**2 / 
                                         (self.mass * c**2)**2) 
        
        #: *Energy (program) in [eV]* :math:`: \quad E_n`
        #:
        #: .. math:: \\E = \sqrt{ p^2 + \left(\frac{mc^2}{e}\right)^2 }
        self.energy_program = np.sqrt(self.momentum_program**2 + 
                                      (self.mass * c**2 / self.charge)**2)
        
        # Revolution period 
        self.T0 = self.ring_circumference / np.dot(ring_length_list/self.ring_circumference, self.beta_rel_program) / c 
        
        # Revolution frequency 
        self.f_rev = 1 / self.T0 
        
        #: *Slippage factor (order 0)* :math:`: \quad \eta_{0,n}`
        #:
        #: .. math:: \eta_0 = \alpha_0 - \frac{1}{\gamma^2}
        self.eta0 = 0
        
        #: *Slippage factor (order 1)* :math:`: \quad \eta_{1,n}`
        #:
        #: .. math:: \eta_1 = \frac{3\beta^2}{2\gamma^2} + \alpha_1 - \alpha_0\eta_0
        self.eta1 = 0
        
        #: *Slippage factor (order 2)* :math:`: \quad \eta_{2,n}`
        #:
        #: .. math:: \eta_2 = -\frac{\beta^2\left(5\beta^2-1\right)}{2\gamma^2} + \alpha_2 - 2\alpha_0\alpha_1 + \frac{\alpha_1}{\gamma^2} + \alpha_0^2\eta_0 - \frac{3\beta^2\alpha_0}{2\gamma^2}
        self.eta2 = 0
        
        # Warning that higher orders for alpha will not be used
        if len(self.alpha[0]) > 3:
            print 'WARNING : Momentum compaction factor is held only up to \
                   2nd order'
        
        
        if not self.momentum_program.shape[1] == self.n_turns + 1:
            raise RuntimeError('The input momentum program does not \
                                match the proper length (n_turns+1)')
        
        
        # Processing the slippage factor
        self.eta_generation()
                
                
    def eta_generation(self):
        '''
        | *Pre-processing of the slippage factor parameters with respect to the input momentum compaction factor (up to 2nd order) and the momentum program.*
        | *For eta coefficients, see Lee: Accelerator Physics (Wiley).*
        '''
        
        for i in xrange(len(self.alpha[0])):
            getattr(self, '_eta' + str(i))()

    
    def _eta0(self):
        '''
        *Calculation of the slippage factor (order 0) with respect to the
        momentum program and momentum compaction factor.*
        ''' 

        self.eta0 = np.empty([self.n_sections, self.n_turns+1])        
        for i in range(0, self.n_sections):
            self.eta0[i] = self.alpha[i,0] - self.gamma_rel_program[i]**-2
        
   
    
    def _eta1(self):
        '''
        *Calculation of the slippage factor (order 1) with respect to the
        momentum program and momentum compaction factor.*
        ''' 
                
        self.eta1 = np.empty([self.n_sections, self.n_turns+1])        
        for i in range(0, self.n_sections):
            self.eta1[i] = 3 * self.beta_rel_program[i]**2 / (2 * self.gamma_rel_program[i]**2) + \
                    self.alpha[i,1] - \
                    self.alpha[i,0] * self.eta0[i]
        
    
    
    def _eta2(self):
        '''
        *Calculation of the slippage factor (order 2) with respect to the
        momentum program and momentum compaction factor.*
        ''' 
                
        self.eta2 = np.empty([self.n_sections, self.n_turns+1])        
        for i in range(0, self.n_sections):
            self.eta2[i] = - self.beta_rel_program[i]**2 * (5 * self.beta_rel_program[i]**2 - 1) / (2 * self.gamma_rel_program[i]**2) + \
                    self.alpha[i,2] - 2 * self.alpha[i,0] * self.alpha[i,1] + \
                    self.alpha[i,1] / self.gamma_rel_program[i]**2 + \
                    self.alpha[i,0]**2 * self.eta0[i] - \
                    3 * self.beta_rel_program[i]**2 * self.alpha[i,0] / (2 * self.gamma_rel_program[i]**2)
         
        
        
