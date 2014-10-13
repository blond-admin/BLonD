'''
**Module gathering all the general input parameters used for the simulation**

:Authors: **Alexandre Lasheen**, **Danilo Quartullo**
'''

from __future__ import division
from scipy.constants import m_p, m_e, e, c
import numpy as np
import warnings


class GeneralParameters(object):
    '''
    *Object containing all the general input parameters used for the simulation.*
    '''

    def __init__(self, n_turns, ring_length, alpha, momentum, 
                 particle_type, user_mass = None, user_charge = None, 
                 particle_type_2 = None, user_mass_2 = None, 
                 user_charge_2 = None, number_of_sections = 1):
        
        #: | *Number of sections defines how many longitudinal maps are done per turn.*
        #: | *Default is one.*
        self.n_sections = number_of_sections
        
        #: | *Particle type*
        #: | *Recognized types: 'proton' and 'user_input' to input mass and charge manually.*
        #: | *Particle mass in [kg]* :math:`: \quad m`
        #: | *Particle charge in [C]* :math:`: \quad q`
        self.particle_type = particle_type
        
        #: *Second particle type: optional; does not affect the momentum, 
        #: energy, beta, and gamma.*
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
        #: | *In case of several sections without acceleration, the input is as follow : [[momentum_section_1], [momentum_section_2]]
        #: | *In case of several sections with momentum program, the input is as follow: [momentum_program_section_1, momentum_program_section_2]*
        self.momentum = np.array(momentum, ndmin =2)

        #: | *Momentum compaction factor (up to 2nd order) for each RF section* :math:`: \quad \alpha_i`
        #: | *Should be given as a list for multi-station (each element of the list should be a list that can contain up to 2nd order but must contain the same orders)*
        self.alpha = np.array(alpha, ndmin =2) 
        
        #: *Number of orders for the momentum compaction*
        self.alpha_order = self.alpha.shape[1]

        #: | *Ring length contains the length of the RF sections, in [m]*
        #: | *Should be given as a list for multi-station*
        self.ring_length = ring_length
        if isinstance(self.ring_length, float):
            self.ring_length = [self.ring_length]
        
        #: | *Ring circumference is the sum of section lengths* :math:`: \quad C = \sum_k L_k`
        self.ring_circumference = np.sum(self.ring_length)
        
        #: *Ring radius in [m]* :math:`: \quad R`
        self.ring_radius = self.ring_circumference / (2 * np.pi)         
        
        # Check consistency of input data; raise error if not consistent        
        if self.n_sections != len(self.ring_length) or \
           self.n_sections != self.alpha.shape[0] or \
           self.n_sections != self.momentum.shape[0]:
            raise RuntimeError('ERROR: Number of sections, ring length, alpha, \
                               and/or momentum data do not match!')    
        
        if self.n_sections > 1:
            if self.momentum.shape[1] == 1:
                self.momentum = self.momentum * np.ones(self.n_turns + 1)
        else:
            if self.momentum.size == 1:
                self.momentum = self.momentum * np.ones(self.n_turns + 1)

        if not self.momentum.shape[1] == self.n_turns + 1:
                raise RuntimeError('The input momentum program does not match \
                                    the proper length (n_turns+1)')
            
        
        #: *Relativistic beta (program)* :math:`: \quad \beta_n`
        #:
        #: .. math:: \beta = \sqrt{ 1 + \frac{1}{1 + \left(\frac{mc^2}{ep}\right)^2} }
        self.beta_r = np.sqrt(1 / (1 + (self.mass * c**2)**2 / 
                              (self.momentum * self.charge)**2))
        
        #: *Relativistic gamma (program)* :math:`: \quad \gamma_n`
        #:
        #: .. math:: \gamma = \sqrt{ 1 + \left(\frac{ep}{mc^2}\right)^2 }
        self.gamma_r = np.sqrt(1 + (self.momentum * self.charge)**2 / 
                               (self.mass * c**2)**2) 
        
        #: *Energy (program) in [eV]* :math:`: \quad E_n`
        #:
        #: .. math:: \\E = \sqrt{ p^2 + \left(\frac{mc^2}{e}\right)^2 }
        self.energy = np.sqrt(self.momentum**2 + 
                              (self.mass * c**2 / self.charge)**2)
        
        #: *Revolution period [s]* :math:`: \quad T_0 = \frac{C}{\beta c}`
        self.t_rev = np.dot(self.ring_length, 1/(self.beta_r*c))
 
        #: *Revolution frequency [Hz]* :math:`: \quad f_0 = \frac{1}{T_0}`
        self.f_rev = 1 / self.t_rev
         
        #: *Revolution angular frequency [rad/s]* :math:`: \quad \omega_0 = 2\pi f_0`
        self.omega_rev = 2 * np.pi * self.f_rev
        
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
        
        for i in xrange(self.alpha_order):
            getattr(self, '_eta' + str(i))()

    
    def _eta0(self):
        '''
        *Calculation of the slippage factor (order 0) with respect to the
        momentum program and momentum compaction factor.*
        ''' 

        self.eta0 = np.empty([self.n_sections, self.n_turns+1])        
        for i in range(0, self.n_sections):
            self.eta0[i] = self.alpha[i,0] - self.gamma_r[i]**-2
        
   
    
    def _eta1(self):
        '''
        *Calculation of the slippage factor (order 1) with respect to the
        momentum program and momentum compaction factor.*
        ''' 
                
        self.eta1 = np.empty([self.n_sections, self.n_turns+1])        
        for i in range(0, self.n_sections):
            self.eta1[i] = 3 * self.beta_r[i]**2 / (2 * self.gamma_r[i]**2) + \
                           self.alpha[i,1] - \
                           self.alpha[i,0] * self.eta0[i]
        
    
    
    def _eta2(self):
        '''
        *Calculation of the slippage factor (order 2) with respect to the
        momentum program and momentum compaction factor.*
        ''' 
                
        self.eta2 = np.empty([self.n_sections, self.n_turns+1])        
        for i in range(0, self.n_sections):
            self.eta2[i] = - self.beta_r[i]**2 * (5 * self.beta_r[i]**2 - 1) / (2 * self.gamma_r[i]**2) + \
                           self.alpha[i,2] - 2 * self.alpha[i,0] * self.alpha[i,1] + \
                           self.alpha[i,1] / self.gamma_r[i]**2 + \
                           self.alpha[i,0]**2 * self.eta0[i] - \
                           3 * self.beta_r[i]**2 * self.alpha[i,0] / (2 * self.gamma_r[i]**2)
         
    
    
