
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
    """ Class containing the general properties of the synchrotron that are 
    independent of the RF system or the beam. The index 'n' denotes time steps,
    'k' ring segments and 'i' momentum compaction orders.
    
    Parameters
    ----------
    n_turns : int
        Number of turns [1] to be simulated.
    ring_length : float
        Length [m] of the n_sections ring segments of the synchrotron.
        Input as a list for multiple RF stations.
    alpha : float (opt: float array)
        Momentum compaction factor [1]; can be input as single float (only 0th 
        order element) or float array (up to 2nd order elements). In case of 
        several sections without higher orders, input: [[alpha_section_1], 
        [alpha_section_2], etc.]. In case of several sections and higher order
        alphas, input: [alpha_array_section_1, alpha_array_section_2, etc.], 
        :math:`: \quad \alpha_{k,i}`.
    momentum : float (opt: float array/matrix)
        Design synchronous particle momementum [eV] on the design orbit. Input
        for each RF section :math:`: \quad p_{s,k}^n`. Can be input as a single
        constant float, or as a program of (n_turns + 1) turns. In case of 
        several sections without acceleration, input: [[momentum_section_1], 
        [momentum_section_2], etc.]. In case of several sections with 
        acceleration, input: [momentum_program_section_1, 
        momentum_program_section_2, etc.]. Can be input also as a tuple of time
        and momentum, see also 'cumulative_times'.
    particle_type : string
        Primary particle type that is reference for the momentum; Recognized 
        types are 'proton' and 'electron'. Use 'user_input' to input mass and 
        charge manually.
    n_sections : int
        Optional: number of RF sections [1] over the ring; default is 1.
        
    Attributes
    ----------
    ring_circumference : float
        Circumference [m] of the synchrotron. Sum of ring segment lengths, 
        :math:`: \quad C = \sum_k L_k`.
    ring_radius : float
        Radius [m] of the synchrotron, :math:`: \quad R = C/(2 \pi)`
    alpha_order : int
        Number of orders of the momentum compaction factor
    eta_0 : float
        Zeroth order slippage factor [1] :math:`: \quad \eta_{k,0}`
        .. math:: \eta_0 = \alpha_0 - \frac{1}{\gamma_s^2}
    eta_1 : float
        First order slippage factor [1] :math:`: \quad \eta_{k,1}`
        .. math:: \eta_1 = \frac{3\beta_s^2}{2\gamma_s^2} + \alpha_1 - \alpha_0\eta_0
    eta_2 : float
        Second order slippage factor [1] :math:`: \quad \eta_{k,2}`
        .. math:: \eta_2 = -\frac{\beta_s^2\left(5\beta_s^2-1\right)}{2\gamma_s^2} + \alpha_2 - 2\alpha_0\alpha_1 + \frac{\alpha_1}{\gamma_s^2} + \alpha_0^2\eta_0 - \frac{3\beta_s^2\alpha_0}{2\gamma_s^2}
    cumulative_times : float array
        Cycle time moments [s] taken from preprocess ramp method. Possibility
        to extract cycle parameters at these moments using 'parameters_at_time'
    mass : float
        Primary particle mass [eV].
    charge : float
        Primary particle charge [e].
    beta : float matrix
        Synchronous relativistic beta program [1] for each segment of the ring
        :math:`: \quad \beta_{s,k}^n`
        .. math:: \beta_s = \frac{1}{\sqrt{1 + \left(\frac{m}{p_s}\right)^2} }
    gamma : float matrix
        Synchronous relativistic gamma program [1] for each segment of the ring
        :math:`: \quad \gamma_{s,k}^n`
        .. math:: \gamma_s = \sqrt{ 1 + \left(\frac{p_s}{m}\right)^2 }
    energy : float matrix
        Synchronous total energy program [eV] for each segment of the ring
        :math:`: \quad E_{s,k}^n`
        .. math:: E_s = \sqrt{ p_s^2 + m^2 }
    kin_energy : float matrix
        Synchronous kinetic energy program [eV] for each segment of the ring
        .. math:: E_s^kin = \sqrt{ p_s^2 + m^2 } - m
    t_rev : float array
        Revolution period [s] turn by turn.
        :math:`: \quad T_0 = \frac{C}{\beta_s c}`
    f_rev : float array
        Revolution frequency [Hz] :math:`: \quad f_0 = \frac{1}{T_0}`
    omega_rev : float array
        Revolution angular frequency [1/s] :math:`: \quad \omega_0 = 2\pi f_0`

    Examples
    --------
    To declare a single-stationed synchrotron at constant energy:
    >>> from input_parameters.general_parameters import GeneralParameters
    >>>
    >>> n_turns = 10
    >>> C = 26659
    >>> alpha = 3.21e-4
    >>> momentum = 450e9
    >>> general_parameters = GeneralParameters(n_turns, C, eta, momentum, 'proton')

    To declare a double-stationed synchrotron at constant energy and higher-
    order momentum compaction factors:
    >>> from input_parameters.general_parameters import GeneralParameters
    >>>
    >>> n_turns = 10
    >>> C = [13000, 13659]
    >>> alpha = [[3.21e-4, 2.e-5, 5.e-7], [2.89e-4, 1.e-5, 5.e-7]]
    >>> momentum = 450e9
    >>> general_parameters = GeneralParameters(n_turns, C, eta, momentum, 'proton')

    """
    
    def __init__(self, n_turns, ring_length, alpha, momentum, particle_type, 
                 user_mass = None, user_charge = None, n_sections = 1): 
        
        self.n_turns = int(n_turns) 
        self.n_sections = int(n_sections)
        
        # Ring length and checks
        self.ring_length = np.array(ring_length, ndmin = 1, dtype = float)
        self.ring_circumference = np.sum(self.ring_length)
        self.ring_radius = self.ring_circumference/(2*np.pi)         
        if self.n_sections != len(self.ring_length): 
            raise RuntimeError('ERROR: Number of sections and ring length'+
                               ' size do not match!')    
        
        # Momentum compaction, checks, and derived slippage factors
        self.alpha = np.array(alpha, ndmin = 2, dtype = float) 
        self.alpha_order = int(self.alpha.shape[1])
        if self.alpha_order > 3:
            warnings.filterwarnings("once")
            warnings.warn('WARNING: Momentum compaction factor is implemented'+
                          ' only up to 2nd order. Higher orders are ignored.')
            self.alpha_order = 3         
        if self.n_sections != self.alpha.shape[0]:
            raise RuntimeError('ERROR: Number of sections and the momentum'+
                               ' compaction factor size do not match!')    
                
        # Synchronous momentum and checks
        if type(momentum)==tuple:
            self.momentum = np.array(momentum[1], ndmin = 2)
            self.cumulative_times = momentum[0]
        else:
            self.momentum = np.array(momentum, ndmin = 2)
        if self.n_sections != self.momentum.shape[0]:
            raise RuntimeError('ERROR: Number of sections and momentum data'+
                               ' do not match!')           
        if self.n_sections > 1:
            if self.momentum.shape[1] == 1:
                self.momentum = self.momentum*np.ones(self.n_turns + 1)
        else:
            if self.momentum.size == 1:
                self.momentum = self.momentum*np.ones(self.n_turns + 1)
        if not self.momentum.shape[1] == self.n_turns + 1:
                raise RuntimeError('ERROR: The momentum program does not'+ 
                                   'match the proper length (n_turns+1)')
         
        # Particle type, checks, and derived mass and charge
        self.particle_type = str(particle_type)        
        if self.particle_type is 'proton':
            self.mass =  float(m_p*c**2/e) 
            self.charge = float(1) 
        elif self.particle_type is 'electron':
            self.mass =  float(m_e*c**2/e)
            self.charge = float(-1)
        elif self.particle_type is 'user_input':
            if user_mass > 0. and user_charge > 0.:
                self.mass = float(user_mass)
                self.charge = float(user_charge)
            else:
                raise RuntimeError('ERROR: Particle mass and/or charge not'+
                                   ' recognized!')
        else:
            raise RuntimeError('ERROR: Particle type not recognized!')
        
        # Derived from momentum
        self.beta = np.sqrt(1/(1 + (self.mass/self.momentum)**2))
        self.gamma = np.sqrt(1 + (self.momentum/self.mass)**2) 
        self.energy = np.sqrt(self.momentum**2 + self.mass**2)
        self.kin_energy = np.sqrt(self.momentum**2 + self.mass**2) - self.mass
        
        # N.B. self.cycle_time in the else statement should start always with 0
        if type(momentum)==tuple:
            self.cycle_time = self.cumulative_times
            self.t_rev = np.append(np.diff(self.cycle_time),
                                   self.ring_circumference/
                                   (self.beta[0][-1]*c))
        else:    
            self.t_rev = np.dot(self.ring_length, 1/(self.beta*c))
            self.cycle_time = np.cumsum(self.t_rev)
        self.f_rev = 1/self.t_rev
        self.omega_rev = 2*np.pi*self.f_rev

        # Slippage factor derived from alpha, beta, gamma
        self.eta_0 = float(0)
        self.eta_1 = float(0)
        self.eta_2 = float(0)
        self.eta_generation()
        
                
                
    def add_species(self, particle_type_2, user_mass_2 = None, 
                    user_charge_2 = None):
        """ Function to declare an optional second particle type
    
        Parameters
        ----------
        particle_type_2 : string
            secondary particle type that is not a reference for the momentum; 
            Recognized types are 'proton' and 'electron'. 
            Use 'user_input' to input mass and charge manually.    

        Attributes
        ----------
        mass2 : float
            primary particle mass [eV].
        charge2 : float
            primary particle charge [e].
        """
        
        self.particle_type_2 = str(particle_type_2)
        if self.particle_type_2 is 'proton':
            self.mass2 =  float(m_p*c**2/e)
            self.charge2 = float(1)
        elif self.particle_type_2 is 'electron':
            self.mass2 =  float(m_e*c**2/e)
            self.charge2 = float(-1)
        elif self.particle_type_2 is 'user_input':
            if user_mass_2 > 0. and user_charge_2 > 0.:
                self.mass2 = float(user_mass_2)
                self.charge2 = float(user_charge_2)
            else:
                raise RuntimeError('ERROR: Particle mass and/or charge not'+
                                   ' recognized!')
        else:
            raise RuntimeError('ERROR: Second particle type not recognized!')

    
    def eta_generation(self):
        """ Function to generate the slippage factors (zeroth, first, and 
        second orders) from the momentum compaction and the relativistic beta 
        and gamma program through the cycle.
        
        References
        ----------
        .. [2] "Accelerator Physics," S. Y. Lee, World Scientific, 
                Third Edition, 2012.
        """
        
        for i in range(self.alpha_order):
            getattr(self, '_eta' + str(i))()

    
    def _eta0(self):
        """ Function to calculate the zeroth order slippage factor eta_0 """

        self.eta_0 = np.empty([self.n_sections, self.n_turns+1])        
        for i in range(0, self.n_sections):
            self.eta_0[i] = self.alpha[i,0] - self.gamma[i]**-2 
   
    
    def _eta1(self):
        """ Function to calculate the first order slippage factor eta_1 """
                
        self.eta_1 = np.empty([self.n_sections, self.n_turns+1])        
        for i in range(0, self.n_sections):
            self.eta_1[i] = 3*self.beta[i]**2/(2*self.gamma[i]**2) + \
                self.alpha[i,1] - self.alpha[i,0]*self.eta_0[i]
        
        
    def _eta2(self):
        """ Function to calculate the second order slippage factor eta_2 """
                
        self.eta_2 = np.empty([self.n_sections, self.n_turns+1])        
        for i in range(0, self.n_sections):
            self.eta_2[i] = - self.beta[i]**2*(5*self.beta[i]**2 - 1)/ \
                (2*self.gamma[i]**2) + self.alpha[i,2] - 2*self.alpha[i,0]* \
                self.alpha[i,1] + self.alpha[i,1]/self.gamma[i]**2 + \
                self.alpha[i,0]**2*self.eta_0[i] - 3*self.beta[i]**2* \
                self.alpha[i,0]/(2*self.gamma[i]**2)


    def parameters_at_time(self, cycle_time):
        """ Function to return various cycle parameters at a specific moment in
        time.
            
        Parameters
        ----------
        cycle_time : float array
            moments of time at which cycle parameters are to be calculated [s].  
              
        Attributes
        ----------
        parameters : dictionary
            contains 'momentum', 'beta', 'gamma', 'energy', 'kin_energy',
            'f_rev', 't_rev'. 'omega_rev', 'eta_0', and 'delta_E' interpolated
            to the moments contained in the 'cycle_time' array
        """

        parameters = {}
        parameters['momentum'] = np.interp(cycle_time, self.cumulative_times, 
                                           self.momentum[0])
        parameters['beta'] = np.interp(cycle_time, self.cumulative_times, 
                                       self.beta[0])
        parameters['gamma'] = np.interp(cycle_time, self.cumulative_times, 
                                        self.gamma[0])
        parameters['energy'] = np.interp(cycle_time, self.cumulative_times, 
                                         self.energy[0])
        parameters['kin_energy'] = np.interp(cycle_time, self.cumulative_times, 
                                             self.kin_energy[0])
        parameters['f_rev'] = np.interp(cycle_time, self.cumulative_times, 
                                        self.f_rev)
        parameters['t_rev'] = np.interp(cycle_time, self.cumulative_times, 
                                        self.t_rev)
        parameters['omega_rev'] = np.interp(cycle_time, self.cumulative_times, 
                                            self.omega_rev)
        parameters['eta_0'] = np.interp(cycle_time, self.cumulative_times, 
                                        self.eta_0[0])
        parameters['delta_E'] = np.interp(cycle_time, 
                                          self.cumulative_times[1:], 
                                          np.diff(self.energy[0]))

        return parameters



