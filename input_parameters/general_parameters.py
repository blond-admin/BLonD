# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
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
from scipy.constants import c
from beams.beams import Proton
from input_parameters.preprocess import PreprocessRamp



class GeneralParameters(object):
    r""" Class containing the general properties of the synchrotron that are
    independent of the RF system or the beam.
    
    The index :math:`n` denotes time steps, :math:`k` ring segments/sections
    and :math:`i` momentum compaction orders.
    
    Parameters
    ----------
    n_turns : int
        Number of turns :math:`n` [1] to be simulated
    ring_length : float
        Length [m] of the n_sections ring segments of the synchrotron.
        Input as a list for multiple RF stations
    alpha : float (opt: float array)
        Momentum compaction factor :math:`\alpha_{k,i}` [1]; can be input as
        single float (only 0th order element) or float array (up to 2nd order
        elements). In case of several sections without higher orders, input:
        [[alpha_section_1], [alpha_section_2], etc.]. In case of several
        sections and higher order alphas, input: [alpha_array_section_1,
        alpha_array_section_2, etc.]
    synchronous_data : float (opt: float array/matrix)
        Design synchronous particle momentum (default) [eV] or kinetic or
        total energy [eV] on the design orbit. Input for each RF section
        :math:`p_{s,k,n}`. Can be input as a single constant float, or as a
        program of (n_turns + 1) turns. In case of several sections without
        acceleration, input: [[momentum_section_1], [momentum_section_2],
        etc.]. In case of several sections with acceleration, input:
        [momentum_program_section_1, momentum_program_section_2, etc.]. Can
        be input also as a tuple of time and momentum, see also
        'cycle_time' and 'PreprocessRamp'
    synchronous_data_type : str
        Choice of 'synchronous_data' type; can be 'momentum' (default),
        'total_energy' or 'kinetic_energy'
    Particle : class
        A Particle-based class defining the primary, synchronous particle (mass
        and charge) that is reference for the momentum/energy in the ring. 
        Default is Proton().
    n_sections : int
        Optional: number of RF sections [1] over the ring; default is 1  
    PreprocessRamp : class
        A PreprocessRamp-based class defining smoothing, interpolation, etc. 
        options for synchronous_data that comes as a tuple.
           
    Attributes
    ----------
    ring_circumference : float
        Circumference of the synchrotron. Sum of ring segment lengths,
        :math:`C = \sum_k L_k` [m]    
    ring_radius : float
        Radius of the synchrotron, :math:`R = C/(2 \pi)` [m]
    alpha_order : int
        Number of orders of the momentum compaction factor
    eta_0 : float matrix
        Zeroth order slippage factor :math:`\eta_{0,k,n} = \alpha_{0,k,n} - 
        \frac{1}{\gamma_{s,k,n}^2}` [1]
    eta_1 : float matrix
        First order slippage factor :math:`\eta_{1,k,n} = 
        \frac{3\beta_{s,k,n}^2}{2\gamma_{s,k,n}^2} + \alpha_{1,k,n} - 
        \alpha_{0,k,n}\eta_{0,k,n}` [1]
    eta_2 : float matrix
        Second order slippage factor :math:`\eta_{2,k,n} = 
        -\frac{\beta_{s,k,n}^2\left(5\beta_{s,k,n}^2-1\right)}
        {2\gamma_{s,k,n}^2} + \alpha_{2,k,n} - 2\alpha_{0,k,n}\alpha_{1,k,n} 
        + \frac{\alpha_{1,k,n}}{\gamma_{s,k,n}^2} + \alpha_{0,k}^2\eta_{0,k,n} 
        - \frac{3\beta_{s,k,n}^2\alpha_{0,k,n}}{2\gamma_{s,k,n}^2}` [1]
    mass : float
        Primary particle mass :math:`m` [eV]
    charge : float
        Primary particle charge :math:`q` [e]
    beta : float matrix
        Synchronous relativistic beta program for each segment of the
        ring :math:`\beta_{s,k}^n = \frac{1}{\sqrt{1 
        + \left(\frac{m}{p_{s,k,n}}\right)^2} }` [1]
    gamma : float matrix
        Synchronous relativistic gamma program for each segment of the ring
        :math:`\gamma_{s,k,n} = \sqrt{ 1 
        + \left(\frac{p_{s,k,n}}{m}\right)^2 }` [1] 
    energy : float matrix
        Synchronous total energy program for each segment of the ring
        :math:`E_{s,k,n} = \sqrt{ p_{s,k,n}^2 + m^2 }` [eV]
    kin_energy : float matrix
        Synchronous kinetic energy program for each segment of the ring
        :math:`E_{s,kin} = \sqrt{ p_{s,k,n}^2 + m^2 } - m` [eV]
    delta_E : float matrix
        Derivative of synchronous total energy w.r.t. time, for all sections,
        :math:`: \quad E_{s,k,n+1}- E_{s,k,n}` [eV]
    t_rev : float array
        Revolution period turn by turn.
        :math:`T_{0,n} = \frac{C}{\beta_{s,n} c}` [s]
    f_rev : float array
        Revolution frequency :math:`f_{0,n} = \frac{1}{T_{0,n}}` [Hz]
    omega_rev : float array
        Revolution angular frequency :math:`\omega_{0,n} = 2\pi f_{0,n}` [1/s]
    cycle_time : float array
        Cumulative cycle time, turn by turn, :math:`t_n = \sum_n T_{0,n}` [s]. 
        Possibility to extract cycle parameters at these moments using 
        'parameters_at_time'.

    Examples
    --------
    >>> # To declare a single-stationed synchrotron at constant energy:
    >>> # Particle type Proton
    >>> from input_parameters.general_parameters import GeneralParameters
    >>>
    >>> n_turns = 10
    >>> C = 26659
    >>> alpha = 3.21e-4
    >>> momentum = 450e9
    >>> general_parameters = GeneralParameters(n_turns, C, alpha, momentum) 
    >>>                                        
    >>>
    >>> # To declare a double-stationed synchrotron at constant energy and 
    >>> # higher-order momentum compaction factors; particle Electron:
    >>> from beams.beams import Electron
    >>> from input_parameters.general_parameters import GeneralParameters
    >>>
    >>> n_turns = 10
    >>> C = [13000, 13659]
    >>> alpha = [[3.21e-4, 2.e-5, 5.e-7], [2.89e-4, 1.e-5, 5.e-7]]
    >>> momentum = 450e9
    >>> general_parameters = GeneralParameters(n_turns, C, alpha, momentum, 
    >>>                                        Particle = Electron())
    
    """
    
    def __init__(self, n_turns, ring_length, alpha, synchronous_data, 
                 synchronous_data_type = 'momentum', Particle = Proton(), 
                 n_sections = 1, PreprocessRamp = PreprocessRamp()): 
        
        self.n_turns = int(n_turns) 
        self.n_sections = int(n_sections)
        
        # Ring length and checks
        self.ring_length = np.array(ring_length, ndmin = 1, dtype = float)
        self.ring_circumference = np.sum(self.ring_length)
        self.ring_radius = self.ring_circumference/(2*np.pi)         
        if self.n_sections != len(self.ring_length): 
            raise RuntimeError("ERROR in GeneralParameters: Number of"+
                " sections and ring length size do not match!")    
        
        # Momentum compaction, checks, and derived slippage factors
        self.alpha = np.array(alpha, ndmin = 2, dtype = float) 
        self.alpha_order = int(self.alpha.shape[1])
        if self.alpha_order > 3:
            warnings.filterwarnings("once")
            warnings.warn("WARNING in GeneralParameters: Momentum compaction"+
                " factor is implemented only up to 2nd order. Higher orders"+
                " are ignored.")
            self.alpha_order = 3         
        if self.n_sections != self.alpha.shape[0]:
            raise RuntimeError("ERROR in GeneralParameters: Number of"+
                " sections and size of momentum compaction do not match!")    

        # Primary particle mass and charge used for energy calculations
        self.mass = Particle.mass
        self.charge = Particle.charge
        
        # If tuple, separate time and synchronous data
        if type(synchronous_data)==tuple:
            self.cycle_time = synchronous_data[0]
            self.momentum = synchronous_data[1]
            synchronous_data = synchronous_data[1]
            if len(self.cycle_time) != len(self.momentum):
                raise RuntimeError("ERROR in GeneralParameters: synchronous"+
                    " data does not match the time data")
        # Convert synchronous data to momentum, if necessary
        self.momentum = PreprocessRamp.convert_data(synchronous_data, 
            Particle = Particle, synchronous_data_type = synchronous_data_type)

        # Synchronous momentum and checks
        if type(synchronous_data)==tuple:
            self.cycle_time, self.momentum = PreprocessRamp.preprocess(
                self.mass, self.ring_circumference, self.cycle_time, 
                self.momentum)
        else:
            self.momentum = np.array(self.momentum, ndmin = 2)
        if self.n_sections != self.momentum.shape[0]:
            raise RuntimeError("ERROR in GeneralParameters: Number of"+
                " sections and momentum data do not match!")           
        if self.n_sections > 1:
            if self.momentum.shape[1] == 1:
                self.momentum = self.momentum*np.ones(self.n_turns + 1)
        else:
            if self.momentum.size == 1:
                self.momentum = self.momentum*np.ones(self.n_turns + 1)
        if not self.momentum.shape[1] == self.n_turns + 1:
                raise RuntimeError("ERROR in GeneralParameters: The momentum"+
                    " program does not match the proper length (n_turns+1)")
         
        # Derived from momentum
        self.beta = np.sqrt(1/(1 + (self.mass/self.momentum)**2))
        self.gamma = np.sqrt(1 + (self.momentum/self.mass)**2) 
        self.energy = np.sqrt(self.momentum**2 + self.mass**2)
        self.kin_energy = np.sqrt(self.momentum**2 + self.mass**2) - self.mass
        self.delta_E = np.diff(self.energy, axis=1)
        self.t_rev = np.dot(self.ring_length, 1/(self.beta*c))
        self.cycle_time = np.cumsum(self.t_rev) # Always starts with zero
        self.f_rev = 1/self.t_rev
        self.omega_rev = 2*np.pi*self.f_rev

        # Slippage factor derived from alpha, beta, gamma
        self.eta_generation()
        
    
    def eta_generation(self):
        """ Function to generate the slippage factors (zeroth, first, and 
        second orders, see [1]_) from the momentum compaction and the 
        relativistic beta and gamma program through the cycle.
        
        References
        ----------
        .. [1] "Accelerator Physics," S. Y. Lee, World Scientific, 
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


    def parameters_at_time(self, cycle_moments):
        """ Function to return various cycle parameters at a specific moment in
        time. The cycle time is defined to start at zero in turn zero. 
            
        Parameters
        ----------
        cycle_moments : float array
            Moments of time at which cycle parameters are to be calculated [s].  
              
        Attributes
        ----------
        parameters : dictionary
            Contains 'momentum', 'beta', 'gamma', 'energy', 'kin_energy',
            'f_rev', 't_rev'. 'omega_rev', 'eta_0', and 'delta_E' interpolated
            to the moments contained in the 'cycle_moments' array
            
        """

        parameters = {}
        parameters['momentum'] = np.interp(cycle_moments, self.cycle_time, 
                                           self.momentum[0])
        parameters['beta'] = np.interp(cycle_moments, self.cycle_time, 
                                       self.beta[0])
        parameters['gamma'] = np.interp(cycle_moments, self.cycle_time, 
                                        self.gamma[0])
        parameters['energy'] = np.interp(cycle_moments, self.cycle_time, 
                                         self.energy[0])
        parameters['kin_energy'] = np.interp(cycle_moments, self.cycle_time, 
                                             self.kin_energy[0])
        parameters['f_rev'] = np.interp(cycle_moments, self.cycle_time, 
                                        self.f_rev)
        parameters['t_rev'] = np.interp(cycle_moments, self.cycle_time, 
                                        self.t_rev)
        parameters['omega_rev'] = np.interp(cycle_moments, self.cycle_time, 
                                            self.omega_rev)
        parameters['eta_0'] = np.interp(cycle_moments, self.cycle_time, 
                                        self.eta_0[0])
        parameters['delta_E'] = np.interp(cycle_moments, 
                                          self.cycle_time[1:], 
                                          np.diff(self.energy[0]))

        return parameters



