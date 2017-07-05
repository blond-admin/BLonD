# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module gathering and processing all RF parameters used in the simulation.**

:Authors: **Alexandre Lasheen**, **Danilo Quartullo**, **Helga Timko**
'''

from __future__ import division, print_function
from builtins import str, range, object
import numpy as np
from scipy.constants import c
from scipy.integrate import cumtrapz
from beams.beams import Proton



def input_check(input_value, expected_length):
    r"""Function to check the length of the input. The input can be a float, 
    int, np.ndarray and list. If len(input_value) == 1, transform it to a 
    constant array. If len(input_value) != expected_length and != 1, raise an 
    error"""
    
    if isinstance(input_value, float):
        return input_value * np.ones(expected_length)
    elif isinstance(input_value, int):
        return input_value * np.ones(expected_length)
    elif isinstance(input_value, np.ndarray) and input_value.size == 1:
        return input_value * np.ones(expected_length)
    elif isinstance(input_value, list) and len(input_value) == 1:
        return input_value[0] * np.ones(expected_length)
    elif len(input_value) == expected_length:
        return np.array(input_value)
    else:
        raise RuntimeError("ERROR: " + str(input_value) + " does not match " 
                           + str(expected_length))
    
    

class RFSectionParameters(object):
    r""" Class containing all the RF parameters for all the RF systems in one 
    ring segment or RF section.

    **How to use RF programs:**

    * For 1 RF system and constant values of V, h, or phi, input a single value
    * For 1 RF system and varying values of V, h, or phi, input an array of 
      n_turns values
    * For several RF systems and constant values of V, h, or phi, input lists 
      of single values 
    * For several RF systems and varying values of V, h, or phi, input lists 
      of arrays of n_turns values
    * For pre-processing, pass a list of times-voltages, times-harmonics, 
      and/or times-phases for **each** RF system and define the 
      PreprocessRFParams class, i.e. [time_1, ..., time_n, data_1, ..., data_n]  
      
    Optional: RF frequency other than the design frequency. In this case, need
    to use a beam phase loop for correct RF phase!
    
    The index :math:`n` denotes time steps, :math:`l` the index of the RF 
    systems in the section.
    
    **N.B. for negative eta the RF phase has to be shifted by Pi w.r.t the time
    reference.**
    
    Parameters
    ----------
    GeneralParameters : class
        A GeneralParameters type class
    n_turns : int 
        Inherited from
        :py:attr:`input_parameters.general_parameters.GeneralParameters`
    ring_circumference : float
        Inherited from
        :py:attr:`input_parameters.general_parameters.GeneralParameters`
    section_length : float
        Length :math:`L_k` of the RF section; inherited from 
        :py:attr:`input_parameters.general_parameters.GeneralParameters`
    length_ratio : float
        Fractional RF section length :math:`L_k/C`
    t_rev : float array
        Inherited from
        :py:attr:`input_parameters.general_parameters.GeneralParameters`
    momentum : float array
        Momentum program of the present RF section; inherited from
        :py:attr:`input_parameters.general_parameters.GeneralParameters`
    beta : float array
        Relativistic beta of the present RF section; inherited from
        :py:attr:`input_parameters.general_parameters.GeneralParameters`
    gamma : float array
        Relativistic gamma of the present RF section; inherited from
        :py:attr:`input_parameters.general_parameters.GeneralParameters`
    energy : float array
        Total energy of the present RF section; inherited from
        :py:attr:`input_parameters.general_parameters.GeneralParameters`
    delta_E : float array
        Time derivative of total energy of the present section; inherited from
        :py:attr:`input_parameters.general_parameters.GeneralParameters`
    alpha_order : int
        Inherited from
        :py:attr:`input_parameters.general_parameters.GeneralParameters`
    eta_0 : float array
        Zeroth order slippage factor of the present section; inherited from
        :py:attr:`input_parameters.general_parameters.GeneralParameters`
    eta_1 : float array
        First order slippage factor of the present section; inherited from
        :py:attr:`input_parameters.general_parameters.GeneralParameters`
    eta_2 : float array
        Second order slippage factor of the present section; inherited from
        :py:attr:`input_parameters.general_parameters.GeneralParameters`
    sign_eta_0 : float array
        Sign of the eta_0 array
    n_rf : int
        Number of harmonic RF systems in the section :math:`l`
    harmonic : float (opt: float array/matrix)
        Harmonic number of the RF system, :math:`h_{l,n}` [1]. For input 
        options, see above
    voltage : float (opt: float array/matrix)
        RF cavity voltage as seen by the beam, :math:`V_{l,n}` [V]. For input 
        options, see above
    phi_rf_d : float (opt: float array/matrix)
        Programmed/designed RF cavity phase, 
        :math:`\phi_{d,l,n}` [rad]. For input options, see above
    phi_noise : float (opt: float array/matrix)
        Optional, programmed RF cavity phase noise, :math:`\phi_{N,l,n}` [rad].
        Added to all RF systems in the station. For input options, see above
    omega_rf : float (opt: float array/matrix)
        Actual RF revolution frequency, :math:`\omega_{rf,l,n}` [rad]. 
        For input options, see above. The default value is the design frequency
        :math:`\omega_{rf,l,n} = \omega_{d,l,n}`
    Particle : class
        A Particle type class defining the primary, synchronous particle (mass
        and charge) that is used to calculate phi_s and Qs; default is Proton()
    PreprocessRFParams : class
        A PreprocessRFParams-based class defining smoothing, interpolation, 
        etc. options for harmonic, voltage, and/or phi_rf_d programme to be 
        interpolated to a turn-by-turn programme
        
    Attributes
    ----------
    counter : int
        Counter of the current simulation time step; defined as a list in 
        order to be passed by reference
    section_index : int                    
        Unique index :math:`k` of the RF section the present class is defined 
        for. Input in the range 1..n_sections (see 
        :py:class:`input_parameters.general_parameters.GeneralParameters`). 
        Inside the code, indices 0..n_sections-1 are used.
    phi_rf : float matrix
        Actual RF cavity phase of each harmonic system, 
        :math:`\phi_{rf,l,n}` [rad]. Initially the same as the designed phase.
    dphi_rf : float matrix
        Accumulated RF phase error of each harmonic system 
        :math:`\Delta \phi_{rf,l,n}` [rad]
    omega_rf_d : float matrix
        Design RF frequency of the RF systems in the station 
        :math:`\omega_{d,l,n} = \frac{h_{l,n} \beta_{l,n} c}{R_{s,n}}` [Hz]
    t_rf : float matrix
        RF period :math:`\frac{2 \pi}{\omega_{rf,l,n}}` [s]
    phi_s : float array
        Synchronous phase for this section, calculated in 
        :py:func:`input_parameters.rf_parameters.calculate_phi_s`
    Q_s : float array
        Synchrotron tune for this section, calculated in 
        :py:func:`input_parameters.rf_parameters.calculate_Q_s`
    omega_s0 : float array
        Central synchronous angular frequency corresponding to Q_s (single
        harmonic, no intensity effects) 
        :math:`\omega_{s,0} = Q_s \omega_{\text{rev}}` [1/s], where 
        :math:`\omega_{\text{rev}}` is defined in 
        :py:class:`input_parameters.general_parameters.GeneralParameters`)  


    Examples
    --------
    >>> # To declare a double-harmonic RF system for protons:
    >>>
    >>> n_turns = 10
    >>> C = 26659
    >>> alpha = 3.21e-4
    >>> momentum = 450e9
    >>> general_parameters = GeneralParameters(n_turns, C, alpha, momentum) 
    >>> rf_parameters = RFSectionParameters(general_parameters, 2, [35640, 
    >>>     71280], [6e6, 6e5], [0, 0])

        
    """
    
    def __init__(self, GeneralParameters, n_rf, harmonic, voltage, phi_rf_d, 
                 phi_noise = None, omega_rf = None, section_index = 1, 
                 accelerating_systems = 'as_single', Particle = Proton(), 
                 PreprocessRFParams = None):
        
        # Imported from GeneralParameters
        self.n_turns = GeneralParameters.n_turns
        self.ring_circumference = GeneralParameters.ring_circumference
        self.section_length = GeneralParameters.ring_length[self.section_index]
        self.length_ratio = float(self.section_length/self.ring_circumference)
        self.t_rev = GeneralParameters.t_rev
        self.momentum = GeneralParameters.momentum[self.section_index]
        self.beta = GeneralParameters.beta[self.section_index]
        self.gamma = GeneralParameters.gamma[self.section_index]
        self.energy = GeneralParameters.energy[self.section_index]
        self.delta_E = GeneralParameters.delta_E[self.section_index]
        self.alpha_order = GeneralParameters.alpha_order
        self.eta_0 = 0
        self.eta_1 = 0
        self.eta_2 = 0
        for i in range( self.alpha_order ):
            dummy = getattr(GeneralParameters, 'eta_' + str(i))
            setattr(self, "eta_%s" %i, dummy[self.section_index])
        self.sign_eta_0 = np.sign(self.eta_0)   

        # Different indices
        self.counter = [int(0)]
        self.section_index = int(section_index - 1)
        if self.section_index < 0 \
            or self.section_index > GeneralParameters.n_sections - 1:
            raise RuntimeError("ERROR in RFSectionParameters: section_index"+
                " out of allowed range!")    
        self.n_rf = n_rf
 
        # Process RF programs
        self.harmonic = harmonic
        self.voltage = voltage
        self.phi_rf_d = phi_rf_d
        self.omega_rf = omega_rf
        rf_params = ['harmonic', 'voltage', 'phi_rf_d', 'omega_rf']
        for rf_param in rf_params:
            # Option 1: pre-process
            if PreprocessRFParams:
                if PreprocessRFParams.__getattribute__(rf_param) == True:
                    if len(self.__getattribute__(rf_param)) == 2*self.n_rf:
                        # Overwrite with interpolated values
                        self.__setattribute__(rf_param, 
                            PreprocessRFParams.preprocess(GeneralParameters, 
                            self.__getattribute__(rf_param)[0:self.n_rf], #time
                            self.__getattribute__(rf_param)[self.n_rf:])) #data
                    else:
                        raise RuntimeError("ERROR in RFSectionParameters:"+
                            " harmonic to be pre-processed should have length"+
                            " of 2*n_rf!")
                else:
                    input_check(self.__getattribute__(rf_param))
        if phi_noise:
            input_check(self.phi_noise)
            
# BEGIN MOVE TO INPUT CHECK... ************************************************               
        # Option 2: cast the input into appropriate shape: the input is 
        # analyzed and structured in order to have lists whose length is 
        # matching the number of RF systems in the section.      
        if self.n_rf == 1:
            self.harmonic = [harmonic] 
            self.voltage = [voltage]
            self.phi_rf_d = [phi_rf_d]
            if phi_noise != None:
                self.phi_noise = [phi_noise]
            if omega_rf != None:
                self.omega_rf = [omega_rf]                 
        else:
            self.harmonic = harmonic
            self.voltage = voltage 
            self.phi_rf_d = phi_rf_d
            if phi_noise != None:
                self.phi_noise = phi_noise
            if omega_rf != None:
                self.omega_rf = omega_rf
        # Run input_check() on all RF systems
        for i in range(self.n_rf):
            self.harmonic[i] = input_check(self.harmonic[i], self.n_turns+1)
            self.voltage[i] = input_check(self.voltage[i], self.n_turns+1)
            self.phi_rf_d[i] = input_check(self.phi_rf_d[i], 
                                             self.n_turns+1)
            if phi_noise != None:
                self.phi_noise[i] = input_check(self.phi_noise[i], 
                                                self.n_turns+1) 
            if omega_rf != None:
                self.omega_rf[i] = input_check(self.omega_rf[i], 
                                               self.n_turns+1) 
        # Convert to 2D numpy matrix
        self.harmonic = np.array(self.harmonic, ndmin =2)
        self.voltage = np.array(self.voltage, ndmin =2)
        self.phi_rf_d = np.array(self.phi_rf_d, ndmin =2)
        if phi_noise != None:
            self.phi_noise = np.array(self.phi_noise, ndmin =2) 
        if omega_rf != None:
            self.omega_rf = np.array(self.omega_rf, ndmin =2) 
# END MOVE TO INPUT CHECK... **************************************************               
        
        # RF (feedback) properties
        self.phi_rf = np.array(self.phi_rf_d) 
        self.dphi_rf = np.zeros(self.n_rf)
        self.omega_rf_d = 2.*np.pi*self.beta*c*self.harmonic/ \
                          (self.ring_circumference)
        if omega_rf == None:
            self.omega_rf = np.array(self.omega_rf_d)                  
        self.t_rf = 2*np.pi / self.omega_rf[0]

        # From helper functions
        self.phi_s = calculate_phi_s(self, Particle, accelerating_systems)
        self.Q_s = calculate_Q_s(self, Particle)   
        self.omega_s0 = self.Q_s*GeneralParameters.omega_rev

       
    def eta_tracking(self, beam, counter, dE):
        r"""Function to calculate the slippage factor as a function of the 
        energy offset :math:`\Delta E` of the particle. The slippage factor 
        of the :math:`i`th order is :math:`\eta(\delta) = \sum_{i}(\eta_i \, 
        \delta^i) = \sum_{i} \left(\eta_i \, \left[ \frac{\Delta E}
        {\beta_s^2 E_s} \right]^i \right)`
    
        """
        
        if self.alpha_order == 1:
            return self.eta_0[counter]
        else:
            eta = 0
            delta = dE/(beam.beta**2 * beam.energy)
            for i in range( self.alpha_order ):
                eta_i = getattr(self, 'eta_' + str(i))[counter]
                eta  += eta_i * (delta**i)
            return eta  



def calculate_Q_s(RFSectionParameters, Particle = Proton()):
    r""" Function calculating the turn-by-turn synchrotron tune for 
    single-harmonic RF, without intensity effects. 
        
    Parameters
    ----------
    RFSectionParameters : class
        An RFSectionParameters type class.  
    Particle : class
        A Particle type class; default is Proton().
          
    Returns
    -------
    float
        Synchrotron tune.
        
    """

    return np.sqrt( RFSectionParameters.harmonic[0]*Particle.charge* \
        RFSectionParameters.voltage[0]*np.abs(RFSectionParameters.eta_0* \
        np.cos(RFSectionParameters.phi_s)) / \
        (2*np.pi*RFSectionParameters.beta**2*RFSectionParameters.energy) )



def calculate_phi_s(RFSectionParameters, Particle = Proton(),
                    accelerating_systems = 'as_single'):
    r"""Function calculating the turn-by-turn synchronous phase according to 
    the parameters in the RFSectionParameters object. The phase is expressed in
    the lowest RF harmonic and with respect to the RF bucket (see the equations
    of motion defined for BLonD). The returned value is given in the range [0,
    2*Pi]. Below transition, the RF wave is shifted by Pi w.r.t. the time 
    reference.
    
    The accelerating_systems option can be set to
    * 'as_single' (default): the synchronous phase is calculated analytically 
    taking into account the phase program (RFSectionParameters.phi_offset).
    * 'all': the synchronous phase is calculated numerically by finding the 
    minimum of the potential well; no intensity effects included. In case of 
    several minima, the deepest is taken. WARNING: in case of RF harmonics with
    comparable voltages, this may lead to inconsistent values of phi_s.
    * 'first': not yet implemented. Its purpose should be to adjust the 
    RFSectionParameters.phi_offset of the higher harmonics so that only the 
    main harmonic is accelerating.
    
    Parameters
    ----------
    RFSectionParameters : class
        An RFSectionParameters type class.  
    Particle : class
        A Particle type class; default is Proton().
    accelerating_systems : str
        Choice of accelerating systems; or options, see list above.
          
    Returns
    -------
    float
        Synchronous phase.
            
    """
    
    eta0 = RFSectionParameters.eta_0
    
    if accelerating_systems == 'as_single':
        
        denergy = np.append(RFSectionParameters.delta_E, 
                            RFSectionParameters.delta_E[-1])             
        acceleration_ratio = denergy/(Particle.charge*
                                      RFSectionParameters.voltage[0,:])
        acceleration_test = np.where((acceleration_ratio > -1)*
                                     (acceleration_ratio < 1) == False)[0]
        
        # Validity check on acceleration_ratio        
        if acceleration_test.size > 0:
            print("WARNING in calculate_phi_s(): acceleration is not possible"+
                  " (momentum increment is too big or voltage too low) at"+
                  " index " + str(acceleration_test))
        
        phi_s = np.arcsin(acceleration_ratio)

        # Identify where eta swaps sign
        eta0_middle_points = (eta0[1:] + eta0[:-1])/2
        eta0_middle_points = np.append(eta0_middle_points, eta0[-1])             
        index = np.where(eta0_middle_points > 0)[0]
        index_below = np.where(eta0_middle_points < 0)[0]
        
        # Project phi_s in correct range 
        phi_s[index] = (np.pi - phi_s[index]) % (2*np.pi)
        phi_s[index_below] = (np.pi + phi_s[index_below]) % (2*np.pi)
        
        return phi_s 
     
    elif accelerating_systems == 'all':
            
        phi_s = np.zeros(len(RFSectionParameters.voltage[0,1:]))
        
        for indexTurn in range(len(RFSectionParameters.delta_E)):
            
            totalRF = 0
            if np.sign(eta0[indexTurn]) > 0:
                phase_array = np.linspace(
                    -RFSectionParameters.phi_RF[0,indexTurn+1], 
                    -RFSectionParameters.phi_RF[0,indexTurn+1] + 2*np.pi, 1000) 
            else:
                phase_array = np.linspace(
                    -RFSectionParameters.phi_RF[0,indexTurn+1] - np.pi, 
                    -RFSectionParameters.phi_RF[0,indexTurn+1] + np.pi, 1000) 

            for indexRF in range(len(RFSectionParameters.voltage[:,indexTurn+1])):
                totalRF += RFSectionParameters.voltage[indexRF,indexTurn+1]* \
                    np.sin(RFSectionParameters.harmonic[indexRF,indexTurn+1]/ \
                    np.min(RFSectionParameters.harmonic[:,indexTurn+1])* \
                    phase_array + \
                    RFSectionParameters.phi_RF[indexRF,indexTurn+1]) 
                
            potential_well = - cumtrapz( np.sign(eta0[indexTurn])*(totalRF - 
                RFSectionParameters.delta_E[indexTurn]/abs(Particle.charge)), 
                dx=phase_array[1]-phase_array[0], initial=0 )

            phi_s[indexTurn] = np.mean(phase_array[potential_well == 
                                                   np.min(potential_well)])

        phi_s = np.insert(phi_s, 0, phi_s[0]) + RFSectionParameters.phi_RF[0,:]
        phi_s[eta0 < 0] += np.pi
        phi_s = phi_s % (2*np.pi)
        
        return phi_s
    
    elif accelerating_systems == 'first':
       
        print("WARNING in calculate_phi_s(): accelerating_systems 'first'"+
              " not yet implemented")
        pass
    else:
        raise RuntimeError("ERROR in calculate_phi_s(): unrecognised"+
                           " accelerating_systems option")




