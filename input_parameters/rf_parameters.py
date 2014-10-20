'''
**Module gathering and processing all the RF parameters to be given to the 
other modules.**

:Authors: **Alexandre Lasheen**, **Danilo Quartullo**
'''

from __future__ import division
import numpy as np


def input_check(input_value, expected_length):
    '''
    | *Function to check the length of the input*
    | *The input can be a float, int, np.ndarray and list*
    | *If len(input_value) == 1, transform it to a constant array*
    | *If len(input_value) != expected_length and != 1, raise an error*
    '''
    
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
        raise RuntimeError(str(input_value) + ' does not match ' + str(expected_length))
    
    

class RFSectionParameters(object):
    '''
    *Object gathering all the RF parameters for one section (see section
    definition in tracker.RingAndRFSection), and pre-processing 
    them in order to be used in the tracker.py module.*
    
    :How to use RF programs:

      - For 1 RF system and constant values of V, h or phi, just input the single value
      - For 1 RF system and varying values of V, h or phi, input an array of n_turns values
      - For several RF systems and constant values of V, h or phi, input lists of single values 
      - For several RF systems and varying values of V, h or phi, input lists of arrays of n_turns values
    '''
    
    def __init__(self, GeneralParameters, n_rf, 
                 harmonic, voltage, phi_offset, section_index = 1):
        
        #: | *Counter to keep track of time step (used in momentum and voltage)*
        #: | *Definined as a list in order to be passed by reference.*
        self.counter = [0]
                
        #: | *Index of the RF section -- has to be unique*
        #: | *Counter for RF section is:* :math:`k`
        #: | *In the user input, the section_index goes from 1 to k*
        #: | *This index is then corrected in the constructor in order to go from 0 to k-1 (as Python indexing starts from 0)*
        self.section_index = section_index - 1
        
        #: | *Number of turns for the simulation*
        #: | *Counter for turns is:* :math:`n`
        self.n_turns = GeneralParameters.n_turns
        
        #: *Copy of the ring circumference (from GeneralParameters)*
        self.ring_cirumference = GeneralParameters.ring_circumference
        
        #: *Length of the section in [m]* :math:`: \quad L_k`
        self.section_length = GeneralParameters.ring_length[self.section_index]
        
        #: *Length ratio of the section wrt the circumference*
        self.length_ratio = self.section_length / GeneralParameters.ring_circumference
        
        #: *Momentum program of the section in [eV/c]* :math:`: \quad p_{k,n}`
        self.momentum = GeneralParameters.momentum[self.section_index]
        
        #: *Momentum increment (acceleration/deceleration) between two turns,
        #: for one section in [eV/c]* :math:`: \quad \Delta p_{n\rightarrow n+1}`
        self.p_increment = np.diff(self.momentum)
        
        #: *Copy of the relativistic beta for the section (from 
        #: GeneralParameters)* :math:`: \quad \beta_{k,n}`
        self.beta_r = GeneralParameters.beta_r[self.section_index]
        
        #: *Copy of the relativistic gamma for the section (from 
        #: GeneralParameters)* :math:`: \quad \gamma_{k,n}`
        self.gamma_r = GeneralParameters.gamma_r[self.section_index]
        
        #: *Copy of the relativistic energy for the section (from 
        #: GeneralParameters)* :math:`: \quad E_{k,n}`
        self.energy = GeneralParameters.energy[self.section_index]
        
        #: *Slippage factor (order 0) for the given RF section*
        self.eta_0 = 0
        #: *Slippage factor (order 1) for the given RF section*
        self.eta_1 = 0
        #: *Slippage factor (order 2) for the given RF section*
        self.eta_2 = 0
        
        #: *Copy of the order of alpha for the section (from GeneralParameters)*
        self.alpha_order = GeneralParameters.alpha_order
        for i in xrange( self.alpha_order ):
            dummy = getattr(GeneralParameters, 'eta' + str(i))
            setattr(self, "eta_%s" %i, dummy[self.section_index])
            
        
        #: | *Number of RF systems in the section* :math:`: \quad n_{RF}`
        #: | *Counter for RF is:* :math:`j`
        self.n_rf = n_rf
        
        #: | *Harmonic number list* :math:`: \quad h_{j,k,n}`
        #: | *See note above on how to input RF programs.*
        self.harmonic = 0
        
        #: | *Voltage program list in [V]* :math:`: \quad V_{j,k,n}`
        #: | *See note above on how to input RF programs.*
        self.voltage = 0
        
        #: | *Phase offset list in [rad]* :math:`: \quad \phi_{j,k,n}`
        #: | *See note above on how to input RF programs.*
        self.phi_offset = 0
                
        ### Pre-processing the inputs
        # The input is analysed and structured in order to have lists, which
        # length are matching the number of RF systems considered in this
        # section.
        # For one rf system, single values for h, V_rf, and phi will assume
        # that these values will remain constant for all the simulation.
        # These can be inputed directly as arrays in order to have programs
        # (the length of the arrays will then be checked)
        if n_rf == 1:
            self.harmonic = [harmonic] 
            self.voltage = [voltage]
            self.phi_offset = [phi_offset] 
        else:
            self.harmonic = harmonic
            self.voltage = voltage 
            self.phi_offset = phi_offset
        
        for i in range(self.n_rf):
            self.harmonic[i] = input_check(self.harmonic[i], self.n_turns+1)
            self.voltage[i] = input_check(self.voltage[i], self.n_turns+1)
            self.phi_offset[i] = input_check(self.phi_offset[i], self.n_turns+1)
        
        # Convert to numpy matrix
        self.harmonic = np.array(self.harmonic, ndmin =2)
        self.voltage = np.array(self.voltage, ndmin =2)
        self.phi_offset = np.array(self.phi_offset, ndmin =2)
            
        #: *Synchronous phase for this section, calculated from the gamma
        #: transition and the momentum program.*
        self.phi_s = calc_phi_s(self)   
    
    
    def eta_tracking(self, counter, delta):
        '''
        *The slippage factor is calculated as a function of the relative momentum
        (delta) of the beam. By definition, the slippage factor is:*
        
        .. math:: 
            \eta = \sum_{i}(\eta_i \, \delta^i)
    
        '''
        
        if self.alpha_order == 1:
            return self.eta_0[counter]
        else:
            eta = 0
            for i in xrange( self.alpha_order ):
                eta_i = getattr(self, 'eta_' + str(i))[counter]
                eta  += eta_i * (delta**i)
            return eta  


def calc_phi_s(RFSectionParameters, accelerating_systems = 'all'):
    '''
    | *The synchronous phase calculated from the rate of momentum change.*
    | *Below transition, for decelerating bucket: phi_s is in (-Pi/2,0)*
    | *Below transition, for accelerating bucket: phi_s is in (0,Pi/2)*
    | *Above transition, for accelerating bucket: phi_s is in (Pi/2,Pi)*
    | *Above transition, for decelerating bucket: phi_s is in (Pi,3Pi/2)*
    | *The synchronous phase is calculated at a certain moment.*
    | *Uses beta, energy averaged over the turn.*
    '''
    
    eta0 = RFSectionParameters.eta_0
         
    if RFSectionParameters.n_rf == 1:
                     
        acceleration_ratio = RFSectionParameters.beta_r[1:] * RFSectionParameters.p_increment \
            / RFSectionParameters.voltage[0,1:] 
        
        acceleration_test = np.where((acceleration_ratio > -1) * (acceleration_ratio < 1) == False)[0]
                
        if acceleration_test.size > 0:
            raise RuntimeError('Acceleration is not possible (momentum increment is too big or voltage too low) at index ' + str(acceleration_test))
        
        # For the initial phi_s, add the first value a second time, also in index   
        phi_s = np.arcsin( np.concatenate((np.array([acceleration_ratio[0]]), 
                                           acceleration_ratio)) )
        
        index = np.where((eta0[1:] + eta0[0:-1])/2 > 0)[0] + 1
        if len(index) > 0:
            if index[0] == 1:
                index = np.concatenate((np.array([0]), index))
            phi_s[index] = np.pi - phi_s[index]

        return phi_s 
     
    else:
        '''
        To be implemented
        '''
        if accelerating_systems == 'all':
            '''
            In this case, all the rf_systems are accelerating, phi_s is calculated accordingly
            with respect to the fundamental frequency
            '''
            pass
        elif accelerating_systems == 'first':
            '''
            Only the first rf_system is accelerating, so we have to correct the phi_offset of the
            other rf_systems in order that the p_increment is only caused by the first RF
            '''
            pass
        else:
            raise RuntimeError('Did not recognize the option accelerating_systems in calc_phi_s function')
        
        # This part only works when you have no acceleration and several RF systems
        if eta0[0] > 0:
            return np.pi*np.ones(RFSectionParameters.n_turns)
        elif eta0[0] < 0:
            return 0*np.ones(RFSectionParameters.n_turns)
         
 
    
            

            
            
    
