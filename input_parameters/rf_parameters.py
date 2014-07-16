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
    definition in longitudinal_tracker.Ring_and_RF_section), and pre-processing 
    them in order to be used in the longitudinal_tracker.py module.
    It can be added to another RF_section_parameters object by the 
    Sum_RF_section_parameters object in order to concatenate all the parameters
    for one full ring.*
    '''
    
    def __init__(self, general_parameters, section_number, n_rf, 
                 harmonic, voltage, phi_offset):
        
        #: | *Counter to keep track of time step (used in momentum and voltage)*
        #: | *It is defined as a list in order to be passed by reference to other modules.*
        self.counter = [0]
        
        #: *Number of the RF section (from 1 to n) -- has to be unique*
        self.sno = section_number - 1
        
        #: | *Number of turns for the simulation*
        #: | *Counter for turns is:* :math:`n`
        self.n_turns = general_parameters.n_turns
        
        #: *Length of the section in [m]* :math:`: \quad L_i`
        self.section_length = general_parameters.ring_length[self.sno]
        self.length_ratio = self.section_length/general_parameters.ring_circumference
        
        #: | *Momentum program in [eV/c]* :math:`: \quad p_{j,n}`
        self.momentum = general_parameters.momentum[self.sno]
        
        
        #: *Momentum increment (acceleration/deceleration) between two turns,
        #: for one section in [eV/c]* :math:`: \quad \Delta p_{n\rightarrow n+1}`
        self.p_increment = np.diff(self.momentum)
        
        #: *Copy of the relativistic parameters*
        self.beta_r = general_parameters.beta_r[self.sno]
        
        self.beta_av = (self.beta_r[1:] + self.beta_r[0:-1])/2
        
        self.gamma_r = general_parameters.gamma_r[self.sno]
        self.energy = general_parameters.energy[self.sno]

        #: | *Slippage factors for the given RF section*
        self.alpha_order = len(general_parameters.alpha[self.sno])
        for i in xrange( self.alpha_order ):
            dummy = getattr(general_parameters, 'eta' + str(i))
            setattr(self, "eta_%s" %i, dummy[self.sno])     
        
        #: | *Number of RF systems in the section* :math:`: \quad n_{RF}`
        #: | *Counter for RF is:* :math:`j`
        self.n_rf = n_rf
        
        #: | *Harmonic number list* :math:`: \quad h_{j,n}`
        #: | *Voltage program list in [V]* :math:`: \quad V_{j,n}`
        #: | *Phase offset list in [rad]* :math:`: \quad \phi_{j,n}`
        #: | *Check consistency of input array lengths*
        
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
            if (not n_rf == len(harmonic) == 
                len(voltage) == len(phi_offset)):
                raise RuntimeError('The RF parameters to define \
                                    RF_section_parameters are not homogeneous \
                                    (n_rf is not matching the input)')
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


    def eta_tracking(self, delta):
        
        '''
        *The slippage factor is calculated as a function of the relative momentum
        (delta) of the beam. By definition, the slippage factor is:*
        
        .. math:: 
            \eta = \sum_{i}(\eta_i \, \delta^i)
    
        '''
        
        if self.alpha_order == 1:
            return self.eta_0[self.counter]
        else:
            eta = 0
            for i in xrange( self.alpha_order ):
                eta_i = getattr(self, 'eta_' + str(i))[self.counter]
                eta  += eta_i * (delta**i)
            return eta          


def calc_phi_s(RF_section_parameters, accelerating_systems = 'all'):
    """The synchronous phase calculated from the rate of momentum change.
    Below transition, for decelerating bucket: phi_s is in (-Pi/2,0)
    Below transition, for accelerating bucket: phi_s is in (0,Pi/2)
    Above transition, for accelerating bucket: phi_s is in (Pi/2,Pi)
    Above transition, for decelerating bucket: phi_s is in (Pi,3Pi/2)
    The synchronous phase is calculated at a certain moment.
    Uses beta, energy averaged over the turn."""

    
    eta0 = RF_section_parameters.eta_0
         
    if RF_section_parameters.n_rf == 1:
                     
        acceleration_ratio = RF_section_parameters.beta_av * RF_section_parameters.p_increment \
            / (RF_section_parameters.voltage[0,0:-1] + RF_section_parameters.voltage[0,1:]) * 2 
        
        acceleration_test = np.where((acceleration_ratio > -1) * (acceleration_ratio < 1) == False)[0]
                
        if acceleration_test.size > 0:
            raise RuntimeError('Acceleration is not possible (momentum increment is too big or voltage too low) at index ' + str(acceleration_test))
           
        phi_s = np.arcsin(acceleration_ratio)
        
        index = np.where((eta0[1:] + eta0[0:-1])/2 > 0)       
        phi_s[index] = np.pi - phi_s

        
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
         
 
    
            

            
            
    
