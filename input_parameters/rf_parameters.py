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
    
    def __init__(self, general_parameters, section_number, n_rf_systems, 
                 harmonic_number_list, voltage_program_list, phi_offset_list):
        
        #: *Number of the RF section (from 1 to n) -- has to be unique*
        self.sno = section_number - 1
        
        #: | *Number of turns for the simulation*
        #: | *Counter for turns is:* :math:`n`
        self.n_turns = general_parameters.n_turns
        
        #: *Length of the section in [m]* :math:`: \quad L_i`
        self.section_length = general_parameters.ring_length_list[self.sno]
        self.length_ratio = self.section_length/general_parameters.ring_circumference
        
        #: | *Momentum program in [eV/c]* :math:`: \quad p_{j,n}`
        self.momentum_program = general_parameters.momentum_program[self.sno]
        
        #: *Momentum increment (acceleration/deceleration) between two turns,
        #: for one section in [eV/c]* :math:`: \quad \Delta p_{n\rightarrow n+1}`
        self.p_increment = np.diff(self.momentum_program)
        
        #: *Copy of the relativistic parameters*
        self.beta_r = general_parameters.beta_rel_program[self.sno]
        self.beta_av = (self.beta_r[1:] + self.beta_r[0:-1])/2
        self.gamma_r = general_parameters.gamma_rel_program[self.sno]
        self.energy = general_parameters.energy_program[self.sno]

        #: | *Slippage factors for the given RF section*
        self.alpha_order = len(general_parameters.alpha[self.sno])
        for i in xrange( self.alpha_order ):
            dummy = getattr(general_parameters, 'eta' + str(i))
            setattr(self, "eta_%s" %i, dummy[self.sno])     
        
        #: | *Number of RF systems in the section* :math:`: \quad n_{RF}`
        #: | *Counter for RF is:* :math:`j`
        self.n_rf_systems = n_rf_systems
        
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
        if n_rf_systems == 1:
            self.harmonic_number_list = [harmonic_number_list] 
            self.voltage_program_list = [voltage_program_list] 
            self.phi_offset_list = [phi_offset_list] 
        else:
            if (not n_rf_systems == len(harmonic_number_list) == 
                len(voltage_program_list) == len(phi_offset_list)):
                raise RuntimeError('The RF parameters to define \
                                    RF_section_parameters are not homogeneous \
                                    (n_rf_systems is not matching the input)')
            self.harmonic_number_list = harmonic_number_list
            self.voltage_program_list = voltage_program_list 
            self.phi_offset_list = phi_offset_list
        
        for i in range(self.n_rf_systems):
            self.harmonic_number_list[i] = input_check(self.harmonic_number_list[i], self.n_turns)
            self.voltage_program_list[i] = input_check(self.voltage_program_list[i], self.n_turns)
            self.phi_offset_list[i] = input_check(self.phi_offset_list[i], self.n_turns)
            

            
            

            
            
    
