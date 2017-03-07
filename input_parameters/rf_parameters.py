
# Copyright 2016 CERN. This software is distributed under the
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
    *Object gathering all the RF parameters for one section (sections defined in
    tracker.RingAndRFSection), and pre-processing them for later use.*
    
    :How to use RF programs:

      - For 1 RF system and constant values of V, h or phi, just input the single value
      - For 1 RF system and varying values of V, h or phi, input an array of n_turns values
      - For several RF systems and constant values of V, h or phi, input lists of single values 
      - For several RF systems and varying values of V, h or phi, input lists of arrays of n_turns values
      
    Optional: RF frequency other than the design frequency. In this case, need
    to use the Phase Loop for correct RF phase!
    '''
    
    def __init__(self, GeneralParameters, n_rf, harmonic, voltage, phi_offset, 
                 phi_noise = None, omega_rf = None, section_index = 1, accelerating_systems = 'as_single'):
        
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
        
        #: *Import ring circumference [m] (from GeneralParameters)*
        self.ring_circumference = GeneralParameters.ring_circumference
        
        #: *Import section length [m] (from GeneralParameters)*
        self.section_length = GeneralParameters.ring_length[self.section_index]
        
        #: *Length ratio of the section wrt the circumference*
        self.length_ratio = self.section_length/self.ring_circumference
        
        #: *Import revolution period [s] (from GeneralParameters)*       
        self.t_rev = GeneralParameters.t_rev
        
        #: *Import momentum program [eV] (from GeneralParameters)*
        self.momentum = GeneralParameters.momentum[self.section_index]
        
        #: *Import synchronous relativistic beta [1] (from GeneralParameters)*
        self.beta = GeneralParameters.beta[self.section_index]
        
        #: *Import synchronous relativistic gamma [1] (from GeneralParameters)*
        self.gamma = GeneralParameters.gamma[self.section_index]
        
        #: *Import synchronous total energy [eV] (from GeneralParameters)*
        self.energy = GeneralParameters.energy[self.section_index]
        
        #: *Energy increment (acceleration/deceleration) between two turns,
        #: for one section in [eV]* :math:`: \quad E_s^{n+1}- E_s^n`
        self.E_increment = np.diff(self.energy)
        
        #: *Import particle mass [e] (from GeneralParameters)*
        self.charge = GeneralParameters.charge
        
        #: *Slippage factor (0th order) for the given RF section*
        self.eta_0 = 0
        #: *Slippage factor (1st order) for the given RF section*
        self.eta_1 = 0
        #: *Slippage factor (2nd order) for the given RF section*
        self.eta_2 = 0
        
        #: *Import alpha order for the section (from GeneralParameters)*
        self.alpha_order = GeneralParameters.alpha_order
        for i in range( self.alpha_order ):
            dummy = getattr(GeneralParameters, 'eta_' + str(i))
            setattr(self, "eta_%s" %i, dummy[self.section_index])
        #: *Sign of eta_0*
        self.sign_eta_0 = np.sign(self.eta_0)   
        
        #: | *Number of RF systems in the section* :math:`: \quad n_{\mathsf{rf}}`
        #: | *Counter for RF is:* :math:`j`
        self.n_rf = n_rf
        
        #: | *RF harmonic number list* :math:`: \quad h_{j,k}^n`
        #: | *See note above on how to input RF programs.*
        self.harmonic = 0
        
        #: | *RF voltage program list in [eV]* :math:`: \quad V_{j,k}^n`
        #: | *See note above on how to input RF programs.*
        self.voltage = 0
        
        #: | *Design RF phase offset list in [rad]* :math:`: \quad \phi_{j,k}^n`
        #: | *See note above on how to input RF programs.*
        self.phi_offset = 0

        #: | *Phase noise array (optional). Added to all RF systems.*
        self.phi_noise = phi_noise
                
        ### Pre-processing the inputs
        # The input is analyzed and structured in order to have lists, which
        # length are matching the number of RF systems considered in this
        # section.
        # For one RF system, single values for h, V_rf, and phi will assume
        # that these values will remain constant for all the simulation.
        # These can be input directly as arrays in order to have programs
        # (the length of the arrays will then be checked)
        if n_rf == 1:
            self.harmonic = [harmonic] 
            self.voltage = [voltage]
            self.phi_offset = [phi_offset]
            if phi_noise != None:
                self.phi_noise = [phi_noise]
            if omega_rf != None:
                self.omega_RF = [omega_rf]
                 
        else:
            self.harmonic = harmonic
            self.voltage = voltage 
            self.phi_offset = phi_offset
            if phi_noise != None:
                self.phi_noise = phi_noise
            if omega_rf != None:
                self.omega_RF = omega_rf
        
        for i in range(self.n_rf):
            self.harmonic[i] = input_check(self.harmonic[i], self.n_turns+1)
            self.voltage[i] = input_check(self.voltage[i], self.n_turns+1)
            self.phi_offset[i] = input_check(self.phi_offset[i], self.n_turns+1)
            if phi_noise != None:
                self.phi_noise[i] = input_check(self.phi_noise[i], self.n_turns+1) 
            if omega_rf != None:
                self.omega_RF[i] = input_check(self.omega_RF[i], self.n_turns+1) 
        
        # Convert to numpy matrix
        self.harmonic = np.array(self.harmonic, ndmin =2)
        self.voltage = np.array(self.voltage, ndmin =2)
        self.phi_offset = np.array(self.phi_offset, ndmin =2)
        if phi_noise != None:
            self.phi_noise = np.array(self.phi_noise, ndmin =2) 
        if omega_rf != None:
            self.omega_RF = np.array(self.omega_RF, ndmin =2) 
            
        #: *Synchronous phase for this section, calculated from the transition
        #: energy and the momentum program.*
        self.phi_s = calc_phi_s(self, accelerating_systems)   

        
        #: *Synchrotron tune [1]*                         
        self.Qs = np.sqrt( self.harmonic[0]*self.charge*self.voltage[0]*np.abs(self.eta_0*np.cos(self.phi_s)) / \
                                 (2*np.pi*self.beta**2*self.energy) ) 
        
        #: *Central angular synchronous frequency, w/o intensity effects [1/s]*
        self.omega_s0 = self.Qs*GeneralParameters.omega_rev

        #: *Design RF frequency of the RF systems in the station [Hz]*        
        self.omega_RF_d = 2.*np.pi*self.beta*c*self.harmonic/ \
                          (self.ring_circumference)
                          
        #: *Initial, actual RF frequency of the RF systems in the station [Hz]*
        if omega_rf == None:
            self.omega_RF = np.array(self.omega_RF_d)                  

        #: *Initial, actual RF phase of each harmonic system*
        self.phi_RF = np.array(self.phi_offset) 
        
        #: *Accumulated RF phase error of each harmonic system*
        self.dphi_RF = np.zeros(self.n_rf)
        
        #: *Accumulated RF phase error of each harmonic system*
        self.dphi_RF_steering = np.zeros(self.n_rf)
        
        self.t_RF = 2*np.pi / self.omega_RF[0]
        
 
        
        
    def eta_tracking(self, beam, counter, dE):
        '''
        *The slippage factor is calculated as a function of the energy offset
        (dE) of the beam particle. By definition, the slippage factor in ith 
        order is:*
        
        .. math:: 
            \\eta(\\delta) = \\sum_{i}(\\eta_i \\, \\delta^i) = \\sum_{i} \\left(\\eta_i \\, \\left[ \\frac{\\Delta E}{\\beta_s^2 E_s} \\right]^i \\right)
    
        '''
        
        if self.alpha_order == 1:
            return self.eta_0[counter]
        else:
            eta = 0
            delta = dE/(beam.beta**2 * beam.energy)
            for i in range( self.alpha_order ):
                eta_i = getattr(self, 'eta_' + str(i))[counter]
                eta  += eta_i * (delta**i)
            return eta  



def calc_phi_s(RFSectionParameters, accelerating_systems = 'as_single'):
    '''
    Calculation of the synchronous phase at every turn
    according to the parameters in the RFSectionParameters object. The
    phase is expressed in the lowest RF harmonic and with respect to the
    RF bucket (see the equations of motion defined for BLonD)
    The returned value is given in the range [0,2pi].
    Below transition, the RF wave is shifted by Pi w.r.t. the time reference.
    
    If the accelerating_systems option is set to 'as_single', the synchronous
    phase is calculated analytically.
    
    If the accelerating_systems is set to 'all', the synchronous phase
    is calculated numerically by finding the minimum of the potential well.
    In case of several minima, the deepest is taken. WARNING: in case of 
    RF harmonics with comparable voltages, this may lead to inconsistent 
    values of phi_s.
    
    The option accelerating_systems set to 'first' is not yet implemented.
    Its purpose should be to adjust the RFSectionParameters.phi_offset of the
    higher harmonics so that only the main harmonic is accelerating.
    '''
    
    eta0 = RFSectionParameters.eta_0
    
    if accelerating_systems == 'as_single':
        denergy = np.append(RFSectionParameters.E_increment, 
                            RFSectionParameters.E_increment[-1])             
        acceleration_ratio = denergy/ \
                             (RFSectionParameters.charge*
                              RFSectionParameters.voltage[0,:])

        acceleration_test = np.where((acceleration_ratio > -1)*\
                                     (acceleration_ratio < 1) == False)[0]
                
        if acceleration_test.size > 0:
            print('Warning!!! Acceleration is not possible (momentum increment is too big or voltage too low) at index ' + str(acceleration_test))
        
        phi_s = np.arcsin(acceleration_ratio)

        eta0_middle_points = (eta0[1:] + eta0[:-1])/2
        eta0_middle_points = np.append(eta0_middle_points, eta0[-1])             
        index = np.where(eta0_middle_points > 0)[0]
        index_below = np.where(eta0_middle_points < 0)[0]
         
        phi_s[index] = (np.pi - phi_s[index]) % (2*np.pi)
        phi_s[index_below] = (np.pi + phi_s[index_below]) % (2*np.pi)
        
        return phi_s 
     
    else:
        
        if accelerating_systems == 'all':
            '''
            In this case, all the RF systems are accelerating, phi_s is 
            calculated accordingly with respect to the fundamental frequency 
            (the minimum of the potential well is taken)
            '''         
            
            phi_s = np.zeros(len(RFSectionParameters.voltage[0,1:]))
            
            for indexTurn in range(len(RFSectionParameters.E_increment)):
                
                totalRF = 0
                if np.sign(eta0[indexTurn])>0:
                    phase_array = np.linspace(-RFSectionParameters.phi_RF[0,indexTurn+1], -RFSectionParameters.phi_RF[0,indexTurn+1] + 2*np.pi, 1000) 
                else:
                    phase_array = np.linspace(-RFSectionParameters.phi_RF[0,indexTurn+1] - np.pi, -RFSectionParameters.phi_RF[0,indexTurn+1] + np.pi, 1000) 

                for indexRF in range(len(RFSectionParameters.voltage[:,indexTurn+1])):
                    totalRF += RFSectionParameters.voltage[indexRF,indexTurn+1] * np.sin(RFSectionParameters.harmonic[indexRF,indexTurn+1]/np.min(RFSectionParameters.harmonic[:,indexTurn+1]) * phase_array + RFSectionParameters.phi_RF[indexRF,indexTurn+1]) #
                    
                potential_well = - cumtrapz(np.sign(eta0[indexTurn])*(totalRF - RFSectionParameters.E_increment[indexTurn]/abs(RFSectionParameters.charge)), dx=phase_array[1]-phase_array[0], initial=0)

                phi_s[indexTurn] = np.mean(phase_array[potential_well==np.min(potential_well)])

            phi_s = np.insert(phi_s, 0, phi_s[0])+RFSectionParameters.phi_RF[0,:]
            phi_s[eta0<0] += np.pi
            phi_s = phi_s % (2*np.pi)
            
            return phi_s
        
        elif accelerating_systems == 'first':
            '''
            Only the first RF system is accelerating, so we have to correct the 
            phi_RF of the other rf_systems such that p_increment relates 
            only to the first RF
            '''
            pass
        else:
            raise RuntimeError('Did not recognize the option accelerating_systems in calc_phi_s function')
        
        # This part only works when you have no acceleration and several RF systems
        if eta0[0] > 0:
            return np.pi*np.ones(RFSectionParameters.n_turns)
        elif eta0[0] < 0:
            return 0*np.ones(RFSectionParameters.n_turns)
