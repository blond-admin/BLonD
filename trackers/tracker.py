
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module containing all the elements to track the RF frequency and phase and the
beam in phase space.**

:Authors:  **Helga Timko**, **Alexandre Lasheen**, **Danilo Quartullo**
'''

from __future__ import division
import numpy as np
from scipy.integrate import cumtrapz
import ctypes
from setup_cpp import libfib
from scipy.constants import c


class FullRingAndRF(object):
    '''
    *Definition of the full ring and RF parameters in order to be able to have
    a full turn information (used in the hamiltonian for example).*
    '''
    
    def __init__(self, RingAndRFSection_list):
        
        #: *List of the total RingAndRFSection objects*
        self.RingAndRFSection_list = RingAndRFSection_list
        
        #: *Total potential well in [V]*
        self.potential_well = 0
        
        #: *Total potential well theta coordinates in [rad]*
        self.potential_well_coordinates = 0
        
        #: *Ring circumference in [m]*
        self.ring_circumference = 0
        for RingAndRFSectionElement in self.RingAndRFSection_list:
            self.ring_circumference += RingAndRFSectionElement.section_length
            
        #: *Ring radius in [m]*
        self.ring_radius = self.ring_circumference / (2*np.pi)
        
        
        
    def potential_well_generation(self, turn = 0, n_points = 1e5, 
                                  main_harmonic_option = 'lowest_freq', 
                                  theta_margin_percent = 0.):
        '''
        *Method to generate the potential well out of the RF systems. The 
        assumption made is that all the RF voltages are averaged over
        one turn. The potential well is then approximated over one turn,
        which is not the exact potential. This approximation should be
        fine enough to generate a bunch (the mismatch should be small and
        damped fast enough). The default main harmonic is defined to be the
        lowest one in frequency. The user can change this option if it is
        not the case for his simulations (other options are: 'highest_voltage',
        or inputing directly the value of the desired main harmonic). 
        A margin on the theta array can be applied in order
        to be able to see the min/max that might be exactly on the edges of the
        frame (by adding a % to the length of the frame, this is set to 0 by default. 
        It assumes also that the slippage factor is the same in the whole ring.*
        '''
        
        voltages = np.array([])
        harmonics = np.array([])
        phi_offsets = np.array([])
        sync_phases = np.array([])
                 
        for RingAndRFSectionElement in self.RingAndRFSection_list:
            for rf_system in range(RingAndRFSectionElement.n_rf):
                voltages = np.append(voltages, RingAndRFSectionElement.voltage[rf_system, turn])
                harmonics = np.append(harmonics, RingAndRFSectionElement.harmonic[rf_system, turn])
                phi_offsets = np.append(phi_offsets, RingAndRFSectionElement.phi_RF[rf_system, turn])
                sync_phases = np.append(sync_phases, RingAndRFSectionElement.phi_s[turn])
                        
        voltages = np.array(voltages, ndmin = 2)
        harmonics = np.array(harmonics, ndmin = 2)
        phi_offsets = np.array(phi_offsets, ndmin = 2)
        sync_phases = np.array(sync_phases, ndmin = 2)
        
        if main_harmonic_option is 'lowest_freq':
            main_harmonic = np.min(harmonics)
        elif main_harmonic_option is 'highest_voltage':
            main_harmonic = np.min(harmonics[voltages == np.max(voltages)])
        elif isinstance(main_harmonic_option, int) or isinstance(main_harmonic_option, float):
            if harmonics[harmonics == main_harmonic_option].size == 0:
                raise RuntimeError('The desired harmonic to compute the potential well does not match the RF parameters...')
            main_harmonic = np.min(harmonics[harmonics == main_harmonic_option])
            
        theta_array_margin = theta_margin_percent * 2 * np.pi/main_harmonic
        
        slippage_factor = self.RingAndRFSection_list[0].eta_0[turn]
        beta = self.RingAndRFSection_list[0].beta[turn]
        
        if slippage_factor > 0:
            first_theta = 0 - theta_array_margin / 2
            last_theta = 2*np.pi/main_harmonic + theta_array_margin / 2
            transition_factor = - 1
        else:
            first_theta = - np.pi/main_harmonic - theta_array_margin / 2
            last_theta = np.pi/main_harmonic + theta_array_margin / 2
            transition_factor = 1
            
        theta_array = np.linspace(first_theta, last_theta, n_points)
                
        total_voltage = np.sum(voltages.T * np.sin(harmonics.T * theta_array + phi_offsets.T) - voltages.T * np.sin(sync_phases.T), axis = 0)
        
        eom_factor_potential = (beta * c) / (self.ring_circumference)
        
        potential_well = transition_factor * np.insert(cumtrapz(total_voltage, dx=theta_array[1]-theta_array[0]),0,0)
        
        self.potential_well_coordinates = theta_array
        self.potential_well = eom_factor_potential * potential_well
        
        
    def track(self):
        '''
        *Loops over all the RingAndRFSection.track methods.*
        '''
        
        for RingAndRFSectionElement in self.RingAndRFSection_list:
            RingAndRFSectionElement.track()



class RingAndRFSection(object):
    '''
    *Definition of an RF station and part of the ring until the next station, 
    see figure.*
    
    .. image:: ring_and_RFstation.png
        :align: center
        :width: 600
        :height: 600
        
    *The time step is fixed to be one turn, but the tracking can consist of 
    multiple RingAndRFSection objects. In this case, the user should make sure 
    that the lengths of the stations sum up exactly to the circumference or use
    the FullRingAndRF object in order to let the code pre-process the 
    parameters. Each RF station may contain several RF harmonic systems which 
    are considered to be in the same location. First, the energy kick of the RF
    station is applied, and then the particle arrival time to the next station
    is updated. The change in RF phase and frequency due to control loops is 
    tracked as well.*
    '''
        
    def __init__(self, RFSectionParameters, Beam, solver = 'simple', 
                 PhaseLoop = None):
        
        #: *Import of RFSectionParameters object*
        self.rf_params = RFSectionParameters

        #: *Import of Beam object*
        self.beam = Beam

        #: *Import PhaseLoop object*                
        self.PL = PhaseLoop   
        
        ### Import RF section parameters #######################################
        #: *Import section index (from RFSectionParameters)*        
        self.section_index = RFSectionParameters.section_index
        
        #: *Import counter (from RFSectionParameters)*        
        self.counter = RFSectionParameters.counter 
              
        #: *Import length ratio (from RFSectionParameters)*
        self.length_ratio = RFSectionParameters.length_ratio
        
        #: *Import section length (from RFSectionParameters)*
        self.section_length = RFSectionParameters.section_length
        
        #: *Import revolution period (from GeneralParameters)*       
        self.t_rev = RFSectionParameters.t_rev

        #: *Import the number of RF systems (from RFSectionParameters)*
        self.n_rf = RFSectionParameters.n_rf
        
        #: *Import beta (from RFSectionParameters)*
        self.beta = RFSectionParameters.beta
        
        #: *Import RF harmonic number program (from RFSectionParameters)*
        self.harmonic = RFSectionParameters.harmonic 
               
        #: *Import RF voltage program [GV] (from RFSectionParameters)*
        self.voltage = RFSectionParameters.voltage  
           
        #: *Import RF phase noise [rad] (from RFSectionParameters)*
        self.phi_noise = RFSectionParameters.phi_noise
        
        #: *Import RF phase [rad] (from RFSectionParameters)*
        self.phi_RF = RFSectionParameters.phi_RF
        
        #: *Import phi_s [rad] (from RFSectionParameters)*
        self.phi_s = RFSectionParameters.phi_s
        
        #: *Import actual RF frequency [1/s] (from RFSectionParameters)*
        self.omega_RF = RFSectionParameters.omega_RF
        
        #: *Slippage factor (0th order) for the given RF section*
        self.eta_0 = RFSectionParameters.eta_0
        
        #: *Slippage factor (1st order) for the given RF section*
        self.eta_1 = RFSectionParameters.eta_1
        
        #: *Slippage factor (2nd order) for the given RF section*
        self.eta_2 = RFSectionParameters.eta_2
        
        #: *Import alpha order (from RFSectionParameters)*                
        self.alpha_order = RFSectionParameters.alpha_order
        
        #: *Fill unused eta arrays with zeros*
        for i in xrange( self.alpha_order, 3 ):
            setattr(self, "eta_%s" %i, np.zeros(RFSectionParameters.n_turns+1))       
        ### End of import of RF section parameters #############################
            
        #: *Synchronous energy change* :math:`: \quad - \delta E_s`
        self.acceleration_kick = - RFSectionParameters.E_increment  
        
        
        #: | *Choice of drift solver options*
        self.solver = solver
        if self.solver != 'simple' and self.solver != 'full':
            raise RuntimeError("ERROR: Choice of longitudinal solver not recognized! Aborting...")
            
        #: | *Set to 'full' if higher orders of eta are used*
        if self.alpha_order > 1:
            self.solver = 'full'
                     
        
    def kick(self):
        '''
        *Update of the particle energy due to the RF kick in a given RF station. 
        The kicks are summed over the different harmonic RF systems in the 
        station. The cavity phase can be shifted by the user via phi_offset.
        The main RF (harmonic[0]) has by definition phase=0 at time=0. The 
        phases of all other RF systems are defined w.r.t. to the main RF.
        The increment in energy is given by the discrete equation of motion:*
        
        .. math::
            \Delta E^{n+1} = \Delta E^n + \sum_{k=0}^{n_{\mathsf{rf}}-1}{e V_k^n \\sin{\\left(\omega_{\mathsf{rf,k}}^n \\Delta t^n + \phi_{\mathsf{rf,k}}^n \\right)}} - (E_s^{n+1} - E_s^n) 
            
        '''
        
        v_kick = np.ascontiguousarray(self.voltage[:, self.counter[0]])
        o_kick = np.ascontiguousarray(self.omega_RF[:, self.counter[0]])
        p_kick = np.ascontiguousarray(self.phi_RF[:, self.counter[0]])
        
        libfib.kick(self.beam.dt.ctypes.data_as(ctypes.c_void_p), 
            self.beam.dE.ctypes.data_as(ctypes.c_void_p), 
            ctypes.c_int(self.n_rf), v_kick.ctypes.data_as(ctypes.c_void_p), 
            o_kick.ctypes.data_as(ctypes.c_void_p), 
            p_kick.ctypes.data_as(ctypes.c_void_p), 
            ctypes.c_uint(self.beam.n_macroparticles), 
            ctypes.c_double(self.acceleration_kick[self.counter[0]]))
        
   
    def drift(self):
        '''
        *Update of particle arrival time to the RF station. If only the zeroth 
        order slippage factor is given, 'simple' and 'full' solvers are 
        available. The 'simple' solver is somewhat faster. Otherwise, the solver
        is automatically 'full' and calculates the frequency slippage up to 
        second order.*
        
        *The corresponding equations are:*
        
        .. math::
            \\Delta t^{n+1} = \\Delta t^{n} + \\frac{L}{C} T_0^{n+1} \\left(\\frac{1}{1 - \\eta(\\delta^n)\\delta^n} - 1\\right) \quad \\text{(full)}
            
        .. math::
            \\Delta t^{n+1} = \\Delta t^{n} + \\frac{L}{C} T_0^{n+1}\\eta_0\\delta^n \quad \\text{(simple)}
        
        '''
        
        libfib.drift(self.beam.dt.ctypes.data_as(ctypes.c_void_p), 
            self.beam.dE.ctypes.data_as(ctypes.c_void_p), 
            ctypes.c_char_p(self.solver),
            ctypes.c_double(self.t_rev[self.counter[0]]),
            ctypes.c_double(self.length_ratio), 
            ctypes.c_double(self.alpha_order), 
            ctypes.c_double(self.eta_0[self.counter[0]]), 
            ctypes.c_double(self.eta_1[self.counter[0]]),
            ctypes.c_double(self.eta_2[self.counter[0]]), 
            ctypes.c_double(self.beam.beta), ctypes.c_double(self.beam.energy), 
            ctypes.c_uint(self.beam.n_macroparticles))

                
    def track(self):
        '''
        *Tracking method for the section. Applies first the kick, then the 
        drift. Calls also RF feedbacks if applicable. Updates the counter of the
        corresponding RFSectionParameters class and the energy-related 
        variables of the Beam class.*
        '''
        
        # Add phase noise directly to the cavity RF phase
        if self.phi_noise != None:
            self.phi_RF[:,self.counter[0]] += self.phi_noise[:,self.counter[0]]

        # Determine phase loop correction on RF phase and frequency
        if self.PL != None:
            self.PL.track()  

        # Kick
        self.kick()
        
        # Drift
        self.drift()
        
        # Increment by one the turn counter
        self.counter[0] += 1
        
        # Updating the beam synchronous momentum etc.
        self.beam.beta = self.rf_params.beta[self.counter[0]]
        self.beam.gamma = self.rf_params.gamma[self.counter[0]]
        self.beam.energy = self.rf_params.energy[self.counter[0]]
        self.beam.momentum = self.rf_params.momentum[self.counter[0]]
