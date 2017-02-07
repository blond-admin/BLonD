# Copyright 2016 CERN. This software is distributed under the
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
from builtins import range, object
import numpy as np
from scipy.integrate import cumtrapz
import ctypes
from setup_cpp import libblond

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
                                  dt_margin_percent = 0., time_array=None):

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
        A margin on the time array can be applied in order
        to be able to see the min/max that might be exactly on the edges of the
        frame (by adding a % to the length of the frame, this is set to 0 by default. 
        It assumes also that the slippage factor is the same in the whole ring.*
        '''
        
        voltages = np.array([])
        omega_rf = np.array([])
        phi_offsets = np.array([])
                 
        for RingAndRFSectionElement in self.RingAndRFSection_list:
            charge = RingAndRFSectionElement.charge
            for rf_system in range(RingAndRFSectionElement.n_rf):
                voltages = np.append(voltages, RingAndRFSectionElement.voltage[rf_system, turn])
                omega_rf = np.append(omega_rf, RingAndRFSectionElement.omega_RF[rf_system, turn])
                phi_offsets = np.append(phi_offsets, RingAndRFSectionElement.phi_RF[rf_system, turn])
                        
        voltages = np.array(voltages, ndmin = 2)
        omega_rf = np.array(omega_rf, ndmin = 2)
        phi_offsets = np.array(phi_offsets, ndmin = 2)
        
        if main_harmonic_option is 'lowest_freq':
            main_omega_rf = np.min(omega_rf)
        elif main_harmonic_option is 'highest_voltage':
            main_omega_rf = np.min(omega_rf[voltages == np.max(voltages)])
        elif isinstance(main_harmonic_option, int) or isinstance(main_harmonic_option, float):
            if omega_rf[omega_rf == main_harmonic_option].size == 0:
                raise RuntimeError('The desired harmonic to compute the potential well does not match the RF parameters...')
            main_omega_rf = np.min(omega_rf[omega_rf == main_harmonic_option])
            
        slippage_factor = self.RingAndRFSection_list[0].eta_0[turn]
        
        if time_array is None:            
            time_array_margin = dt_margin_percent * 2 * np.pi/main_omega_rf
            
            first_dt = - time_array_margin / 2
            last_dt = 2 * np.pi/main_omega_rf + time_array_margin / 2
                
            time_array = np.linspace(first_dt, last_dt, n_points)
                
        self.total_voltage = np.sum(voltages.T * np.sin(omega_rf.T * time_array + phi_offsets.T), axis=0)
        
        eom_factor_potential = np.sign(slippage_factor) * charge / (RingAndRFSectionElement.t_rev[turn])
        
        potential_well = -cumtrapz(eom_factor_potential * (self.total_voltage - (-RingAndRFSectionElement.acceleration_kick[turn])/abs(charge)), dx=time_array[1]-time_array[0],initial=0)
        potential_well = potential_well - np.min(potential_well)
        
        self.potential_well_coordinates = time_array
        self.potential_well = potential_well

        
        
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
        
    def __init__(self, RFSectionParameters, Beam, solver = b'simple', 
                 PhaseLoop = None, NoiseFB = None, periodicity = False, dE_max = None, rf_kick_interp=False, Slices=None, TotalInducedVoltage=None):
        
        #: *Import of RFSectionParameters object*
        self.rf_params = RFSectionParameters

        #: *Import of Beam object*
        self.beam = Beam
        
        #: *Import PhaseLoop object*                
        self.PL = PhaseLoop   
        
        #: *Import NoiseFB object*                
        self.noiseFB = NoiseFB

        ### Import RF section parameters #######################################
        #: *Import section index (from RFSectionParameters)*        
        self.section_index = RFSectionParameters.section_index
        
        #: *Import counter (from RFSectionParameters)*        
        self.counter = RFSectionParameters.counter 
              
        #: *Import length ratio (from RFSectionParameters)*
        self.length_ratio = RFSectionParameters.length_ratio
        
        #: *Import section length (from RFSectionParameters)* # needed for FullRingAndRF
        self.section_length = RFSectionParameters.section_length
        
        #: *Import revolution period (from GeneralParameters)*       
        self.t_rev = RFSectionParameters.t_rev

        #: *Import the number of RF systems (from RFSectionParameters)*
        self.n_rf = RFSectionParameters.n_rf
        
        #: *Import beta (from RFSectionParameters)* # needed for FullRingAndRF
        self.beta = RFSectionParameters.beta
        
        #: *Import particle charge (from RFSectionParameters)* 
        self.charge = RFSectionParameters.charge
        
        #: *Import RF harmonic number program (from RFSectionParameters)*
        self.harmonic = RFSectionParameters.harmonic 
               
        #: *Import RF voltage program [V] (from RFSectionParameters)*
        self.voltage = RFSectionParameters.voltage  
           
        #: *Import RF phase noise [rad] (from RFSectionParameters)*
        self.phi_noise = RFSectionParameters.phi_noise
        
        #: *Import RF phase [rad] (from RFSectionParameters)*
        self.phi_RF = RFSectionParameters.phi_RF
        
        #: *Import phi_s [rad] (from RFSectionParameters)* # needed for FullRingAndRF
        self.phi_s = RFSectionParameters.phi_s
        
        #: *Import actual RF frequency [1/s] (from RFSectionParameters)*
        self.omega_RF = RFSectionParameters.omega_RF
        
        #: *Slippage factor (0th order) for the given RF section*
        self.eta_0 = RFSectionParameters.eta_0
        
        #: *Slippage factor (1st order) for the given RF section*
        self.eta_1 = RFSectionParameters.eta_1
        
        #: *Slippage factor (2nd order) for the given RF section*
        self.eta_2 = RFSectionParameters.eta_2
        
        #: *Slippage factor (2nd order) for the given RF section*
        self.sign_eta_0 = RFSectionParameters.sign_eta_0
        
        #: *Import alpha order (from RFSectionParameters)*                
        self.alpha_order = RFSectionParameters.alpha_order
        
        #: *Fill unused eta arrays with zeros*
        for i in range( self.alpha_order, 3 ):
            setattr(self, "eta_%s" %i, np.zeros(RFSectionParameters.n_turns+1))       
        ### End of import of RF section parameters #############################
            
        #: *Synchronous energy change* :math:`: \quad - \delta E_s`
        self.acceleration_kick = - RFSectionParameters.E_increment  
        
        #: | *Choice of drift solver options*
        self.solver = solver
        if self.solver != b'simple' and self.solver != b'full':
            raise RuntimeError("ERROR: Choice of longitudinal solver not recognized! Aborting...")
            
        #: | *Set to 'full' if higher orders of eta are used*
        if self.alpha_order > 1:
            self.solver = 'full'
        
        # Set the horizontal cut
        self.dE_max = dE_max
        
        # Periodicity setting up
        self.periodicity = periodicity
            
        # Use interpolate to apply kick
        self.rf_kick_interp = rf_kick_interp
        self.slices = Slices
        self.TotalInducedVoltage = TotalInducedVoltage
        
        if self.rf_kick_interp and self.slices is None:
            raise RuntimeError('ERROR: A slices object is needed in the RingAndRFSection to use the kick_interp option')
        
 
    def kick(self, beam_dt, beam_dE, index):
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
        
        voltage_kick = np.ascontiguousarray(self.charge*
                                      self.voltage[:, index])
        omegaRF_kick = np.ascontiguousarray(self.omega_RF[:, index])
        phiRF_kick = np.ascontiguousarray(self.phi_RF[:, index])
        
        libblond.kick(beam_dt.ctypes.data_as(ctypes.c_void_p), 
            beam_dE.ctypes.data_as(ctypes.c_void_p), 
            ctypes.c_int(self.n_rf), voltage_kick.ctypes.data_as(ctypes.c_void_p), 
            omegaRF_kick.ctypes.data_as(ctypes.c_void_p), 
            phiRF_kick.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(len(beam_dt)), 
            ctypes.c_double(self.acceleration_kick[index]))
        
   
    def drift(self, beam_dt, beam_dE, index):
        '''
        *Update of particle arrival time to the RF station. If only the zeroth 
        order slippage factor is given, 'simple' and 'full' solvers are 
        available. The 'simple' solver is somewhat faster. Otherwise, the solver
        is automatically 'full' and calculates the frequency slippage up to 
        second order.*
        
        *The corresponding equations are:*
        
        .. math::
            \\Delta t^{n+1} = \\Delta t^{n} + \\frac{L}{C} T_0^{n+1} \\left(\\frac{1}{1 - \\eta(\\delta^{n+1})\\delta^{n+1}} - 1\\right) \quad \\text{(full)}
            
        .. math::
            \\Delta t^{n+1} = \\Delta t^{n} + \\frac{L}{C} T_0^{n+1}\\eta_0\\delta^{n+1} \quad \\text{(simple)}
        
        '''
        
        libblond.drift(beam_dt.ctypes.data_as(ctypes.c_void_p), 
            beam_dE.ctypes.data_as(ctypes.c_void_p), 
            ctypes.c_char_p(self.solver),
            ctypes.c_double(self.t_rev[index]),
            ctypes.c_double(self.length_ratio), 
            ctypes.c_double(self.alpha_order), 
            ctypes.c_double(self.eta_0[index]), 
            ctypes.c_double(self.eta_1[index]),
            ctypes.c_double(self.eta_2[index]), 
            ctypes.c_double(self.rf_params.beta[index]), 
            ctypes.c_double(self.rf_params.energy[index]), 
            ctypes.c_int(len(beam_dt)))


    def rf_voltage_calculation(self, turn, Slices):
        '''
        *Calculating the RF voltage seen by the beam at a given turn, needs a Slices object.
        '''
        
        voltages = np.array([])
        omega_rf = np.array([])
        phi_RF = np.array([])
        
        for rf_system in range(self.n_rf):
                voltages = np.append(voltages, self.voltage[rf_system, turn])
                omega_rf = np.append(omega_rf, self.omega_RF[rf_system, turn])
                phi_RF = np.append(phi_RF, self.phi_RF[rf_system, turn])
                        
        voltages = np.array(voltages, ndmin = 2)
        omega_rf = np.array(omega_rf, ndmin = 2)
        phi_RF = np.array(phi_RF, ndmin = 2)
        
        self.rf_voltage = np.sum(voltages.T * np.sin(omega_rf.T * Slices.bin_centers + phi_RF.T), axis = 0)
        
                
    def track(self):
        '''
        *Tracking method for the section. Applies first the kick, then the 
        drift. Calls also RF feedbacks if applicable. Updates the counter of the
        corresponding RFSectionParameters class and the energy-related 
        variables of the Beam class.*
        '''
        
        # Add phase noise directly to the cavity RF phase
        if self.phi_noise != None:
            if self.noiseFB != None:
                self.phi_RF[:,self.counter[0]] += \
                    self.noiseFB.x*self.phi_noise[:,self.counter[0]]
            else:
                self.phi_RF[:,self.counter[0]] += \
                    self.phi_noise[:,self.counter[0]]

        # Determine phase loop correction on RF phase and frequency
        if self.PL != None and self.counter[0]>=self.PL.delay:
            self.PL.track()  
        
        if self.periodicity:
            
            # Distinguish the particles inside the frame from the particles on the
            # right of the frame.
            self.indices_right_outside = np.where(self.beam.dt > self.t_rev[self.counter[0]+1])[0]
            self.indices_inside_frame = np.where(self.beam.dt < self.t_rev[self.counter[0]+1])[0]
            
            if len(self.indices_right_outside)>0:
                # Change reference of all the particles on the right of the current
                # frame; these particles skip one kick and drift
                self.beam.dt[self.indices_right_outside] -= self.t_rev[self.counter[0]+1]
                # Syncronize the bunch with the particles that are on the right of
                # the current frame applying kick and drift to the bunch; after that 
                # all the particles are in the new updated frame
                self.insiders_dt = np.ascontiguousarray(self.beam.dt[self.indices_inside_frame])
                self.insiders_dE = np.ascontiguousarray(self.beam.dE[self.indices_inside_frame])
                self.kick(self.insiders_dt, self.insiders_dE, self.counter[0])
                self.drift(self.insiders_dt, self.insiders_dE, self.counter[0]+1)
                self.beam.dt[self.indices_inside_frame] = self.insiders_dt
                self.beam.dE[self.indices_inside_frame] = self.insiders_dE
                # Check all the particles on the left of the just updated frame and 
                # apply a second kick and drift to them with the previous wave after
                # having changed reference.
                self.indices_left_outside = np.where(self.beam.dt < 0)[0]
                
            else:
                self.kick(self.beam.dt, self.beam.dE, self.counter[0])
                self.drift(self.beam.dt, self.beam.dE, self.counter[0]+1)
                # Check all the particles on the left of the just updated frame and 
                # apply a second kick and drift to them with the previous wave after
                # having changed reference.
                self.indices_left_outside = np.where(self.beam.dt < 0)[0]
                
            if len(self.indices_left_outside)>0:
                left_outsiders_dt = np.ascontiguousarray(self.beam.dt[self.indices_left_outside])
                left_outsiders_dE = np.ascontiguousarray(self.beam.dE[self.indices_left_outside])
                left_outsiders_dt += self.t_rev[self.counter[0]+1]
                self.kick(left_outsiders_dt, left_outsiders_dE, self.counter[0])
                self.drift(left_outsiders_dt, left_outsiders_dE, self.counter[0]+1)
                self.beam.dt[self.indices_left_outside] = left_outsiders_dt
                self.beam.dE[self.indices_left_outside] = left_outsiders_dE
                        
            # Orizzontal cut: this method really eliminates particles from the
            # code
            if self.dE_max!=None:
                itemindex = np.where(self.beam.dE > -self.dE_max)[0]
                if len(itemindex) < self.beam.n_macroparticles:
                    self.beam.dt = np.ascontiguousarray(self.beam.dt[itemindex])
                    self.beam.dE = np.ascontiguousarray(self.beam.dE[itemindex])
                    self.beam.n_macroparticles = len(self.beam.dt)
                
        else:
            
            if self.rf_kick_interp:
                self.rf_voltage_calculation(self.counter[0], self.slices)
                if self.TotalInducedVoltage is not None:
                    self.total_voltage = self.rf_voltage + self.TotalInducedVoltage.induced_voltage
                else:
                    self.total_voltage = self.rf_voltage
                libblond.linear_interp_kick(self.beam.dt.ctypes.data_as(ctypes.c_void_p),
                                  self.beam.dE.ctypes.data_as(ctypes.c_void_p), 
                                  (self.beam.charge * self.total_voltage).ctypes.data_as(ctypes.c_void_p), 
                                  self.slices.bin_centers.ctypes.data_as(ctypes.c_void_p), 
                                  ctypes.c_int(self.slices.n_slices),
                                  ctypes.c_int(self.beam.n_macroparticles),
                                  ctypes.c_double(self.acceleration_kick[self.counter[0]]))
                
            else:
                self.kick(self.beam.dt, self.beam.dE, self.counter[0])
            
            self.drift(self.beam.dt, self.beam.dE, self.counter[0]+1)
            
            # Orizzontal cut: this method really eliminates particles from the
            # code
            if self.dE_max!=None:
                itemindex = np.where(self.beam.dE > -self.dE_max)[0]
                if len(itemindex) < self.beam.n_macroparticles:
                    self.beam.dt = np.ascontiguousarray(self.beam.dt[itemindex])
                    self.beam.dE = np.ascontiguousarray(self.beam.dE[itemindex])
                    self.beam.n_macroparticles = len(self.beam.dt)
    
        # Increment by one the turn counter
        self.counter[0] += 1
        
        # Updating the beam synchronous momentum etc.
        self.beam.beta = self.rf_params.beta[self.counter[0]]
        self.beam.gamma = self.rf_params.gamma[self.counter[0]]
        self.beam.energy = self.rf_params.energy[self.counter[0]]
        self.beam.momentum = self.rf_params.momentum[self.counter[0]]
