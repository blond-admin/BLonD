'''
**Module containing all the elements to track the beam in the longitudinal plane.**

:Authors: **Danilo Quartullo**, **Helga Timko**, **Adrian Oeftiger**, **Alexandre Lasheen**
'''

from __future__ import division
import numpy as np
from scipy.constants import c
from scipy.integrate import cumtrapz


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
        
        #: *Total potential well theta coordinates in [rad] *
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
                phi_offsets = np.append(phi_offsets, RingAndRFSectionElement.phi_offset[rf_system, turn])
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
        beta_r = self.RingAndRFSection_list[0].beta_r[turn]
        
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
        
        eom_factor_potential = (beta_r * c) / (self.ring_circumference)
        
        potential_well = transition_factor * np.insert(cumtrapz(total_voltage, dx=theta_array[1]-theta_array[0]),0,0)
        
        self.potential_well_coordinates = theta_array
        self.potential_well = eom_factor_potential * potential_well
        
        
    def track(self, beam):
        '''
        *Loops over all the RingAndRFSection.track methods.*
        '''
        
        for RingAndRFSectionElement in self.RingAndRFSection_list:
            RingAndRFSectionElement.track(beam)


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
    the FullRingAndRF object in order to let the code pre-process the parameters.
    Each RF station may contain several RF harmonic systems which are considered
    to be in the same location. First, a kick from the cavity voltage(s) is applied, 
    then an accelerating kick in case the momentum program presents variations, 
    and finally a drift kick between stations.*
    '''
        
    def __init__(self, RFSectionParameters, solver = 'full', PhaseLoop = None):
        
        #: *Copy of the counter (from RFSectionParameters)*
        self.counter = RFSectionParameters.counter
        
        ### Import RF section parameters for RF kick
        #: *Copy of length (from RFSectionParameters)*
        self.section_length = RFSectionParameters.section_length
        #: *Copy of length ratio (from RFSectionParameters)*
        self.length_ratio = RFSectionParameters.length_ratio
        #: *Copy of the number of RF systems (from RFSectionParameters)*
        self.n_rf = RFSectionParameters.n_rf
        #: *Copy of harmonic number program (from RFSectionParameters)*
        self.harmonic = RFSectionParameters.harmonic
        #: *Copy of voltage program in [V] (from RFSectionParameters)*
        self.voltage = RFSectionParameters.voltage
        #: *Copy of phi_offset program in [rad] (from RFSectionParameters)*
        self.phi_offset = RFSectionParameters.phi_offset
        #: *Copy of phi_s program in [rad] (from RFSectionParameters)*
        self.phi_s = RFSectionParameters.phi_s
        
        ### Import RF section parameters for accelerating kick
        #: *Copy of the momentum program in [eV/c] (from RFSectionParameters)*
        self.momentum = RFSectionParameters.momentum
        #: *Copy of the momentum increment in [eV/c] (from RFSectionParameters)*
        self.p_increment = RFSectionParameters.p_increment
        #: *Copy of the relativistic beta (from RFSectionParameters)*
        self.beta_r = RFSectionParameters.beta_r
        #: *Copy of the relativistic gamma (from RFSectionParameters)*        
        self.gamma_r = RFSectionParameters.gamma_r
        #: *Copy of the relativistic energy (from RFSectionParameters)*                
        self.energy = RFSectionParameters.energy
        
        #: *Acceleration kick* :math:`: \quad - <\beta> \Delta p`
        self.acceleration_kick = - self.beta_r[1:] * self.p_increment  
        
        ### Import RF section parameters for the drift
        #: *Slippage factor (order 0) for the given RF section*
        self.eta_0 = RFSectionParameters.eta_0
        #: *Slippage factor (order 1) for the given RF section*
        self.eta_1 = RFSectionParameters.eta_1
        #: *Slippage factor (order 2) for the given RF section*
        self.eta_2 = RFSectionParameters.eta_2
        #: *Copy of the slippage factor order number (from RFSectionParameters)*                
        self.alpha_order = RFSectionParameters.alpha_order
            
        #: *Beta ratio*  :math:`: \quad \frac{\beta_{n+1}}{\beta_{n}}`  
        self.beta_ratio = self.beta_r[1:] / self.beta_r[0:-1]
        
        #: | *Choice of solver for the drift*
        #: | *Set to 'simple' if only 0th order of slippage factor eta*
        #: | *Set to 'full' if higher orders of slippage factor eta*
        self.solver = solver
        #if self.alpha_order == 1:
        #    self.solver = 'simple'
        
        self.rf_params = RFSectionParameters

        ### Parameters for the Phase Loop
        #: *Copy of the section index (from RFSectionParameters)*        
        self.section_index = RFSectionParameters.section_index
        
        #: *Design RF frequency of the main RF system in the station*        
        self.omega_RF = 2.*np.pi*self.beta_r*c*self.harmonic[0]/ \
                        (RFSectionParameters.ring_cirumference)
        
        #: *Phase Loop class*                
        self.PL = PhaseLoop         
        
                   
    def kick(self, beam):
        '''
        *The Kick represents the kick(s) by an RF station at a certain position 
        of the ring. The kicks are summed over the different harmonic RF systems 
        in the station. The cavity phase can be shifted by the user via phi_offset.
        The increment in energy is given by the discrete equation of motion:*
        
        .. math::
            \Delta E_{n+1} = \Delta E_n + \sum_{j=0}^{n_{RF}}{V_{j,n}\,\sin{\\left(h_{j,n}\,\\theta + \phi_{j,n}\\right)}}
            
        '''

        for i in range(self.n_rf):
            beam.dE += self.voltage[i,self.counter[0]] * \
                       np.sin(self.harmonic[i,self.counter[0]] * 
                              beam.theta + self.phi_offset[i,self.counter[0]])
   
    
    def kick_acceleration(self, beam):
        '''
        *KickAcceleration gives a single accelerating kick to the bunch. 
        The accelerating kick is defined by the change in the design momentum 
        (synchronous momentum). 
        The acceleration is assumed to be distributed over the length of the 
        RF station, so the average beta is used in the calculation of the kick.
        An extra increment in the equation of motion with respect to the Kick
        object is given by:*
        
        .. math::
            \Delta E_{n+1} = \Delta E_n + <\\beta> \Delta p_{n\\rightarrow n+1}
            
        '''
        
        beam.dE += self.acceleration_kick[self.counter[0]]

        
    def drift(self, beam):
        '''
        *The drift updates the longitudinal coordinate of the particle after 
        applying the energy kick. The two options of tracking are: full, 
        corresponding to the cases where beta the slippage factor may be of 
        higher orders; and simple, where the slippage factor is of order 0 (the
        code is then faster).*
        
        *Corresponding to the equations:*
        
        .. math::
            \\theta_{n+1} = \\frac{\\beta_{n+1}}{\\beta_n}\\theta_n + 2\\pi\\left(\\frac{1}{1 - \\eta\\delta_n} - 1\\right)\\frac{L}{C} \quad \\text{(full)}
            
        .. math::
            \\approx> \\theta_{n+1} = \\frac{\\beta_{n+1}}{\\beta_n}\\theta_n + 2\\pi\\eta_0\\delta_n\\frac{L}{C} \quad \\text{(simple)}
        
        '''
        
        # Determine frequency correction from feedback loops
        if self.PL == None:
            # No Phase Loop, no Radial Loop
            omega_r1 = self.beta_ratio[self.counter[0]]
            omega_r2 = 0.
            omega_r3 = 1.
            
        else:
            self.PL.track(beam, self)
            # Sum up corrections from previous and current time step
            # Sum up corrections from PL, RL, etc. here
            corr_next = self.PL.domega_RF_next
            corr_prev = self.PL.domega_RF_prev
               
            omega_r1 = (self.omega_RF[self.counter[0]+1] - corr_next) / \
                       (self.omega_RF[self.counter[0]]- corr_prev)
            omega_r2 = - corr_next/self.omega_RF[self.counter[0]+1]
            omega_r3 = 1. + omega_r2

       
        # Choose solver
        if self.solver == 'full': 
           
            beam.theta = omega_r1*beam.theta + 2*np.pi*self.length_ratio* \
                (omega_r3*(1/(1 - self.rf_params.eta_tracking(self.counter[0]+1, beam.delta) 
                *beam.delta) - 1) + omega_r2)            
                                            
        elif self.solver == 'simple':

            beam.theta = omega_r1*beam.theta + 2*np.pi*self.length_ratio* \
                (omega_r3*self.eta_0[self.counter[0]+1]*beam.delta + omega_r2)
                         
        else:
            raise RuntimeError("ERROR: Choice of longitudinal solver not \
                               recognized! Aborting...")
                
                
    def track(self, beam):
        '''
        | *Tracking method for the section, applies the equations in this order:*
        | *kick -> kick_acceleration -> drift*
        |
        | *Updates the relativistic information of the beam.*
        '''
        
        self.kick(beam)
        self.kick_acceleration(beam)
        self.drift(beam)
        
        self.counter[0] += 1
        
        # Updating the beam synchronous momentum etc.
        beam.beta_r = self.beta_r[self.counter[0]]
        beam.gamma_r = self.gamma_r[self.counter[0]]
        beam.energy = self.energy[self.counter[0]]
        beam.momentum = self.momentum[self.counter[0]]
      


class LinearMap(object):
    '''
    Linear Map represented by a Courant-Snyder transportation matrix.
    Qs is forced to be constant.
    '''

    def __init__(self, GeneralParameters, Qs):
        
        #: *Copy of the relativistic beta (from GeneralParameters)*
        self.beta_r = GeneralParameters.beta_r[0,0]        
        #: *Copy of the ring circumference (from GeneralParameters)*
        self.ring_circumference = GeneralParameters.ring_circumference        
        #: *Copy of the 0th order slippage factor (from GeneralParameters)*
        self.eta = GeneralParameters.eta0[0,0]
        
        #: *Synchrotron tune (constant)*
        self.Qs = Qs
        
        #: *Copy of the revolution angular frequency (from GeneralParameters)*
        self.omega_0 = GeneralParameters.omega_rev[0]
        
        #: *Synchrotron angular frequency in [rad/s]* 
        #: :math:`: \quad \omega_s = Q_s \omega_0`
        self.omega_s = self.Qs * self.omega_0
        
        self.dQs = 2 * np.pi * self.Qs
        self.cosdQs = np.cos(self.dQs)
        self.sindQs = np.sin(self.dQs)
        

    def track(self, beam):

        z0 = beam.z
        delta0 = beam.delta

        beam.z = z0 * self.cosdQs - self.eta * c / self.omega_s * delta0 * self.sindQs
        beam.delta = delta0 * self.cosdQs + self.omega_s / self.eta / c * z0 * self.sindQs
        
        

