
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''

**Utilities to calculate Hamiltonian, separatrix, total voltage for the full ring.**

:Authors: **Danilo Quartullo**, **Helga Timko**, **Alexandre Lasheen**
'''


from __future__ import division
import warnings
import numpy as np
import copy
from scipy.constants import c
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt


def synchrotron_frequency_distribution(Beam, FullRingAndRF, main_harmonic_option = 'lowest_freq', 
                                 lineDensityInput = None, TotalInducedVoltage = None, smoothOption = None):
    '''
    *Function to compute the frequency distribution of a distribution for a certain
    RF system and optional intensity effects. The potential well (and induced
    potential) are not updated by this function, thus it has to be called* 
    **after** *the potential well (and induced potential) generation.*
    
    *If used with induced potential, be careful that noise can be an issue. An
    analytical line density can be inputed by inputing the output of the 
    matched_from_line_density function output in the longitudinal_distributions 
    module.*
    
    *A smoothing function is included (running mean) in order to smooth
    noise and numerical errors due to linear interpolation, the user can input the 
    number of pixels to smooth with smoothOption = N.*
    
    *The particle distribution of synchrotron frequencies in the beam are also
    outputed. Would you like to have this result for an analytical beam profile,
    you can have the particle distribution with or without intensity effects.
    To have it without intensity effects, please pass the option lineDensityInput 
    with the following parameters:*
    
    lineDensityInput = [theta_array, line_density]
    or
    lineDensityInput = beam_generation_output[1] (generated without intensity effects)
    
    * With intensity effects you can use the TotalInducedVoltage option by passing
    the following parameters:*
    
    TotalInducedVoltage = beam_generation_output[1]
     
    *with beam_generation_output being the output of the 
    matched_from_line_density and matched_from_distribution_density functions 
    in the distribution module.*
    '''
    
    # Initialize variables depending on the accelerator parameters
    slippage_factor = FullRingAndRF.RingAndRFSection_list[0].eta_0[0]
    eom_factor_dE = (np.pi * abs(slippage_factor) * c) / \
                    (FullRingAndRF.ring_circumference * Beam.beta * Beam.energy)
    eom_factor_potential = np.sign(FullRingAndRF.RingAndRFSection_list[0].eta_0[0]) * (Beam.charge*Beam.beta * c) / (FullRingAndRF.ring_circumference)
     
    # Generate potential well
    n_points_potential = int(1e4)
    FullRingAndRF.potential_well_generation(n_points = n_points_potential, 
                                            theta_margin_percent = 0.05, 
                                            main_harmonic_option = main_harmonic_option)
    potential_well_array = FullRingAndRF.potential_well
    theta_coord_array = FullRingAndRF.potential_well_coordinates
    
    induced_potential_final = 0
    
    # Calculating the induced potential
    if TotalInducedVoltage is not None and lineDensityInput is not None:
        raise RuntimeError('The synchrotron_frequency_distribution cannot take \
                            both lineDensity and TotalInducedVoltage options, \
                            please choose one or the other (the line density \
                            with intensity effects is already taken into \
                            account with the TotalInducedVoltage option !)')
    
    if TotalInducedVoltage is not None:
        
        induced_voltage_object = copy.deepcopy(TotalInducedVoltage)
        
        induced_voltage = induced_voltage_object.induced_voltage
        theta_induced_voltage = TotalInducedVoltage.slices.bin_centers / (Beam.ring_radius/(Beam.beta*c))
        
        # Computing induced potential
        induced_potential = - eom_factor_potential * np.insert(cumtrapz(induced_voltage, dx=theta_induced_voltage[1] - theta_induced_voltage[0]),0,0)
        
        # Interpolating the potential well
        induced_potential_final = np.interp(theta_coord_array, theta_induced_voltage, induced_potential)
        
        theta_line_density = np.array(theta_induced_voltage)
        line_density_input = TotalInducedVoltage.slices.n_macroparticles
        
    if lineDensityInput is not None:
        theta_line_density = lineDensityInput[0]
        line_density_input = lineDensityInput[1]
                                
    # Induced voltage contribution
    total_potential = potential_well_array + induced_potential_final
        
    # Process the potential well in order to take a frame around the separatrix
    theta_coord_sep, potential_well_sep = potential_well_cut(theta_coord_array, total_potential)
        
    line_density = np.interp(theta_coord_sep, theta_line_density, line_density_input)
        
    potential_well_sep = potential_well_sep - np.min(potential_well_sep)
    synchronous_phase_index = np.where(potential_well_sep == np.min(potential_well_sep))[0]
    theta_resolution = theta_coord_sep[1] - theta_coord_sep[0] 
    
    # Computing the action J by integrating the dE trajectories
    J_array_dE0 = np.zeros(len(potential_well_sep))
     
    warnings.filterwarnings("ignore")

    for i in range(0, len(potential_well_sep)):
        dE_trajectory = np.sqrt((potential_well_sep[i] - potential_well_sep)/eom_factor_dE)
        dE_trajectory[np.isnan(dE_trajectory)] = 0
        
        # Careful: Action is integrated over theta
        J_array_dE0[i] = 2 / (2*np.pi) * np.trapz(dE_trajectory, dx=theta_resolution)
             
    warnings.filterwarnings("default")

    # Computing the sync_freq_distribution (if to handle cases where maximum is in 2 consecutive points)
    if len(synchronous_phase_index) > 1:
        H_array_left = potential_well_sep[0:synchronous_phase_index[0]+1]
        H_array_right = potential_well_sep[synchronous_phase_index[1]:]
        J_array_left = J_array_dE0[0:synchronous_phase_index[0]+1]
        J_array_right = J_array_dE0[synchronous_phase_index[1]:]
        delta_theta_left = theta_coord_sep[0:synchronous_phase_index[0]+1]
        delta_theta_right = theta_coord_sep[synchronous_phase_index[1]:]
        synchronous_theta = np.mean(theta_coord_sep[synchronous_phase_index])
        line_density_left = line_density[0:synchronous_phase_index[0]+1]
        line_density_right = line_density[synchronous_phase_index[1]:]
    else:
        H_array_left = potential_well_sep[0:synchronous_phase_index[0]+1]
        H_array_right = potential_well_sep[synchronous_phase_index[0]:]   
        J_array_left = J_array_dE0[0:synchronous_phase_index[0]+1]
        J_array_right = J_array_dE0[synchronous_phase_index[0]:]   
        delta_theta_left = theta_coord_sep[0:synchronous_phase_index[0]+1]
        delta_theta_right = theta_coord_sep[synchronous_phase_index[0]:]
        synchronous_theta = theta_coord_sep[synchronous_phase_index]
        line_density_left = line_density[0:synchronous_phase_index[0]+1]
        line_density_right = line_density[synchronous_phase_index[0]:]
        
    delta_theta_left = delta_theta_left[-1] - delta_theta_left
    delta_theta_right = delta_theta_right - delta_theta_right[0]
           
    if smoothOption is not None:
        H_array_left = np.convolve(H_array_left, np.ones(smoothOption)/smoothOption, mode='valid')
        J_array_left = np.convolve(J_array_left, np.ones(smoothOption)/smoothOption, mode='valid')
        H_array_right = np.convolve(H_array_right, np.ones(smoothOption)/smoothOption, mode='valid')
        J_array_right = np.convolve(J_array_right, np.ones(smoothOption)/smoothOption, mode='valid')
        delta_theta_left_before_smooth = delta_theta_left
        delta_theta_right_before_smooth = delta_theta_right
        delta_theta_left = (delta_theta_left + (smoothOption-1) * (delta_theta_left[1] - delta_theta_left[0])/2)[0:len(delta_theta_left)-smoothOption+1]
        delta_theta_right = (delta_theta_right + (smoothOption-1) * (delta_theta_right[1] - delta_theta_right[0])/2)[0:len(delta_theta_right)-smoothOption+1]
    
    # Taking only the centers, because of the derivative fs= dH/dJ / (2*pi)
    delta_theta_left_final = (delta_theta_left + (delta_theta_left[1] - delta_theta_left[0])/2)[0:-1]
    delta_theta_left = np.fliplr([delta_theta_left])[0]
    delta_theta_left_final = np.fliplr([delta_theta_left_final])[0]
    delta_theta_right_final = (delta_theta_right + (delta_theta_right[1] - delta_theta_right[0])/2)[0:-1]
        
    sync_freq_distribution_left = np.diff(H_array_left)/np.diff(J_array_left) / (2*np.pi)
    sync_freq_distribution_left = np.fliplr([sync_freq_distribution_left])[0]
    sync_freq_distribution_right = np.diff(H_array_right)/np.diff(J_array_right) / (2*np.pi)
        
    emittance_array_left = J_array_left * FullRingAndRF.ring_radius / (Beam.beta * c) * (2*np.pi)
    emittance_array_left = np.fliplr([emittance_array_left])[0]
    emittance_array_left = np.interp(delta_theta_left_final, delta_theta_left, emittance_array_left)

    emittance_array_right = J_array_right * FullRingAndRF.ring_radius / (Beam.beta * c) * (2*np.pi)     
    emittance_array_right = np.interp(delta_theta_right_final, delta_theta_right, emittance_array_right)
    
    if smoothOption is not None:
        line_density_left = np.fliplr([line_density_left])[0]
        delta_theta_left_before_smooth = np.fliplr([delta_theta_left_before_smooth])[0]
        line_density_left = np.interp(delta_theta_left_final, delta_theta_left_before_smooth, line_density_left)
        line_density_right = np.interp(delta_theta_right_final, delta_theta_right_before_smooth, line_density_right)
    else:
        line_density_left = np.fliplr([line_density_left])[0]
        line_density_left = np.interp(delta_theta_left_final, delta_theta_left, line_density_left)
        line_density_right = np.interp(delta_theta_right_final, delta_theta_right, line_density_right)
         
    line_density_normalizing_factor = np.sum(line_density_left) + np.sum(line_density_right)
    line_density_left = line_density_left / line_density_normalizing_factor * Beam.n_macroparticles
    line_density_right = line_density_right / line_density_normalizing_factor * Beam.n_macroparticles
    
    particleDistributionFreq_left = np.interp(synchronous_theta - (Beam.dt[Beam.dt<synchronous_theta*(Beam.ring_radius/(Beam.beta*c))])/(Beam.ring_radius/(Beam.beta*c)), delta_theta_left_final, sync_freq_distribution_left)
    particleDistributionFreq_right = np.interp((Beam.dt[Beam.dt>synchronous_theta*(Beam.ring_radius/(Beam.beta*c))])/(Beam.ring_radius/(Beam.beta*c)) - synchronous_theta, delta_theta_right_final, sync_freq_distribution_right)
    particleDistributionFreq = np.concatenate(([particleDistributionFreq_left, particleDistributionFreq_right]))   
              
    return [sync_freq_distribution_left, sync_freq_distribution_right], [emittance_array_left, emittance_array_right], [delta_theta_left_final, delta_theta_right_final], [line_density_left, line_density_right], particleDistributionFreq, synchronous_theta


class synchrotron_frequency_tracker(object):
    '''
    *This class can be added to the tracking map to track a certain
    number of particles (defined by the user) and to store the evolution
    of their coordinates in phase space in order to compute their synchrotron
    frequency as a function of their amplitude in theta.*
    
    *As the time step between two turns can change with acceleration, make sure
    that the momentum program is set to be constant when using this function,
    or that beta_rel~1.*
    
    *The user can input the minimum and maximum theta for the theta_coordinate_range
    option as [min, max]. The input n_macroparticles will be generated with
    linear spacing between these values. One can also input the theta_coordinate_range
    as the coordinates of all particles, but the length of the array should 
    match the n_macroparticles value.*
    '''

    def __init__(self, GeneralParameters, n_macroparticles, theta_coordinate_range, FullRingAndRF, 
                 TotalInducedVoltage = None):
        
        #: *Number of macroparticles used in the synchrotron_frequency_tracker method*
        self.n_macroparticles = int(n_macroparticles)
        
        #: *Copy of the input FullRingAndRF object to retrieve the accelerator programs*
        self.FullRingAndRF = copy.deepcopy(FullRingAndRF)
        
        #: *Copy of the input TotalInducedVoltage object to retrieve the intensity effects
        #: (the synchrotron_frequency_tracker particles are not contributing to the
        #: induced voltage).*
        self.TotalInducedVoltage = None
        if TotalInducedVoltage is not None:
            self.TotalInducedVoltage = TotalInducedVoltage
            intensity = TotalInducedVoltage.slices.Beam.intensity
        else:
            intensity = 0.
            
        from beams.beams import Beam
        #: *Beam object containing the same physical information as the real beam,
        #: but containing only the coordinates of the particles for which the 
        #: synchrotron frequency are computed.*
        self.Beam = Beam(GeneralParameters, n_macroparticles, intensity)
        
        # Generating the distribution from the user input
        if len(theta_coordinate_range) == 2:
            self.Beam.dt = np.linspace(theta_coordinate_range[0], theta_coordinate_range[1], n_macroparticles) * (self.Beam.ring_radius/(self.Beam.beta*c))
        else:
            if len(theta_coordinate_range) != n_macroparticles:
                raise RuntimeError('The input n_macroparticles does not match with the length of the theta_coordinates')
            else:
                self.Beam.dt = np.array(theta_coordinate_range) * (self.Beam.ring_radius/(self.Beam.beta*c))
                        
        self.Beam.dE = np.zeros(int(n_macroparticles))
 
        for RFsection in self.FullRingAndRF.RingAndRFSection_list:
            RFsection.beam = self.Beam
        
        #: *Revolution period in [s]*
        self.timeStep = GeneralParameters.t_rev[0]
        
        #: *Number of turns of the simulation (+1 to include the input parameters)*
        self.nTurns = GeneralParameters.n_turns+1
        
        #: *Saving the theta coordinates of the particles while tracking*
        self.theta_save = np.zeros((self.nTurns, int(n_macroparticles)))
        
        #: *Saving the dE coordinates of the particles while tracking*
        self.dE_save = np.zeros((self.nTurns, int(n_macroparticles)))
        
        #: *Tracking counter*
        self.counter = 0
          
        # The first save coordinates are the input coordinates      
        self.theta_save[self.counter] = self.Beam.dt / (self.Beam.ring_radius/(self.Beam.beta*c))
        self.dE_save[self.counter] = self.Beam.dE
    
            
    def track(self):
        '''
        *Method to track the particles with or without intensity effects.*
        '''
    
        self.FullRingAndRF.track()
        
        if self.TotalInducedVoltage is not None:
            self.TotalInducedVoltage.track_ghosts_particles(self.Beam)
            
        self.counter = self.counter + 1
        
        self.theta_save[self.counter] = self.Beam.dt / (self.Beam.ring_radius/(self.Beam.beta*c))
        self.dE_save[self.counter] = self.Beam.dE
        
            
    def frequency_calculation(self, n_sampling=100000, start_turn = None, end_turn = None):
        '''
        *Method to compute the fft of the particle oscillations in theta and dE
        to obtain their synchrotron frequencies. The particles for which
        the amplitude of oscillations is extending the minimum and maximum
        theta from user input are considered to be lost and their synchrotron
        frequencies are not calculated.*
        '''
        
        n_sampling = int(n_sampling)
        
        #: *Saving the synchrotron frequency from the theta oscillations for each particle*
        self.frequency_theta_save = np.zeros(int(self.n_macroparticles))
        
        #: *Saving the synchrotron frequency from the dE oscillations for each particle*
        self.frequency_dE_save = np.zeros(int(self.n_macroparticles))
        
        #: *Saving the maximum of oscillations in theta for each particle 
        #: (theta amplitude on the right side of the bunch)*
        self.max_theta_save = np.zeros(int(self.n_macroparticles))
        
        #: *Saving the minimum of oscillations in theta for each particle 
        #: (theta amplitude on the left side of the bunch)*
        self.min_theta_save = np.zeros(int(self.n_macroparticles))
        
        # Maximum theta for which the particles are considered to be lost        
        max_theta_range = np.max(self.theta_save[0,:])
        
        # Minimum theta for which the particles are considered to be lost
        min_theta_range = np.min(self.theta_save[0,:])
        
        #: *Frequency array for the synchrotron frequency distribution*
        self.frequency_array = np.fft.rfftfreq(n_sampling, self.timeStep)
        
        if start_turn is None:
            start_turn = 0
        
        if end_turn is None:
            end_turn = self.nTurns + 1
        
        # Computing the synchrotron frequency of each particle from the maximum
        # peak of the FFT.
        for indexParticle in range(0, self.n_macroparticles):
            self.max_theta_save[indexParticle] = np.max(self.theta_save[start_turn:end_turn,indexParticle])
            self.min_theta_save[indexParticle] = np.min(self.theta_save[start_turn:end_turn,indexParticle])
            
            if (self.max_theta_save[indexParticle]<max_theta_range) and (self.min_theta_save[indexParticle]>min_theta_range):
            
                theta_save_fft = abs(np.fft.rfft(self.theta_save[start_turn:end_turn,indexParticle] - np.mean(self.theta_save[start_turn:end_turn,indexParticle]), n_sampling))
                dE_save_fft = abs(np.fft.rfft(self.dE_save[start_turn:end_turn,indexParticle] - np.mean(self.dE_save[start_turn:end_turn,indexParticle]), n_sampling))
        
                self.frequency_theta_save[indexParticle] = self.frequency_array[theta_save_fft==np.max(theta_save_fft)]
                self.frequency_dE_save[indexParticle] = self.frequency_array[dE_save_fft==np.max(dE_save_fft)]



def total_voltage(RFsection_list, harmonic = 'first'):
    """
    Total voltage from all the RF stations and systems in the ring.
    To be generalized.
    """
    
    n_sections = len(RFsection_list)
    
    #: *Sums up only the voltage of the first harmonic RF, 
    #: taking into account relative phases*
    if harmonic == 'first':
        Vcos = RFsection_list[0].voltage[0]*np.cos(RFsection_list[0].phi_RF[0])
        Vsin = RFsection_list[0].voltage[0]*np.sin(RFsection_list[0].phi_RF[0])
        if n_sections > 1:
            for i in range(1, n_sections):
                print RFsection_list[i].voltage[0]
                Vcos += RFsection_list[i].voltage[0]*np.cos(RFsection_list[i].phi_RF[0])
                Vsin += RFsection_list[i].voltage[0]*np.sin(RFsection_list[i].phi_RF[0])
        Vtot = np.sqrt(Vcos**2 + Vsin**2)
        return Vtot
    
    #: *To be implemented*
    elif harmonic == "all":
        return 0

    else:
        warnings.filterwarnings("once")
        warnings.warn("WARNING: In total_voltage, harmonic choice not recognize!")
    


def hamiltonian(GeneralParameters, RFSectionParameters, Beam, dt, dE, 
                total_voltage = None):
    """Single RF sinusoidal Hamiltonian.
    For the time being, for single RF section only or from total voltage.
    Uses beta, energy averaged over the turn.
    To be generalized."""
     
   
    warnings.filterwarnings("once")
    
    if GeneralParameters.n_sections > 1:
        warnings.warn("WARNING: The Hamiltonian is not yet properly computed for several sections!")
    if RFSectionParameters.n_rf > 1:
        warnings.warn("WARNING: The Hamiltonian will be calculated for the first harmonic only!")

         
    counter = RFSectionParameters.counter[0]
    h0 = RFSectionParameters.harmonic[0,counter]
    if total_voltage == None:
        V0 = RFSectionParameters.voltage[0,counter]
    else: 
        V0 = total_voltage[counter]
    V0 *= RFSectionParameters.charge
    
    c1 = RFSectionParameters.eta_tracking(Beam, counter, dE)*c*np.pi/ \
         (GeneralParameters.ring_circumference*Beam.beta*Beam.energy )
    c2 = c*Beam.beta*V0/(h0*GeneralParameters.ring_circumference)
     
    phi_s = RFSectionParameters.phi_s[counter] 
    phi_b = RFSectionParameters.omega_RF[0,counter]*dt + \
            RFSectionParameters.phi_RF[0,counter] 
    
    eta0 = RFSectionParameters.eta_0[counter]
    
    # Modulo 2 Pi of bunch phase
    if eta0 < 0:
        phi_b = phase_modulo_below_transition(phi_b)
    elif eta0 > 0:
        phi_b = phase_modulo_above_transition(phi_b)    

    return c1 * dE**2 + c2 * (np.cos(phi_b) - np.cos(phi_s) + 
                               (phi_b - phi_s) * np.sin(phi_s))
         
 
 
def separatrix(GeneralParameters, RFSectionParameters, dt, total_voltage = None):
    """Single RF sinusoidal separatrix.
    For the time being, for single RF section only or from total voltage.
    Uses beta, energy averaged over the turn.
    To be generalized."""
 
    warnings.filterwarnings("once")
     
    if GeneralParameters.n_sections > 1:
        warnings.warn("WARNING: The separatrix is not yet properly computed for several sections!")
       
    # Import RF and ring parameters at this moment 
    counter = RFSectionParameters.counter[0]
    voltage = GeneralParameters.charge*RFSectionParameters.voltage[:,counter]
    omega_RF = RFSectionParameters.omega_RF[:,counter]
    phi_RF = RFSectionParameters.phi_RF[:,counter]

    eta0 = RFSectionParameters.eta_0[counter]
    beta_sq = RFSectionParameters.beta[counter]**2     
    energy = RFSectionParameters.energy[counter]

    # Projects time array into the range [-T_RF/2+t_RF, T_RF/2+t_RF]
    # if below transition and into the range [t_RF, t_RF+T_RF] if above transition.
    # T_RF = 2*pi/omega_RF, t_RF = - phi_RF/omega_RF
    if eta0 < 0:
        dt = time_modulo(dt, (phi_RF[0] - np.pi)/omega_RF[0], 
                         2.*np.pi/omega_RF[0])
    elif eta0 > 0:
        dt = time_modulo(dt, phi_RF[0]/omega_RF[0], 2.*np.pi/omega_RF[0])

    
    # Single-harmonic RF system
    if RFSectionParameters.n_rf == 1:
     
        h0 = RFSectionParameters.harmonic[0,counter]
     
        if total_voltage == None:
            V0 = voltage[0]
        else: 
            V0 = total_voltage[counter]
      
        phi_s = RFSectionParameters.phi_s[counter]
        phi_b = omega_RF[0]*dt + phi_RF[0]
          
        separatrix_array = np.sqrt(beta_sq*energy*V0/(np.pi*eta0*h0)* 
                                    (-np.cos(phi_b) - np.cos(phi_s) + 
                                     (np.pi - phi_s - phi_b)*np.sin(phi_s)))

    # Multi-harmonic RF system
    else:
        
        voltage = GeneralParameters.charge*RFSectionParameters.voltage[:,counter]
        omega_RF = RFSectionParameters.omega_RF[:,counter]
        phi_RF = RFSectionParameters.phi_RF[:,counter]     
        try:
            denergy = RFSectionParameters.E_increment[counter]
        except:
            denergy = RFSectionParameters.E_increment[-1]
        T0 = GeneralParameters.t_rev[counter]
        index_voltage = np.min(np.where(voltage>0)[0])
        T_RF0 = 2*np.pi/omega_RF[index_voltage]
        
        # Find unstable fixed point
        
        dt_ufp = np.linspace(-phi_RF[index_voltage]/omega_RF[index_voltage] - T_RF0/1000, 
                             T_RF0 - phi_RF[index_voltage]/omega_RF[index_voltage] + T_RF0/1000, 1002)
        
        if eta0 < 0:
            dt_ufp -= 0.5*T_RF0
        Vtot = np.zeros(len(dt_ufp))
        
        # Construct waveform
        for i in range(RFSectionParameters.n_rf):
            temp = np.sin(omega_RF[i]*dt_ufp + phi_RF[i])
            Vtot += voltage[i]*temp
        Vtot -= denergy
        
        # Find zero crossings
        zero_crossings = np.where(np.diff(np.sign(Vtot)))[0]
        
        # Interpolate UFP
        if eta0 < 0:
            i = -1
            ind  = zero_crossings[i]
            while (Vtot[ind+1] -  Vtot[ind]) > 0:
                i -= 1
                ind = zero_crossings[i]
        else:
            i = 0
            ind = zero_crossings[i]
            while (Vtot[ind+1] -  Vtot[ind]) < 0:
                i += 1
                ind = zero_crossings[i]
        dt_ufp = dt_ufp[ind] + Vtot[ind]/(Vtot[ind] - Vtot[ind+1])* \
                 (dt_ufp[ind+1] - dt_ufp[ind])
        
        # Construct separatrix
        Vtot = np.zeros(len(dt))
        for i in range(RFSectionParameters.n_rf):
            Vtot += voltage[i]*(np.cos(omega_RF[i]*dt_ufp + phi_RF[i]) - 
                                np.cos(omega_RF[i]*dt + phi_RF[i]))/omega_RF[i]
    
        separatrix_array = np.sqrt(2*beta_sq*energy/(eta0*T0)* \
                                    (Vtot + denergy*(dt_ufp - dt)))
         
    return separatrix_array
 
 
 
def is_in_separatrix(GeneralParameters, RFSectionParameters, Beam, dt, dE, total_voltage = None):
    """Condition for being inside the separatrix.
    For the time being, for single RF section only or from total voltage.
    Single RF sinusoidal.
    Uses beta, energy averaged over the turn.
    To be generalized."""
     
    warnings.filterwarnings("once")
    
    if GeneralParameters.n_sections > 1:
        warnings.warn("WARNING: is_in_separatrix is not yet properly computed for several sections!")
    if RFSectionParameters.n_rf > 1:
        warnings.warn("WARNING: is_in_separatrix will be calculated for the first harmonic only!")
    
         
    counter = RFSectionParameters.counter[0]
    dt_sep = (np.pi - RFSectionParameters.phi_s[counter] 
              - RFSectionParameters.phi_RF[0,counter])/ \
              RFSectionParameters.omega_RF[0,counter]
     
    Hsep = hamiltonian(GeneralParameters, RFSectionParameters, Beam, dt_sep, 0, 
                       total_voltage = None) 
    isin = np.fabs(hamiltonian(GeneralParameters, RFSectionParameters, Beam, dt, 
                               dE, total_voltage = None)) < np.fabs(Hsep)
     
    return isin
        


def minmax_location(x,f):
    '''
    *Function to locate the minima and maxima of the f(x) numerical function.*
    '''
    
    f_derivative = np.diff(f)
    x_derivative = x[0:-1] + (x[1]-x[0])/2
    f_derivative = np.interp(x, x_derivative,f_derivative)
    
    f_derivative_second = np.diff(f_derivative)
    f_derivative_second = np.interp(x, x_derivative,f_derivative_second)
    
    warnings.filterwarnings("ignore")
    f_derivative_zeros = np.unique(np.append(np.where(f_derivative == 0), np.where(f_derivative[1:]/f_derivative[0:-1] < 0)))
        
    min_x_position = (x[f_derivative_zeros[f_derivative_second[f_derivative_zeros]>0] + 1] + x[f_derivative_zeros[f_derivative_second[f_derivative_zeros]>0]])/2
    max_x_position = (x[f_derivative_zeros[f_derivative_second[f_derivative_zeros]<0] + 1] + x[f_derivative_zeros[f_derivative_second[f_derivative_zeros]<0]])/2
    
    min_values = np.interp(min_x_position, x, f)
    max_values = np.interp(max_x_position, x, f)

    warnings.filterwarnings("default")
                                          
    return [min_x_position, max_x_position], [min_values, max_values]


def potential_well_cut(theta_coord_array, potential_array):
    '''
    *Function to cut the potential well in order to take only the separatrix
    (several cases according to the number of min/max).*
    '''
    
    # Check for the min/max of the potential well
    minmax_positions, minmax_values = minmax_location(theta_coord_array, 
                                                      potential_array)
    min_theta_positions = minmax_positions[0]
    max_theta_positions = minmax_positions[1]
    max_potential_values = minmax_values[1]
    n_minima = len(min_theta_positions)
    n_maxima = len(max_theta_positions)
    
    if n_minima == 0:
        raise RuntimeError('The potential well has no minima...')
    if n_minima > n_maxima and n_maxima == 1:
        raise RuntimeError('The potential well has more minima than maxima, and only one maximum')
    if n_maxima == 0:
        print ('Warning: The maximum of the potential well could not be found... \
                You may reconsider the options to calculate the potential well \
                as the main harmonic is probably not the expected one. \
                You may also increase the percentage of margin to compute \
                the potentiel well. The full potential well will be taken')
    elif n_maxima == 1:
        if min_theta_positions[0] > max_theta_positions[0]:
            saved_indexes = (potential_array < max_potential_values[0]) * \
                            (theta_coord_array > max_theta_positions[0])
            theta_coord_sep = theta_coord_array[saved_indexes]
            potential_well_sep = potential_array[saved_indexes]
            if potential_array[-1] < potential_array[0]:
                raise RuntimeError('The potential well is not well defined. \
                                    You may reconsider the options to calculate \
                                    the potential well as the main harmonic is \
                                    probably not the expected one.')
        else:
            saved_indexes = (potential_array < max_potential_values[0]) * \
                            (theta_coord_array < max_theta_positions[0])
            theta_coord_sep = theta_coord_array[saved_indexes]
            potential_well_sep = potential_array[saved_indexes]
            if potential_array[-1] > potential_array[0]:
                raise RuntimeError('The potential well is not well defined. \
                                    You may reconsider the options to calculate \
                                    the potential well as the main harmonic is \
                                    probably not the expected one.')
    elif n_maxima == 2:
        lower_maximum_value = np.min(max_potential_values)
        higher_maximum_value = np.max(max_potential_values)
        lower_maximum_theta = max_theta_positions[max_potential_values == lower_maximum_value]
        higher_maximum_theta = max_theta_positions[max_potential_values == higher_maximum_value]
        if len(lower_maximum_theta)==2:
            saved_indexes = (potential_array < lower_maximum_value) * \
                            (theta_coord_array > lower_maximum_theta[0]) * \
                            (theta_coord_array < lower_maximum_theta[1])
            theta_coord_sep = theta_coord_array[saved_indexes]
            potential_well_sep = potential_array[saved_indexes]
        elif min_theta_positions[0] > lower_maximum_theta:
            saved_indexes = (potential_array < lower_maximum_value) * \
                            (theta_coord_array > lower_maximum_theta) * \
                            (theta_coord_array < higher_maximum_theta)
            theta_coord_sep = theta_coord_array[saved_indexes]
            potential_well_sep = potential_array[saved_indexes]
        else:
            saved_indexes = (potential_array < lower_maximum_value) * \
                            (theta_coord_array < lower_maximum_theta) * \
                            (theta_coord_array > higher_maximum_theta)
            theta_coord_sep = theta_coord_array[saved_indexes]
            potential_well_sep = potential_array[saved_indexes]
    elif n_maxima > 2:
        left_max_theta = np.min(max_theta_positions)
        right_max_theta = np.max(max_theta_positions)
        left_max_value = max_potential_values[max_theta_positions==left_max_theta]
        right_max_value = max_potential_values[max_theta_positions==right_max_theta]
        separatrix_value = np.min([left_max_value, right_max_value])
        saved_indexes = (theta_coord_array > left_max_theta) * (theta_coord_array < right_max_theta) * (potential_array < separatrix_value)
        theta_coord_sep = theta_coord_array[saved_indexes]
        potential_well_sep = potential_array[saved_indexes]
        
        
    return theta_coord_sep, potential_well_sep
        
        

def phase_modulo_above_transition(phi):
    '''
    *Projects a phase array into the range -Pi/2 to +3*Pi/2.*
    '''
    
    return phi - 2.*np.pi*np.floor(phi/(2.*np.pi))


 
def phase_modulo_below_transition(phi):
    '''
    *Projects a phase array into the range -Pi/2 to +3*Pi/2.*
    '''
    
    return phi - 2.*np.pi*(np.floor(phi/(2.*np.pi) + 0.5))
        


def time_modulo(dt, dt_offset, T):
    '''
    *Returns dt projected onto the desired interval.*
    '''
    
    return dt - T*np.floor((dt + dt_offset)/T)


 
        


