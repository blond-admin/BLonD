
# Copyright 2014 CERN. This software is distributed under the
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
                    (FullRingAndRF.ring_circumference * Beam.beta_r * Beam.energy)
    eom_factor_potential = np.sign(FullRingAndRF.RingAndRFSection_list[0].eta_0[0]) * (Beam.beta_r * c) / (FullRingAndRF.ring_circumference)
     
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
                            account yith the TotalInducedVoltage option !)')
    
    if TotalInducedVoltage is not None:
        
        induced_voltage_object = copy.deepcopy(TotalInducedVoltage)
        
        induced_voltage = induced_voltage_object.induced_voltage
        theta_induced_voltage = TotalInducedVoltage.slices.convert_coordinates(TotalInducedVoltage.slices.bins_centers, TotalInducedVoltage.slices.slicing_coord, 'theta')
        
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
        
    emittance_array_left = J_array_left * FullRingAndRF.ring_radius / (Beam.beta_r * c) * (2*np.pi)
    emittance_array_left = np.fliplr([emittance_array_left])[0]
    emittance_array_left = np.interp(delta_theta_left_final, delta_theta_left, emittance_array_left)

    emittance_array_right = J_array_right * FullRingAndRF.ring_radius / (Beam.beta_r * c) * (2*np.pi)     
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
    
    particleDistributionFreq_left = np.interp(synchronous_theta - Beam.theta[Beam.theta<synchronous_theta], delta_theta_left_final, sync_freq_distribution_left)
    particleDistributionFreq_right = np.interp(Beam.theta[Beam.theta>synchronous_theta] - synchronous_theta, delta_theta_right_final, sync_freq_distribution_right)
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
        self.n_macroparticles = n_macroparticles
        
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
            self.Beam.theta = np.linspace(theta_coordinate_range[0], theta_coordinate_range[1], n_macroparticles)
        else:
            if len(theta_coordinate_range) != n_macroparticles:
                raise RuntimeError('The input n_macroparticles does not match with the length of the theta_coordinates')
            else:
                self.Beam.theta = np.array(theta_coordinate_range)
                        
        self.Beam.dE = np.zeros(int(n_macroparticles))
        
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
        self.theta_save[self.counter] = self.Beam.theta
        self.dE_save[self.counter] = self.Beam.dE
    
            
    def track(self, Beam):
        '''
        *Method to track the particles with or without intensity effects.*
        '''
    
        self.FullRingAndRF.track(self.Beam)
        
        if self.TotalInducedVoltage is not None:
            self.TotalInducedVoltage.track(self.Beam, no_update_induced_voltage = True)
            
        self.counter = self.counter + 1
        
        self.theta_save[self.counter] = self.Beam.theta
        self.dE_save[self.counter] = self.Beam.dE
        
            
    def frequency_calculation(self, n_sampling=100000):
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
        
        # Computing the synchrotron frequency of each particle from the maximum
        # peak of the FFT.
        for indexParticle in range(0, self.n_macroparticles):
            self.max_theta_save[indexParticle] = np.max(self.theta_save[:,indexParticle])
            self.min_theta_save[indexParticle] = np.min(self.theta_save[:,indexParticle])
            
            if (self.max_theta_save[indexParticle]<max_theta_range) and (self.min_theta_save[indexParticle]>min_theta_range):
            
                theta_save_fft = abs(np.fft.rfft(self.theta_save[:,indexParticle] - np.mean(self.theta_save[:,indexParticle]), n_sampling))
                dE_save_fft = abs(np.fft.rfft(self.dE_save[:,indexParticle] - np.mean(self.dE_save[:,indexParticle]), n_sampling))
        
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
        Vcos = RFsection_list[0].voltage[0]*np.cos(RFsection_list[0].phi_offset[0])
        Vsin = RFsection_list[0].voltage[0]*np.sin(RFsection_list[0].phi_offset[0])
        if n_sections > 1:
            for i in range(1, n_sections):
                print RFsection_list[i].voltage[0]
                Vcos += RFsection_list[i].voltage[0]*np.cos(RFsection_list[i].phi_offset[0])
                Vsin += RFsection_list[i].voltage[0]*np.sin(RFsection_list[i].phi_offset[0])
        Vtot = np.sqrt(Vcos**2 + Vsin**2)
        return Vtot
    
    #: *To be implemented*
    elif harmonic == "all":
        return 0

    else:
        warnings.filterwarnings("once")
        warnings.warn("WARNING: In total_voltage, harmonic choice not recognize!")
    


def hamiltonian(GeneralParameters, RFSectionParameters, theta, dE, delta, 
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
    
    c1 = RFSectionParameters.eta_tracking(counter, delta) * c * np.pi / (GeneralParameters.ring_circumference * 
         RFSectionParameters.beta_r[counter] * RFSectionParameters.energy[counter] )
    c2 = c * RFSectionParameters.beta_r[counter] * V0 / (h0 * GeneralParameters.ring_circumference)
     
    phi_s = RFSectionParameters.phi_s[counter]  
    
    return c1 * dE**2 + c2 * (np.cos(h0 * theta) - np.cos(phi_s) + 
                               (h0 * theta - phi_s) * np.sin(phi_s))
         
 
 
def separatrix(GeneralParameters, RFSectionParameters, theta, total_voltage = None):
    """Single RF sinusoidal separatrix.
    For the time being, for single RF section only or from total voltage.
    Uses beta, energy averaged over the turn.
    To be generalized."""
 
    warnings.filterwarnings("once")
     
    if GeneralParameters.n_sections > 1:
        warnings.warn("WARNING: The separatrix is not yet properly computed for several sections!")
    if RFSectionParameters.n_rf > 1:
        warnings.warn("WARNING: The separatrix will be calculated for the first harmonic only!")    

     
     
    counter = RFSectionParameters.counter[0]
    h0 = RFSectionParameters.harmonic[0,counter]

    if total_voltage == None:
        V0 = RFSectionParameters.voltage[0,counter]
    else: 
        V0 = total_voltage[counter]
 
    phi_s = RFSectionParameters.phi_s[counter]
     
    beta = RFSectionParameters.beta_r[counter]
     
    energy = RFSectionParameters.energy[counter]
     
    eta0 = RFSectionParameters.eta_0[counter]
      
    separatrix_array = np.sqrt(beta**2 * energy * V0 / (np.pi * eta0 * h0) * 
                       (-np.cos(h0 * theta) - np.cos(phi_s) + 
                       (np.pi - phi_s - h0 * theta) * np.sin(phi_s)))
         
    return separatrix_array
 
 
 
def is_in_separatrix(GeneralParameters, RFSectionParameters, theta, dE, delta, total_voltage = None):
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
    h0 = RFSectionParameters.harmonic[0,counter]     
    phi_s = RFSectionParameters.phi_s[counter] 
     
    Hsep = hamiltonian(GeneralParameters, RFSectionParameters, (np.pi - phi_s) / h0, 0, 0, total_voltage = None) 
    isin = np.fabs(hamiltonian(GeneralParameters, RFSectionParameters, theta, dE, delta, total_voltage = None)) < np.fabs(Hsep)
     
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
        if min_theta_positions[0] > lower_maximum_theta:
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
#             raise RuntimeError('Work in progress, case to be included in the future...')
        left_max_theta = np.min(max_theta_positions)
        right_max_theta = np.max(max_theta_positions)
        saved_indexes = (theta_coord_array > left_max_theta) * (theta_coord_array < right_max_theta)
        theta_coord_sep = theta_coord_array[saved_indexes]
        potential_well_sep = potential_array[saved_indexes]
        
        
    return theta_coord_sep, potential_well_sep
        
        

        
        
