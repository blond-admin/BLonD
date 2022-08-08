
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


from __future__ import division, print_function
from builtins import range, object
import warnings
import numpy as np
import copy
from scipy.constants import c
from scipy.integrate import cumtrapz
from ..utils import bmath as bm


def synchrotron_frequency_distribution(Beam, FullRingAndRF, main_harmonic_option = 'lowest_freq', 
                                 turn = 0, TotalInducedVoltage = None, smoothOption = None):
    '''
    *Function to compute the frequency distribution of a distribution for a certain
    RF system and optional intensity effects. The potential well (and induced
    potential) are not updated by this function, thus it has to be called* 
    **after** *the potential well (and induced potential) generation.*
    
    *If used with induced potential, be careful that noise can be an issue. An
    analytical line density can be inputed by using the TotalInducedVoltage 
    option and passing the following parameters:*
    
    TotalInducedVoltage = beam_generation_output[1]
     
    *with beam_generation_output being the output of the 
    matched_from_line_density and matched_from_distribution_density functions 
    in the distribution module.*
    
    *A smoothing function is included (running mean) in order to smooth
    noise and numerical errors due to linear interpolation, the user can input the 
    number of pixels to smooth with smoothOption = N.*
    
    *The particle distribution in synchrotron frequencies of the beam is also
    outputed.*
    '''
    
    # Initialize variables depending on the accelerator parameters
    slippage_factor = FullRingAndRF.RingAndRFSection_list[0].rf_params.eta_0[0]
                        
    eom_factor_dE = abs(slippage_factor) / (2*Beam.beta**2. * Beam.energy)
    eom_factor_potential = np.sign(slippage_factor) * Beam.Particle.charge / (FullRingAndRF.RingAndRFSection_list[0].rf_params.t_rev[0])

    # Generate potential well
    n_points_potential = int(1e4)
    FullRingAndRF.potential_well_generation(n_points = n_points_potential, 
                                            turn = turn, dt_margin_percent = 0.05, 
                                            main_harmonic_option = main_harmonic_option)
    potential_well_array = FullRingAndRF.potential_well
    time_coord_array = FullRingAndRF.potential_well_coordinates
    
    induced_potential_final = 0
    
    # Calculating the induced potential    
    if TotalInducedVoltage is not None:
        
        induced_voltage_object = copy.deepcopy(TotalInducedVoltage)
        
        induced_voltage = induced_voltage_object.induced_voltage
        time_induced_voltage = TotalInducedVoltage.profile.bin_centers
        
        # Computing induced potential
        induced_potential = - eom_factor_potential * np.insert(cumtrapz(induced_voltage, dx=time_induced_voltage[1] - time_induced_voltage[0]),0,0)
        
        # Interpolating the potential well
        induced_potential_final = np.interp(time_coord_array, time_induced_voltage, induced_potential)
                                    
    # Induced voltage contribution
    total_potential = potential_well_array + induced_potential_final
    
    # Process the potential well in order to take a frame around the separatrix
    time_coord_sep, potential_well_sep = potential_well_cut(time_coord_array, total_potential)
    
    potential_well_sep = potential_well_sep - np.min(potential_well_sep)
    synchronous_phase_index = np.where(potential_well_sep == np.min(potential_well_sep))[0]
    
    # Computing the action J by integrating the dE trajectories
    J_array_dE0 = np.zeros(len(potential_well_sep))
     
    warnings.filterwarnings("ignore")

    for i in range(0, len(potential_well_sep)):
        # Find left and right time coordinates for a given hamiltonian 
        # value
        time_indexes = np.where(potential_well_sep <= 
                                potential_well_sep[i])[0]
        left_time = time_coord_sep[np.max((0,time_indexes[0]))]
        right_time = time_coord_sep[np.min((time_indexes[-1],
                                                   len(time_coord_sep)-1))]
        # Potential well calculation with high resolution in that frame
        time_potential_high_res = np.linspace(float(left_time), float(right_time),
                                              n_points_potential)
        FullRingAndRF.potential_well_generation(
                                 n_points=n_points_potential,
                                 time_array=time_potential_high_res,
                                 main_harmonic_option=main_harmonic_option)
        pot_well_high_res = FullRingAndRF.potential_well   
        if TotalInducedVoltage is not None:
            pot_well_high_res += np.interp(time_potential_high_res,
                                       time_induced_voltage, induced_potential)
            pot_well_high_res -= pot_well_high_res.min()
        # Integration to calculate action
        dE_trajectory = np.sqrt((potential_well_sep[i] -
                                 pot_well_high_res) / eom_factor_dE)
        dE_trajectory[np.isnan(dE_trajectory)] = 0
        J_array_dE0[i] = 1 / np.pi * np.trapz(dE_trajectory,
                dx=time_potential_high_res[1] - time_potential_high_res[0])
    
    warnings.filterwarnings("default")
    
    # Computing the sync_freq_distribution (if to handle cases where maximum is in 2 consecutive points)
    if len(synchronous_phase_index) > 1:
        H_array_left = potential_well_sep[0:synchronous_phase_index[0]+1]
        H_array_right = potential_well_sep[synchronous_phase_index[1]:]
        J_array_left = J_array_dE0[0:synchronous_phase_index[0]+1]
        J_array_right = J_array_dE0[synchronous_phase_index[1]:]
        delta_time_left = time_coord_sep[0:synchronous_phase_index[0]+1]
        delta_time_right = time_coord_sep[synchronous_phase_index[1]:]
        synchronous_time = np.mean(time_coord_sep[synchronous_phase_index])
    else:
        H_array_left = potential_well_sep[0:synchronous_phase_index[0]+1]
        H_array_right = potential_well_sep[synchronous_phase_index[0]:]   
        J_array_left = J_array_dE0[0:synchronous_phase_index[0]+1]
        J_array_right = J_array_dE0[synchronous_phase_index[0]:]   
        delta_time_left = time_coord_sep[0:synchronous_phase_index[0]+1]
        delta_time_right = time_coord_sep[synchronous_phase_index[0]:]
        synchronous_time = time_coord_sep[synchronous_phase_index]
    
    delta_time_left = delta_time_left[-1] - delta_time_left
    delta_time_right = delta_time_right - delta_time_right[0]
    
    if smoothOption is not None:
        H_array_left = np.convolve(H_array_left, np.ones(smoothOption)/smoothOption, mode='valid')
        J_array_left = np.convolve(J_array_left, np.ones(smoothOption)/smoothOption, mode='valid')
        H_array_right = np.convolve(H_array_right, np.ones(smoothOption)/smoothOption, mode='valid')
        J_array_right = np.convolve(J_array_right, np.ones(smoothOption)/smoothOption, mode='valid')
        delta_time_left = (delta_time_left + (smoothOption-1) * (delta_time_left[1] - delta_time_left[0])/2)[0:len(delta_time_left)-smoothOption+1]
        delta_time_right = (delta_time_right + (smoothOption-1) * (delta_time_right[1] - delta_time_right[0])/2)[0:len(delta_time_right)-smoothOption+1]
    
    delta_time_left = np.fliplr([delta_time_left])[0]
    
    # Calculation of fs as fs= dH/dJ / (2*pi)
    sync_freq_distribution_left = np.gradient(H_array_left)/np.gradient(J_array_left) / (2*np.pi)
    sync_freq_distribution_left = np.fliplr([sync_freq_distribution_left])[0]
    sync_freq_distribution_right = np.gradient(H_array_right)/np.gradient(J_array_right) / (2*np.pi)
    
    # Emittance arrays
    emittance_array_left = J_array_left * (2*np.pi)
    emittance_array_left = np.fliplr([emittance_array_left])[0]
    emittance_array_right = J_array_right * (2*np.pi) 
    
    # Calculating particle distribution in synchrotron frequency 
    H_particles = eom_factor_dE * Beam.dE**2 + np.interp(Beam.dt, time_coord_array, total_potential)
    sync_freq_distribution = np.concatenate((sync_freq_distribution_left, sync_freq_distribution_right))
    H_array = np.concatenate((np.fliplr([H_array_left])[0], H_array_right))
    sync_freq_distribution = sync_freq_distribution[H_array.argsort()]
    H_array.sort()
    
    particleDistributionFreq = np.interp(H_particles, H_array, sync_freq_distribution)
    
    return [sync_freq_distribution_left, sync_freq_distribution_right], \
            [emittance_array_left, emittance_array_right], \
            [delta_time_left, delta_time_right], \
            particleDistributionFreq, synchronous_time


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

    def __init__(self, Ring, n_macroparticles, theta_coordinate_range, FullRingAndRF, 
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
            intensity = TotalInducedVoltage.profiles.Beam.intensity
        else:
            intensity = 0.
            
        from beam.beam import Beam
        #: *Beam object containing the same physical information as the real beam,
        #: but containing only the coordinates of the particles for which the 
        #: synchrotron frequency are computed.*
        self.Beam = Beam(Ring, n_macroparticles, intensity)
        
        # Generating the distribution from the user input
        if len(theta_coordinate_range) == 2:
            self.Beam.dt = np.linspace(float(theta_coordinate_range[0]),
                                       float(theta_coordinate_range[1]), n_macroparticles)\
                                       * (self.Beam.ring_radius/(self.Beam.beta*c))
        else:
            if len(theta_coordinate_range) != n_macroparticles:
                #SynchrotronMotionError
                raise RuntimeError('The input n_macroparticles does not match with the length of the theta_coordinates')
            else:
                self.Beam.dt = np.array(theta_coordinate_range) * (self.Beam.ring_radius/(self.Beam.beta*c))
                        
        self.Beam.dE = np.zeros(int(n_macroparticles))
 
        for RFsection in self.FullRingAndRF.RingAndRFSection_list:
            RFsection.beam = self.Beam
        
        #: *Revolution period in [s]*
        self.timeStep = Ring.t_rev[0]
        
        #: *Number of turns of the simulation (+1 to include the input parameters)*
        self.nTurns = Ring.n_turns+1
        
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
        Vcos = RFsection_list[0].voltage[0]*np.cos(RFsection_list[0].phi_rf[0])
        Vsin = RFsection_list[0].voltage[0]*np.sin(RFsection_list[0].phi_rf[0])
        if n_sections > 1:
            for i in range(1, n_sections):
                print(RFsection_list[i].voltage[0])
                Vcos += RFsection_list[i].voltage[0]*np.cos(RFsection_list[i].phi_rf[0])
                Vsin += RFsection_list[i].voltage[0]*np.sin(RFsection_list[i].phi_rf[0])
        Vtot = np.sqrt(Vcos**2 + Vsin**2)
        return Vtot
    
    #: *To be implemented*
    elif harmonic == "all":
        return 0

    else:
        warnings.filterwarnings("once")
        warnings.warn("WARNING: In total_voltage, harmonic choice not recognize!")
    


def hamiltonian(Ring, RFStation, Beam, dt, dE, 
                total_voltage = None):
    """Single RF sinusoidal Hamiltonian.
    For the time being, for single RF section only or from total voltage.
    Uses beta, energy averaged over the turn.
    To be generalized."""
     
   
    warnings.filterwarnings("once")
    
    if Ring.n_sections > 1:
        warnings.warn("WARNING: The Hamiltonian is not yet properly computed for several sections!")
    if RFStation.n_rf > 1:
        warnings.warn("WARNING: The Hamiltonian will be calculated for the first harmonic only!")

         
    counter = RFStation.counter[0]
    h0 = RFStation.harmonic[0,counter]
    if total_voltage == None:
        V0 = float(RFStation.voltage[0,counter])
    else: 
        V0 = float(total_voltage[counter])
    V0 *= RFStation.Particle.charge
    
    c1 = RFStation.eta_tracking(Beam, counter, dE)*c*np.pi/ \
         (Ring.ring_circumference*Beam.beta*Beam.energy )
    c2 = c*Beam.beta*V0/(h0*Ring.ring_circumference)
     
    phi_s = RFStation.phi_s[counter] 
    phi_b = RFStation.omega_rf[0,counter]*dt + \
            RFStation.phi_rf_d[0,counter] 
    
    eta0 = RFStation.eta_0[counter]
    
    # Modulo 2 Pi of bunch phase
    if eta0 < 0:
        phi_b = phase_modulo_below_transition(phi_b)
    elif eta0 > 0:
        phi_b = phase_modulo_above_transition(phi_b)    

    return c1 * dE**2 + c2 * (bm.cos(phi_b) - bm.cos(phi_s) + 
                               (phi_b - phi_s) * bm.sin(phi_s))
         
 
 
def separatrix(Ring, RFStation, dt):
    r""" Function to calculate the ideal separatrix without intensity effects.
    For single or multiple RF systems. For the time being, multiple RF sections
    are not yet implemented.
    
    Parameters
    ---------- 
    Ring : class
        A Ring type class
    RFStation : class
        An RFStation type class
    dt : float array
        Time coordinates the separatrix is to be calculated for
        
    Returns
    -------
    float array
        Energy coordinates of the separatrix corresponding to dt
        
    """
 
    warnings.filterwarnings("once")
     
    if Ring.n_sections > 1:
        warnings.warn("WARNING in separatrix(): the usage of several RF" +
                      " sections is not yet implemented!")
       
    # Import RF and ring parameters at this moment 
    counter = RFStation.counter[0]
    voltage = Ring.Particle.charge*RFStation.voltage[:,counter]
    omega_rf = RFStation.omega_rf[:,counter]
    phi_rf = RFStation.phi_rf[:,counter]

    eta_0 = RFStation.eta_0[counter]
    beta_sq = RFStation.beta[counter]**2     
    energy = RFStation.energy[counter]
    try:
        delta_E = RFStation.delta_E[counter]
    except:
        delta_E = RFStation.delta_E[-1]
    T_0 = Ring.t_rev[counter]
    index = np.min( np.where(voltage > 0)[0] )
    T_rf_0 = 2*np.pi/omega_rf[index]


    # Projects time array into the range [t_RF, t_RF+T_RF] below and above
    # transition, where T_RF = 2*pi/omega_RF, t_RF = - phi_RF/omega_RF.
    # Note that the RF wave is shifted by Pi for eta < 0
    if eta_0 < 0:
        dt = time_modulo(dt, (phi_rf[0] - np.pi)/omega_rf[0], 
                         2.*np.pi/omega_rf[0])
    elif eta_0 > 0:
        dt = time_modulo(dt, phi_rf[0]/omega_rf[0], 2.*np.pi/omega_rf[0])
    
    # Unstable fixed point in single-harmonic RF system
    if RFStation.n_rf == 1:
     
        dt_s = RFStation.phi_s[counter]/omega_rf[0]
        if eta_0 < 0:
            dt_RF = -(phi_rf[0] - np.pi)/omega_rf[0]
        else:
            dt_RF = -phi_rf[0]/omega_rf[0]
            
        dt_ufp = dt_RF + 0.5*T_rf_0 - dt_s
        if eta_0*delta_E < 0:
            dt_ufp += T_rf_0

    # Unstable fixed point in multi-harmonic RF system
    else:
        
        dt_ufp = np.linspace(-float(phi_rf[index]/omega_rf[index] - T_rf_0/1000), 
            float(T_rf_0 - phi_rf[index]/omega_rf[index] + T_rf_0/1000), 1002)

        if eta_0 < 0:
            dt_ufp += 0.5*T_rf_0 # Shift in RF phase below transition
        Vtot = np.zeros(len(dt_ufp))
        
        # Construct waveform
        for i in range(RFStation.n_rf):
            Vtot += voltage[i]*np.sin(omega_rf[i]*dt_ufp + phi_rf[i])
        Vtot -= delta_E
        
        # Find zero crossings
        zero_crossings = np.where(np.diff(np.sign(Vtot)))[0]
        
        # Interpolate UFP
        if eta_0 < 0:
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
    for i in range(RFStation.n_rf):
        Vtot += voltage[i]*(np.cos(omega_rf[i]*dt_ufp + phi_rf[i]) - 
                            np.cos(omega_rf[i]*dt + phi_rf[i]))/omega_rf[i]
                            
    separatrix_sq = 2*beta_sq*energy/(eta_0*T_0)*(Vtot + delta_E*(dt_ufp - dt))
    pos_ind = np.where(separatrix_sq >= 0)[0]
    separatrix_array = np.empty((len(separatrix_sq)))*np.nan
    separatrix_array[pos_ind] = np.sqrt(separatrix_sq[pos_ind])
         
    return separatrix_array
 
 
 
def is_in_separatrix(Ring, RFStation, Beam, dt, dE, 
                     total_voltage = None):
    r"""Function checking whether coordinate pair(s) are inside the separatrix. 
    Uses the single-RF sinusoidal Hamiltonian.
    
    Parameters
    ---------- 
    Ring : class
        A Ring type class
    RFStation : class
        An RFStation type class
    Beam : class
        A Beam type class
    dt : float array
        Time coordinates of the particles to be checked
    dE : float array
        Energy coordinates of the particles to be checked
    total_voltage : float array
        Total voltage to be used if not single-harmonic RF
        
    Returns
    -------
    bool array
        True/False array for the given coordinates
        
    """
     
    warnings.filterwarnings("once")
    
    if Ring.n_sections > 1:
        warnings.warn("WARNING: in is_in_separatrix(): the usage of several"+
                      " sections is not yet implemented!")
    if RFStation.n_rf > 1:
        warnings.warn("WARNING in is_in_separatrix(): taking into account" +
                      " the first harmonic only!")
    
         
    counter = RFStation.counter[0]
    dt_sep = (np.pi - RFStation.phi_s[counter] 
              - RFStation.phi_rf_d[0,counter])/ \
              RFStation.omega_rf[0,counter]
     
    Hsep = hamiltonian(Ring, RFStation, Beam, dt_sep, 0, 
                       total_voltage = None) 
    isin = bm.fabs(hamiltonian(Ring, RFStation, Beam, 
                               dt, dE, total_voltage = None)) < bm.fabs(Hsep)
     
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


def potential_well_cut(time_potential, potential_array):
    '''
    *Function to cut the potential well in order to take only the separatrix
    (several cases according to the number of min/max).*
    '''
    
    # Check for the min/max of the potential well
    minmax_positions, minmax_values = minmax_location(time_potential, 
                                                      potential_array)
    min_time_positions = minmax_positions[0]
    max_time_positions = minmax_positions[1]
    max_potential_values = minmax_values[1]
    n_minima = len(min_time_positions)
    n_maxima = len(max_time_positions)
    
    if n_minima == 0:
        #PotentialWellError
        raise RuntimeError('The potential well has no minima...')
    if n_minima > n_maxima and n_maxima == 1:
        #PotentialWellError
        raise RuntimeError('The potential well has more minima than maxima, and only one maximum')
    if n_maxima == 0:
        print ('Warning: The maximum of the potential well could not be found... \
                You may reconsider the options to calculate the potential well \
                as the main harmonic is probably not the expected one. \
                You may also increase the percentage of margin to compute \
                the potentiel well. The full potential well will be taken')
    elif n_maxima == 1:
        if min_time_positions[0] > max_time_positions[0]:
            saved_indexes = (potential_array < max_potential_values[0]) * \
                            (time_potential > max_time_positions[0])
            time_potential_sep = time_potential[saved_indexes]
            potential_well_sep = potential_array[saved_indexes]
            if potential_array[-1] < potential_array[0]:
                #PotentialWellError
                raise RuntimeError('The potential well is not well defined. \
                                    You may reconsider the options to calculate \
                                    the potential well as the main harmonic is \
                                    probably not the expected one.')
        else:
            saved_indexes = (potential_array < max_potential_values[0]) * \
                            (time_potential < max_time_positions[0])
            time_potential_sep = time_potential[saved_indexes]
            potential_well_sep = potential_array[saved_indexes]
            if potential_array[-1] > potential_array[0]:
                #PotentialWellError
                raise RuntimeError('The potential well is not well defined. \
                                    You may reconsider the options to calculate \
                                    the potential well as the main harmonic is \
                                    probably not the expected one.')
    elif n_maxima == 2:
        lower_maximum_value = np.min(max_potential_values)
        higher_maximum_value = np.max(max_potential_values)
        lower_maximum_time = max_time_positions[max_potential_values == lower_maximum_value]
        higher_maximum_time = max_time_positions[max_potential_values == higher_maximum_value]
        if len(lower_maximum_time)==2:
            saved_indexes = (potential_array < lower_maximum_value) * \
                            (time_potential > lower_maximum_time[0]) * \
                            (time_potential < lower_maximum_time[1])
            time_potential_sep = time_potential[saved_indexes]
            potential_well_sep = potential_array[saved_indexes]
        elif min_time_positions[0] > lower_maximum_time:
            saved_indexes = (potential_array < lower_maximum_value) * \
                            (time_potential > lower_maximum_time) * \
                            (time_potential < higher_maximum_time)
            time_potential_sep = time_potential[saved_indexes]
            potential_well_sep = potential_array[saved_indexes]
        else:
            saved_indexes = (potential_array < lower_maximum_value) * \
                            (time_potential < lower_maximum_time) * \
                            (time_potential > higher_maximum_time)
            time_potential_sep = time_potential[saved_indexes]
            potential_well_sep = potential_array[saved_indexes]
    elif n_maxima > 2:
        left_max_time = np.min(max_time_positions)
        right_max_time = np.max(max_time_positions)
        left_max_value = max_potential_values[max_time_positions==left_max_time]
        right_max_value = max_potential_values[max_time_positions==right_max_time]
        separatrix_value = np.min([left_max_value, right_max_value])
        saved_indexes = (time_potential > left_max_time) * (time_potential < right_max_time) * (potential_array < separatrix_value)
        time_potential_sep = time_potential[saved_indexes]
        potential_well_sep = potential_array[saved_indexes]
        
        
    return time_potential_sep, potential_well_sep

def phase_modulo_above_transition(phi):
    '''
    *Projects a phase array into the range -Pi/2 to +3*Pi/2.*
    '''
    
    return phi - 2.*np.pi*bm.floor(phi/(2.*np.pi))


 
def phase_modulo_below_transition(phi):
    '''
    *Projects a phase array into the range -Pi/2 to +3*Pi/2.*
    '''
    
    return phi - 2.*np.pi*(bm.floor(phi/(2.*np.pi) + 0.5))
        


def time_modulo(dt, dt_offset, T):
    '''
    *Returns dt projected onto the desired interval.*
    '''
    
    return dt - T*bm.floor((dt + dt_offset)/T)
