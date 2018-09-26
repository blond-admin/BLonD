'''
**Module to generate multibunch distributions**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**, **Theodoros Argyropoulos**
'''

from __future__ import division, print_function, absolute_import
from builtins import range
import numpy as np
import copy
import matplotlib.pyplot as plt
from ..beam.beam import Beam
from ..beam.distributions import matched_from_distribution_function, matched_from_line_density



def matched_from_distribution_density_multibunch(beam, Ring, FullRingAndRF, distribution_options_list,
                                      n_bunches, bunch_spacing_buckets,
                                      intensity_list = None,
                                      minimum_n_macroparticles = None,
                                      main_harmonic_option = 'lowest_freq', 
                                      TotalInducedVoltage = None,
                                      n_iterations_input = 1,
                                      plot_option = False, seed=None):
    '''
    *Function to generate a multi-bunch beam using the matched_from_distribution_density
    function for each bunch. The extra parameters to include are the number of 
    bunches and the spacing between two bunches (assumed constant presently).
    Moreover, the distribution_options_list corresponds to the distribution_options
    of the matched_from_distribution_density function. It can be inputed as
    a dictionary just like the matched_from_distribution_density function (assuming
    the same parameters for all bunches), or as a list of length n_bunches
    to have different parameters for each bunch.*
    '''  

    
    if intensity_list is None:
        intensity_per_bunch = beam.intensity/n_bunches * np.ones(n_bunches)
        n_macroparticles_per_bunch = beam.n_macroparticles/n_bunches * np.ones(n_bunches)
    else:
        intensity_per_bunch = np.array(intensity_list)
        if minimum_n_macroparticles is None:
            n_macroparticles_per_bunch = np.round(beam.n_macroparticles/beam.intensity * intensity_per_bunch)
        else:
            n_macroparticles_per_bunch = np.round(minimum_n_macroparticles/np.min(intensity_per_bunch) * intensity_per_bunch)
    
    if np.sum(intensity_per_bunch) != beam.intensity:
        print('WARNING !! The total intensity per bunch does not match the total intensity of the beam, the beam.intensity will be overwritten')
        beam.intensity = np.sum(intensity_per_bunch)
    
    
    if np.sum(n_macroparticles_per_bunch) != beam.n_macroparticles:
        print('WARNING !! The number of macroparticles per bunch does not match the total number of the beam, the beam.n_macroparticles will be overwritten')
        beam.n_macroparticles = int(np.sum(n_macroparticles_per_bunch))

    voltages = np.array([])
    harmonics = np.array([])
        
    for RingAndRFSectionElement in FullRingAndRF.RingAndRFSection_list:
            for rf_system in range(RingAndRFSectionElement.n_rf):
                voltages = np.append(voltages, RingAndRFSectionElement.voltage[rf_system, 0])
                harmonics = np.append(harmonics, RingAndRFSectionElement.harmonic[rf_system, 0])
                
    if main_harmonic_option is 'lowest_freq':
            main_harmonic = np.min(harmonics)
    elif main_harmonic_option is 'highest_voltage':
        main_harmonic = np.min(harmonics[voltages == np.max(voltages)])
    elif isinstance(main_harmonic_option, int) or isinstance(main_harmonic_option, float):
        if harmonics[harmonics == main_harmonic_option].size == 0:
            raise RuntimeError('The desired harmonic to compute the potential well does not match the RF parameters...')
        main_harmonic = np.min(harmonics[harmonics == main_harmonic_option])
         
    bucket_size_tau = 2 * np.pi / (main_harmonic * Ring.omega_rev[0])

    beamIteration = Beam(Ring, 1, 0.)
    
    extraVoltageDict = None
    
    if TotalInducedVoltage is not None:
        bucket_tolerance = 0.40
        TotalInducedVoltageIteration = copy.deepcopy(TotalInducedVoltage)
        TotalInducedVoltageIteration.profile.Beam = beamIteration
        
    
    for indexBunch in range(0, n_bunches):
        
        print('Generating bunch no %d' %(indexBunch+1))
        
        bunch = Beam(Ring, int(n_macroparticles_per_bunch[indexBunch]), intensity_per_bunch[indexBunch])
        
        if isinstance(distribution_options_list, list):
            distribution_options = distribution_options_list[indexBunch]
        elif isinstance(distribution_options_list, dict):
            distribution_options = distribution_options_list
        else:
            raise RuntimeError('The input distribution_options_list option of the matched_from_distribution_density_multibunch \
            function should either be a dictionary as requested by the matched_from_distribution_density \
            function, or a list of dictionaries containing n_bunches elements')
        
        if 'type' in distribution_options:
            distribution_type = distribution_options['type']
        else:
            distribution_type = None
        
        if 'exponent' in distribution_options:
            distribution_exponent = distribution_options['exponent']
        else:
            distribution_exponent = None
        
        if 'emittance' in distribution_options:
            emittance = distribution_options['emittance']
        else:
            emittance = None
        
        if 'bunch_length' in distribution_options:
            bunch_length = distribution_options['bunch_length']
        else:
            bunch_length = None
        
        if 'bunch_length_fit' in distribution_options:
            bunch_length_fit = distribution_options['bunch_length_fit']
        else:
            bunch_length_fit = None
        
        if 'density_variable' in distribution_options:
            distribution_variable = distribution_options['density_variable']
        else:
            distribution_variable = None            
        
        if distribution_options['type'] is 'user_input':
            distribution_function_input = distribution_options['function']
        else:
            distribution_function_input = None
        
        if distribution_options['type'] is 'user_input_table':
            distribution_user_table = {
              'user_table_action': distribution_options['user_table_action'],
              'user_table_density': distribution_options['user_table_density']}
        else:
            distribution_user_table = None
            
        matched_from_distribution_function(bunch, FullRingAndRF,
                       distribution_function_input=distribution_function_input,
                       distribution_user_table=distribution_user_table,
                       main_harmonic_option=main_harmonic_option,
                       TotalInducedVoltage=TotalInducedVoltage,
                       n_iterations=n_iterations_input,
                       extraVoltageDict=extraVoltageDict,
                       distribution_exponent=distribution_exponent,
                       distribution_type=distribution_type,
                       emittance=emittance, bunch_length=bunch_length,
                       bunch_length_fit=bunch_length_fit,
                       distribution_variable=distribution_variable, seed=seed)

        if indexBunch==0:
            beamIteration.dt = bunch.dt
            beamIteration.dE = bunch.dE
        else:
            beamIteration.dt = np.append(beamIteration.dt, bunch.dt + 
                        (indexBunch * bunch_spacing_buckets * bucket_size_tau))
            beamIteration.dE = np.append(beamIteration.dE, bunch.dE)
        
        beamIteration.n_macroparticles = int(np.sum(n_macroparticles_per_bunch[:indexBunch+1]))
        beamIteration.intensity = np.sum(intensity_per_bunch[:indexBunch+1])
        beamIteration.ratio = beamIteration.intensity / beamIteration.n_macroparticles
        
        
        if TotalInducedVoltage is not None:
            TotalInducedVoltageIteration.profile.track()
            TotalInducedVoltageIteration.induced_voltage_sum()

            left_edge = ((indexBunch + 1) * bunch_spacing_buckets * 
                          bucket_size_tau - bucket_tolerance * bucket_size_tau)
            right_edge = (((indexBunch + 1) * bunch_spacing_buckets + 1) * 
                          bucket_size_tau + bucket_tolerance * bucket_size_tau)
            
            bin_centers = TotalInducedVoltageIteration.profile.bin_centers
            
            tau_induced_voltage_next_bunch = bin_centers[
                        (bin_centers > left_edge) * (bin_centers < right_edge)]
            induced_voltage_next_bunch = \
                        TotalInducedVoltageIteration.induced_voltage[
                        (bin_centers > left_edge) * (bin_centers < right_edge)]

            time_induced_voltage_next_bunch = (tau_induced_voltage_next_bunch -
                      (indexBunch+1) * bunch_spacing_buckets * bucket_size_tau)
            
            extraVoltageDict = {'time_array':time_induced_voltage_next_bunch,
                                'voltage_array':induced_voltage_next_bunch}

            
        if plot_option:
            plt.figure('Bunch train + induced voltage')
            plt.clf()
            plt.plot(TotalInducedVoltageIteration.profile.bin_centers,
                     TotalInducedVoltageIteration.profile.n_macroparticles / 
                     (1.*np.max(TotalInducedVoltageIteration.profile.n_macroparticles)) *
                     np.max(TotalInducedVoltageIteration.induced_voltage))
            plt.plot(TotalInducedVoltageIteration.profile.bin_centers,
                     TotalInducedVoltageIteration.induced_voltage)
            plt.show()
                
    beam.dt = beamIteration.dt

    beam.dE = beamIteration.dE
    
    
def matched_from_line_density_multibunch(beam, Ring,
                        FullRingAndRF, line_density_options_list, n_bunches,
                        bunch_spacing_buckets, intensity_list=None,
                        minimum_n_macroparticles=None,
                        main_harmonic_option='lowest_freq',
                        TotalInducedVoltage=None, half_option='first',
                        plot_option=False, seed=None):
    '''
    *Function to generate a multi-bunch beam using the matched_from_distribution_density
    function for each bunch. The extra parameters to include are the number of 
    bunches and the spacing between two bunches (assumed constant presently).
    Moreover, the line_density_options_list corresponds to the distribution_options
    of the matched_from_line_density function. It can be inputed as
    a dictionary just like the matched_from_line_density function (assuming
    the same parameters for all bunches), or as a list of length n_bunches
    to have different parameters for each bunch.*
    '''
    
    if intensity_list is None:
        intensity_per_bunch = beam.intensity/n_bunches * np.ones(n_bunches)
        n_macroparticles_per_bunch = beam.n_macroparticles/n_bunches * np.ones(n_bunches)
    else:
        intensity_per_bunch = np.array(intensity_list)
        if minimum_n_macroparticles is None:
            n_macroparticles_per_bunch = np.round(beam.n_macroparticles/beam.intensity * intensity_per_bunch)
        else:
            n_macroparticles_per_bunch = np.round(minimum_n_macroparticles/np.min(intensity_per_bunch) * intensity_per_bunch)
        
    if np.sum(intensity_per_bunch) != beam.intensity:
        print('WARNING !! The total intensity per bunch does not match the total intensity of the beam, the beam.intensity will be overwritten')
        beam.intensity = np.sum(intensity_per_bunch)
    
    if np.sum(n_macroparticles_per_bunch) != beam.n_macroparticles:
        print('WARNING !! The number of macroparticles per bunch does not match the total number of the beam, the beam.n_macroparticles will be overwritten')
        beam.n_macroparticles = int(np.sum(n_macroparticles_per_bunch))
    
    voltages = np.array([])
    harmonics = np.array([])
        
    for RingAndRFSectionElement in FullRingAndRF.RingAndRFSection_list:
            for rf_system in range(RingAndRFSectionElement.n_rf):
                voltages = np.append(voltages, RingAndRFSectionElement.voltage[rf_system, 0])
                harmonics = np.append(harmonics, RingAndRFSectionElement.harmonic[rf_system, 0])
                
    if main_harmonic_option is 'lowest_freq':
            main_harmonic = np.min(harmonics)
    elif main_harmonic_option is 'highest_voltage':
        main_harmonic = np.min(harmonics[voltages == np.max(voltages)])
    elif isinstance(main_harmonic_option, int) or isinstance(main_harmonic_option, float):
        if harmonics[harmonics == main_harmonic_option].size == 0:
            raise RuntimeError('The desired harmonic to compute the potential well does not match the RF parameters...')
        main_harmonic = np.min(harmonics[harmonics == main_harmonic_option])
            
    bucket_size_tau = 2 * np.pi / (main_harmonic * Ring.omega_rev[0])
    
    beamIteration = Beam(Ring, 1, 0.)
    
    extraVoltageDict = None
    
    if TotalInducedVoltage is not None:
        TotalInducedVoltageIteration = copy.deepcopy(TotalInducedVoltage)
        TotalInducedVoltageIteration.profile.Beam = beamIteration
        
    
    for indexBunch in range(0, n_bunches):
        
        print('Generating bunch no %d' %(indexBunch+1))
        
        bunch = Beam(Ring,
                     int(n_macroparticles_per_bunch[indexBunch]),
                     intensity_per_bunch[indexBunch])
        
        if isinstance(line_density_options_list, list):
            line_density_options = line_density_options_list[indexBunch]
        elif isinstance(line_density_options_list, dict):
            line_density_options = line_density_options_list
        else:
            raise RuntimeError('The input line_density_options_list option ' +
                               'of the matched_from_line_density_multibunch ' +
                               'function should either be a dictionary as ' +
                               'requested by the matched_from_line_density ' +
                               'function, or a list of dictionaries ' +
                               'containing n_bunches elements')
        
        
        if 'bunch_length' in line_density_options:
            bunch_length = line_density_options['bunch_length']
        else:
            bunch_length = None
                              
        if 'type' in line_density_options:
            line_density_type = line_density_options['type']
        else:
            line_density_type = None
        
        if 'exponent' in line_density_options:
            line_density_exponent = line_density_options['exponent']
        else:
            line_density_exponent = None
        
        if line_density_options['type'] is 'user_input':
            line_density_input = {
                        'time_line_den': line_density_options['time_line_den'],
                        'line_density': line_density_options['line_density']}
        else:
            line_density_input = None
        
        matched_from_line_density(bunch, FullRingAndRF,
                              line_density_input=line_density_input,
                              main_harmonic_option=main_harmonic_option,
                              TotalInducedVoltage=TotalInducedVoltage,
                              plot=plot_option, half_option=half_option,
                              extraVoltageDict=extraVoltageDict,
                              bunch_length=bunch_length,
                              line_density_type=line_density_type,
                              line_density_exponent=line_density_exponent,
                              seed=seed)

        if indexBunch==0:
            beamIteration.dt = bunch.dt
            beamIteration.dE = bunch.dE
        else:
            beamIteration.dt = np.append(beamIteration.dt, bunch.dt +
                        (indexBunch * bunch_spacing_buckets * bucket_size_tau))
            beamIteration.dE = np.append(beamIteration.dE, bunch.dE)
        
        beamIteration.n_macroparticles = int(np.sum(n_macroparticles_per_bunch[:indexBunch+1]))
        beamIteration.intensity = np.sum(intensity_per_bunch[:indexBunch+1])
        beamIteration.ratio = beamIteration.intensity/beamIteration.n_macroparticles
        
        if TotalInducedVoltage is not None:
            TotalInducedVoltageIteration.profile.track()
            TotalInducedVoltageIteration.induced_voltage_sum()        
            
            bucket_tolerance = 0.40  
            
            left_edge = (indexBunch+1) * bunch_spacing_buckets * bucket_size_tau - bucket_tolerance * bucket_size_tau
            right_edge = ((indexBunch+1) * bunch_spacing_buckets +1)* bucket_size_tau + bucket_tolerance * bucket_size_tau
            
            tau_induced_voltage_next_bunch = TotalInducedVoltageIteration.profile.bin_centers[(TotalInducedVoltageIteration.profile.bin_centers > left_edge)*(TotalInducedVoltageIteration.profile.bin_centers < right_edge)]
            induced_voltage_next_bunch = TotalInducedVoltageIteration.induced_voltage[(TotalInducedVoltageIteration.profile.bin_centers > left_edge)*(TotalInducedVoltageIteration.profile.bin_centers < right_edge)]
            
            time_induced_voltage_next_bunch = (tau_induced_voltage_next_bunch - (indexBunch+1) * bunch_spacing_buckets * bucket_size_tau)
            
            extraVoltageDict = {'time_array':time_induced_voltage_next_bunch, 'voltage_array':induced_voltage_next_bunch}         
            
            
    if plot_option:
        plt.figure('Bunch train + induced voltage')
        plt.clf()
        plt.plot(TotalInducedVoltageIteration.profile.bin_centers, TotalInducedVoltageIteration.profile.n_macroparticles / (1.*np.max(TotalInducedVoltageIteration.profile.n_macroparticles))*np.max(TotalInducedVoltageIteration.induced_voltage))
        plt.plot(TotalInducedVoltageIteration.profile.bin_centers, TotalInducedVoltageIteration.induced_voltage)
        plt.show()
                
    beam.dt = beamIteration.dt
    beam.dE = beamIteration.dE
