
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to generate distributions**

:Authors: **Danilo Quartullo**, **Helga Timko**, **Alexandre Lasheen**, **Juan Esteban Muller**, **Theodoros Argyropoulos**
'''

from __future__ import division
import numpy as np
import warnings
import copy
import matplotlib.pyplot as plt
from scipy.constants import c
from trackers.utilities import is_in_separatrix
from slices import Slices
from scipy.integrate import cumtrapz
from trackers.utilities import potential_well_cut, minmax_location
import sys




def matched_from_line_density(Beam, FullRingAndRF, line_density_options, 
                              main_harmonic_option = 'lowest_freq', 
                              TotalInducedVoltage = None,
                              plot_option = False, half_option = 'first'):
    '''
    *Function to generate a beam by inputing the line density. The distribution
    density is then reconstructed with the Abel transform and the particles
    randomly generated.*
    '''
    
    if not line_density_options.has_key('exponent'):  
        line_density_options['exponent'] = None
        
    # Initialize variables depending on the accelerator parameters
    slippage_factor = FullRingAndRF.RingAndRFSection_list[0].eta_0[0]
    eom_factor_dE = (np.pi * abs(slippage_factor) * c) / \
                    (FullRingAndRF.ring_circumference * Beam.beta_r * Beam.energy)
    eom_factor_potential = np.sign(FullRingAndRF.RingAndRFSection_list[0].eta_0[0]) * (Beam.beta_r * c) / (FullRingAndRF.ring_circumference)
     
    # Generate potential well
    n_points_potential = int(1e4)
    FullRingAndRF.potential_well_generation(n_points = n_points_potential, 
                                            theta_margin_percent = 0.25, 
                                            main_harmonic_option = main_harmonic_option)
    potential_well_array = FullRingAndRF.potential_well
    theta_coord_array = FullRingAndRF.potential_well_coordinates
    
    if line_density_options['type'] is not 'user_input':
        # Theta coordinates for the line density
        n_points_line_den = int(1e4)
        theta_line_den = np.linspace(theta_coord_array[0], theta_coord_array[-1], n_points_line_den)
        line_den_resolution = theta_line_den[1] - theta_line_den[0]
                        
        # Normalizing the line density                
        line_density = line_density_function(theta_line_den, line_density_options['type'], line_density_options['bunch_length'], exponent = line_density_options['exponent'],
                                             bunch_position = (theta_coord_array[0]+theta_coord_array[-1])/2)
        
        line_density = line_density / np.sum(line_density) * Beam.n_macroparticles
        line_density = line_density - np.min(line_density)

    elif line_density_options['type'] is 'user_input':
        # Theta coordinates for the line density
        theta_line_den = line_density_options['theta_line_den']
        n_points_line_den = len(theta_line_den)
        line_den_resolution = theta_line_den[1] - theta_line_den[0]
                        
        # Normalizing the line density                
        line_density = line_density_options['line_density']
        line_density = line_density / np.sum(line_density) * Beam.n_macroparticles
        line_density = line_density - np.min(line_density)
    else:
        raise RuntimeError('The input for the matched_from_line_density function was not recognized')
    
    induced_potential_final = 0
    n_iterations = 1
    
    if TotalInducedVoltage is not None:
        # Calculating the induced voltage
        induced_voltage_object = copy.deepcopy(TotalInducedVoltage)
        
        # Inputing new line density
        induced_voltage_object.slices.n_macroparticles = line_density
        induced_voltage_object.slices.bins_centers = theta_line_den * FullRingAndRF.ring_radius / (Beam.beta_r * c)
        induced_voltage_object.slices.edges = np.linspace(induced_voltage_object.slices.bins_centers[0]-(induced_voltage_object.slices.bins_centers[1]-induced_voltage_object.slices.bins_centers[0])/2,induced_voltage_object.slices.bins_centers[-1]+(induced_voltage_object.slices.bins_centers[1]-induced_voltage_object.slices.bins_centers[0])/2,len(induced_voltage_object.slices.bins_centers)+1)
        induced_voltage_object.slices.n_slices = n_points_line_den
        induced_voltage_object.slices.fit_option = 'off'
        
        # Re-calculating the sources of wakes/impedances according to this slicing
        induced_voltage_object.reprocess(induced_voltage_object.slices)
        
        # Calculating the induced voltage
        induced_voltage_length = int(1.5*n_points_line_den)
        induced_voltage = induced_voltage_object.induced_voltage_sum(Beam, length = induced_voltage_length)
        theta_induced_voltage = np.linspace(theta_line_den[0], theta_line_den[0] + (induced_voltage_length - 1) * line_den_resolution, induced_voltage_length)
      
        # Calculating the induced potential
        induced_potential = - eom_factor_potential * np.insert(cumtrapz(induced_voltage, dx=theta_induced_voltage[1] - theta_induced_voltage[0]),0,0)
        
        # Changing number of iterations
        n_iterations = 100
    
    # Centering the bunch in the potential well
    for i in range(0, n_iterations):
        
        if TotalInducedVoltage is not None:
            # Interpolating the potential well
            induced_potential_final = np.interp(theta_coord_array, theta_induced_voltage, induced_potential)
            
        # Induced voltage contribution
        total_potential = potential_well_array + induced_potential_final
        
        # Process the potential well in order to take a frame around the separatrix
        theta_coord_sep, potential_well_sep = potential_well_cut(theta_coord_array, total_potential)
        
        minmax_positions_potential, minmax_values_potential = minmax_location(theta_coord_sep, potential_well_sep)
        minmax_positions_profile, minmax_values_profile = minmax_location(theta_line_den[line_density != 0], line_density[line_density != 0])

        n_minima_potential = len(minmax_positions_potential[0])
        n_maxima_profile = len(minmax_positions_profile[1])
        
        # Warnings
        if n_maxima_profile > 1:
            print 'Warning: the profile has serveral max, the highest one is taken. Be sure the profile is monotonous and not too noisy.'
            max_profile_pos = minmax_positions_profile[1][np.where(minmax_values_profile[1] == np.max(minmax_values_profile[1]))]
        else:
            max_profile_pos = minmax_positions_profile[1]
        if n_minima_potential > 1:
            print 'Warning: the potential well has serveral min, the deepest one is taken. The induced potential is probably splitting the potential well.'
            min_potential_pos = minmax_positions_potential[0][np.where(minmax_values_potential[0] == np.min(minmax_values_potential[0]))]
        else:
            min_potential_pos = minmax_positions_potential[0]
        
        # Moving the bunch (not for the last iteration if intensity effects are present)
        if TotalInducedVoltage is None:
            theta_line_den = theta_line_den - (max_profile_pos - min_potential_pos)
            max_profile_pos = max_profile_pos - (max_profile_pos - min_potential_pos)
        if i != n_iterations - 1:
            theta_line_den = theta_line_den - (max_profile_pos - min_potential_pos)
            theta_induced_voltage = np.linspace(theta_line_den[0], theta_line_den[0] + (induced_voltage_length - 1) * line_den_resolution, induced_voltage_length)
    
            
    # Taking the first/second half of line density and potential
    n_points_abel = int(1e4)
    
    abel_both_step = 1
    if half_option is 'both':
        abel_both_step = 2
        density_function_average = np.zeros((n_points_abel,2))
        hamiltonian_average = np.zeros((n_points_abel,2))
        
    for abel_index in range(0, abel_both_step):

        if half_option is 'first':
            half_indexes = np.where((theta_line_den >= theta_line_den[0]) * (theta_line_den <= max_profile_pos))
        if half_option is 'second':
            half_indexes = np.where((theta_line_den >= max_profile_pos) * (theta_line_den <= theta_line_den[-1]))
        if half_option is 'both' and abel_index == 0:
            half_indexes = np.where((theta_line_den >= theta_line_den[0]) * (theta_line_den <= max_profile_pos))
        if half_option is 'both' and abel_index == 1:
            half_indexes = np.where((theta_line_den >= max_profile_pos) * (theta_line_den <= theta_line_den[-1]))
        
        line_den_half = line_density[half_indexes]
        theta_coord_half = theta_line_den[half_indexes]
        potential_half = np.interp(theta_coord_half, theta_coord_sep, potential_well_sep)
        potential_half = potential_half - np.min(potential_half)
    
        # Derivative of the line density
        line_den_diff = np.diff(line_den_half) / (theta_coord_half[1] - theta_coord_half[0])
        
        theta_line_den_diff = theta_coord_half[:-1] + (theta_coord_half[1] - theta_coord_half[0]) / 2
        line_den_diff = np.interp(theta_coord_half, theta_line_den_diff, line_den_diff, left = 0, right = 0)
    
        # Interpolating the line density derivative and potential well for Abel transform
        theta_abel = np.linspace(theta_coord_half[0], theta_coord_half[-1], n_points_abel)
        line_den_diff_abel = np.interp(theta_abel, theta_coord_half, line_den_diff)
        potential_abel = np.interp(theta_abel, theta_coord_half, potential_half)
        
        density_function = np.zeros(n_points_abel)
        hamiltonian_coord = np.zeros(n_points_abel) 
        
        # Abel transform
        warnings.filterwarnings("ignore")
        
        if (half_option is 'first') or (half_option is 'both' and abel_index == 0):
            for i in range(0, n_points_abel):
                integrand = line_den_diff_abel[0:i+1] / np.sqrt(potential_abel[0:i+1] - potential_abel[i])
                        
                if len(integrand)>2:
                    integrand[-1] = integrand[-2] + (integrand[-2] - integrand[-3])
                elif len(integrand)>1:
                    integrand[-1] = integrand[-2]
                else:
                    integrand = np.array([0])
                    
                density_function[i] = np.sqrt(eom_factor_dE) / np.pi * np.trapz(integrand, dx = theta_coord_half[1] - theta_coord_half[0])
        
                hamiltonian_coord[i] = potential_abel[i]
                
        if (half_option is 'second') or (half_option is 'both' and abel_index == 1):
            for i in range(0, n_points_abel):
                integrand = line_den_diff_abel[i:] / np.sqrt(potential_abel[i:] - potential_abel[i])
    
                if len(integrand)>2:
                    integrand[0] = integrand[1] + (integrand[2] - integrand[1])
                if len(integrand)>1:
                    integrand[0] = integrand[1]
                else:
                    integrand = np.array([0])
    
                density_function[i] = - np.sqrt(eom_factor_dE) / np.pi * np.trapz(integrand, dx = theta_coord_half[1] - theta_coord_half[0])
                hamiltonian_coord[i] = potential_abel[i]
        
        warnings.filterwarnings("default")
    
        # Cleaning the density function from unphysical results
        density_function[np.isnan(density_function)] = 0
        density_function[density_function<0] = 0
        
        if half_option is 'both':
            hamiltonian_average[:,abel_index] = hamiltonian_coord
            density_function_average[:,abel_index] = density_function
            
            
    if half_option is 'both':
        hamiltonian_coord = hamiltonian_average[:,0]
        density_function = (density_function_average[:,0] + np.interp(hamiltonian_coord, hamiltonian_average[:,1], density_function_average[:,1])) / 2
        
    # Compute deltaE frame corresponding to the separatrix
    max_potential = np.max(potential_half)
    max_deltaE = np.sqrt(max_potential / eom_factor_dE)
    
    # Initializing the grids by reducing the resolution to a 
    # n_points_grid*n_points_grid frame.
    n_points_grid = int(1e3)
    grid_indexes = np.arange(0,n_points_grid) * len(theta_line_den) / n_points_grid
    theta_coord_indexes = np.arange(0, len(theta_line_den))
    theta_coord_for_grid = np.interp(grid_indexes, theta_coord_indexes, theta_line_den)
    deltaE_for_grid = np.linspace(-max_deltaE, max_deltaE, n_points_grid)
    potential_well_for_grid = np.interp(theta_coord_for_grid, theta_coord_sep, potential_well_sep)
    potential_well_for_grid = potential_well_for_grid - np.min(potential_well_for_grid)
    
    theta_grid, deltaE_grid = np.meshgrid(theta_coord_for_grid, deltaE_for_grid)
    potential_well_grid = np.meshgrid(potential_well_for_grid, potential_well_for_grid)[0]
    
    hamiltonian_grid = eom_factor_dE * deltaE_grid**2 + potential_well_grid

    # Sort the density function and generate the density grid
    hamiltonian_argsort = np.argsort(hamiltonian_coord)
    hamiltonian_coord = hamiltonian_coord.take(hamiltonian_argsort)
    density_function = density_function.take(hamiltonian_argsort)
    density_grid = np.interp(hamiltonian_grid, hamiltonian_coord, density_function)
    
    # Normalizing density
    density_grid = density_grid / np.sum(density_grid)
    reconstructed_line_den = np.sum(density_grid, axis=0)
    
    # Ploting the result
    if plot_option:
        plt.figure()
        plt.plot(theta_line_den, line_density)        
        plt.plot(theta_coord_for_grid, reconstructed_line_den / np.max(reconstructed_line_den) * np.max(line_density))
        plt.title('Line densities')
        plt.show()
    
    # Populating the bunch
    indexes = np.random.choice(range(0,np.size(density_grid)), Beam.n_macroparticles, p=density_grid.flatten())
    Beam.theta = np.ascontiguousarray(theta_grid.flatten()[indexes] + (np.random.rand(Beam.n_macroparticles) - 0.5) * (theta_coord_for_grid[1]-theta_coord_for_grid[0]))
    Beam.dE = np.ascontiguousarray(deltaE_grid.flatten()[indexes] + (np.random.rand(Beam.n_macroparticles) - 0.5) * (deltaE_for_grid[1]-deltaE_for_grid[0]))
    
    if TotalInducedVoltage is not None:
        # Inputing new line density
        induced_voltage_object.slices.n_macroparticles = reconstructed_line_den * Beam.n_macroparticles
        induced_voltage_object.slices.bins_centers = theta_coord_for_grid * FullRingAndRF.ring_radius / (Beam.beta_r * c)
        induced_voltage_object.slices.edges = np.linspace(induced_voltage_object.slices.bins_centers[0]-(induced_voltage_object.slices.bins_centers[1]-induced_voltage_object.slices.bins_centers[0])/2,induced_voltage_object.slices.bins_centers[-1]+(induced_voltage_object.slices.bins_centers[1]-induced_voltage_object.slices.bins_centers[0])/2,len(induced_voltage_object.slices.bins_centers)+1)
        induced_voltage_object.slices.n_slices = len(theta_coord_for_grid)
        induced_voltage_object.slices.fit_option = 'off'
        
        # Re-calculating the sources of wakes/impedances according to this slicing
        induced_voltage_object.reprocess(induced_voltage_object.slices)
        
        # Calculating the induced voltage
        induced_voltage_object.induced_voltage_sum(Beam)
        
        return [hamiltonian_coord, density_function], induced_voltage_object
    else:
        return [hamiltonian_coord, density_function], [theta_line_den, line_density]



def matched_from_distribution_density(Beam, FullRingAndRF, distribution_options,
                                      main_harmonic_option = 'lowest_freq', 
                                      TotalInducedVoltage = None,
                                      n_iterations_input = 1):
    '''
    *Function to generate a beam by inputing the distribution density (by
    choosing the type of distribution and the emittance). 
    The potential well is preprocessed to check for the min/max and center
    the frame around the separatrix.
    An error will be raised if there is not a full potential well (2 max 
    and 1 min at least), or if there are several wells (more than 2 max and 
    1 min, this case will be treated in the future).
    A margin of 5% is applied in order to be able to catch the min/max of the 
    potential well that might be on the edge of the frame. 
    The slippage factor should be updated to take the higher orders.
    Outputs should be added in order for the user to check step by step if
    his bunch is going to be well generated. More detailed 'step by step' 
    documentation should be implemented
    The user can input a custom distribution function by setting the parameter
    distribution_options['type'] = 'user_input' and passing the function in the
    parameter distribution_options['function'], with the following definition:
    distribution_density_function(action_array, dist_type, length, exponent = None).
    The user can also add an input table by setting the parameter 
    distribution_options['type'] = 'user_input_table', 
    distribution_options['user_table_action'] = array of action (in H or in J)
    and distribution_options['user_table_density']*
    '''
    
    if not distribution_options.has_key('exponent'):  
        distribution_options['exponent'] = None
    
    # Loading the distribution function if provided by the user
    if distribution_options['type'] is 'user_input':
        distribution_density_function = distribution_options['function']
    else:
        distribution_density_function = _distribution_density_function
    
    # Initialize variables depending on the accelerator parameters
    slippage_factor = abs(FullRingAndRF.RingAndRFSection_list[0].eta_0[0])
    eom_factor_dE = (np.pi * slippage_factor * c) / \
                    (FullRingAndRF.ring_circumference * Beam.beta_r * Beam.energy)
    eom_factor_potential = np.sign(FullRingAndRF.RingAndRFSection_list[0].eta_0[0]) * (Beam.beta_r * c) / (FullRingAndRF.ring_circumference)
    
    # Generate potential well
    n_points_potential = int(1e5)
    FullRingAndRF.potential_well_generation(n_points = n_points_potential, 
                                            theta_margin_percent = 0.25, 
                                            main_harmonic_option = main_harmonic_option)
    potential_well_array = FullRingAndRF.potential_well
    theta_coord_array = FullRingAndRF.potential_well_coordinates
    theta_resolution = theta_coord_array[1] - theta_coord_array[0]
    
    induced_potential = 0
    total_potential = potential_well_array + induced_potential
    
    n_iterations = n_iterations_input
    if not TotalInducedVoltage:
        n_iterations = 1
        
    for i in range(0, n_iterations):
        
        old_potential = copy.deepcopy(total_potential)
        # Adding the induced potential to the RF potential
        total_potential = potential_well_array + induced_potential
        
        sse = np.sqrt(np.sum((old_potential-total_potential)**2))

        print 'Matching the bunch... (iteration: ' + str(i) + ' and sse: ' + str(sse) +')'
                
        # Process the potential well in order to take a frame around the separatrix
        theta_coord_sep, potential_well_sep = potential_well_cut(theta_coord_array, total_potential)
        
        # Potential is shifted to put the minimum on 0
        potential_well_sep = potential_well_sep - np.min(potential_well_sep)
        n_points_potential = len(potential_well_sep)
        
        # Compute deltaE frame corresponding to the separatrix
        max_potential = np.max(potential_well_sep)
        max_deltaE = np.sqrt(max_potential / eom_factor_dE)
    
        # Saving the Hamilotian values corresponding to dE=0 (with high resolution
        # to be used in the integral to compute J further)
        H_array_dE0 = potential_well_sep
        
        # Initializing the grids by reducing the resolution to a 
        # n_points_grid*n_points_grid frame.
        n_points_grid = int(1e3)
        potential_well_indexes = np.arange(0,n_points_potential)
        grid_indexes = np.arange(0,n_points_grid) * n_points_potential / n_points_grid
        theta_coord_low_res = np.interp(grid_indexes, potential_well_indexes, theta_coord_sep)
        deltaE_coord_array = np.linspace(-max_deltaE, max_deltaE, n_points_grid)
        potential_well_low_res = np.interp(grid_indexes, potential_well_indexes, potential_well_sep)
        theta_grid, deltaE_grid = np.meshgrid(theta_coord_low_res, deltaE_coord_array)
        potential_well_grid = np.meshgrid(potential_well_low_res, potential_well_low_res)[0]
        
        # Computing the action J by integrating the dE trajectories
        J_array_dE0 = np.zeros(n_points_grid)
        
        warnings.filterwarnings("ignore")
        
        for i in range(0, n_points_grid):
            dE_trajectory = np.sqrt((potential_well_low_res[i] - H_array_dE0)/eom_factor_dE)
            dE_trajectory[np.isnan(dE_trajectory)] = 0
            J_array_dE0[i] = 2 / (2*np.pi) * np.trapz(dE_trajectory, dx=theta_resolution * 
                                                      FullRingAndRF.ring_radius / 
                                                      (Beam.beta_r * c)) 

                
        warnings.filterwarnings("default")
        
        # Sorting the H and J functions in order to be able to interpolate the function J(H)
        H_array_dE0 = potential_well_low_res
        sorted_H_dE0 = H_array_dE0[H_array_dE0.argsort()]
        sorted_J_dE0 = J_array_dE0[H_array_dE0.argsort()]
        
        # Calculating the H and J grid
        H_grid = eom_factor_dE * deltaE_grid**2 + potential_well_grid
        J_grid = np.interp(H_grid, sorted_H_dE0, sorted_J_dE0, left = 0, right = np.inf)
        
        # Computing bunch length as a function of H/J if needed
        # Bunch length can be calculated as 4-rms, Gaussian fit, or FWHM
        density_variable_option = distribution_options['density_variable']
        if distribution_options.has_key('bunch_length'):        
            time_low_res = theta_coord_low_res * FullRingAndRF.ring_radius / (Beam.beta_r * c)
            tau = 0.0
            # Choice of either H or J as the variable used
            if density_variable_option is 'density_from_J':
                X_low = sorted_J_dE0[0]
                X_hi = sorted_J_dE0[n_points_grid - 1]
            elif density_variable_option is 'density_from_H':
                X_low = sorted_H_dE0[0]
                X_hi = sorted_H_dE0[n_points_grid - 1]
            else:
                raise RuntimeError('The density_variable option was not recognized')
            
            bunch_length_accuracy = (time_low_res[1] - time_low_res[0]) / 10
            
            # Iteration to find H0/J0 from the bunch length
            while np.abs(distribution_options['bunch_length']-tau) > bunch_length_accuracy:
                # Takes middle point of the interval [X_low,X_hi]                
                X0 = 0.5 * (X_low + X_hi)
                
                # Calculating the line density for the parameter X0
                if density_variable_option is 'density_from_J':
                    density_grid = distribution_density_function(J_grid, distribution_options['type'], X0, distribution_options['exponent'])
                elif density_variable_option is 'density_from_H':
                    density_grid = distribution_density_function(H_grid, distribution_options['type'], X0, distribution_options['exponent'])                
                
                density_grid = density_grid / np.sum(density_grid)                
                line_density = np.sum(density_grid, axis = 0)
                
                # Calculating the bunch length of that line density
                if (line_density>0).any():
                    tau = 4.0 * np.sqrt(np.sum((time_low_res - np.sum(line_density * time_low_res) / np.sum(line_density))**2 * line_density) / np.sum(line_density))            
                    
                    if distribution_options.has_key('bunch_length_fit'):
                        slices = Slices(Beam,n_points_grid)
                        slices.n_macroparticles = line_density
                        slices.bins_centers = time_low_res
                        slices.edges = np.linspace(slices.bins_centers[0]-(slices.bins_centers[1]-slices.bins_centers[0])/2,slices.bins_centers[-1]+(slices.bins_centers[1]-slices.bins_centers[0])/2,len(slices.bins_centers)+1)
                        
                        if distribution_options['bunch_length_fit'] is 'gauss':                              
                            slices.bl_gauss = tau
                            slices.bp_gauss = np.sum(line_density * time_low_res) / np.sum(line_density)
                            slices.gaussian_fit()
                            tau = slices.bl_gauss
                        elif distribution_options['bunch_length_fit'] is 'fwhm': 
                            slices.fwhm()
                            tau = slices.bl_fwhm
                
                # Update of the interval for the next iteration
                if tau >= distribution_options['bunch_length']:
                    X_hi = X0
                else:
                    X_low = X0
            
            if density_variable_option is 'density_from_J':
                J0 = X0
            elif density_variable_option is 'density_from_H':
                H0 = X0
        
        # Computing the density grid
        if distribution_options['type'] is not 'user_input_table':
            if density_variable_option is 'density_from_J':
                if distribution_options.has_key('emittance'):
                    J0 = distribution_options['emittance']/ (2*np.pi)
                density_grid = distribution_density_function(J_grid, distribution_options['type'], J0, distribution_options['exponent'])
            elif density_variable_option is 'density_from_H':
                if distribution_options.has_key('emittance'):
                    H0 = np.interp(distribution_options['emittance'] / (2*np.pi), sorted_J_dE0, sorted_H_dE0)
                density_grid = distribution_density_function(H_grid, distribution_options['type'], H0, distribution_options['exponent'])
        else:
            if density_variable_option is 'density_from_J':
                density_grid = np.interp(J_grid, distribution_options['user_table_action'], distribution_options['user_table_density'])
            elif density_variable_option is 'density_from_H':
                density_grid = np.interp(H_grid, distribution_options['user_table_action'], distribution_options['user_table_density'])
        
        # Normalizing the grid
        density_grid = density_grid / np.sum(density_grid)
        
        # Induced voltage contribution
        if TotalInducedVoltage is not None:
            # Calculating the line density
            line_density = np.sum(density_grid, axis = 0)
            line_density = line_density / np.sum(line_density) * Beam.n_macroparticles

            # Calculating the induced voltage
            induced_voltage_object = copy.deepcopy(TotalInducedVoltage)
                        
            # Inputing new line density
            induced_voltage_object.slices.n_macroparticles = line_density
            induced_voltage_object.slices.bins_centers = theta_coord_low_res * FullRingAndRF.ring_radius / (Beam.beta_r * c)
            induced_voltage_object.slices.edges = np.linspace(induced_voltage_object.slices.bins_centers[0]-(induced_voltage_object.slices.bins_centers[1]-induced_voltage_object.slices.bins_centers[0])/2,induced_voltage_object.slices.bins_centers[-1]+(induced_voltage_object.slices.bins_centers[1]-induced_voltage_object.slices.bins_centers[0])/2,len(induced_voltage_object.slices.bins_centers)+1)
            induced_voltage_object.slices.n_slices = n_points_grid
            induced_voltage_object.slices.fit_option = 'off'
            
            # Re-calculating the sources of wakes/impedances according to this slicing
            induced_voltage_object.reprocess(induced_voltage_object.slices)
            
            # Calculating the induced voltage
            induced_voltage_length_sep = int(np.ceil((theta_coord_array[-1] -  theta_coord_low_res[0]) / (theta_coord_low_res[1] - theta_coord_low_res[0])))
            induced_voltage = induced_voltage_object.induced_voltage_sum(Beam, length = induced_voltage_length_sep)
            theta_induced_voltage = np.linspace(theta_coord_low_res[0], theta_coord_low_res[0] + (induced_voltage_length_sep - 1) * (theta_coord_low_res[1] - theta_coord_low_res[0]), induced_voltage_length_sep)
            
            # Calculating the induced potential
            induced_potential_low_res = - eom_factor_potential * np.insert(cumtrapz(induced_voltage, dx=theta_induced_voltage[1]-theta_induced_voltage[0]),0,0)
            induced_potential = np.interp(theta_coord_array, theta_induced_voltage, induced_potential_low_res)
        else:
            line_density = np.sum(density_grid, axis = 0)
            line_density = line_density / np.sum(line_density) * Beam.n_macroparticles
         
    # Populating the bunch
    indexes = np.random.choice(range(0,np.size(density_grid)), Beam.n_macroparticles, p=density_grid.flatten())
     
    Beam.theta = np.ascontiguousarray(theta_grid.flatten()[indexes] + (np.random.rand(Beam.n_macroparticles) - 0.5) * (theta_coord_low_res[1]-theta_coord_low_res[0]))
    Beam.dE = np.ascontiguousarray(deltaE_grid.flatten()[indexes] + (np.random.rand(Beam.n_macroparticles) - 0.5) * (deltaE_coord_array[1]-deltaE_coord_array[0]))

    if TotalInducedVoltage is not None:
        return [theta_coord_low_res, line_density], induced_voltage_object
    else:
        return [theta_coord_low_res, line_density]
    


def _distribution_density_function(action_array, dist_type, length, exponent = None):
    '''
    *Distribution density (formulas from Laclare).*
    '''
    
    if dist_type in ['binomial', 'waterbag', 'parabolic_amplitude', 'parabolic_line']:
        if dist_type is 'waterbag':
            exponent = 0
        elif dist_type is 'parabolic_amplitude':
            exponent = 1
        elif dist_type is 'parabolic_line':
            exponent = 0.5
        
        warnings.filterwarnings("ignore")
        density_function = (1 - action_array / length)**exponent
        warnings.filterwarnings("default")
        density_function[action_array > length] = 0
        return density_function
    
    elif dist_type is 'gaussian':
        density_function = np.exp(- 2 * action_array / length)
        return density_function

    else:
        raise RuntimeError('The dist_type option was not recognized')
    


def line_density_function(coord_array, dist_type, bunch_length, bunch_position = 0, exponent = None):
    '''
    *Line density*
    '''
    
    if dist_type in ['binomial', 'waterbag', 'parabolic_amplitude', 'parabolic_line']:
        if dist_type is 'waterbag':
            exponent = 0
        elif dist_type is 'parabolic_amplitude':
            exponent = 1
        elif dist_type is 'parabolic_line':
            exponent = 0.5
        
        warnings.filterwarnings("ignore")
        density_function = (1 - ((coord_array - bunch_position) / (bunch_length/2))**2)**(exponent+0.5)
        warnings.filterwarnings("default")
        density_function[np.abs(coord_array - bunch_position) > bunch_length/2 ] = 0
        return density_function
    
    elif dist_type is 'gaussian':
        sigma = bunch_length/4
        density_function = np.exp(- (coord_array - bunch_position)**2 /(2*sigma**2))
        return density_function
    
    elif dist_type is 'cosine_squared':
        warnings.filterwarnings("ignore")
        density_function = np.cos(np.pi * (coord_array - bunch_position) / bunch_length)**2
        warnings.filterwarnings("default")
        density_function[np.abs(coord_array - bunch_position) > bunch_length/2 ] = 0
        return density_function   



def longitudinal_bigaussian(GeneralParameters, RFSectionParameters, beam, 
                            sigma_x, sigma_y = None, xunit = None, yunit = None, 
                            seed = None, reinsertion = 'off'):
    
    warnings.filterwarnings("once")
    if GeneralParameters.n_sections > 1:
        warnings.warn("WARNING: longitudinal_bigaussian is not yet properly computed for several sections!")
    if RFSectionParameters.n_rf > 1:
        warnings.warn("longitudinal_bigaussian for multiple RF is not yet implemented")
    
    counter = RFSectionParameters.counter[0]
    
    harmonic = RFSectionParameters.harmonic[0,counter]
    energy = RFSectionParameters.energy[counter]
    beta = RFSectionParameters.beta_r[counter]
    if sigma_y == None:
        voltage = RFSectionParameters.voltage[0,counter]
        eta0 = RFSectionParameters.eta_0[counter]
    
    if xunit == None or xunit == 'rad':
        sigma_theta = sigma_x
    elif xunit == 'm':
        sigma_theta = sigma_x / (- beam.ring_radius * harmonic) 
    elif xunit == 'ns':       
        sigma_theta = sigma_x * beta * c * 1.e-9 / beam.ring_radius
    
    phi_s = RFSectionParameters.phi_s[counter]
    
    if sigma_y != None:    
        if yunit == None or yunit == 'eV':
            sigma_dE = sigma_y
        elif yunit == '1':    
            sigma_dE = sigma_y * beta**2 * energy
    else:
        phi_b = harmonic*sigma_theta + phi_s
        sigma_dE = np.sqrt( voltage * energy * beta**2  
             * (np.cos(phi_b) - np.cos(phi_s) + (phi_b - phi_s) * np.sin(phi_s)) 
             / (np.pi * harmonic * eta0) )
        
    
    beam.sigma_theta = sigma_theta
    beam.sigma_dE = sigma_dE
    
    np.random.seed(seed)
    beam.theta = sigma_theta * np.random.randn(beam.n_macroparticles) \
                        + phi_s/harmonic
    beam.dE = sigma_dE * np.random.randn(beam.n_macroparticles)
    
    if reinsertion is 'on':
    
        itemindex = np.where(is_in_separatrix(GeneralParameters, RFSectionParameters,
                                     beam.theta, beam.dE, beam.delta) == False)[0]
         
        while itemindex.size != 0:
         
            beam.theta[itemindex] = sigma_theta * np.random.randn(itemindex.size) \
                    + phi_s/harmonic
            beam.dE[itemindex] = sigma_dE * np.random.randn(itemindex.size)
            itemindex = np.where(is_in_separatrix(GeneralParameters, 
                                RFSectionParameters, beam.theta, beam.dE, beam.delta) 
                                 == False)[0]

  

