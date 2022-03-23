'''
**Module to generate multibunch distributions**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**, **Theodoros Argyropoulos**
'''

from __future__ import division, print_function, absolute_import
from builtins import range
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import gc
from ..utils import bmath as bm

from ..beam.beam import Beam
from ..beam.distributions import matched_from_distribution_function,\
                           matched_from_line_density, populate_bunch,\
                           distribution_function, potential_well_cut,\
                           X0_from_bunch_length

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

    if main_harmonic_option == 'lowest_freq':
            main_harmonic = np.min(harmonics)
    elif main_harmonic_option == 'highest_voltage':
        main_harmonic = np.min(harmonics[voltages == np.max(voltages)])
    elif isinstance(main_harmonic_option, int) or isinstance(main_harmonic_option, float):
        if harmonics[harmonics == main_harmonic_option].size == 0:
            #GenerationError
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
            #DistributionError
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

        if distribution_options['type'] == 'user_input':
            distribution_function_input = distribution_options['function']
        else:
            distribution_function_input = None

        if distribution_options['type'] == 'user_input_table':
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
                
    beam.dt = beamIteration.dt.astype(dtype=bm.precision.real_t, order='C', copy=False)
    beam.dE = beamIteration.dE.astype(dtype=bm.precision.real_t, order='C', copy=False)
    gc.collect()


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

    if main_harmonic_option == 'lowest_freq':
            main_harmonic = np.min(harmonics)
    elif main_harmonic_option == 'highest_voltage':
        main_harmonic = np.min(harmonics[voltages == np.max(voltages)])
    elif isinstance(main_harmonic_option, int) or isinstance(main_harmonic_option, float):
        if harmonics[harmonics == main_harmonic_option].size == 0:
            #GenerationError
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
            #GenerationError
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

        if line_density_options['type'] == 'user_input':
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
                
    beam.dt = beamIteration.dt.astype(dtype=bm.precision.real_t, order='C', copy=False)
    beam.dE = beamIteration.dE.astype(dtype=bm.precision.real_t, order='C', copy=False)
    gc.collect()


def match_beam_from_distribution(beam, FullRingAndRF, GeneralParameters,
                                  distribution_options, n_bunches,
                                  bunch_spacing_buckets,
                                  main_harmonic_option='lowest_freq',
                                  TotalInducedVoltage=None, n_iterations=1,
                                  n_points_potential=1e4,
                                  dt_margin_percent=0.40, seed=None,):
    '''
    *This function generates n equaly spaced bunches for a stationary
    distribution and try to match them with intensity effects.*

    *The corresponding distributions are specified by their exponent:*

    .. math::
        g_0(J) \\sim (1-J/J_0)^{\\text{exponent}}}

    *Knowing the distribution, to generate the phase space:
    - Compute the potential U
    - The value of H can be computed thanks to U
    - The action J can be integrated over the whole phase space
    - 2piJ = emittance, this restrict the value of J0 (or H0)
    - with g0(H) we can randomize the macroparticles*
    '''
#------------------------------------------------------------------------
# USEFUL VARIABLES
#------------------------------------------------------------------------
    # Slicing necessary only with intensity effects
    if TotalInducedVoltage is not None:
        profile = TotalInducedVoltage.profile

    # Ring informations, Trev, energy, RF parameters ...
    rf_params = FullRingAndRF.RingAndRFSection_list[0].rf_params
    t_rev = rf_params.t_rev[0]
    n_rf = rf_params.n_rf
    beta = rf_params.beta[0]
    E = rf_params.energy[0]
    charge = rf_params.charge
#    acceleration_kick = FullRingAndRF.RingAndRFSection_list[0].acceleration_kick[0]

    # Minimum omega_rf is used to compute the size of the bucket
    omega_rf = []
    for i in range(n_rf):
        omega_rf += [rf_params.omega_rf[i][0]]
    omega_rf = np.array(omega_rf)

    eta_0 = rf_params.eta_0[0]

    # Coefficient of Kin and Pot part of the hamiltonian
    normalization_DeltaE = np.abs(eta_0) / (2.*beta**2*E)
    normalization_potential = np.sign(eta_0)*charge/t_rev

    intensity_per_bunch = beam.intensity/n_bunches
    n_macro_per_bunch = int(beam.n_macroparticles/n_bunches)
    bucket_size_tau = 2*np.pi/(np.min(omega_rf))

#------------------------------------------------------------------------
# GENERATES N BUNCHES WITHOUT INTENSITY EFFECTS
#------------------------------------------------------------------------

    FullRingAndRF.potential_well_generation(n_points=n_points_potential,
                                    dt_margin_percent=dt_margin_percent,
                                    main_harmonic_option=main_harmonic_option)

    # Restrict the potential well inside the separatrix and put min on 0
    potential_well_coordinates, potential_well = potential_well_cut(\
        FullRingAndRF.potential_well_coordinates,\
        FullRingAndRF.potential_well)
    potential_well = potential_well - np.min(potential_well)

    # Temporary beam, everything is done in the first bucket and then
    # shifted to plug into the real beam.
    temporary_beam = Beam(GeneralParameters, n_macro_per_bunch, intensity_per_bunch)

    # Bunches placed in all the buckets without intensity effects
    # Loop the match function to have "different" bunches in each bucket
    matched_bunch_list = []
    for indexBunch in range(n_bunches):
        (time_grid, deltaE_grid, distribution, time_resolution,
            energy_resolution, single_profile) = match_a_bunch(
                normalization_DeltaE, temporary_beam,
                potential_well_coordinates,
                potential_well, seed, distribution_options,
                full_ring_and_RF=FullRingAndRF)
        matched_bunch_list.append(
            (time_grid, deltaE_grid, distribution, time_resolution,
             energy_resolution, single_profile))

    print(str(n_bunches)+' stationary bunches without intensity generated')
#------------------------------------------------------------------------
# REMATCH THE BUNCHES WITH INTENSITY EFFECTS
#------------------------------------------------------------------------
    if TotalInducedVoltage is not None:
        print('Applying intensity effects ...')
        previous_well = potential_well
        for it in range(n_iterations):
            conv = 0.
            # Compute the induced voltage/potential for all the beam
            profile.n_macroparticles[:] = 0
            for indexBunch in range(n_bunches):
                profile.n_macroparticles += np.interp(
                    profile.bin_centers,
                    potential_well_coordinates +
                    indexBunch*bunch_spacing_buckets*bucket_size_tau,
                    matched_bunch_list[indexBunch][5],
                    left=0, right=0)
            profile.n_macroparticles[:] *= 1/(np.sum(profile.n_macroparticles)) * beam.n_macroparticles

            TotalInducedVoltage.induced_voltage_sum()

            induced_voltage_coordinates = TotalInducedVoltage.time_array
            induced_voltage = TotalInducedVoltage.induced_voltage
            induced_potential = - normalization_potential * cumtrapz(
                induced_voltage,
                dx=induced_voltage_coordinates[1] -
                induced_voltage_coordinates[0],
                initial=0)

            for indexBunch in range(n_bunches):
                # Extract the induced potential for the specific bucket
                induced_potential_bunch = np.interp(potential_well_coordinates\
                + indexBunch*bunch_spacing_buckets*bucket_size_tau,\
                induced_voltage_coordinates, induced_potential)

                distorted_pot_well = potential_well+induced_potential_bunch
                distorted_pot_well -= np.min(distorted_pot_well)

                # Recompute the phase space distribution for the new
                # perturbed potential (containing induced_potential_bunch)
                matched_bunch_list[indexBunch] = match_a_bunch(
                    normalization_DeltaE, temporary_beam,
                    potential_well_coordinates,
                    distorted_pot_well, seed,
                    distribution_options,
                    full_ring_and_RF=FullRingAndRF)

            conv = np.sqrt(np.sum((previous_well-distorted_pot_well)**2.)) / len(distorted_pot_well)
            previous_well = distorted_pot_well

            print('iteration ' + str(it+1) + ', convergence parameter = ' + str(conv))

            profile.n_macroparticles[:] = 0
            for indexBunch in range(n_bunches):
                profile.n_macroparticles += np.interp(
                    profile.bin_centers,
                    potential_well_coordinates +
                    indexBunch*bunch_spacing_buckets*bucket_size_tau,
                    matched_bunch_list[indexBunch][5],
                    left=0, right=0)
            profile.n_macroparticles[:] *= 1/(np.sum(profile.n_macroparticles)) * beam.n_macroparticles

            TotalInducedVoltage.induced_voltage_sum()

    for indexBunch in range(n_bunches):

        (time_grid, deltaE_grid, distribution, time_resolution,
         energy_resolution, single_profile) = matched_bunch_list[indexBunch]
        populate_bunch(temporary_beam, time_grid, deltaE_grid, distribution,
                       time_resolution, energy_resolution, seed)
                
        length_dt = len(temporary_beam.dt)
        length_dE = len(temporary_beam.dE)
        
        beam.dt[indexBunch*length_dt:(indexBunch+1)*length_dt] = np.array(
            temporary_beam.dt)+(indexBunch *bunch_spacing_buckets *bucket_size_tau)
        beam.dE[indexBunch*length_dE:(indexBunch+1)*length_dE] = np.array(
            temporary_beam.dE)
    
    beam.dt = beam.dt.astype(dtype=bm.precision.real_t, order='C', copy=False)
    beam.dE = beam.dE.astype(dtype=bm.precision.real_t, order='C', copy=False)
    gc.collect()


def match_beam_from_distribution_multibatch(beam, FullRingAndRF, GeneralParameters,
                                  distribution_options, n_bunches,
                                  bunch_spacing_buckets, n_batch,
                                  batch_spacing_buckets,
                                  main_harmonic_option='lowest_freq',
                                  TotalInducedVoltage=None, n_iterations=1,
                                  n_points_potential=1e4,
                                  dt_margin_percent=0.40, seed=None):
    '''
    *This function generates n equaly spaced bunches for a stationary
    distribution and try to match them with intensity effects.*

    *Then it copies the batch n_batch times with spacing batch_spacing_buckets*

    *The corresponding distributions are specified by their exponent:*

    .. math::
        g_0(J) \\sim (1-J/J_0)^{\\text{exponent}}}

    *Knowing the distribution, to generate the phase space:
    - Compute the potential U
    - The value of H can be computed thanks to U
    - The action J can be integrated over the whole phase space
    - 2piJ = emittance, this restrict the value of J0 (or H0)
    - with g0(H) we can randomize the macroparticles*
    '''
#------------------------------------------------------------------------
# USEFUL VARIABLES
#------------------------------------------------------------------------
    # Ring informations, Trev, energy, RF parameters ...
    rf_params = FullRingAndRF.RingAndRFSection_list[0].rf_params
    n_rf = rf_params.n_rf
    # Slicing necessary only with intensity effects
    if TotalInducedVoltage is not None:
        profile = TotalInducedVoltage.profile

        t_rev = rf_params.t_rev[0]
        beta = rf_params.beta[0]
        E = rf_params.energy[0]
        charge = rf_params.charge
        eta_0 = rf_params.eta_0[0]

        normalization_DeltaE = np.abs(eta_0) / (2.*beta**2*E)
        normalization_potential = np.sign(eta_0)*charge/t_rev

    # Ring informations, Trev, energy, RF parameters ...
#    beta = rf_params.beta[0]
#    E = rf_params.energy[0]
#    charge = rf_params.charge
#    acceleration_kick = FullRingAndRF.RingAndRFSection_list[0].acceleration_kick[0]

    # Minimum omega_rf is used to compute the size of the bucket
    omega_rf = []
    for i in range(n_rf):
        omega_rf += [rf_params.omega_rf[i][0]]
    omega_rf = np.array(omega_rf)

#    eta_0 = rf_params.eta_0[0]

#    # Coefficient of Kin and Pot part of the hamiltonian
#    normalization_DeltaE = np.abs(eta_0) / (2.*beta**2*E)
#    normalization_potential = np.sign(eta_0)*charge/t_rev

    intensity_per_bunch = beam.intensity/n_bunches/n_batch
    n_macro_per_bunch = int(beam.n_macroparticles/n_bunches/n_batch)
    bucket_size_tau = 2*np.pi/(np.min(omega_rf))

    temporary_batch = Beam(GeneralParameters, int(n_macro_per_bunch*n_bunches), (intensity_per_bunch*n_bunches))

#    print(temporary_batch.dt)
    match_beam_from_distribution(temporary_batch, FullRingAndRF, GeneralParameters,
                                  distribution_options, n_bunches,bunch_spacing_buckets,
                                  TotalInducedVoltage=None, n_iterations=n_iterations,
                                  n_points_potential=n_points_potential)

#    matched_from_distribution_density_multibunch(temporary_batch, GeneralParameters, FullRingAndRF, distribution_options,
#                                          n_bunches, bunch_spacing_buckets,
#                                          TotalInducedVoltage = TotalInducedVoltage,
#                                          n_iterations_input = n_iterations)
    length_dt = len(temporary_batch.dt)
    print(length_dt)
    for index_batch in range(n_batch):
        beam.dt[index_batch*length_dt:(index_batch+1)*length_dt] = temporary_batch.dt + index_batch*(n_bunches-1)*bunch_spacing_buckets*bucket_size_tau + (index_batch)*batch_spacing_buckets*bucket_size_tau
        beam.dE[index_batch*length_dt:(index_batch+1)*length_dt] = temporary_batch.dE

    plt.figure('copymultibatch')
    plt.plot(beam.dt[::100],beam.dE[::100],'b.')
    plt.figure('temporarybatch')
    plt.plot(temporary_batch.dt[::100],temporary_batch.dE[::100],'b.')
    plt.figure('profile before induced voltage')
    profile.track()
    plt.plot(profile.bin_centers,profile.n_macroparticles)
    plt.figure('beamInSlice')
    plt.plot(profile.Beam.dt[::100],profile.Beam.dE[::100],'b.')
#------------------------------------------------------------------------
# REMATCH THE BUNCHES WITH INTENSITY EFFECTS
#------------------------------------------------------------------------
    if TotalInducedVoltage is not None:
#        TotalInducedVoltage.profile.Beam.dt[:len(beam.dt)] = beam.dt
#        TotalInducedVoltage.profile.Beam.dE[:len(beam.dE)] = beam.dE
        print('Applying intensity effects ...')
        for it in range(n_iterations):
            conv = 0.
            # Compute the induced voltage/potential for all the beam
            profile.track()
            TotalInducedVoltage.induced_voltage_sum()

            plt.figure('profile before induced voltage')
            profile.track()
            plt.plot(profile.bin_centers,profile.n_macroparticles)
#
#            plt.figure('inducedvoltage before induced voltage')
#            profile.track()
#            plt.plot(TotalInducedVoltage.time_array,TotalInducedVoltage.induced_voltage)
#
            induced_voltage_coordinates = TotalInducedVoltage.time_array
            induced_voltage = TotalInducedVoltage.induced_voltage
            induced_potential = - normalization_potential * cumtrapz(induced_voltage, dx=induced_voltage_coordinates[1] - induced_voltage_coordinates[0], initial=0)

            plt.figure('testInducedVolt')
            plt.plot(induced_voltage_coordinates,induced_voltage)
            plt.figure('testInducedPot')
            plt.plot(induced_voltage_coordinates,induced_potential)

            FullRingAndRF.potential_well_generation(n_points=n_points_potential,
                                            dt_margin_percent=dt_margin_percent,
                                            main_harmonic_option=main_harmonic_option)

            # Restrict the potential well inside the separatrix and put min on 0
            potential_well_coordinates, potential_well = potential_well_cut(\
                FullRingAndRF.potential_well_coordinates,\
                FullRingAndRF.potential_well)
            potential_well = potential_well - np.min(potential_well)

            temporary_beam = Beam(GeneralParameters, n_macro_per_bunch, intensity_per_bunch)
            for indexBatch in range(n_batch):
                for indexBunch in range(n_bunches):
                    # Extract the induced potential for the specific bucket
                    induced_potential_bunch = np.interp(potential_well_coordinates\
                    + indexBunch*bunch_spacing_buckets*bucket_size_tau\
                    + indexBatch*(batch_spacing_buckets + (n_bunches-1)*bunch_spacing_buckets)*bucket_size_tau,\
                    induced_voltage_coordinates, induced_potential)

                    # Recompute the phase space distribution for the new
                    # perturbed potential (containing induced_potential_bunch)
                    match_a_bunch(normalization_DeltaE, temporary_beam,
                                  potential_well_coordinates,
                                  potential_well+induced_potential_bunch, seed,
                                  distribution_options,
                                  full_ring_and_RF=FullRingAndRF)

                    dt = temporary_beam.dt
                    dE = temporary_beam.dE

                    # Compute RMS emittance to observe convergence
                    conv += np.pi*np.std(dt)*np.std(dE)

                    length_dt = len(dt)
                    length_dE = len(dE)
                    beam.dt[(indexBunch+n_bunches*indexBatch)*length_dt:(indexBunch+n_bunches*indexBatch+1)*length_dt] = dt+(indexBunch *bunch_spacing_buckets *bucket_size_tau) + indexBatch*(batch_spacing_buckets + (n_bunches-1)*bunch_spacing_buckets)*bucket_size_tau
                    beam.dE[(indexBunch+n_bunches*indexBatch)*length_dE:(indexBunch+n_bunches*indexBatch+1)*length_dE] = dE


            print('iteration ' + str(it) + ', average RMS emittance (4sigma) = ' + str(4*conv/n_bunches))
            profile.track()
            TotalInducedVoltage.induced_voltage_sum()


def compute_X_grid(normalization_DeltaE, time_array, potential_well,
                   distribution_variable):

    # Delta Energy array
    max_DeltaE = np.sqrt(np.max(potential_well)/normalization_DeltaE)
    coord_array_DeltaE = np.linspace(-float(max_DeltaE),float(max_DeltaE), len(time_array))

    # Resolution in time and energy
    time_resolution = time_array[1]-time_array[0]
    energy_resolution = coord_array_DeltaE[1]-coord_array_DeltaE[0]

    # Grid
    time_grid, deltaE_grid = np.meshgrid(time_array, coord_array_DeltaE)
    potential_well_grid = np.meshgrid(potential_well, potential_well)[0]
    H_grid = normalization_DeltaE * deltaE_grid**2 + potential_well_grid

    # Compute the action J
    J_array = np.zeros(shape=potential_well.shape, dtype=float)
    for i in range(len(J_array)):
        DELTA = np.sqrt((potential_well[i]-potential_well)[potential_well <= potential_well[i]]/normalization_DeltaE)
        J_array[i] = 1./np.pi*np.trapz(DELTA, dx=time_array[1]-time_array[0])

    # Compute J grid
    sorted_H = potential_well[potential_well.argsort()]
    sorted_J = J_array[potential_well.argsort()]

    if distribution_variable == 'Action':
        J_grid = np.interp(H_grid, sorted_H, sorted_J,\
                           left=0, right=np.inf)
        return sorted_H, sorted_J, J_grid, time_grid, deltaE_grid,\
               time_resolution, energy_resolution
    else:
        return sorted_H, sorted_J, H_grid, time_grid, deltaE_grid,\
                       time_resolution, energy_resolution

def compute_H0(emittance, H, J):
    #  Estimation of H corresponding to the emittance
    return np.interp(emittance / (2.*np.pi), J, H)

def match_a_bunch(normalization_DeltaE, beam, potential_well_coordinates,\
                  potential_well, seed, distribution_options,\
                  full_ring_and_RF=None):

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
        distribution_variable = 'Hamiltonian'

    H, J, X_grid, time_grid, deltaE_grid, time_resolution, energy_resolution =\
    compute_X_grid(normalization_DeltaE, potential_well_coordinates,
                   potential_well,distribution_variable)

    # Choice of either H or J as the variable used
    if distribution_variable == 'Action':
        sorted_X = J
    elif distribution_variable == 'Hamiltonian':
        sorted_X = H
    else:
        #DistributionError
        raise SystemError('distribution_variable should be Action or Hamiltonian')

    if bunch_length is not None:
        n_points_grid = X_grid.shape[0]
        X0 = X0_from_bunch_length(bunch_length, bunch_length_fit, X_grid, sorted_X,
                         n_points_grid, potential_well_coordinates,
                         distribution_function, distribution_type,
                         distribution_exponent, beam, full_ring_and_RF)
    elif emittance is not None:
        X0 = compute_H0(emittance, H, J)
    else:
        #DistributionError
        raise SystemError('You should specify either bunch_length or emittance')

    distribution = distribution_function(X_grid, distribution_type, X0, exponent=distribution_exponent)
    distribution[X_grid>np.max(H)] = 0
    distribution = distribution / np.sum(distribution)

    profile = np.sum(distribution, axis=0)
    
    return (time_grid, deltaE_grid, distribution, time_resolution,
            energy_resolution, profile)
