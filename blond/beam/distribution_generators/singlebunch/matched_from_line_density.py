# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Module to generate distributions**

:Authors: **Danilo Quartullo**, **Helga Timko**, **Alexandre Lasheen**,
          **Juan F. Esteban Mueller**, **Theodoros Argyropoulos**,
          **Joel Repond**
"""

from __future__ import annotations

import copy
import gc
import warnings
from os import PathLike
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import scipy
from packaging.version import Version

if Version(scipy.__version__) >= Version("1.14"):
    from scipy.integrate import cumulative_trapezoid as cumtrapz
else:
    from scipy.integrate import cumtrapz
from .methods import populate_bunch

from ....trackers.utilities import minmax_location, potential_well_cut
from ....utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Literal, Optional

    from numpy.typing import NDArray

    from ...beam import Beam
    from ....impedances.impedance import TotalInducedVoltage
    from ....trackers.tracker import FullRingAndRF, MainHarmonicOptionType
    from ....utils.types import (LineDensityInputType,
                                 ExtraVoltageDictType, HalfOptionType, LineDensityDistType)


def __line_density_by_exponent(bunch_length: float,
                               bunch_position: float,
                               coord_array: NDArray,
                               exponent: float) -> NDArray:
    warnings.filterwarnings("ignore")
    line_density_ = ((1 - (2.0 * (coord_array - bunch_position) /
                           bunch_length) ** 2) ** (exponent + 0.5))
    warnings.filterwarnings("default")
    line_density_[np.abs(coord_array - bunch_position)
                  > bunch_length / 2] = 0
    return line_density_


def line_density(coord_array: NDArray, dist_type: str, bunch_length: float,
                 bunch_position: float = 0.0,
                 exponent: Optional[float] = None) -> NDArray:
    """
    *Line density*
    """

    if dist_type == 'waterbag':
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        exponent = 0
        line_density_ = __line_density_by_exponent(bunch_length, bunch_position, coord_array, exponent)

    elif dist_type == 'parabolic_amplitude':
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        exponent = 1
        line_density_ = __line_density_by_exponent(bunch_length, bunch_position, coord_array, exponent)

    elif dist_type == 'parabolic_line':
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        exponent = 0.5
        line_density_ = __line_density_by_exponent(bunch_length, bunch_position, coord_array, exponent)
    elif dist_type == 'binomial':
        assert exponent is not None, "Please specify exponent!"
        line_density_ = __line_density_by_exponent(bunch_length, bunch_position, coord_array, exponent)

    elif dist_type == 'gaussian':
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        sigma = bunch_length / 4
        line_density_ = np.exp(-(coord_array - bunch_position)
                                ** 2 / (2 * sigma ** 2))

    elif dist_type == 'cosine_squared':
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        warnings.filterwarnings("ignore")
        line_density_ = (np.cos(np.pi * (coord_array - bunch_position) /
                                bunch_length) ** 2)
        warnings.filterwarnings("default")
        line_density_[np.abs(coord_array - bunch_position)
                      > bunch_length / 2] = 0
    else:
        # DistributionError
        raise RuntimeError('The dist_type option was not recognized')

    return line_density_


@handle_legacy_kwargs
def matched_from_line_density(beam: Beam,
                              full_ring_and_rf: FullRingAndRF,
                              line_density_input: Optional[LineDensityInputType] = None,
                              main_harmonic_option: MainHarmonicOptionType = 'lowest_freq',
                              total_induced_voltage: Optional[TotalInducedVoltage] = None,
                              plot: bool = False,
                              figdir: PathLike | str = 'fig',
                              half_option: HalfOptionType = 'first',
                              extra_voltage_dict: Optional[ExtraVoltageDictType] = None,
                              n_iterations: int = 100,
                              n_points_potential: int = int(1e4),
                              n_points_grid: int = int(1e3),
                              dt_margin_percent: float = 0.40,
                              n_points_abel: float = 1e4,
                              bunch_length: Optional[float] = None,
                              line_density_type: LineDensityDistType | Literal['user_input'] | None = None,
                              line_density_exponent: Optional[float] = None,
                              seed: Optional[int] = None,
                              process_pot_well: bool = True) \
        -> (tuple[list[NDArray], TotalInducedVoltage]
            | tuple[list[NDArray, NDArray], list[NDArray, NDArray]]):
    """
    *Function to generate a beam by inputting the line density. The distribution
    function is then reconstructed with the Abel transform and the particles
    randomly generated.*
    """

    # Initialize variables depending on the accelerator parameters
    slippage_factor = full_ring_and_rf.ring_and_rf_section[0].rf_params.eta_0[0]

    eom_factor_dE = abs(slippage_factor) / (2 * beam.beta ** 2. * beam.energy)
    eom_factor_potential = (np.sign(slippage_factor) * beam.particle.charge
                            / full_ring_and_rf.ring_and_rf_section[0].rf_params.t_rev[0])

    #: *Number of points to be used in the potential well calculation*
    n_points_potential = int(n_points_potential)
    # Generate potential well
    full_ring_and_rf.potential_well_generation(
        n_points=n_points_potential,
        dt_margin_percent=dt_margin_percent,
        main_harmonic_option=main_harmonic_option)
    potential_well = full_ring_and_rf.potential_well
    time_potential = full_ring_and_rf.potential_well_coordinates

    extra_potential = 0

    if extra_voltage_dict is not None:
        extra_voltage_time_input = extra_voltage_dict['time_array']
        extra_voltage_input = extra_voltage_dict['voltage_array']
        extra_potential_input = - (eom_factor_potential
                                   * cumtrapz(extra_voltage_input,
                                              dx=float(extra_voltage_time_input[1])
                                                 - float(extra_voltage_time_input[0]),
                                              initial=0))
        extra_potential = np.interp(time_potential, extra_voltage_time_input,
                                    extra_potential_input)

    if line_density_type != 'user_input':
        # Time coordinates for the line density
        n_points_line_den = int(1e4)
        time_line_den = np.linspace(float(time_potential[0]),
                                    float(time_potential[-1]),
                                    n_points_line_den)
        line_den_resolution = time_line_den[1] - time_line_den[0]

        # Normalizing the line density
        line_density_ = line_density(time_line_den, line_density_type,
                                     bunch_length, exponent=line_density_exponent,
                                     bunch_position=float(time_potential[0]
                                                          + time_potential[-1])
                                                    / 2)

        line_density_ -= np.min(line_density_)
        line_density_ *= beam.n_macroparticles / np.sum(line_density_)

    elif line_density_type == 'user_input':
        # Time coordinates for the line density
        time_line_den = line_density_input['time_line_den']
        n_points_line_den = len(time_line_den)
        line_den_resolution = time_line_den[1] - time_line_den[0]

        # Normalizing the line density
        line_density_ = line_density_input['line_density']
        line_density_ -= np.min(line_density_)
        line_density_ *= beam.n_macroparticles / np.sum(line_density_)
    else:
        # GenerationError
        raise RuntimeError('The input for the matched_from_line_density ' +
                           'function was not recognized')

    induced_potential_final = 0

    if total_induced_voltage is not None:
        # Calculating the induced voltage
        induced_voltage_object = copy.deepcopy(total_induced_voltage)
        profile = induced_voltage_object.profile

        # Inputting new line density
        profile.cut_options.cut_left = time_line_den[0] - \
                                       0.5 * line_den_resolution
        profile.cut_options.cut_right = time_line_den[-1] + \
                                        0.5 * line_den_resolution
        profile.cut_options.n_slices = n_points_line_den
        profile.cut_options.cuts_unit = 's'
        profile.cut_options.set_cuts()
        profile.set_slices_parameters()
        profile.n_macroparticles = line_density_

        # Re-calculating the sources of wakes/impedances according to this
        # slicing
        induced_voltage_object.reprocess()

        # Calculating the induced voltage
        induced_voltage_object.induced_voltage_sum()
        induced_voltage = induced_voltage_object.induced_voltage

        # Calculating the induced potential
        induced_potential = -(eom_factor_potential
                              * cumtrapz(induced_voltage, dx=profile.bin_size,
                                         initial=0))

    # Centering the bunch in the potential well
    for i in range(0, n_iterations):
        if total_induced_voltage is not None:
            # Interpolating the potential well
            induced_potential_final = np.interp(time_potential,
                                                profile.bin_centers,
                                                induced_potential)

        # Induced voltage contribution
        total_potential = (potential_well + induced_potential_final +
                           extra_potential)

        # Potential well calculation around the separatrix
        if not process_pot_well:
            time_potential_sep, potential_well_sep = time_potential, total_potential
        else:
            time_potential_sep, potential_well_sep = potential_well_cut(
                time_potential, total_potential)

        minmax_positions_potential, minmax_values_potential = \
            minmax_location(time_potential_sep, potential_well_sep)
        minmax_positions_profile, minmax_values_profile = \
            minmax_location(time_line_den[line_density_ != 0],
                            line_density_[line_density_ != 0])

        n_minima_potential = len(minmax_positions_potential[0])
        n_maxima_profile = len(minmax_positions_profile[1])

        # Warnings
        if n_maxima_profile > 1:
            print('Warning: the profile has several max, the highest one ' +
                  'is taken. Be sure the profile is monotonous and not too noisy.')
            max_profile_pos = minmax_positions_profile[1][np.where(
                minmax_values_profile[1] == minmax_values_profile[1].max())]
        else:
            max_profile_pos = minmax_positions_profile[1]
        if n_minima_potential > 1:
            print('Warning: the potential well has several min, the deepest ' +
                  'one is taken. The induced potential is probably splitting ' +
                  'the potential well.')
            min_potential_pos = minmax_positions_potential[0][np.where(
                minmax_values_potential[0] == minmax_values_potential[0].min())]
        else:
            min_potential_pos = minmax_positions_potential[0]

        # Moving the bunch (not for the last iteration if intensity effects
        # are present)
        if total_induced_voltage is None:
            time_line_den -= max_profile_pos - min_potential_pos
            max_profile_pos -= max_profile_pos - min_potential_pos
        elif i != n_iterations - 1:
            time_line_den -= max_profile_pos - min_potential_pos
            # Update profile
            profile.cut_options.cut_left -= max_profile_pos - min_potential_pos
            profile.cut_options.cut_right -= max_profile_pos - min_potential_pos
            profile.cut_options.set_cuts()
            profile.set_slices_parameters()

    # Taking the first/second half of line density and potential
    n_points_abel = int(n_points_abel)

    abel_both_step = 1
    if half_option == 'both':
        abel_both_step = 2
        distribution_function_average = np.zeros((n_points_abel, 2))
        hamiltonian_average = np.zeros((n_points_abel, 2))

    for abel_index in range(0, abel_both_step):
        if half_option == 'first':
            half_indexes = np.where((time_line_den >= time_line_den[0]) *
                                    (time_line_den <= max_profile_pos))
        if half_option == 'second':
            half_indexes = np.where((time_line_den >= max_profile_pos) *
                                    (time_line_den <= time_line_den[-1]))
        if half_option == 'both' and abel_index == 0:
            half_indexes = np.where((time_line_den >= time_line_den[0]) *
                                    (time_line_den <= max_profile_pos))
        if half_option == 'both' and abel_index == 1:
            half_indexes = np.where((time_line_den >= max_profile_pos) *
                                    (time_line_den <= time_line_den[-1]))

        line_den_half = line_density_[half_indexes]
        time_half = time_line_den[half_indexes]
        potential_half = np.interp(time_half, time_potential_sep,
                                   potential_well_sep)
        potential_half = potential_half - np.min(potential_half)

        # Derivative of the line density
        line_den_diff = np.diff(line_den_half) / line_den_resolution

        time_line_den_diff = time_half[:-1] + line_den_resolution / 2
        line_den_diff = np.interp(time_half, time_line_den_diff, line_den_diff,
                                  left=0, right=0)

        # Interpolating the line density derivative and potential well for
        # Abel transform
        time_abel = np.linspace(
            float(time_half[0]), float(time_half[-1]), n_points_abel)
        line_den_diff_abel = np.interp(time_abel, time_half, line_den_diff)
        potential_abel = np.interp(time_abel, time_half, potential_half)

        distribution_function_ = np.zeros(n_points_abel)
        hamiltonian_coord = np.zeros(n_points_abel)

        # Abel transform
        warnings.filterwarnings("ignore")

        if (half_option == 'first') or (half_option == 'both' and
                                        abel_index == 0):
            for i in range(0, n_points_abel):
                integrand = (line_den_diff_abel[:i + 1] /
                             np.sqrt(potential_abel[:i + 1] - potential_abel[i]))

                if len(integrand) > 2:
                    integrand[-1] = integrand[-2] + (integrand[-2] -
                                                     integrand[-3])
                elif len(integrand) > 1:
                    integrand[-1] = integrand[-2]
                else:
                    integrand = np.array([0])

                distribution_function_[i] = (np.sqrt(eom_factor_dE) / np.pi *
                                             np.trapezoid(integrand,
                                                          dx=line_den_resolution))

                hamiltonian_coord[i] = potential_abel[i]

        if (half_option == 'second') or (half_option == 'both' and
                                         abel_index == 1):
            for i in range(0, n_points_abel):
                integrand = (line_den_diff_abel[i:] /
                             np.sqrt(potential_abel[i:] - potential_abel[i]))

                if len(integrand) > 2:
                    integrand[0] = integrand[1] + (integrand[2] - integrand[1])
                if len(integrand) > 1:
                    integrand[0] = integrand[1]
                else:
                    integrand = np.array([0])

                distribution_function_[i] = -(np.sqrt(eom_factor_dE) / np.pi *
                                              np.trapezoid(integrand,
                                                           dx=line_den_resolution))
                hamiltonian_coord[i] = potential_abel[i]

        warnings.filterwarnings("default")

        # Cleaning the distribution function from unphysical results
        distribution_function_[np.isnan(distribution_function_)] = 0
        distribution_function_[distribution_function_ < 0] = 0

        if half_option == 'both':
            hamiltonian_average[:, abel_index] = hamiltonian_coord
            distribution_function_average[:, abel_index] = \
                distribution_function_

    if half_option == 'both':
        hamiltonian_coord = hamiltonian_average[:, 0]
        distribution_function_ = ((distribution_function_average[:, 0]
                                   + np.interp(
                    hamiltonian_coord,
                    hamiltonian_average[:, 1],
                    distribution_function_average[:, 1]))
                                  / 2)

    # Compute deltaE frame corresponding to the separatrix
    max_potential = np.max(potential_half)
    max_deltaE = np.sqrt(max_potential / eom_factor_dE)

    # Initializing the grids by reducing the resolution to a
    # n_points_grid*n_points_grid frame
    time_for_grid = np.linspace(float(time_line_den[0]),
                                float(time_line_den[-1]),
                                n_points_grid)
    deltaE_for_grid = np.linspace(-float(max_deltaE),
                                  float(max_deltaE), n_points_grid)
    potential_well_for_grid = np.interp(time_for_grid, time_potential_sep,
                                        potential_well_sep)
    potential_well_for_grid = (potential_well_for_grid -
                               potential_well_for_grid.min())

    time_grid, deltaE_grid = np.meshgrid(time_for_grid, deltaE_for_grid)
    potential_well_grid = np.meshgrid(potential_well_for_grid,
                                      potential_well_for_grid)[0]

    hamiltonian_grid = eom_factor_dE * deltaE_grid ** 2 + potential_well_grid

    # Sort the distribution function and generate the density grid
    hamiltonian_argsort = np.argsort(hamiltonian_coord)
    hamiltonian_coord = hamiltonian_coord.take(hamiltonian_argsort)
    distribution_function_ = distribution_function_.take(hamiltonian_argsort)
    density_grid = np.interp(hamiltonian_grid, hamiltonian_coord,
                             distribution_function_)

    density_grid[np.isnan(density_grid)] = 0
    density_grid[density_grid < 0] = 0
    # Normalizing density
    density_grid = density_grid / np.sum(density_grid)
    reconstructed_line_den = np.sum(density_grid, axis=0)

    # Ploting the result
    if plot:
        plt.figure('Generated bunch')
        plt.plot(time_line_den, line_density_)
        plt.plot(time_for_grid, reconstructed_line_den /
                 np.max(reconstructed_line_den) * np.max(line_density_))
        plt.title('Line densities')
        if plot == 'show':
            plt.show()
        elif plot == 'savefig':
            fign = figdir + '/generated_bunch.png'
            plt.savefig(fign)

    # Populating the bunch
    populate_bunch(beam, time_grid, deltaE_grid, density_grid,
                   float(time_for_grid[1] - time_for_grid[0]),
                   float(deltaE_for_grid[1] - deltaE_for_grid[0]), seed)

    if total_induced_voltage is not None:
        # Inputting new line density
        profile.cut_options.cut_left = (time_for_grid[0] - 0.5
                                        * (time_for_grid[1] - time_for_grid[0]))
        profile.cut_options.cut_right = time_for_grid[-1] + 0.5 * (
                time_for_grid[1] - time_for_grid[0])
        profile.cut_options.n_slices = n_points_grid
        profile.cut_options.set_cuts()
        profile.set_slices_parameters()
        profile.n_macroparticles = reconstructed_line_den * beam.n_macroparticles

        # Re-calculating the sources of wakes/impedances according to this
        # slicing
        induced_voltage_object.reprocess()

        # Calculating the induced voltage
        induced_voltage_object.induced_voltage_sum()
        gc.collect()
        return [hamiltonian_coord, distribution_function_], \
            induced_voltage_object

    gc.collect()
    return [hamiltonian_coord, distribution_function_], \
        [time_line_den, line_density_]
