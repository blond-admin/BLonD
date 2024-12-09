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
from typing import TYPE_CHECKING

import numpy as np
import scipy
from packaging.version import Version

if Version(scipy.__version__) >= Version("1.14"):
    from scipy.integrate import cumulative_trapezoid as cumtrapz
else:
    from scipy.integrate import cumtrapz
from .methods import populate_bunch
from ..methods import x0_from_bunch_length
from ....trackers.utilities import potential_well_cut
from ....utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Callable, Optional

    from numpy.typing import NDArray

    from ...beam import Beam
    from ....impedances.impedance import TotalInducedVoltage
    from ....trackers.tracker import FullRingAndRF, MainHarmonicOptionType
    from ....utils.types import (DistributionUserTableType, ExtraVoltageDictType, DistributionVariableType,
                                 BunchLengthFitTypes)
    from ....utils.types import DistTypeDistFunction


def __distribution_function_by_exponent(action_array: NDArray, exponent: float,
                                        length: float) -> NDArray:
    warnings.filterwarnings("ignore")
    distribution_function_ = (1 - action_array / length) ** exponent
    warnings.filterwarnings("default")
    distribution_function_[action_array > length] = 0
    return distribution_function_


def distribution_function(action_array: NDArray,
                          dist_type: DistTypeDistFunction,
                          length: float,
                          exponent: Optional[float] = None):
    """
    *Distribution function (formulas from Laclare).*
    """

    if dist_type == 'waterbag':
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        exponent = 0
        distribution_ = __distribution_function_by_exponent(action_array,
                                                            exponent, length)

    elif dist_type == 'parabolic_amplitude':
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        exponent = 1
        distribution_ = __distribution_function_by_exponent(action_array,
                                                            exponent, length)

    elif dist_type == 'parabolic_line':
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        exponent = 0.5
        distribution_ = __distribution_function_by_exponent(action_array,
                                                            exponent, length)

    elif dist_type == 'binomial':
        assert exponent is not None, "Please specify exponent"
        distribution_ = __distribution_function_by_exponent(action_array,
                                                            exponent, length)

    elif dist_type == 'gaussian':
        distribution_ = np.exp(- 2 * action_array / length)

    else:
        # DistributionError
        raise RuntimeError('The dist_type option was not recognized')

    return distribution_


@handle_legacy_kwargs
def matched_from_distribution_function(beam: Beam, full_ring_and_rf: FullRingAndRF,
                                       distribution_function_input: Optional[Callable] = None,
                                       # needs better specification
                                       distribution_user_table: Optional[DistributionUserTableType] = None,
                                       main_harmonic_option: MainHarmonicOptionType = 'lowest_freq',
                                       total_induced_voltage: Optional[TotalInducedVoltage] = None,
                                       n_iterations: int = 1,
                                       n_points_potential: float = 1e4,
                                       n_points_grid: int = int(1e3),
                                       dt_margin_percent: float = 0.40,
                                       extra_voltage_dict: Optional[ExtraVoltageDictType] = None,
                                       seed: Optional[int] = None,
                                       distribution_exponent: Optional[float] = None,
                                       distribution_type: Optional[str] = None,
                                       emittance: Optional[float] = None,
                                       bunch_length: Optional[float] = None,
                                       bunch_length_fit: Optional[BunchLengthFitTypes] = None,
                                       distribution_variable: DistributionVariableType = 'Hamiltonian',
                                       process_pot_well: bool = True,
                                       turn_number: int = 0) \
        -> tuple[list[NDArray], TotalInducedVoltage] | list[NDArray]:
    """
    *Function to generate a beam by inputting the distribution function (by
    choosing the type of distribution and the emittance).
    The potential well is preprocessed to check for the min/max and center
    the frame around the separatrix.
    An error will be raised if there is not a full potential well (2 max
    and 1 min at least), or if there are several wells (more than 2 max and
    1 min, this case will be treated in the future).
    An adjustable margin (40% by default) is applied in order to be able to
    catch the min/max of the potential well that might be on the edge of the
    frame. The slippage factor should be updated to take the higher orders.
    Outputs should be added in order for the user to check step by step if
    his bunch is going to be well generated. More detailed 'step by step'
    documentation should be implemented
    The user can input a custom distribution function by setting the parameter
    distribution_type = 'user_input' and passing the function in the
    parameter distribution_options['function'], with the following definition:
    distribution_function(action_array, dist_type, length, exponent=None).
    The user can also add an input table by setting the parameter
    distribution_type = 'user_input_table',
    distribution_options['user_table_action'] = array of action (in H or in J)
    and distribution_options['user_table_distribution']*
    """
    # TODO cleanup this overloaded function
    # Loading the distribution function if provided by the user
    if distribution_function_input is not None:
        distribution_function_ = distribution_function_input
    else:
        distribution_function_ = distribution_function

    # Initialize variables depending on the accelerator parameters
    slippage_factor = full_ring_and_rf.ring_and_rf_section[0].rf_params.eta_0[turn_number]
    beta = full_ring_and_rf.ring_and_rf_section[0].rf_params.beta[turn_number]
    energy = full_ring_and_rf.ring_and_rf_section[0].rf_params.energy[turn_number]

    eom_factor_dE = abs(slippage_factor) / (2 * beta ** 2. * energy)
    eom_factor_potential = (np.sign(slippage_factor) * beam.particle.charge
                            / (full_ring_and_rf.ring_and_rf_section[0].rf_params.t_rev[turn_number]))

    #: *Number of points to be used in the potential well calculation*
    n_points_potential = int(n_points_potential)
    # Generate potential well
    full_ring_and_rf.potential_well_generation(turn=turn_number,
                                               n_points=n_points_potential,
                                               dt_margin_percent=dt_margin_percent,
                                               main_harmonic_option=main_harmonic_option)
    potential_well: NDArray = full_ring_and_rf.potential_well
    time_potential: NDArray = full_ring_and_rf.potential_well_coordinates

    induced_potential = 0

    # Extra potential from previous bunches (for multi-bunch generation)
    extra_potential = 0
    if extra_voltage_dict is not None:
        extra_voltage_time_input = extra_voltage_dict['time_array']
        extra_voltage_input = extra_voltage_dict['voltage_array']
        extra_potential_input = -(eom_factor_potential
                                  * cumtrapz(extra_voltage_input,
                                             dx=(float(extra_voltage_time_input[1])
                                                 - float(extra_voltage_time_input[0])),
                                             initial=0)
                                  )
        extra_potential = np.interp(time_potential, extra_voltage_time_input,
                                    extra_potential_input)

    total_potential = potential_well + induced_potential + extra_potential

    if not total_induced_voltage:
        n_iterations = 1
    else:
        induced_voltage_object = copy.deepcopy(total_induced_voltage)
        profile = induced_voltage_object.profile

    dE_trajectory = np.zeros(n_points_potential)
    for i in range(n_iterations):
        old_potential = copy.deepcopy(total_potential)

        # Adding the induced potential to the RF potential
        total_potential = (potential_well + induced_potential +
                           extra_potential)

        sse = np.sqrt(np.sum((old_potential - total_potential) ** 2))

        print('Matching the bunch... (iteration: ' + str(i) + ' and sse: ' +
              str(sse) + ')')

        # Process the potential well in order to take a frame around the separatrix
        if not process_pot_well:
            time_potential_sep, potential_well_sep = time_potential, total_potential
        else:
            time_potential_sep, potential_well_sep = potential_well_cut(
                time_potential, total_potential)

        # Potential is shifted to put the minimum on 0
        potential_well_sep = potential_well_sep - np.min(potential_well_sep)

        # Compute deltaE frame corresponding to the separatrix
        max_potential = np.max(potential_well_sep)
        max_deltaE = np.sqrt(max_potential / eom_factor_dE)

        # Initializing the grids by reducing the resolution to a
        # n_points_grid*n_points_grid frame
        time_potential_low_res = np.linspace(float(time_potential_sep[0]),
                                             float(time_potential_sep[-1]),
                                             n_points_grid)
        time_resolution_low = (float(time_potential_low_res[1]) -
                               float(time_potential_low_res[0]))
        deltaE_coord_array = np.linspace(-float(max_deltaE), float(max_deltaE),
                                         n_points_grid)
        potential_well_low_res = np.interp(time_potential_low_res,
                                           time_potential_sep,
                                           potential_well_sep)
        time_grid, deltaE_grid = np.meshgrid(time_potential_low_res,
                                             deltaE_coord_array)
        potential_well_grid = np.meshgrid(potential_well_low_res,
                                          potential_well_low_res)[0]

        # Computing the action J by integrating the dE trajectories
        J_array_dE0 = np.zeros(n_points_grid)

        full_ring_and_RF2 = copy.deepcopy(full_ring_and_rf)
        for j in range(n_points_grid):
            # Find left and right time coordinates for a given hamiltonian
            # value
            time_indexes = np.where(potential_well_low_res <=
                                    potential_well_low_res[j])[0]
            left_time = time_potential_low_res[np.max((0, time_indexes[0]))]
            right_time = time_potential_low_res[np.min((time_indexes[-1],
                                                        n_points_grid - 1))]
            # Potential well calculation with high resolution in that frame
            time_potential_high_res = np.linspace(float(left_time),
                                                  float(right_time),
                                                  n_points_potential)
            full_ring_and_RF2.potential_well_generation(
                n_points=n_points_potential,
                time_array=time_potential_high_res,
                main_harmonic_option=main_harmonic_option)
            pot_well_high_res = full_ring_and_RF2.potential_well

            if total_induced_voltage is not None and i != 0:
                induced_potential_hires = np.interp(
                    time_potential_high_res,
                    time_potential, induced_potential +
                                    extra_potential, left=0, right=0)
                pot_well_high_res += induced_potential_hires
                pot_well_high_res -= pot_well_high_res.min()

            # Integration to calculate action
            dE_trajectory[pot_well_high_res <= potential_well_low_res[j]] = \
                np.sqrt((potential_well_low_res[j]
                         - pot_well_high_res[pot_well_high_res
                                             <= potential_well_low_res[j]])
                        / eom_factor_dE)
            dE_trajectory[pot_well_high_res > potential_well_low_res[j]] = 0
            # todo fix trapz naming
            J_array_dE0[j] = (1 / np.pi
                              * np.trapezoid(dE_trajectory,
                                             dx=time_potential_high_res[1]
                                                - time_potential_high_res[0]))

        # Sorting the H and J functions to be able to interpolate J(H)
        H_array_dE0 = potential_well_low_res
        sorted_H_dE0 = H_array_dE0[H_array_dE0.argsort()]
        sorted_J_dE0 = J_array_dE0[H_array_dE0.argsort()]

        # Calculating the H and J grid
        H_grid = eom_factor_dE * deltaE_grid ** 2 + potential_well_grid
        J_grid = np.interp(H_grid, sorted_H_dE0, sorted_J_dE0, left=0,
                           right=np.inf)

        # Choice of either H or J as the variable used
        if distribution_variable == 'Action':
            sorted_X_dE0 = sorted_J_dE0
            X_grid = J_grid
        elif distribution_variable == 'Hamiltonian':
            sorted_X_dE0 = sorted_H_dE0
            X_grid = H_grid
        else:
            # DistributionError
            raise RuntimeError('The distribution_variable option was not ' +
                               'recognized')

        # Computing bunch length as a function of H/J if needed
        # Bunch length can be calculated as 4-rms, Gaussian fit, or FWHM
        if bunch_length is not None:
            X0 = x0_from_bunch_length(bunch_length, bunch_length_fit,
                                      X_grid, sorted_X_dE0, n_points_grid,
                                      time_potential_low_res,
                                      distribution_function_,
                                      distribution_type, distribution_exponent,
                                      beam, full_ring_and_rf)

        elif emittance is not None:
            if distribution_variable == 'Action':
                X0 = emittance / (2 * np.pi)
            elif distribution_variable == 'Hamiltonian':
                X0 = np.interp(emittance / (2 * np.pi), sorted_J_dE0,
                               sorted_H_dE0)

        # Computing the density grid
        if distribution_user_table is None:
            density_grid = distribution_function_(X_grid, distribution_type,
                                                  X0, distribution_exponent)
        else:
            density_grid = np.interp(
                X_grid,
                distribution_user_table['user_table_action'],
                distribution_user_table['user_table_distribution']
            )

        # Normalizing the grid
        density_grid[H_grid > np.max(H_array_dE0)] = 0
        density_grid = density_grid / np.sum(density_grid)

        # Calculating the line density
        line_density_ = np.sum(density_grid, axis=0)
        line_density_ *= beam.n_macroparticles / np.sum(line_density_)

        # Induced voltage contribution
        if total_induced_voltage is not None:
            # Inputing new line density
            profile.cut_options.cut_left = (time_potential_low_res[0]
                                            - 0.5 * time_resolution_low)
            profile.cut_options.cut_right = (time_potential_low_res[-1]
                                             + 0.5 * time_resolution_low)
            profile.cut_options.n_slices = n_points_grid
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
            induced_potential_low_res = -(eom_factor_potential
                                          * cumtrapz(induced_voltage,
                                                     dx=time_resolution_low,
                                                     initial=0))
            induced_potential = np.interp(time_potential,
                                          time_potential_low_res,
                                          induced_potential_low_res,
                                          left=0, right=0)
        del full_ring_and_RF2
        gc.collect()
    # Populating the bunch
    populate_bunch(beam, time_grid, deltaE_grid, density_grid,
                   time_resolution_low, float(deltaE_coord_array[1]
                                              - deltaE_coord_array[0]), seed)

    if total_induced_voltage is not None:
        return [time_potential_low_res, line_density_], induced_voltage_object
    else:
        return [time_potential_low_res, line_density_]
