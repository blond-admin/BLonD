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
try:
    np.trapezoid
except AttributeError:
    np.trapezoid = np.trapz

from .profile import CutOptions, Profile
from ..trackers.utilities import (
    is_in_separatrix,
    minmax_location,
    potential_well_cut,
)
from ..utils import bmath as bm
from ..utils.legacy_support import handle_legacy_kwargs


if TYPE_CHECKING:
    from typing import Literal, Callable, Optional

    from numpy.typing import NDArray as NumpyArray

    from ..input_parameters.ring import Ring
    from ..input_parameters.rf_parameters import RFStation
    from .beam import Beam
    from ..impedances.impedance import TotalInducedVoltage
    from ..trackers.tracker import FullRingAndRF, MainHarmonicOptionType
    from ..utils.types import (
        DistributionUserTableType,
        LineDensityInputType,
        ExtraVoltageDictType,
        DistributionVariableType,
        HalfOptionType,
        BunchLengthFitTypes,
        LineDensityDistType,
    )
    from ..utils.types import DistTypeDistFunction


@handle_legacy_kwargs
def matched_from_line_density(
    beam: Beam,
    full_ring_and_rf: FullRingAndRF,
    line_density_input: Optional[LineDensityInputType] = None,
    main_harmonic_option: MainHarmonicOptionType = "lowest_freq",
    total_induced_voltage: Optional[TotalInducedVoltage] = None,
    plot: bool = False,
    figdir: PathLike | str = "fig",
    half_option: HalfOptionType = "first",
    extra_voltage_dict: Optional[ExtraVoltageDictType] = None,
    n_iterations: int = 100,
    n_points_potential: int = int(1e4),
    n_points_grid: int = int(1e3),
    dt_margin_percent: float = 0.40,
    n_points_abel: float = 1e4,
    bunch_length: Optional[float] = None,
    line_density_type: LineDensityDistType
    | Literal["user_input"]
    | None = None,
    line_density_exponent: Optional[float] = None,
    seed: Optional[int] = None,
    process_pot_well: bool = True,
) -> (
    tuple[list[NumpyArray], TotalInducedVoltage]
    | tuple[list[NumpyArray, NumpyArray], list[NumpyArray, NumpyArray]]
):
    """
    *Function to generate a beam by inputting the line density. The distribution
    function is then reconstructed with the Abel transform and the particles
    randomly generated.*
    """

    # Initialize variables depending on the accelerator parameters
    slippage_factor = full_ring_and_rf.ring_and_rf_section[0].rf_params.eta_0[
        0
    ]

    eom_factor_dE = abs(slippage_factor) / (2 * beam.beta**2.0 * beam.energy)
    eom_factor_potential = (
        np.sign(slippage_factor)
        * beam.particle.charge
        / full_ring_and_rf.ring_and_rf_section[0].rf_params.t_rev[0]
    )

    #: *Number of points to be used in the potential well calculation*
    n_points_potential = int(n_points_potential)
    # Generate potential well
    full_ring_and_rf.potential_well_generation(
        n_points=n_points_potential,
        dt_margin_percent=dt_margin_percent,
        main_harmonic_option=main_harmonic_option,
    )
    potential_well = full_ring_and_rf.potential_well
    time_potential = full_ring_and_rf.potential_well_coordinates

    extra_potential = 0

    if extra_voltage_dict is not None:
        extra_voltage_time_input = extra_voltage_dict["time_array"]
        extra_voltage_input = extra_voltage_dict["voltage_array"]
        extra_potential_input = -(
            eom_factor_potential
            * cumtrapz(
                extra_voltage_input,
                dx=float(extra_voltage_time_input[1])
                - float(extra_voltage_time_input[0]),
                initial=0,
            )
        )
        extra_potential = np.interp(
            time_potential, extra_voltage_time_input, extra_potential_input
        )

    if line_density_type != "user_input":
        # Time coordinates for the line density
        n_points_line_den = int(1e4)
        time_line_den = np.linspace(
            float(time_potential[0]),
            float(time_potential[-1]),
            n_points_line_den,
        )
        line_den_resolution = time_line_den[1] - time_line_den[0]

        # Normalizing the line density
        line_density_ = line_density(
            time_line_den,
            line_density_type,
            bunch_length,
            exponent=line_density_exponent,
            bunch_position=float(time_potential[0] + time_potential[-1]) / 2,
        )

        line_density_ -= np.min(line_density_)
        line_density_ *= beam.n_macroparticles / np.sum(line_density_)

    elif line_density_type == "user_input":
        # Time coordinates for the line density
        time_line_den = line_density_input["time_line_den"]
        n_points_line_den = len(time_line_den)
        line_den_resolution = time_line_den[1] - time_line_den[0]

        # Normalizing the line density
        line_density_ = line_density_input["line_density"]
        line_density_ -= np.min(line_density_)
        line_density_ *= beam.n_macroparticles / np.sum(line_density_)
    else:
        # GenerationError
        raise RuntimeError(
            "The input for the matched_from_line_density "
            + "function was not recognized"
        )

    induced_potential_final = 0

    if total_induced_voltage is not None:
        # Calculating the induced voltage
        induced_voltage_object = copy.deepcopy(total_induced_voltage)
        profile = induced_voltage_object.profile

        # Inputting new line density
        profile.cut_options.cut_left = (
            time_line_den[0] - 0.5 * line_den_resolution
        )
        profile.cut_options.cut_right = (
            time_line_den[-1] + 0.5 * line_den_resolution
        )
        profile.cut_options.n_slices = n_points_line_den
        profile.cut_options.cuts_unit = "s"
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
        induced_potential = -(
            eom_factor_potential
            * cumtrapz(induced_voltage, dx=profile.bin_size, initial=0)
        )

    # Centering the bunch in the potential well
    for i in range(0, n_iterations):
        if total_induced_voltage is not None:
            # Interpolating the potential well
            induced_potential_final = np.interp(
                time_potential, profile.bin_centers, induced_potential
            )

        # Induced voltage contribution
        total_potential = (
            potential_well + induced_potential_final + extra_potential
        )

        # Potential well calculation around the separatrix
        if not process_pot_well:
            time_potential_sep, potential_well_sep = (
                time_potential,
                total_potential,
            )
        else:
            time_potential_sep, potential_well_sep = potential_well_cut(
                time_potential, total_potential
            )

        minmax_positions_potential, minmax_values_potential = minmax_location(
            time_potential_sep, potential_well_sep
        )
        minmax_positions_profile, minmax_values_profile = minmax_location(
            time_line_den[line_density_ != 0],
            line_density_[line_density_ != 0],
        )

        n_minima_potential = len(minmax_positions_potential[0])
        n_maxima_profile = len(minmax_positions_profile[1])

        # Warnings
        if n_maxima_profile > 1:
            print(
                "Warning: the profile has several max, the highest one "
                + "is taken. Be sure the profile is monotonous and not too noisy."
            )
            max_profile_pos = minmax_positions_profile[1][
                np.where(
                    minmax_values_profile[1] == minmax_values_profile[1].max()
                )
            ]
        else:
            max_profile_pos = minmax_positions_profile[1]
        if n_minima_potential > 1:
            print(
                "Warning: the potential well has several min, the deepest "
                + "one is taken. The induced potential is probably splitting "
                + "the potential well."
            )
            min_potential_pos = minmax_positions_potential[0][
                np.where(
                    minmax_values_potential[0]
                    == minmax_values_potential[0].min()
                )
            ]
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
            profile.cut_options.cut_right -= (
                max_profile_pos - min_potential_pos
            )
            profile.cut_options.set_cuts()
            profile.set_slices_parameters()

    # Taking the first/second half of line density and potential
    n_points_abel = int(n_points_abel)

    abel_both_step = 1
    if half_option == "both":
        abel_both_step = 2
        distribution_function_average = np.zeros((n_points_abel, 2))
        hamiltonian_average = np.zeros((n_points_abel, 2))

    for abel_index in range(0, abel_both_step):
        if half_option == "first":
            half_indexes = np.where(
                (time_line_den >= time_line_den[0])
                * (time_line_den <= max_profile_pos)
            )
        if half_option == "second":
            half_indexes = np.where(
                (time_line_den >= max_profile_pos)
                * (time_line_den <= time_line_den[-1])
            )
        if half_option == "both" and abel_index == 0:
            half_indexes = np.where(
                (time_line_den >= time_line_den[0])
                * (time_line_den <= max_profile_pos)
            )
        if half_option == "both" and abel_index == 1:
            half_indexes = np.where(
                (time_line_den >= max_profile_pos)
                * (time_line_den <= time_line_den[-1])
            )

        line_den_half = line_density_[half_indexes]
        time_half = time_line_den[half_indexes]
        potential_half = np.interp(
            time_half, time_potential_sep, potential_well_sep
        )
        potential_half = potential_half - np.min(potential_half)

        # Derivative of the line density
        line_den_diff = np.diff(line_den_half) / line_den_resolution

        time_line_den_diff = time_half[:-1] + line_den_resolution / 2
        line_den_diff = np.interp(
            time_half, time_line_den_diff, line_den_diff, left=0, right=0
        )

        # Interpolating the line density derivative and potential well for
        # Abel transform
        time_abel = np.linspace(
            float(time_half[0]), float(time_half[-1]), n_points_abel
        )
        line_den_diff_abel = np.interp(time_abel, time_half, line_den_diff)
        potential_abel = np.interp(time_abel, time_half, potential_half)

        distribution_function_ = np.zeros(n_points_abel)
        hamiltonian_coord = np.zeros(n_points_abel)

        # Abel transform
        warnings.filterwarnings("ignore")

        if (half_option == "first") or (
            half_option == "both" and abel_index == 0
        ):
            for i in range(0, n_points_abel):
                integrand = line_den_diff_abel[: i + 1] / np.sqrt(
                    potential_abel[: i + 1] - potential_abel[i]
                )

                if len(integrand) > 2:
                    integrand[-1] = integrand[-2] + (
                        integrand[-2] - integrand[-3]
                    )
                elif len(integrand) > 1:
                    integrand[-1] = integrand[-2]
                else:
                    integrand = np.array([0])

                distribution_function_[i] = (
                    np.sqrt(eom_factor_dE)
                    / np.pi
                    * np.trapezoid(integrand, dx=line_den_resolution)
                )

                hamiltonian_coord[i] = potential_abel[i]

        if (half_option == "second") or (
            half_option == "both" and abel_index == 1
        ):
            for i in range(0, n_points_abel):
                integrand = line_den_diff_abel[i:] / np.sqrt(
                    potential_abel[i:] - potential_abel[i]
                )

                if len(integrand) > 2:
                    integrand[0] = integrand[1] + (integrand[2] - integrand[1])
                if len(integrand) > 1:
                    integrand[0] = integrand[1]
                else:
                    integrand = np.array([0])

                distribution_function_[i] = -(
                    np.sqrt(eom_factor_dE)
                    / np.pi
                    * np.trapezoid(integrand, dx=line_den_resolution)
                )
                hamiltonian_coord[i] = potential_abel[i]

        warnings.filterwarnings("default")

        # Cleaning the distribution function from unphysical results
        distribution_function_[np.isnan(distribution_function_)] = 0
        distribution_function_[distribution_function_ < 0] = 0

        if half_option == "both":
            hamiltonian_average[:, abel_index] = hamiltonian_coord
            distribution_function_average[:, abel_index] = (
                distribution_function_
            )

    if half_option == "both":
        hamiltonian_coord = hamiltonian_average[:, 0]
        distribution_function_ = (
            distribution_function_average[:, 0]
            + np.interp(
                hamiltonian_coord,
                hamiltonian_average[:, 1],
                distribution_function_average[:, 1],
            )
        ) / 2

    # Compute deltaE frame corresponding to the separatrix
    max_potential = np.max(potential_half)
    max_deltaE = np.sqrt(max_potential / eom_factor_dE)

    # Initializing the grids by reducing the resolution to a
    # n_points_grid*n_points_grid frame
    time_for_grid = np.linspace(
        float(time_line_den[0]), float(time_line_den[-1]), n_points_grid
    )
    deltaE_for_grid = np.linspace(
        -float(max_deltaE), float(max_deltaE), n_points_grid
    )
    potential_well_for_grid = np.interp(
        time_for_grid, time_potential_sep, potential_well_sep
    )
    potential_well_for_grid = (
        potential_well_for_grid - potential_well_for_grid.min()
    )

    time_grid, deltaE_grid = np.meshgrid(time_for_grid, deltaE_for_grid)
    potential_well_grid = np.meshgrid(
        potential_well_for_grid, potential_well_for_grid
    )[0]

    hamiltonian_grid = eom_factor_dE * deltaE_grid**2 + potential_well_grid

    # Sort the distribution function and generate the density grid
    hamiltonian_argsort = np.argsort(hamiltonian_coord)
    hamiltonian_coord = hamiltonian_coord.take(hamiltonian_argsort)
    distribution_function_ = distribution_function_.take(hamiltonian_argsort)
    density_grid = np.interp(
        hamiltonian_grid, hamiltonian_coord, distribution_function_
    )

    density_grid[np.isnan(density_grid)] = 0
    density_grid[density_grid < 0] = 0
    # Normalizing density
    density_grid = density_grid / np.sum(density_grid)
    reconstructed_line_den = np.sum(density_grid, axis=0)

    # Ploting the result
    if plot:
        plt.figure("Generated bunch")
        plt.plot(time_line_den, line_density_)
        plt.plot(
            time_for_grid,
            reconstructed_line_den
            / np.max(reconstructed_line_den)
            * np.max(line_density_),
        )
        plt.title("Line densities")
        if plot == "show":
            plt.show()
        elif plot == "savefig":
            fign = figdir + "/generated_bunch.png"
            plt.savefig(fign)

    # Populating the bunch
    populate_bunch(
        beam,
        time_grid,
        deltaE_grid,
        density_grid,
        float(time_for_grid[1] - time_for_grid[0]),
        float(deltaE_for_grid[1] - deltaE_for_grid[0]),
        seed,
    )

    if total_induced_voltage is not None:
        # Inputting new line density
        profile.cut_options.cut_left = time_for_grid[0] - 0.5 * (
            time_for_grid[1] - time_for_grid[0]
        )
        profile.cut_options.cut_right = time_for_grid[-1] + 0.5 * (
            time_for_grid[1] - time_for_grid[0]
        )
        profile.cut_options.n_slices = n_points_grid
        profile.cut_options.set_cuts()
        profile.set_slices_parameters()
        profile.n_macroparticles = (
            reconstructed_line_den * beam.n_macroparticles
        )

        # Re-calculating the sources of wakes/impedances according to this
        # slicing
        induced_voltage_object.reprocess()

        # Calculating the induced voltage
        induced_voltage_object.induced_voltage_sum()
        gc.collect()
        return [
            hamiltonian_coord,
            distribution_function_,
        ], induced_voltage_object

    gc.collect()
    return [hamiltonian_coord, distribution_function_], [
        time_line_den,
        line_density_,
    ]


@handle_legacy_kwargs
def matched_from_distribution_function(
    beam: Beam,
    full_ring_and_rf: FullRingAndRF,
    distribution_function_input: Optional[Callable] = None,
    # needs better specification
    distribution_user_table: Optional[DistributionUserTableType] = None,
    main_harmonic_option: MainHarmonicOptionType = "lowest_freq",
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
    distribution_variable: DistributionVariableType = "Hamiltonian",
    process_pot_well: bool = True,
    turn_number: int = 0,
) -> tuple[list[NumpyArray], TotalInducedVoltage] | list[NumpyArray]:
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
    slippage_factor = full_ring_and_rf.ring_and_rf_section[0].rf_params.eta_0[
        turn_number
    ]
    beta = full_ring_and_rf.ring_and_rf_section[0].rf_params.beta[turn_number]
    energy = full_ring_and_rf.ring_and_rf_section[0].rf_params.energy[
        turn_number
    ]

    eom_factor_dE = abs(slippage_factor) / (2 * beta**2.0 * energy)
    eom_factor_potential = (
        np.sign(slippage_factor)
        * beam.particle.charge
        / (
            full_ring_and_rf.ring_and_rf_section[0].rf_params.t_rev[
                turn_number
            ]
        )
    )

    #: *Number of points to be used in the potential well calculation*
    n_points_potential = int(n_points_potential)
    # Generate potential well
    full_ring_and_rf.potential_well_generation(
        turn=turn_number,
        n_points=n_points_potential,
        dt_margin_percent=dt_margin_percent,
        main_harmonic_option=main_harmonic_option,
    )
    potential_well: NumpyArray = full_ring_and_rf.potential_well
    time_potential: NumpyArray = full_ring_and_rf.potential_well_coordinates

    induced_potential = 0

    # Extra potential from previous bunches (for multi-bunch generation)
    extra_potential = 0
    if extra_voltage_dict is not None:
        extra_voltage_time_input = extra_voltage_dict["time_array"]
        extra_voltage_input = extra_voltage_dict["voltage_array"]
        extra_potential_input = -(
            eom_factor_potential
            * cumtrapz(
                extra_voltage_input,
                dx=(
                    float(extra_voltage_time_input[1])
                    - float(extra_voltage_time_input[0])
                ),
                initial=0,
            )
        )
        extra_potential = np.interp(
            time_potential, extra_voltage_time_input, extra_potential_input
        )

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
        total_potential = potential_well + induced_potential + extra_potential

        sse = np.sqrt(np.sum((old_potential - total_potential) ** 2))

        print(
            "Matching the bunch... (iteration: "
            + str(i)
            + " and sse: "
            + str(sse)
            + ")"
        )

        # Process the potential well in order to take a frame around the separatrix
        if not process_pot_well:
            time_potential_sep, potential_well_sep = (
                time_potential,
                total_potential,
            )
        else:
            time_potential_sep, potential_well_sep = potential_well_cut(
                time_potential, total_potential
            )

        # Potential is shifted to put the minimum on 0
        potential_well_sep = potential_well_sep - np.min(potential_well_sep)

        # Compute deltaE frame corresponding to the separatrix
        max_potential = np.max(potential_well_sep)
        max_deltaE = np.sqrt(max_potential / eom_factor_dE)

        # Initializing the grids by reducing the resolution to a
        # n_points_grid*n_points_grid frame
        time_potential_low_res = np.linspace(
            float(time_potential_sep[0]),
            float(time_potential_sep[-1]),
            n_points_grid,
        )
        time_resolution_low = float(time_potential_low_res[1]) - float(
            time_potential_low_res[0]
        )
        deltaE_coord_array = np.linspace(
            -float(max_deltaE), float(max_deltaE), n_points_grid
        )
        potential_well_low_res = np.interp(
            time_potential_low_res, time_potential_sep, potential_well_sep
        )
        time_grid, deltaE_grid = np.meshgrid(
            time_potential_low_res, deltaE_coord_array
        )
        potential_well_grid = np.meshgrid(
            potential_well_low_res, potential_well_low_res
        )[0]

        # Computing the action J by integrating the dE trajectories
        J_array_dE0 = np.zeros(n_points_grid)

        full_ring_and_RF2 = copy.deepcopy(full_ring_and_rf)
        for j in range(n_points_grid):
            # Find left and right time coordinates for a given hamiltonian
            # value
            time_indexes = np.where(
                potential_well_low_res <= potential_well_low_res[j]
            )[0]
            left_time = time_potential_low_res[np.max((0, time_indexes[0]))]
            right_time = time_potential_low_res[
                np.min((time_indexes[-1], n_points_grid - 1))
            ]
            # Potential well calculation with high resolution in that frame
            time_potential_high_res = np.linspace(
                float(left_time), float(right_time), n_points_potential
            )
            full_ring_and_RF2.potential_well_generation(
                n_points=n_points_potential,
                time_array=time_potential_high_res,
                main_harmonic_option=main_harmonic_option,
            )
            pot_well_high_res = full_ring_and_RF2.potential_well

            if total_induced_voltage is not None and i != 0:
                induced_potential_hires = np.interp(
                    time_potential_high_res,
                    time_potential,
                    induced_potential + extra_potential,
                    left=0,
                    right=0,
                )
                pot_well_high_res += induced_potential_hires
                pot_well_high_res -= pot_well_high_res.min()

            # Integration to calculate action
            dE_trajectory[pot_well_high_res <= potential_well_low_res[j]] = (
                np.sqrt(
                    (
                        potential_well_low_res[j]
                        - pot_well_high_res[
                            pot_well_high_res <= potential_well_low_res[j]
                        ]
                    )
                    / eom_factor_dE
                )
            )
            dE_trajectory[pot_well_high_res > potential_well_low_res[j]] = 0

            J_array_dE0[j] = (
                1
                / np.pi
                * np.trapezoid(
                    dE_trajectory,
                    dx=time_potential_high_res[1] - time_potential_high_res[0],
                )
            )

        # Sorting the H and J functions to be able to interpolate J(H)
        H_array_dE0 = potential_well_low_res
        sorted_H_dE0 = H_array_dE0[H_array_dE0.argsort()]
        sorted_J_dE0 = J_array_dE0[H_array_dE0.argsort()]

        # Calculating the H and J grid
        H_grid = eom_factor_dE * deltaE_grid**2 + potential_well_grid
        J_grid = np.interp(
            H_grid, sorted_H_dE0, sorted_J_dE0, left=0, right=np.inf
        )

        # Choice of either H or J as the variable used
        if distribution_variable == "Action":
            sorted_X_dE0 = sorted_J_dE0
            X_grid = J_grid
        elif distribution_variable == "Hamiltonian":
            sorted_X_dE0 = sorted_H_dE0
            X_grid = H_grid
        else:
            # DistributionError
            raise RuntimeError(
                "The distribution_variable option was not " + "recognized"
            )

        # Computing bunch length as a function of H/J if needed
        # Bunch length can be calculated as 4-rms, Gaussian fit, or FWHM
        if bunch_length is not None:
            X0 = x0_from_bunch_length(
                bunch_length,
                bunch_length_fit,
                X_grid,
                sorted_X_dE0,
                n_points_grid,
                time_potential_low_res,
                distribution_function_,
                distribution_type,
                distribution_exponent,
                beam,
                full_ring_and_rf,
            )

        elif emittance is not None:
            if distribution_variable == "Action":
                X0 = emittance / (2 * np.pi)
            elif distribution_variable == "Hamiltonian":
                X0 = np.interp(
                    emittance / (2 * np.pi), sorted_J_dE0, sorted_H_dE0
                )

        # Computing the density grid
        if distribution_user_table is None:
            density_grid = distribution_function_(
                X_grid, distribution_type, X0, distribution_exponent
            )
        else:
            density_grid = np.interp(
                X_grid,
                distribution_user_table["user_table_action"],
                distribution_user_table["user_table_distribution"],
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
            profile.cut_options.cut_left = (
                time_potential_low_res[0] - 0.5 * time_resolution_low
            )
            profile.cut_options.cut_right = (
                time_potential_low_res[-1] + 0.5 * time_resolution_low
            )
            profile.cut_options.n_slices = n_points_grid
            profile.cut_options.cuts_unit = "s"
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
            induced_potential_low_res = -(
                eom_factor_potential
                * cumtrapz(induced_voltage, dx=time_resolution_low, initial=0)
            )
            induced_potential = np.interp(
                time_potential,
                time_potential_low_res,
                induced_potential_low_res,
                left=0,
                right=0,
            )
        del full_ring_and_RF2
        gc.collect()
    # Populating the bunch
    populate_bunch(
        beam,
        time_grid,
        deltaE_grid,
        density_grid,
        time_resolution_low,
        float(deltaE_coord_array[1] - deltaE_coord_array[0]),
        seed,
    )

    if total_induced_voltage is not None:
        return [time_potential_low_res, line_density_], induced_voltage_object
    else:
        return [time_potential_low_res, line_density_]


@handle_legacy_kwargs
def x0_from_bunch_length(
    bunch_length: float,
    bunch_length_fit: BunchLengthFitTypes,
    X_grid: NumpyArray,
    sorted_X_dE0: NumpyArray,
    n_points_grid: int,
    time_potential_low_res: NumpyArray,
    distribution_function_: Callable,
    # TODO this is just distribution_function with strange way, is this intended?
    distribution_type: str,
    distribution_exponent: float,
    beam: Beam,
    full_ring_and_rf: FullRingAndRF,
) -> float:
    """
    Function to find the corresponding H0 or J0 for a given bunch length.
    Used by matched_from_distribution_function()
    """  # todo documentation
    # todo improve readability
    tau = 0.0

    # Initial values for iteration
    x_low = sorted_X_dE0[0]
    x_hi = sorted_X_dE0[-1]
    X_min = sorted_X_dE0[0]
    X_max = sorted_X_dE0[-1]
    X_accuracy = (sorted_X_dE0[1] - sorted_X_dE0[0]) / 2.0

    bin_size = time_potential_low_res[1] - time_potential_low_res[0]

    # Iteration to find H0/J0 from the bunch length
    while np.abs(bunch_length - tau) > bin_size:
        # Takes middle point of the interval [X_low,X_hi]
        X0 = 0.5 * (x_low + x_hi)

        if bunch_length_fit == "full":
            bunch_indices = np.where(np.sum(X_grid <= X0, axis=0))[0]
            tau = float(
                time_potential_low_res[bunch_indices][-1]
                - time_potential_low_res[bunch_indices][0]
            )
        else:
            # Calculating the line density for the parameter X0
            density_grid = distribution_function_(
                X_grid, distribution_type, X0, distribution_exponent
            )

            density_grid = density_grid / np.sum(density_grid)
            line_density_ = np.sum(density_grid, axis=0)

            # Calculating the bunch length of that line density
            if (line_density_ > 0).any():
                tau = 4.0 * np.sqrt(
                    np.sum(
                        (
                            time_potential_low_res
                            - np.sum(line_density_ * time_potential_low_res)
                            / np.sum(line_density_)
                        )
                        ** 2
                        * line_density_
                    )
                    / np.sum(line_density_)
                )

                if bunch_length_fit is not None:
                    profile = Profile(
                        beam,
                        cut_options=CutOptions(
                            cut_left=(
                                time_potential_low_res[0] - 0.5 * bin_size
                            ),
                            cut_right=(
                                time_potential_low_res[-1] + 0.5 * bin_size
                            ),
                            n_slices=n_points_grid,
                            rf_station=full_ring_and_rf.ring_and_rf_section[
                                0
                            ].rf_params,
                        ),
                    )
                    #                     profile = Profile(
                    #                       full_ring_and_RF.RingAndRFSection_list[0].rf_params,
                    #                       beam, n_points_grid, cut_left=time_potential_low_res[0] -
                    #                       0.5*bin_size , cut_right=time_potential_low_res[-1] +
                    #                       0.5*bin_size)

                    profile.n_macroparticles = line_density_

                    if bunch_length_fit == "gauss":
                        raise NotImplementedError()
                        # FIXME this wont work as gaussian_fit is undefined. What shall this function do?
                        profile.bl_gauss = tau
                        profile.bp_gauss = np.sum(
                            line_density_ * time_potential_low_res
                        ) / np.sum(line_density_)
                        profile.gaussian_fit()  # FIXME
                        tau = profile.bl_gauss
                    elif bunch_length_fit == "fwhm":
                        profile.fwhm()
                        tau = profile.bunchLength

        # Update of the interval for the next iteration
        if tau >= bunch_length:
            x_hi = X0
        else:
            x_low = X0

        if (X_max - X0) < X_accuracy:
            print(
                "WARNING: The bucket is too small to have the "
                + "desired bunch length! Input is %.2e, " % bunch_length
                + "the generation gave %.2e, " % tau
                + "the error is %.2e" % (bunch_length - tau)
            )
            break

        if (X0 - X_min) < X_accuracy:
            print(
                "WARNING: The desired bunch length is too small "
                + "to be generated accurately!"
            )

    #    return 0.5 * (X_low + X_hi)
    return X0


@handle_legacy_kwargs
def populate_bunch(
    beam: Beam,
    time_grid: NumpyArray,
    deltaE_grid: NumpyArray,
    density_grid: NumpyArray,
    time_step: float,
    deltaE_step: float,
    seed: int,
):
    """
    *Method to populate the bunch using a random number generator from the
    particle density in phase space.*
    """
    # Initialise the random number generator
    np.random.seed(seed=seed)
    # Generating particles randomly inside the grid cells according to the
    # provided density_grid
    indexes = np.random.choice(
        np.arange(0, np.size(density_grid)),
        beam.n_macroparticles,
        p=density_grid.flatten(),
    )

    # Randomize particles inside each grid cell (uniform distribution)
    beam.dt = (
        np.ascontiguousarray(
            time_grid.flatten()[indexes]
            + (np.random.rand(beam.n_macroparticles) - 0.5) * time_step
        )
    ).astype(dtype=bm.precision.real_t, order="C", copy=False)
    beam.dE = (
        np.ascontiguousarray(
            deltaE_grid.flatten()[indexes]
            + (np.random.rand(beam.n_macroparticles) - 0.5) * deltaE_step
        )
    ).astype(dtype=bm.precision.real_t, order="C", copy=False)


def __distribution_function_by_exponent(
    action_array: NumpyArray, exponent: float, length: float
) -> NumpyArray:
    warnings.filterwarnings("ignore")
    distribution_function_ = (1 - action_array / length) ** exponent
    warnings.filterwarnings("default")
    distribution_function_[action_array > length] = 0
    return distribution_function_


def distribution_function(
    action_array: NumpyArray,
    dist_type: DistTypeDistFunction,
    length: float,
    exponent: Optional[float] = None,
) -> NumpyArray:
    """
    *Distribution function (formulas from Laclare).*
    """

    if dist_type == "waterbag":
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        exponent = 0
        distribution_ = __distribution_function_by_exponent(
            action_array, exponent, length
        )

    elif dist_type == "parabolic_amplitude":
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        exponent = 1
        distribution_ = __distribution_function_by_exponent(
            action_array, exponent, length
        )

    elif dist_type == "parabolic_line":
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        exponent = 0.5
        distribution_ = __distribution_function_by_exponent(
            action_array, exponent, length
        )

    elif dist_type == "binomial":
        assert exponent is not None, "Please specify exponent"
        distribution_ = __distribution_function_by_exponent(
            action_array, exponent, length
        )

    elif dist_type == "gaussian":
        distribution_ = np.exp(-2 * action_array / length)

    else:
        # DistributionError
        raise RuntimeError("The dist_type option was not recognized")

    return distribution_


def __line_density_by_exponent(
    bunch_length: float,
    bunch_position: float,
    coord_array: NumpyArray,
    exponent: float,
) -> NumpyArray:
    warnings.filterwarnings("ignore")
    line_density_ = (
        1 - (2.0 * (coord_array - bunch_position) / bunch_length) ** 2
    ) ** (exponent + 0.5)
    warnings.filterwarnings("default")
    line_density_[np.abs(coord_array - bunch_position) > bunch_length / 2] = 0
    return line_density_


def line_density(
    coord_array: NumpyArray,
    dist_type: str,
    bunch_length: float,
    bunch_position: float = 0.0,
    exponent: Optional[float] = None,
) -> NumpyArray:
    """
    *Line density*
    """

    if dist_type == "waterbag":
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        exponent = 0
        line_density_ = __line_density_by_exponent(
            bunch_length, bunch_position, coord_array, exponent
        )

    elif dist_type == "parabolic_amplitude":
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        exponent = 1
        line_density_ = __line_density_by_exponent(
            bunch_length, bunch_position, coord_array, exponent
        )

    elif dist_type == "parabolic_line":
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        exponent = 0.5
        line_density_ = __line_density_by_exponent(
            bunch_length, bunch_position, coord_array, exponent
        )
    elif dist_type == "binomial":
        assert exponent is not None, "Please specify exponent!"
        line_density_ = __line_density_by_exponent(
            bunch_length, bunch_position, coord_array, exponent
        )

    elif dist_type == "gaussian":
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        sigma = bunch_length / 4
        line_density_ = np.exp(
            -((coord_array - bunch_position) ** 2) / (2 * sigma**2)
        )

    elif dist_type == "cosine_squared":
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        warnings.filterwarnings("ignore")
        line_density_ = (
            np.cos(np.pi * (coord_array - bunch_position) / bunch_length) ** 2
        )
        warnings.filterwarnings("default")
        line_density_[
            np.abs(coord_array - bunch_position) > bunch_length / 2
        ] = 0
    else:
        # DistributionError
        raise RuntimeError("The dist_type option was not recognized")

    return line_density_


@handle_legacy_kwargs
def bigaussian(
    ring: Ring,
    rf_station: RFStation,
    beam: Beam,
    sigma_dt: float,
    sigma_dE: Optional[float] = None,
    seed: int = 1234,
    reinsertion: bool = False,
):
    r"""Function generating a Gaussian beam both in time and energy
    coordinates. Fills Beam.dt and Beam.dE arrays.

    Parameters
    ----------
    ring : class
        A Ring type class
    rf_station : class
        An RFStation type class
    beam : class
        A Beam type class
    sigma_dt : float
        R.m.s. extension of the Gaussian in time
    sigma_dE : float (optional)
        R.m.s. extension of the Gaussian in energy; default is None and will
        match the energy coordinate according to bucket height and sigma_dt
    seed : int (optional)
        Fixed seed to have a reproducible distribution
    reinsertion : bool (optional)
        Re-insert particles that are generated outside the separatrix into the
        bucket; default in False

    """

    if sigma_dE is None:
        sigma_dE = _get_dE_from_dt(ring, rf_station, sigma_dt)
    counter = rf_station.counter[0]
    omega_rf = rf_station.omega_rf[0, counter]
    phi_s = rf_station.phi_s[counter]
    phi_rf = rf_station.phi_rf[0, counter]
    eta0 = rf_station.eta_0[counter]

    # RF wave is shifted by Pi below transition
    if eta0 < 0:
        phi_rf -= np.pi

    beam.sigma_dt = sigma_dt
    beam.sigma_dE = sigma_dE

    # Generate coordinates. For reproducibility, a separate random number stream is used for dt and dE
    rng_dt = np.random.default_rng(seed)
    rng_dE = np.random.default_rng(seed + 1)

    beam.dt = (
        sigma_dt
        * rng_dt.normal(size=beam.n_macroparticles).astype(
            dtype=bm.precision.real_t, order="C", copy=False
        )
        + (phi_s - phi_rf) / omega_rf
    )
    beam.dE = sigma_dE * rng_dE.normal(size=beam.n_macroparticles).astype(
        dtype=bm.precision.real_t, order="C"
    )

    # Re-insert if necessary
    if reinsertion:
        itemindex = bm.where(
            is_in_separatrix(ring, rf_station, beam, beam.dt, beam.dE) == False
        )[0]
        while itemindex.size > 0:
            beam.dt[itemindex] = (
                sigma_dt
                * rng_dt.normal(size=itemindex.size).astype(
                    dtype=bm.precision.real_t, order="C", copy=False
                )
                + (phi_s - phi_rf) / omega_rf
            )

            beam.dE[itemindex] = sigma_dE * rng_dE.normal(
                size=itemindex.size
            ).astype(dtype=bm.precision.real_t, order="C")

            itemindex = bm.where(
                is_in_separatrix(ring, rf_station, beam, beam.dt, beam.dE)
                == False
            )[0]


@handle_legacy_kwargs
def parabolic(
    ring: Ring,
    rf_station: RFStation,
    beam: Beam,
    bunch_length: float,
    bunch_position: Optional[float] = None,
    bunch_energy: Optional[float] = None,
    energy_spread: Optional[float] = None,
    seed: int = 1234,
):
    r"""Generate a bunch of particles distributed as a parabola in phase space.

    Parameters
    ----------
    ring : class
        A Ring type class
    rf_station : class
        An RFStation type class
    beam : class
        A Beam type class
    bunch_length : float
        The length in time [s] of the bunch
    bunch_position : float (optional)
        The position in time [s] of the center of mass of the bunch
    bunch_energy : float (optional)
        The position in energy [eV] of the center of mass of the bunch
        (relative to the synchronous energy)
    energy_spread : float (optional)
        The spread in energy [eV] of the bunch
    seed : int (optional)
        Fixed seed to have a reproducible distribution

    """

    # Getting the position and spread if not defined by user
    counter = rf_station.counter[0]
    omega_rf = rf_station.omega_rf[0, counter]
    phi_s = rf_station.phi_s[counter]
    phi_rf = rf_station.phi_rf[0, counter]
    eta0 = rf_station.eta_0[counter]

    # RF wave is shifted by Pi below transition
    if eta0 < 0:
        phi_rf -= np.pi

    if bunch_position is None:
        bunch_position = (phi_s - phi_rf) / omega_rf

    if bunch_energy is None:
        bunch_energy = 0

    if energy_spread is None:
        energy_spread = _get_dE_from_dt(ring, rf_station, bunch_length)

    # Generating time and energy arrays
    time_array = np.linspace(
        bunch_position - bunch_length / 2,
        bunch_position + bunch_length / 2,
        100,
    )
    energy_array = np.linspace(
        bunch_energy - energy_spread / 2, bunch_energy + energy_spread / 2, 100
    )

    # Getting Hamiltonian on a grid
    dt_grid, deltaE_grid = np.meshgrid(time_array, energy_array)

    # Bin sizes
    bin_dt = float(time_array[1] - time_array[0])
    bin_energy = float(energy_array[1] - energy_array[0])

    # Density grid
    isodensity_lines = (
        (dt_grid - bunch_position) / bunch_length * 2
    ) ** 2.0 + ((deltaE_grid - bunch_energy) / energy_spread * 2) ** 2.0
    density_grid = 1 - isodensity_lines**2.0
    density_grid[density_grid < 0] = 0
    density_grid /= np.sum(density_grid)

    populate_bunch(
        beam, dt_grid, deltaE_grid, density_grid, bin_dt, bin_energy, seed
    )


@handle_legacy_kwargs
def _get_dE_from_dt(
    ring: Ring, rf_station: RFStation, dt_amplitude: float
) -> float:
    r"""A routine to evaluate the dE amplitude from dt following a single
    RF Hamiltonian.

    Parameters
    ----------
    ring : class
        A Ring type class
    rf_station : class
        An RFStation type class
    dt_amplitude : float
        Full amplitude of the particle oscillation in [s]

    Returns
    -------
    dE_amplitude : float
        Full amplitude of the particle oscillation in [eV]

    """

    warnings.filterwarnings("once")
    if ring.n_sections > 1:
        warnings.warn(
            "WARNING in bigaussian(): the usage of several"
            + " sections is not yet implemented. Ignoring"
            + " all but the first!"
        )
    if rf_station.n_rf > 1:
        warnings.warn(
            "WARNING in bigaussian(): the usage of multiple RF"
            + " systems is not yet implemented. Ignoring"
            + " higher harmonics!"
        )

    counter = rf_station.counter[0]

    harmonic = rf_station.harmonic[0, counter]
    energy = rf_station.energy[counter]
    beta = rf_station.beta[counter]
    omega_rf = rf_station.omega_rf[0, counter]
    phi_s = rf_station.phi_s[counter]
    phi_rf = rf_station.phi_rf[0, counter]
    eta0 = rf_station.eta_0[counter]

    # RF wave is shifted by Pi below transition
    if eta0 < 0:
        phi_rf -= np.pi

    # Calculate dE_amplitude from dt_amplitude using single-harmonic Hamiltonian
    voltage = rf_station.charge * rf_station.voltage[0, counter]
    eta0 = rf_station.eta_0[counter]

    phi_b = omega_rf * dt_amplitude + phi_s
    dE_amplitude = np.sqrt(
        voltage
        * energy
        * beta**2
        * (np.cos(phi_b) - np.cos(phi_s) + (phi_b - phi_s) * np.sin(phi_s))
        / (np.pi * harmonic * np.fabs(eta0))
    )

    return dE_amplitude
