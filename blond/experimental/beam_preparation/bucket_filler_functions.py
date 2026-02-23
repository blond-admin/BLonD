# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

import warnings
from typing import Callable  # NOQA
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from blond.experimental.beam_preparation.density_functions import (
    gaussian_density,
)
from blond.experimental.beam_preparation.metric_functions import rms_emittance

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


def metric_fitter(
    hamilton: NumpyArray | CupyArray,
    dt_grid: NumpyArray | CupyArray,
    dE_grid: NumpyArray | CupyArray,
    desired_metric: float,
    free_parameter_guess: float,
    metric_function: callable(
        [
            NumpyArray | CupyArray,
            NumpyArray | CupyArray,
            NumpyArray | CupyArray,
        ]
    ),
    density_function: callable([NumpyArray | CupyArray, float]),
    max_metric_diff: float,
    iterations_max: int = 100,
) -> NumpyArray | CupyArray:
    """fits a density distribution with one free parameter to a metric of interest,
    then returns the resulting density

    Parameters
    ----------
    hamilton
        2D hamiltonian, in [eV]
    dt_grid
        time array corresponding to hamiltonian, in [s]
    dE_grid
        energy array corresponding to hamiltonian, in [eV]
    desired_metric
        metric value to fit the distribution to
    free_parameter_guess
        initial guess for the free parameter.
    metric_function
        function that takes a density mass function
        and outputs a metric to be fitted to an ideal value
    density_function
        function to convert a hamiltonian to density
        with one free parameter e.g. rho(H, parameter)
    max_metric_diff
        The allowed difference between the desired metric
        and the metric of the generated bunch
    iterations_max
        maximum number of iterations to search
        before the search loop gets interrupted.

    Returns
    -------
    density
        the density mass function,
        fitted to give the right metric of interest
    """
    # Make a lambda expression for the absolute difference between desired and calculated metrics for any free parameter
    minimization_function = lambda x: np.abs(
        metric_function(density_function(hamilton, x), dt_grid, dE_grid)
        - desired_metric
    )

    # minimize
    optimize_result = minimize(
        minimization_function,
        free_parameter_guess,
        method="Nelder-Mead",
        tol=max_metric_diff,
        # maxiter=iterations_max,
    )

    ideal_free_parameter = optimize_result.x[
        0
    ]  # optimization is only 1 dimensional

    # return best matching distribution density
    density = density_function(hamilton, ideal_free_parameter)
    return density


def generalized_bucket_filler(
    time_grid: NumpyArray | CupyArray,
    deltaE_grid: NumpyArray | CupyArray,
    hamilton_2D: NumpyArray | CupyArray,
    metric_list: list[float],
    intensity_frac_list: list[float],
    n_buckets: int,
    max_metric_diff: float,
    density_function: callable(
        [NumpyArray | CupyArray, float]
    ) = gaussian_density,
    metric_function: callable(
        [
            NumpyArray | CupyArray,
            NumpyArray | CupyArray,
            NumpyArray | CupyArray,
        ]
    ) = rms_emittance,
    free_parameter_guess: float = None,
) -> NumpyArray | CupyArray:
    """Generalized method for generating density distributions for a hamiltonian.
    Notes
    -----
    Allows for a density distribution rho(H, value) with a free parameters value. This value is then
    automatically selected to make the generated bunches
    satisfy some metric that is also user defined.

    This method does not have functionality for identifying the wells. When the number of buckets is specified the time
    axis is simply split into n_buckets equal sized pieces. If the edges of these slices do not neatly align with the
    maxima of the potential, bunches may not be generated correctly. These bounds can be changed by setting
    different bounds in the matcher.

    Parameters
    ----------
    time_grid
        Time coordinates of the hamiltonian, in [s].
    deltaE_grid
        Energy coordinate of the hamiltonian, in [eV].
    hamiltonn_2D
        hamiltonian value, in [eV].
    metric_list
        list of values to for the metric to have for each bunch.
        Metric definition specified by metric function
    intensity_frac_list
        fraction of total intensity for each bunch to have.
    n_buckets
        number of buckets spanned by the time axis.
    max_metric_diff
        the maximum allowed difference between the desired
        metric and the metric of the generated bunch
    density_function
        simplistic function for the density with one free parameter
        rho(H, value). Value and metric must scale monotonically.
    metric_function
        function of a hamiltonian. Must take the density,
        time and energy as 2d arrays
    free_parameter_guess
        The initial guess for the free parameter


    Returns
    -------
    _density
        2D array containing the density distribution of the beam.
    """
    _hamilton = hamilton_2D.copy()
    density = np.zeros(hamilton_2D.shape)

    if (
        free_parameter_guess is None
    ):  # typically the free parameter is in units of eV so setting it to max of hamiltonian makes sense in most cases
        free_parameter_guess = np.max(_hamilton) / 10000

    n_slices_per_bucket = int(time_grid.shape[1] / n_buckets)
    min_hamilton = np.min(_hamilton)
    _hamilton -= min_hamilton

    for bucket in range(n_buckets):
        # split phase space into each bucket
        min_bucket_index = bucket * n_slices_per_bucket
        max_bucket_index = (bucket + 1) * n_slices_per_bucket

        select_bucket = slice(min_bucket_index, max_bucket_index)

        sliced_time = time_grid[select_bucket, :]
        sliced_energy = deltaE_grid[select_bucket, :]
        sliced_hamilton = _hamilton[select_bucket, :]

        # ensure the minimum hamilton in each slice is equal
        sliced_hamilton -= np.min(sliced_hamilton) + min_hamilton

        if (
            intensity_frac_list[bucket] == 0
        ):  # if no intensity in bucket, just set density to zero
            sliced_density = np.zeros(sliced_hamilton.shape)
        else:  # otherwise fill bucket with a density that satisfies a desired metric
            sliced_density = metric_fitter(
                sliced_hamilton,
                sliced_time,
                sliced_energy,
                metric_list[bucket],
                free_parameter_guess,
                metric_function,
                density_function,
                max_metric_diff,
            )
            sliced_density *= intensity_frac_list[bucket]

        density[select_bucket, :] = (
            sliced_density  # concatenate density of each bucket
        )

    return density


def hamilton_to_density_by_max(
    time_grid: NumpyArray | CupyArray,
    deltaE_grid: NumpyArray | CupyArray,
    hamilton_2D: NumpyArray | CupyArray,
    density_modifier: float,
    hamilton_max: float,
) -> NumpyArray | CupyArray:
    """Converts a 2D Hamilton 2D array into a density distribution.

    This function normalizes the input Hamilton by a specified maximum value,
    inverts it to represent particle density (i.e., lower energy = higher density),
    and optionally adjusts the shape of the distribution using a power-law modifier.

    Notes
    -----
    - The density is highest in regions with the lowest Hamilton values.
    - Values in `hamilton_2D` greater than `hamilton_max` are clipped before processing.
    - The function preserves the array type (NumPy or CuPy) of the input.

    Parameters
    ----------
    time_grid : NumpyArray or CupyArray
        A 2D array representing the time axis of the phase space

    deltaE_grid : NumpyArray or CupyArray
        A 2D array representing the energy axis of the phase space

    hamilton_2D : NumpyArray or CupyArray
        A 2D array representing the spatial Hamilton field.

    density_modifier : float
        Exponent applied to the normalized and inverted Hamilton values
        to shape the final density distribution.
        Higher values exaggerate differences in density.

    hamilton_max : float
        The maximum reference value for normalizing the Hamilton.
        Values above this threshold are truncated.

    Returns
    -------
    density : NumpyArray or CupyArray
        A 2D array of the same shape as `hamilton_2D`, representing the
        computed density distribution. Values are scaled between 0 and 1.


    """

    _density = hamilton_2D.copy()  # So the changes stay in this scope

    _density -= _density.min()
    _density /= hamilton_max
    # Now 1 representing the limit between particles/no-particles.
    # Smaller 1 means there should be particles.

    _density[_density > 1] = 1  # Truncate
    # Flip max with min
    _density *= -1
    _density -= _density.min()

    # Modify of the density to be more/less dense in different regions.
    _density **= density_modifier
    return _density
