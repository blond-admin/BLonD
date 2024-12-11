"""
**Module to generate multibunch distributions**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**, **Theodoros Argyropoulos**
"""

from __future__ import annotations

import numpy as np
import scipy
from packaging.version import Version

if Version(scipy.__version__) >= Version("1.14"):
    pass
else:
    pass

from ...beam import Beam
from ...distribution_generators.methods import (x0_from_bunch_length)
from ...distribution_generators.singlebunch.matched_from_distribution_function import (
    distribution_function)

from ....utils.legacy_support import handle_legacy_kwargs

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import NDArray

    from ...beam import Beam
    from ....trackers.tracker import FullRingAndRF
    from ....utils.types import DistributionVariableType
    from ....utils.types import DistributionOptionsType


def compute_x_grid(normalization_DeltaE,  # todo TypeHint
                   time_array: NDArray, potential_well: NDArray,
                   distribution_variable: DistributionVariableType):
    # Delta Energy array
    max_DeltaE = np.sqrt(np.max(potential_well) / normalization_DeltaE)
    coord_array_DeltaE = np.linspace(-float(max_DeltaE), float(max_DeltaE), len(time_array))

    # Resolution in time and energy
    time_resolution = time_array[1] - time_array[0]
    energy_resolution = coord_array_DeltaE[1] - coord_array_DeltaE[0]

    # Grid
    time_grid, deltaE_grid = np.meshgrid(time_array, coord_array_DeltaE)
    potential_well_grid = np.meshgrid(potential_well, potential_well)[0]
    H_grid = normalization_DeltaE * deltaE_grid ** 2 + potential_well_grid

    # Compute the action J
    J_array = np.zeros(shape=potential_well.shape, dtype=float)
    for i in range(len(J_array)):
        DELTA = np.sqrt((potential_well[i]
                         - potential_well)[potential_well <= potential_well[i]]
                        / normalization_DeltaE)
        J_array[i] = 1. / np.pi * np.trapezoid(DELTA, dx=time_array[1]
                                                         - time_array[0])

    # Compute J grid
    sorted_H = potential_well[potential_well.argsort()]
    sorted_J = J_array[potential_well.argsort()]

    if distribution_variable == 'Action':
        J_grid = np.interp(H_grid, sorted_H, sorted_J,
                           left=0, right=np.inf)
        return sorted_H, sorted_J, J_grid, time_grid, deltaE_grid, \
            time_resolution, energy_resolution
    else:  # TODO better elif Hamiltonian
        return sorted_H, sorted_J, H_grid, time_grid, deltaE_grid, \
            time_resolution, energy_resolution


def compute_H0(emittance, H, J):
    #  Estimation of H corresponding to the emittance
    return np.interp(emittance / (2. * np.pi), J, H)


@handle_legacy_kwargs
def match_a_bunch(normalization_DeltaE: float, beam: Beam,
                  potential_well_coordinates: NDArray, potential_well: NDArray,
                  seed: int, distribution_options: DistributionOptionsType,
                  full_ring_and_rf: Optional[FullRingAndRF] = None) \
        -> tuple[NDArray, NDArray, NDArray, float, float, NDArray]:
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

    H, J, X_grid, time_grid, deltaE_grid, time_resolution, energy_resolution = \
        compute_x_grid(normalization_DeltaE, potential_well_coordinates,
                       potential_well, distribution_variable)

    # Choice of either H or J as the variable used
    if distribution_variable == 'Action':
        sorted_X = J
    elif distribution_variable == 'Hamiltonian':
        sorted_X = H
    else:
        # DistributionError
        raise SystemError('distribution_variable should be Action or Hamiltonian')

    if bunch_length is not None:
        n_points_grid = X_grid.shape[0]
        X0 = x0_from_bunch_length(bunch_length, bunch_length_fit, X_grid, sorted_X,
                                  n_points_grid, potential_well_coordinates,
                                  distribution_function, distribution_type,
                                  distribution_exponent, beam, full_ring_and_rf)
    elif emittance is not None:
        X0 = compute_H0(emittance, H, J)
    else:
        # DistributionError
        raise SystemError('You should specify either bunch_length or emittance')

    distribution = distribution_function(X_grid, distribution_type, X0, exponent=distribution_exponent)
    distribution[X_grid > np.max(H)] = 0
    distribution = distribution / np.sum(distribution)

    profile = np.sum(distribution, axis=0)

    return (time_grid, deltaE_grid, distribution, time_resolution,
            energy_resolution, profile)
