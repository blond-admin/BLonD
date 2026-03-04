# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np
from numba import float64, int32, prange, void

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


# antialiasing parameters
BIN_SIGMA = 1.0  # 68% in one bin
BIN_SIGMA_CALC = 3.0  # 99% of stencil will be drawn


@numba.njit(
    void(float64[:], float64[:, :], float64[:], int32, float64[:]),
    parallel=True,
    fastmath=True,
)
def _gen_hist_numba(
    potential_change,
    potential_2D,
    histogram_write,
    mid,
    occupation_per_equipotential_to_density,
):
    """
    Generate a histogram from the 2D potential and occupation per potential.

    Parameters
    ----------
    potential_change
        Precomputed ``gradient(potential_2D[mid,:])``.
    potential_2D
        Central object for this algorithm.
        2D map of which position lies on which potential.
    histogram_write
        The result is written on this array.
    mid
        Precomputed mid from ``potential_change``.
    occupation_per_equipotential_to_density
        Central array for this function.
        Maps the 2D potential to a density and finally to a histogram.

    See Also
    --------
    _gen_density_numba: The underlying 2D density for this histogram.
    """
    histogram_write[:] = 0.0

    num_states = occupation_per_equipotential_to_density.shape[0]
    h_shape_0 = potential_2D.shape[0]
    h_shape_1 = potential_2D.shape[1]

    # Precompute mid-column energies
    hamilton_mid = potential_2D[:, mid]
    h_max_write = hamilton_mid.max()

    # Precompute sigma, cutoff windows, and Gaussian prefactors
    inv_two_sigma2 = np.empty(num_states)
    emin = np.empty(num_states)
    emax = np.empty(num_states)

    for state_i in range(num_states):
        _sigma = BIN_SIGMA * potential_change[state_i]
        inv_two_sigma2[state_i] = -1.0 / (2.0 * _sigma * _sigma)
        emin[state_i] = hamilton_mid[state_i] - BIN_SIGMA_CALC * _sigma
        emax[state_i] = hamilton_mid[state_i] + BIN_SIGMA_CALC * _sigma

    # Main loop
    for h_i in prange(h_shape_0):
        acc_u = 0.0
        h_u_min = potential_2D[h_i, :].min()

        # Cache row pointer for faster access

        for state_i in range(num_states):
            e_max_i = emax[state_i]
            if e_max_i < h_u_min:
                continue
            e_i = hamilton_mid[state_i]
            s_i = occupation_per_equipotential_to_density[state_i]
            inv2s2 = inv_two_sigma2[state_i]
            e_min_i = emin[state_i]

            for h_j in range(h_shape_1):
                h = potential_2D[h_i, h_j]
                if h > h_max_write:
                    continue

                # Skip expensive exp() if outside cutoff
                if h < e_min_i or h > e_max_i:
                    continue

                dE = h - e_i
                acc_u += np.exp(dE * dE * inv2s2) * s_i

        histogram_write[h_i] = acc_u


@numba.njit(
    void(float64[:], float64[:, :], float64[:, :], int32, float64[:]),
    parallel=True,
    fastmath=True,
)
def _gen_density_numba(
    potential_change: NumpyArray,
    density_write: NumpyArray,
    potential_2D: NumpyArray,
    mid: int,
    occupation_per_equipotential_to_density: NumpyArray,
):
    """
    Generate a 2D density from the 2D potential and occupation per potential.

    Parameters
    ----------
    potential_change
        Precomputed ``gradient(potential_2D[mid,:])``.
    density_write
        The result is written on this array.
    potential_2D
        Central object for this algorithm.
        2D map of which position lies on which potential.
    mid
        Precomputed mid from ``potential_change``.
    occupation_per_equipotential_to_density
        Central array for this function.
        Maps the 2D potential to a density.

    See Also
    --------
    _gen_hist_numba: This density can be converted to a 1D histogram directly,
    """
    h_shape_0 = potential_2D.shape[0]
    h_shape_1 = potential_2D.shape[1]
    n_states = occupation_per_equipotential_to_density.shape[0]

    # Preload the mid-column once
    h_mid = potential_2D[:, mid]
    h_max_write = h_mid.max()
    # Precompute sigma, sigma², and cutoff windows
    sigma = np.empty(n_states)
    inv_two_sigma_sq = np.empty(n_states)
    e_min = np.empty(n_states)
    e_max = np.empty(n_states)

    # to remove calculation from inner loop
    for state_i in range(n_states):
        s = BIN_SIGMA * potential_change[state_i]
        sigma[state_i] = s
        inv_two_sigma_sq[state_i] = -1.0 / (2.0 * s * s)
        e_i = h_mid[state_i]
        e_min[state_i] = e_i - BIN_SIGMA_CALC * s
        e_max[state_i] = e_i + BIN_SIGMA_CALC * s

    for idx_2D_flat in prange(h_shape_0 * h_shape_1):
        h_i = idx_2D_flat // h_shape_1
        h_j = idx_2D_flat % h_shape_1
        if h_i >= potential_2D.shape[0]:
            pass
        h_u_v = potential_2D[h_i, h_j]
        if h_u_v > h_max_write:
            continue

        density_write_sum = 0.0

        for state_i in range(n_states):
            # Skip if outside cutoff
            if h_u_v < e_min[state_i] or h_u_v > e_max[state_i]:
                continue

            dE = h_u_v - h_mid[state_i]
            w = np.exp(dE * dE * inv_two_sigma_sq[state_i])
            density_write_sum += (
                w * occupation_per_equipotential_to_density[state_i]
            )

        density_write[h_i, h_j] = density_write_sum
