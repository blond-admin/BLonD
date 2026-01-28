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
def _gen_hist_numba(H_change, hamilton_2D, histogram_write, mid, state_vector):
    histogram_write[:] = 0.0

    num_states = state_vector.shape[0]
    h_shape_0 = hamilton_2D.shape[0]
    h_shape_1 = hamilton_2D.shape[1]

    # Precompute mid-column energies
    hamilton_mid = hamilton_2D[:, mid]

    # Precompute sigma, cutoff windows, and Gaussian prefactors
    inv_two_sigma2 = np.empty(num_states)
    emin = np.empty(num_states)
    emax = np.empty(num_states)

    for i in range(num_states):
        _sigma = BIN_SIGMA * H_change[i]
        inv_two_sigma2[i] = -1.0 / (2.0 * _sigma * _sigma)
        emin[i] = hamilton_mid[i] - BIN_SIGMA_CALC * _sigma
        emax[i] = hamilton_mid[i] + BIN_SIGMA_CALC * _sigma

    # Main loop
    for u in prange(h_shape_0):
        acc_u = 0.0
        h_u_min = hamilton_2D[u, :].min()

        # Cache row pointer for faster access

        for i in range(num_states):
            e_max_i = emax[i]
            if e_max_i < h_u_min:
                continue
            e_i = hamilton_mid[i]
            s_i = state_vector[i]
            inv2s2 = inv_two_sigma2[i]
            e_min_i = emin[i]

            for v in range(h_shape_1):
                h = hamilton_2D[u, v]

                # Skip expensive exp() if outside cutoff
                if h < e_min_i or h > e_max_i:
                    continue

                dE = h - e_i
                acc_u += np.exp(dE * dE * inv2s2) * s_i

        histogram_write[u] = acc_u


@numba.njit(
    void(float64[:], float64[:, :], float64[:], int32, float64[:]),
    parallel=True,
    fastmath=True,
)
def _gen_state_numba(
    H_change, hamilton_2D, histogram, mid, state_vector_write
):
    num_states = state_vector_write.shape[0]
    h_shape_0 = hamilton_2D.shape[0]
    h_shape_1 = hamilton_2D.shape[1]

    # Precompute mid-column energies
    hamilton_mid = hamilton_2D[:, mid]

    # Precompute sigma, cutoff windows, and Gaussian prefactors
    inv_two_sigma2 = np.empty(num_states)
    emin = np.empty(num_states)
    emax = np.empty(num_states)

    for i in range(num_states):
        _sigma = BIN_SIGMA * H_change[i]
        inv_two_sigma2[i] = -1.0 / (2.0 * _sigma * _sigma)
        emin[i] = hamilton_mid[i] - BIN_SIGMA_CALC * _sigma
        emax[i] = hamilton_mid[i] + BIN_SIGMA_CALC * _sigma

    for i in prange(len(state_vector_write)):
        val2 = 0.0
        valn = 0
        e_i = hamilton_mid[i]
        inv2s2 = inv_two_sigma2[i]

        for u in (i,):
            hist_u = histogram[u]
            if hist_u == 0:
                continue

            for v in range(h_shape_1):
                h = hamilton_2D[u, v]
                if h < emin[i] or h > emax[i]:
                    continue

                dE = h - e_i
                val2 += np.exp(dE * dE * inv2s2) * hist_u
                valn += 1
        if val2 > 0:
            state_vector_write[i] = 1 / val2


@numba.njit(
    void(float64[:], float64[:, :], float64[:, :], int32, float64[:]),
    parallel=True,
    fastmath=True,
)
def _gen_density_numba(
    H_change: NumpyArray,
    density_write: NumpyArray,
    hamilton_2D: NumpyArray,
    mid: int,
    state_vector: NumpyArray,
):
    h_shape_0 = hamilton_2D.shape[0]
    h_shape_1 = hamilton_2D.shape[1]
    n_states = state_vector.shape[0]

    # Preload the mid-column once
    h_mid = hamilton_2D[:, mid]

    # Precompute sigma, sigma², and cutoff windows
    sigma = np.empty(n_states)
    inv_two_sigma_sq = np.empty(n_states)
    e_min = np.empty(n_states)
    e_max = np.empty(n_states)

    # to remove calculation from inner loop
    for i in range(n_states):
        s = BIN_SIGMA * H_change[i]
        sigma[i] = s
        inv_two_sigma_sq[i] = -1.0 / (2.0 * s * s)
        e_i = h_mid[i]
        e_min[i] = e_i - BIN_SIGMA_CALC * s
        e_max[i] = e_i + BIN_SIGMA_CALC * s

    for idx in prange(h_shape_0 * h_shape_1):
        u = idx % h_shape_1
        v = idx // h_shape_1
        h_u_v = hamilton_2D[u, v]

        acc = 0.0

        for i in range(n_states):
            # Skip if outside cutoff
            if h_u_v < e_min[i] or h_u_v > e_max[i]:
                continue

            dE = h_u_v - h_mid[i]
            w = np.exp(dE * dE * inv_two_sigma_sq[i])
            acc += w * state_vector[i]

        density_write[u, v] = acc
