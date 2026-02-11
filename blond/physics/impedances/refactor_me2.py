# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import numba
import numpy as np
from numba import complex128, float64, int32, prange, void


@numba.njit()
def apply_single_pole(
    profile: np.ndarray,
    profile_dts: np.ndarray,
    pole: complex,
    residue: complex,
    voltage: np.ndarray,
    states: np.ndarray,
    pole_i: int,
    update_on_bin: np.ndarray,
):
    # y[n] = profile[n] + exp(p * dt) * y[n-1]
    # V[n] = 2 * Re(r * y[n])
    n_bins = len(profile)
    # state = 0.0 + 0.0j
    state = states[pole_i]
    t_previous = states[-1]
    mlk = 0
    update_on_bin_i = update_on_bin[mlk]
    for bin_i in range(n_bins):
        profile_i_ = profile[bin_i]
        if bin_i == update_on_bin_i:
            t_current = profile_dts[bin_i]
            dt = t_current - t_previous
            decay = np.exp(pole * dt)
            mlk += 1
            update_on_bin_i = update_on_bin[mlk]

        state = state * decay + 0.5 * profile_i_
        voltage[bin_i] += 2 * np.real(residue * state)
        state += 0.5 * profile_i_
        t_previous = t_current

    states[pole_i] = state
    states[-1] = t_previous


@numba.njit(
    void(
        float64[:],
        float64[:],
        complex128[:],
        complex128[:],
        complex128[:],
        float64[:],
        float64[:, :],
        int32[:],
    ),
    fastmath=True,
    parallel=True,
)
def apply_poles2(
    # read
    profile,
    profile_dts,
    poles,
    residues,
    # write
    states,
    voltage,
    voltage_threaded,
    update_on_bin,
):
    n_poles = len(poles)

    for i in prange(n_poles):
        thread_i = numba.get_thread_id()

        apply_single_pole(
            profile,
            profile_dts,
            poles[i],
            residues[i],
            voltage_threaded[thread_i, :],
            states,
            i,
            update_on_bin,
        )
    voltage[:] = np.sum(voltage_threaded, axis=0)
