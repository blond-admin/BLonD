# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/
import os

import numba
import numpy as np
from numba import complex128, float64, int32, prange, void

os.environ["NUMBA_BOUNDSCHECK"] = "1"


@numba.njit()
def apply_single_pole(
    profile: np.ndarray,
    profile_dts: np.ndarray,
    pole: complex,
    residue: complex,
    voltage: np.ndarray,
    state: complex,
    update_on_bin: np.ndarray,
    t_start,
    factor,
    thread_i,
):
    # y[n] = profile[n] + exp(p * dt) * y[n-1]
    # V[n] = 2 * Re(r * y[n])
    n_bins = len(profile)
    # state = 0.0 + 0.0j
    i_update = 0
    update_on_bin_i = update_on_bin[i_update]
    for bin_i in range(n_bins):
        profile_i_ = profile[bin_i]

        if bin_i == update_on_bin_i:
            if bin_i == 0:
                t_jump = profile_dts[0] - t_start + 0j
            else:
                t_jump = profile_dts[bin_i] - profile_dts[bin_i - 1] + 0j
            state *= np.exp(pole * t_jump)
            dt = profile_dts[bin_i + 1] - profile_dts[bin_i]
            decay = np.exp(pole * dt)

            i_update += 1
            if i_update < len(update_on_bin):
                update_on_bin_i = update_on_bin[i_update]
        else:
            state *= decay
        state += 0.5 * profile_i_
        voltage[thread_i, bin_i] += 2 * factor * np.real(residue * state)
        state = state + 0.5 * profile_i_

    return state


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
        float64,
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
    factor,
):
    n_poles = len(poles)
    voltage[:] = 0  # reset to zero from previous call
    voltage_threaded[:, :] = 0  # reset to zero from previous call
    if not (voltage_threaded.shape[0] == numba.get_num_threads()):
        raise Exception
    for pole_i in prange(n_poles):
        thread_i = numba.get_thread_id()

        states[pole_i] = apply_single_pole(
            profile,
            profile_dts,
            poles[pole_i],
            residues[pole_i],
            voltage_threaded[:, :],
            states[pole_i],
            update_on_bin,
            states[-1],
            factor,
            thread_i,
        )

    for thread_i in prange(numba.get_num_threads()):
        voltage += voltage_threaded[thread_i, :]
    states[-1] = profile_dts[-1]
