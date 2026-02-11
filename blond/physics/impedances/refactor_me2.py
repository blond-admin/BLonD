# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import numba
import numpy as np


@numba.njit()
def apply_single_pole(
    profile: np.ndarray,
    profile_dts: np.ndarray,
    pole: complex,
    residue: complex,
    voltage: np.ndarray,
    states: np.ndarray,
    pole_i: int,
):
    # y[n] = profile[n] + exp(p * dt) * y[n-1]
    # V[n] = 2 * Re(r * y[n])
    n_bins = len(profile)
    # state = 0.0 + 0.0j
    state = states[pole_i]
    t_previous = states[-1]
    for bin_i in range(n_bins):
        profile_i_ = profile[bin_i]
        t_current = profile_dts[bin_i]
        dt = t_current - t_previous
        decay = np.exp(pole * dt)

        state = state * decay + 0.5 * profile_i_
        voltage[bin_i] += 2 * np.real(residue * state)
        state += 0.5 * profile_i_
        t_previous = t_current

    states[pole_i] = state
    states[-1] = t_previous


from numba import complex128, float64, prange, void


@numba.njit(
    void(
        float64[:],
        float64[:],
        complex128[:],
        complex128[:],
        complex128[:],
        float64[:],
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
):
    n_threads = numba.get_num_threads()  # this prevents caching
    n_poles = len(poles)

    array_tmp = np.zeros((n_threads, len(voltage)))

    for i in prange(n_poles):
        thread_i = numba.get_thread_id()

        apply_single_pole(
            profile,
            profile_dts,
            poles[i],
            residues[i],
            array_tmp[thread_i, :],
            states,
            i,
        )
    voltage[:] = np.sum(array_tmp, axis=0)
