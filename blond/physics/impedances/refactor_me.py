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
def decay_state(
    dt: float,
    poles: np.ndarray,
    states: np.ndarray,
):
    for pole_i in range(len(poles)):
        states *= np.exp(poles * dt)


@numba.njit()
def apply_single_pole(
    profile: np.ndarray,
    dt: float,
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
    decay = np.exp(pole * dt)
    state = states[pole_i]
    for bin_i in range(n_bins):
        profile_i_ = profile[bin_i]
        state *= decay
        state += 0.5 * profile_i_
        voltage[bin_i] += 2 * np.real(residue * state)
        state += 0.5 * profile_i_
    states[pole_i] = state


from numba import complex128, float64, void


@numba.njit(
    void(
        float64[:],
        float64,
        complex128[:],
        complex128[:],
        complex128[:],
        float64[:],
    ),
    fastmath=True,
)
def apply_poles(
    # read
    profile,
    dt,
    poles,
    residues,
    # write
    states,
    voltage,
):
    for i in range(len(residues)):
        apply_single_pole(
            profile, dt, poles[i], residues[i], voltage, states, i
        )
