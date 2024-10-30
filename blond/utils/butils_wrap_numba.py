"""
BLonD physics functions, numba implementations
"""
from __future__ import annotations
import math
import random
from typing import TYPE_CHECKING

import numpy as np
from numba import get_num_threads, get_thread_id
from numba import jit
from numba import prange

if TYPE_CHECKING:
    from numpy.typing import NDArray


# --------------- Similar to kick.cpp -----------------
@jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
def kick(dt: NDArray, dE: NDArray, voltage: NDArray,
         omega_rf: NDArray, phi_rf: NDArray,
         charge: float, n_rf: int, acceleration_kick: float) -> None:
    """
    Function to apply RF kick on the particles with sin function
    """
    for i in prange(len(dt)):
        dE_sum = 0.0
        dti = dt[i]
        for j in range(len(voltage)):
            dE_sum += voltage[j] * np.sin(omega_rf[j] * dti + phi_rf[j])
        dE_sum *= charge
        dE[i] += dE_sum + acceleration_kick


@jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
def rf_volt_comp(voltages: NDArray, omega_rf: NDArray, phi_rf: NDArray,
                 bin_centers: NDArray) -> NDArray:
    """Compute rf voltage at each bin.

    Args:
        voltages (NDArray): _description_
        omega_rf (NDArray): _description_
        phi_rf (NDArray): _description_
        bin_centers (NDArray): _description_

    Returns:
        NDArray: _description_
    """
    rf_voltage = np.zeros(len(bin_centers))

    for j in range(len(voltages)):
        for i in prange(len(bin_centers)):
            rf_voltage[i] += voltages[j] * \
                             np.sin(omega_rf[j] * bin_centers[i] + phi_rf[j])

    return rf_voltage


# ---------------------------------------------------


# --------------- Similar to drift.cpp -----------------
@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def drift(dt: NDArray, dE: NDArray, solver: str, t_rev: float,
          length_ratio: float, alpha_order, eta_0: float,
          eta_1: float, eta_2: float, alpha_0: float,
          alpha_1: float, alpha_2: float, beta: float, energy: float) -> None:
    """
    Function to apply drift equation of motion
    0 == 'simple'
    1 == 'legacy'
    2 == 'exact'
    """

    T = t_rev * length_ratio

    if solver == 'simple':
        coeff = eta_0 / (beta * beta * energy)
        for i in prange(len(dt)):
            dt[i] += T * coeff * dE[i]

    elif solver == 'legacy':
        coeff = 1. / (beta * beta * energy)
        eta0 = eta_0 * coeff
        eta1 = eta_1 * coeff * coeff
        eta2 = eta_2 * coeff * coeff * coeff

        if alpha_order == 0:
            for i in prange(len(dt)):
                dt[i] += T * (1. / (1. - eta0 * dE[i]) - 1.)
        elif alpha_order == 1:
            for i in prange(len(dt)):
                dt[i] += T * (1. / (1. - eta0 * dE[i]
                                    - eta1 * dE[i] * dE[i]) - 1.)
        else:
            for i in prange(len(dt)):
                dt[i] += T * (1. / (1. - eta0 * dE[i]
                                    - eta1 * dE[i] * dE[i]
                                    - eta2 * dE[i] * dE[i] * dE[i]) - 1.)

    else:
        invbetasq = 1 / (beta * beta)
        invenesq = 1 / (energy * energy)
        for i in prange(len(dt)):
            beam_delta = np.sqrt(1. + invbetasq *
                                 (dE[i] * dE[i] * invenesq + 2. * dE[i] / energy)) - 1.

            dt[i] += T * (
                    (1. + alpha_0 * beam_delta +
                     alpha_1 * (beam_delta * beam_delta) +
                     alpha_2 * (beam_delta * beam_delta * beam_delta)) *
                    (1. + dE[i] / energy) / (1. + beam_delta) - 1.)
# ---------------------------------------------------


# --------------- Similar to histogram.cpp -----------------
@jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
def slice_beam(dt: NDArray, profile: NDArray,
               cut_left: float, cut_right: float) -> None:
    """Slice the time coordinate of the beam.

    Args:
        dt (NDArray): _description_
        profile (NDArray): _description_
        cut_left (float): _description_
        cut_right (float): _description_
    """

    n_slices = len(profile)
    n_parts = len(dt)
    inv_bin_width = n_slices / (cut_right - cut_left)
    n_threads = get_num_threads()

    # Per thread private profile to avoid cross-thread synchronization
    local_profile = np.zeros((n_threads, n_slices),
                             dtype=np.int32)

    # Operate in chunks of 512 particles to avoid calling the expensive
    # get_thread_id() function too often
    STEP = 512
    local_target_bin = np.empty((n_threads, STEP), dtype=np.int32)
    total_steps = math.ceil(n_parts / STEP)

    for i in prange(total_steps):
        thr_id = get_thread_id()
        start_i = i * STEP
        loop_count = min(STEP, n_parts - start_i)
        local_target_bin[thr_id][:loop_count] = np.floor(
            (dt[start_i:start_i + loop_count] - cut_left) * inv_bin_width)

        for j in range(loop_count):
            if local_target_bin[thr_id][j] >= 0 and local_target_bin[thr_id][j] < n_slices:
                local_profile[thr_id, local_target_bin[thr_id][j]] += 1

    # reduce the private profiles to the global profile
    for i in prange(n_slices):
        profile[i] = 0.0
        for j in range(n_threads):
            profile[i] += local_profile[j, i]




# --------------- Similar to linear_interp_kick.cpp -----------------
@jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
def linear_interp_kick(dt: NDArray, dE: NDArray, voltage: NDArray,
                       bin_centers: NDArray, charge: float,
                       acceleration_kick: float) -> None:
    """Interpolated kick method.

    Args:
        dt (NDArray): _description_
        dE (NDArray): _description_
        voltage (NDArray): _description_
        bin_centers (NDArray): _description_
        charge (float): _description_
        acceleration_kick (float): _description_
    """
    n_slices = len(bin_centers)
    inv_bin_width = (n_slices - 1) / (bin_centers[-1] - bin_centers[0])

    helper = np.empty(2 * (n_slices - 1), dtype=np.float64)
    for i in prange(n_slices - 1):
        helper[2 * i] = charge * (voltage[i + 1] - voltage[i]) * inv_bin_width
        helper[2 * i + 1] = (charge * voltage[i] - bin_centers[i]
                             * helper[2 * i]) + acceleration_kick

    for i in prange(len(dt)):
        fbin = int(np.floor((dt[i] - bin_centers[0]) * inv_bin_width))  # FIXME THIS IS DEACTIVATED IN PYTHON VERSION
        if (fbin >= 0) and (fbin < n_slices - 1):
            dE[i] += dt[i] * helper[2 * fbin] + helper[2 * fbin + 1]


# ---------------------------------------------------


# --------------- Similar to synchrotron_radiation.cpp -----------------
@jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
def synchrotron_radiation(dE: NDArray, U0: float,
                          n_kicks: int, tau_z: float) -> None:
    """Apply SR

    Args:
        dE (NDArray): _description_
        U0 (float): _description_
        n_kicks (int): _description_
        tau_z (float): _description_
    """
    # Adjust inputs before the loop to reduce computations
    U0 = U0 / n_kicks
    tau_z = tau_z * n_kicks

    # SR damping constant, adjusted for better performance
    const_synch_rad = 1.0 - 2.0 / tau_z

    for i in prange(len(dE)):
        for _ in range(n_kicks): \
            # FIXME THIS SEEMS TO BE VERY DIFFERENT FROM PYTHON IMPLEMENTATION
            dE[i] = dE[i] * const_synch_rad - U0


@jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
def synchrotron_radiation_full(dE: NDArray, U0: float,
                               n_kicks: int, tau_z: float,
                               sigma_dE: float, energy: float) -> None:
    """Apply SR with quantum excitation

    Args:
        dE (NDArray): _description_
        U0 (float): _description_
        n_kicks (int): _description_
        tau_z (float): _description_
        sigma_dE (float): _description_
        energy (float): _description_
    """

    # Adjust inputs before the loop to reduce computations
    U0 = U0 / n_kicks
    tau_z = tau_z * n_kicks

    const_quantum_exc = 2.0 * sigma_dE / np.sqrt(tau_z) * energy
    const_synch_rad = 1.0 - 2.0 / tau_z

    for i in prange(len(dE)):
        # rand_arr = np.random.normal(0.0, 1.0, size=n_kicks)
        for j in range(n_kicks):
            # FIXME THIS SEEMS TO BE VERY DIFFERENT FROM PYTHON IMPLEMENTATION
            dE[i] = dE[i] * const_synch_rad + \
                    const_quantum_exc * random.gauss(0.0, 1.0) - U0


def fast_resonator(R_S: NDArray, Q: NDArray, frequency_array: NDArray,
                   frequency_R: NDArray, impedance: NDArray = None) -> NDArray:
    """
    We're defining and calling a function internally due to issues
    dealing with parallelization and the allocation of the impedance array.
    """

    if impedance is None:
        impedance = np.zeros(len(frequency_array), dtype=np.complex128)

    @jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
    def calc_impedance(R_S: NDArray, Q: NDArray, frequency_array: NDArray,
                       frequency_R: NDArray, impedance: NDArray):
        for freq in prange(1, len(frequency_array)):
            for i in range(len(R_S)):
                # todo speedup with tmp variable
                impedance[freq] += R_S[i] / (1 + 1j * Q[i] * (frequency_array[freq] / frequency_R[i] -
                                                              frequency_R[i] / frequency_array[freq]))

    calc_impedance(R_S, Q, frequency_array, frequency_R, impedance)

    return impedance


from blond.utils.butils_wrap_python import (
    distribution_from_tomoscope as __distribution_from_tomoscope_python,
    sparse_histogram as __sparse_histogram_python,
    beam_phase_fast as __beam_phase_fast_python,
    beam_phase as __beam_phase_python,
    music_track_multiturn as __music_track_multiturn_python,
    music_track as __music_track_python,
    slice_smooth as __slice_smooth_python,
    # as __XXX_python,
)
# Just-in-time (JIT) compilation of Python routines for speedup
# TODO define signature, similar to the C++ callbacks
# TODO implement versions that can be executed in parallel
options = dict(nopython=True, nogil=True, fastmath=True, cache=True)
sparse_histogram = jit(__sparse_histogram_python, **options)
sparse_histogram.__doc__ = __sparse_histogram_python.__doc__

distribution_from_tomoscope = jit(__distribution_from_tomoscope_python, **options)
distribution_from_tomoscope.__doc__ = __distribution_from_tomoscope_python.__doc__

beam_phase_fast = jit(__beam_phase_fast_python, **options)
beam_phase_fast.__doc__ = __beam_phase_fast_python.__doc__

beam_phase = jit(__beam_phase_python, **options)
beam_phase.__doc__ = __beam_phase_python.__doc__

music_track_multiturn = jit(__music_track_multiturn_python, **options)
music_track_multiturn.__doc__ = __music_track_multiturn_python.__doc__

music_track = jit(__music_track_python, **options)
music_track.__doc__ = __music_track_python.__doc__

slice_smooth = jit(__slice_smooth_python, **options)
slice_smooth.__doc__ = __slice_smooth_python.__doc__

# XXX = jit(__XXX_python, **options)
# XXX.__doc__ = __XXX_python.__doc__
