'''
BLonD physics functions, numba implementations
'''

import math
import random

import numpy as np
from numba import get_num_threads, get_thread_id
from numba import jit
from numba import prange


# --------------- Similar to kick.cpp -----------------
@jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
def kick(dt: np.ndarray, dE: np.ndarray, voltage: np.ndarray,
         omega_rf: np.ndarray, phi_rf: np.ndarray,
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
def rf_volt_comp(voltages: np.ndarray, omega_rf: np.ndarray, phi_rf: np.ndarray,
                 bin_centers: np.ndarray) -> np.ndarray:
    """Compute rf voltage at each bin.

    Args:
        voltages (np.ndarray): _description_
        omega_rf (np.ndarray): _description_
        phi_rf (np.ndarray): _description_
        bin_centers (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
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
def drift(dt: np.ndarray, dE: np.ndarray, solver: str, t_rev: float,
          length_ratio: float, alpha_order, eta_0: float,
          eta_1: float, eta_2: float, alpha_0: float,
          alpha_1: float, alpha_2: float, beta: float, energy: float) -> None:
    '''
    Function to apply drift equation of motion
    0 == 'simple'
    1 == 'legacy'
    2 == 'exact'
    '''

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
def slice_beam(dt: np.ndarray, profile: np.ndarray,
               cut_left: float, cut_right: float) -> None:
    """Slice the time coordinate of the beam.

    Args:
        dt (np.ndarray): _description_
        profile (np.ndarray): _description_
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

    # profile[:] = 0.0

    # profile_len = len(profile)
    # inv_bin_width = profile_len / (cut_right - cut_left)
    # target_bin = np.empty(len(dt), dtype=np.int32)
    # for i in prange(len(dt)):
    #     target_bin[i] = int(np.floor((dt[i] - cut_left) * inv_bin_width))

    # for tbin in target_bin:
    #     if tbin >= 0 and tbin < profile_len:
    #         profile[tbin] += 1.0

    # profile[:] = np.histogram(dt, bins=len(profile),
    #                          range=(cut_left, cut_right))[0]


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def slice_smooth(dt: np.ndarray, profile: np.ndarray,
                 cut_left: float, cut_right: float) -> None:
    """Smooth slice method.

    Args:
        dt (np.ndarray): _description_
        profile (np.ndarray): _description_
        cut_left (float): _description_
        cut_right (float): _description_
    """
    # Constants init
    n_slices = len(profile)
    bin_width = (cut_right - cut_left) / n_slices
    inv_bin_width = 1. / bin_width
    const1 = (cut_left + bin_width * 0.5)
    const2 = (cut_right - bin_width * 0.5)

    profile[:] = 0.0

    for a in dt:
        if (a < const1) or (a > const2):
            continue

        fbin = (a - cut_left) * inv_bin_width
        ffbin = int(fbin)
        distToCenter = fbin - ffbin
        if distToCenter > 0.5:
            fffbin = int(fbin + 1)
        else:
            fffbin = int(fbin - 1)
        profile[ffbin] += (0.5 - distToCenter)
        profile[fffbin] += (0.5 + distToCenter)


# ---------------------------------------------------


# --------------- Similar to linear_interp_kick.cpp -----------------
@jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
def linear_interp_kick(dt: np.ndarray, dE: np.ndarray, voltage: np.ndarray,
                       bin_centers: np.ndarray, charge: float,
                       acceleration_kick: float) -> None:
    """Interpolated kick method.

    Args:
        dt (np.ndarray): _description_
        dE (np.ndarray): _description_
        voltage (np.ndarray): _description_
        bin_centers (np.ndarray): _description_
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
        fbin = int(np.floor((dt[i] - bin_centers[0]) * inv_bin_width))
        if (fbin >= 0) and (fbin < n_slices - 1):
            dE[i] += dt[i] * helper[2 * fbin] + helper[2 * fbin + 1]


# ---------------------------------------------------


# --------------- Similar to synchrotron_radiation.cpp -----------------
@jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
def synchrotron_radiation(dE: np.ndarray, U0: float,
                          n_kicks: int, tau_z: float) -> None:
    """Apply SR

    Args:
        dE (np.ndarray): _description_
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
        for _ in range(n_kicks):
            dE[i] = dE[i] * const_synch_rad - U0


@jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
def synchrotron_radiation_full(dE: np.ndarray, U0: float,
                               n_kicks: int, tau_z: float,
                               sigma_dE: float, energy: float) -> None:
    """Apply SR with quantum excitation

    Args:
        dE (np.ndarray): _description_
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
            dE[i] = dE[i] * const_synch_rad + \
                    const_quantum_exc * random.gauss(0.0, 1.0) - U0


# @jit(nopython=False, nogil=True, fastmath=False, parallel=False)
def set_random_seed(seed: int) -> None:
    """Set the seed of the RNG used in synchrotron radiation

    Args:
        seed (int): _description_
    """
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------


# --------------- Similar to music_track.cpp -----------------
@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def music_track(dt: np.ndarray, dE: np.ndarray, induced_voltage: np.ndarray,
                array_parameters: np.ndarray, alpha: float, omega_bar: float,
                const: float, coeff1: float, coeff2: float,
                coeff3: float, coeff4: float) -> None:
    '''
    This function calculates the single-turn induced voltage and updates the
    energies of the particles.

    Parameters
    ----------
    dt : float array
        Longitudinal coordinates [s]
    dE : float array
        Initial energies [V]
    induced_voltage : float array
        array used to store the output of the computation
    array_parameters : float array
        See documentation in music.py
    alpha, omega_bar, const, coeff1, coeff2, coeff3, coeff4 : floats
        See documentation in music.py

    Returns
    -------
    induced_voltage : float array
        Computed induced voltage.
    beam_dE : float array
        Array of energies updated.
    '''

    indices_sorted = np.argsort(dt)
    dt = dt[indices_sorted]
    dE = dE[indices_sorted]

    # MuSiC algorithm
    dE[0] += induced_voltage[0]
    input_first_component = 1.0
    input_second_component = 0.0

    for i in range(len(dt) - 1):
        time_difference = dt[i + 1] - dt[i]

        exp_term = np.exp(-alpha * time_difference)
        cos_term = np.cos(omega_bar * time_difference)
        sin_term = np.sin(omega_bar * time_difference)

        product_first_component = exp_term * \
                                  ((cos_term + coeff1 * sin_term) * input_first_component
                                   + coeff2 * sin_term * input_second_component)
        product_second_component = exp_term * \
                                   (coeff3 * sin_term * input_first_component
                                    + (cos_term + coeff4 * sin_term) * input_second_component)

        induced_voltage[i + 1] = const * (0.5 + product_first_component)
        dE[i + 1] += induced_voltage[i + 1]

        input_first_component = product_first_component + 1.0
        input_second_component = product_second_component

    array_parameters[0] = input_first_component
    array_parameters[1] = input_second_component
    array_parameters[3] = dt[-1]


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def music_track_multiturn(dt: np.ndarray, dE: np.ndarray, induced_voltage: np.ndarray,
                          array_parameters: np.ndarray, alpha: float, omega_bar: float,
                          const: float, coeff1: float, coeff2: float,
                          coeff3: float, coeff4: float) -> None:
    """This function calculates the multi-turn induced voltage and updates the
    energies of the particles.
    Parameters and Returns as for music_track.

    Args:
        dt (np.ndarray): _description_
        dE (np.ndarray): _description_
        induced_voltage (np.ndarray): _description_
        array_parameters (np.ndarray): _description_
        alpha (float): _description_
        omega_bar (float): _description_
        const (float): _description_
        coeff1 (float): _description_
        coeff2 (float): _description_
        coeff3 (float): _description_
        coeff4 (float): _description_
    """
    indices_sorted = np.argsort(dt)
    dt = dt[indices_sorted]
    dE = dE[indices_sorted]
    time_difference_0 = dt[0] + array_parameters[2] - array_parameters[3]
    exp_term = np.exp(-alpha * time_difference_0)
    cos_term = np.cos(omega_bar * time_difference_0)
    sin_term = np.sin(omega_bar * time_difference_0)

    product_first_component = exp_term * (
            (cos_term + coeff1 * sin_term)
            * array_parameters[0]
            + coeff2 * sin_term * array_parameters[1])

    product_second_component = exp_term * (
            coeff3 * sin_term * array_parameters[0]
            + (cos_term + coeff4 * sin_term) * array_parameters[1])

    induced_voltage[0] = const * (0.5 + product_first_component)

    dE[0] += induced_voltage[0]
    input_first_component = product_first_component + 1.0
    input_second_component = product_second_component

    # MuSiC algorithm for the current turn

    for i in range(len(dt) - 1):
        time_difference = dt[i + 1] - dt[i]

        exp_term = np.exp(-alpha * time_difference)
        cos_term = np.cos(omega_bar * time_difference)
        sin_term = np.sin(omega_bar * time_difference)

        product_first_component = exp_term * \
                                  ((cos_term + coeff1 * sin_term) * input_first_component
                                   + coeff2 * sin_term * input_second_component)
        product_second_component = exp_term * \
                                   (coeff3 * sin_term * input_first_component
                                    + (cos_term + coeff4 * sin_term) * input_second_component)

        induced_voltage[i + 1] = const * (0.5 + product_first_component)
        dE[i + 1] += induced_voltage[i + 1]

        input_first_component = product_first_component + 1.0
        input_second_component = product_second_component

    array_parameters[0] = input_first_component
    array_parameters[1] = input_second_component
    array_parameters[3] = dt[-1]


# ---------------------------------------------------


# # --------------- Similar to fast_resonator.cpp -----------------
# @jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
# def fast_resonator(R_S: np.ndarray, Q: np.ndarray, frequency_array: np.ndarray,
#                    frequency_R: np.ndarray, impedance: np.ndarray = None) -> np.ndarray:

#     if impedance is None:
#         impedance = np.zeros(len(frequency_array), dtype=np.complex128)

#     for freq in prange(1, len(frequency_array)):
#         impedance[freq] = 0.0
#         for i in range(len(R_S)):
#             impedance[freq] += R_S[i] / (1 + 1j * Q[i] * (frequency_array[freq] / frequency_R[i] -
#                                                           frequency_R[i] / frequency_array[freq]))

#     return impedance
# # ---------------------------------------------------

# --------------- Similar to fast_resonator.cpp -----------------
def fast_resonator(R_S: np.ndarray, Q: np.ndarray, frequency_array: np.ndarray,
                   frequency_R: np.ndarray, impedance: np.ndarray = None) -> np.ndarray:
    '''
    We're defining and calling a function internally due to issues
    dealing with parallelization and the allocation of the impedance array.
    '''

    if impedance is None:
        impedance = np.zeros(len(frequency_array), dtype=np.complex128)

    @jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
    def calc_impedance(R_S: np.ndarray, Q: np.ndarray, frequency_array: np.ndarray,
                       frequency_R: np.ndarray, impedance: np.ndarray):
        for freq in prange(1, len(frequency_array)):
            for i in range(len(R_S)):
                impedance[freq] += R_S[i] / (1 + 1j * Q[i] * (frequency_array[freq] / frequency_R[i] -
                                                              frequency_R[i] / frequency_array[freq]))

    calc_impedance(R_S, Q, frequency_array, frequency_R, impedance)

    return impedance


# ---------------------------------------------------


# --------------- Similar to beam_phase.cpp -----------------
@jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
def beam_phase(bin_centers: np.ndarray, profile: np.ndarray,
               alpha: float, omegarf: float,
               phirf: float, bin_size: float) -> float:
    scoeff = np.trapz(np.exp(alpha * (bin_centers))
                      * np.sin(omegarf * bin_centers + phirf)
                      * profile, dx=bin_size)
    ccoeff = np.trapz(np.exp(alpha * (bin_centers))
                      * np.cos(omegarf * bin_centers + phirf)
                      * profile, dx=bin_size)

    return scoeff / ccoeff


@jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
def beam_phase_fast(bin_centers: np.ndarray, profile: np.ndarray,
                    omegarf: float, phirf: float, bin_size: float) -> float:
    scoeff = np.trapz(profile * np.sin(omegarf * bin_centers + phirf),
                      dx=bin_size)
    ccoeff = np.trapz(profile * np.cos(omegarf * bin_centers + phirf),
                      dx=bin_size)

    return scoeff / ccoeff


# ---------------------------------------------------


# --------------- Similar to sparse_histogram.cpp -----------------
@jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def sparse_histogram(dt: np.ndarray, profile: np.ndarray,
                     cut_left: np.ndarray, cut_right: np.ndarray,
                     bunch_indexes: np.ndarray, n_slices_bucket: int) -> None:
    '''
    Optimised routine that calculates the histogram for a sparse beam
    Author: Juan F. Esteban Mueller, Danilo Quartullo, Alexandre Lasheen, Markus Schwarz
    '''
    # Only valid for cut_edges = edges
    inv_bucket_length = 1.0 / (cut_right[0] - cut_left[0])
    inv_bin_width = inv_bucket_length * n_slices_bucket

    profile[:] = 0.0

    for a in dt:
        if ((a < cut_left[0]) or (a > cut_right[-1])):
            continue
        # Find bucket in which the particle is and its index
        fbunch = int((a - cut_left[0]) * inv_bucket_length)
        i_bucket = int(bunch_indexes[fbunch])
        if i_bucket == -1:
            continue
        # Find the bin inside the corresponding bucket
        fbin = int((a - cut_left[i_bucket]) * inv_bin_width)
        profile[i_bucket, fbin] += 1.0


# ---------------------------------------------------


# --------------- Similar to tomoscope.cpp -----------------
@jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
def distribution_from_tomoscope(dt: np.ndarray, dE: np.ndarray, probDistr: np.ndarray,
                                seed: int, profLen: int,
                                cutoff: float, x0: float, y0: float,
                                dtBin: float, dEBin: float) -> None:
    '''
    Generation of particle distribution from probability density
    Author: Helga Timko
    '''
    import bisect

    # Initialize random seed
    np.random.seed(seed)

    # Initialize variables
    cutoff2 = cutoff * cutoff
    dtMin = -1.0 * x0 * dtBin
    dEMin = -1.0 * y0 * dEBin

    # Calculate cumulative probability
    # profLen2 = np.uint32(profLen * profLen)
    cumulDistr = np.cumsum(probDistr)

    # Normalize probability distribution
    totProb = cumulDistr[-1]
    cumulDistr = cumulDistr / totProb

    # Generate particle coordinates

    n = 0
    while n < len(dt):
        randProb = np.random.random()
        m = bisect.bisect(cumulDistr, randProb)
        # for m in range(profLen2):
        #     if randProb < cumulDistr[m]:
        #         break
        i = int(m / profLen)
        k = m % profLen

        iPos = i + np.random.random() - 0.5
        kPos = k + np.random.random() - 0.5

        if ((iPos - x0) * (iPos - x0) + (kPos - y0) * (kPos - y0)) < cutoff2:
            dt[n] = dtMin + iPos * dtBin
            dE[n] = dEMin + kPos * dEBin
            n += 1
# ---------------------------------------------------
