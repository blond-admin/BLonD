"""
BLonD physics functions, python-only implementations

@author: alasheen, kiliakis
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray as NumpyArray
from scipy.constants import e

if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import NDArray as NumpyArray

RNG = np.random.default_rng()


# --------------- Similar to kick.cpp -----------------
def kick(
    dt: NumpyArray,
    dE: NumpyArray,
    voltage: NumpyArray,
    omega_rf: NumpyArray,
    phi_rf: NumpyArray,
    charge: float,
    n_rf: int,
    acceleration_kick: float,
):
    """
    Function to apply RF kick on the particles with sin function
    """

    voltage_kick = charge * voltage

    for j in range(n_rf):
        dE += voltage_kick[j] * np.sin(omega_rf[j] * dt + phi_rf[j])

    dE[:] += acceleration_kick


def rf_volt_comp(
    voltages: NumpyArray,
    omega_rf: NumpyArray,
    phi_rf: NumpyArray,
    bin_centers: NumpyArray,
) -> NumpyArray:
    """Compute rf voltage at each bin.

    Args:
        voltages (NumpyArray): _description_
        omega_rf (NumpyArray): _description_
        phi_rf (NumpyArray): _description_
        bin_centers (NumpyArray): _description_

    Returns:
        NumpyArray: _description_
    """
    rf_voltage = np.zeros(len(bin_centers))

    for j in range(len(voltages)):
        rf_voltage += voltages[j] * np.sin(
            omega_rf[j] * bin_centers + phi_rf[j]
        )

    return rf_voltage


# ---------------------------------------------------


# --------------- Similar to drift.cpp -----------------
def drift(
    dt: NumpyArray,
    dE: NumpyArray,
    solver: str,
    t_rev: float,
    length_ratio: float,
    alpha_order,
    eta_0: float,
    eta_1: float,
    eta_2: float,
    alpha_0: float,
    alpha_1: float,
    alpha_2: float,
    beta: float,
    energy: float,
):
    """
    Function to apply drift equation of motion
    """

    # solver_decoded = solver.decode(encoding='utf_8')

    T = t_rev * length_ratio

    if solver == "simple":
        coeff = eta_0 / (beta * beta * energy)
        dt += T * coeff * dE

    elif solver == "legacy":
        coeff = 1.0 / (beta * beta * energy)
        eta0 = eta_0 * coeff
        eta1 = eta_1 * coeff * coeff
        eta2 = eta_2 * coeff * coeff * coeff

        if alpha_order == 0:
            dt += T * (1.0 / (1.0 - eta0 * dE) - 1.0)
        elif alpha_order == 1:
            dt += T * (1.0 / (1.0 - eta0 * dE - eta1 * dE * dE) - 1.0)
        else:
            dt += T * (
                1.0 / (1.0 - eta0 * dE - eta1 * dE * dE - eta2 * dE * dE * dE)
                - 1.0
            )

    else:
        invbetasq = 1 / (beta * beta)
        invenesq = 1 / (energy * energy)
        # double beam_delta;

        beam_delta = (
            np.sqrt(1.0 + invbetasq * (dE * dE * invenesq + 2.0 * dE / energy))
            - 1.0
        )

        dt += T * (
            (
                1.0
                + alpha_0 * beam_delta
                + alpha_1 * (beam_delta * beam_delta)
                + alpha_2 * (beam_delta * beam_delta * beam_delta)
            )
            * (1.0 + dE / energy)
            / (1.0 + beam_delta)
            - 1.0
        )


# ---------------------------------------------------


# --------------- Similar to histogram.cpp -----------------
def slice_beam(
    dt: NumpyArray, profile: NumpyArray, cut_left: float, cut_right: float
):
    """Slice the time coordinate of the beam.

    Args:
        dt (NumpyArray): _description_
        profile (NumpyArray): _description_
        cut_left (float): _description_
        cut_right (float): _description_
    """
    profile[:] = np.histogram(
        dt, bins=len(profile), range=(cut_left, cut_right)
    )[0]


def slice_smooth(
    dt: NumpyArray, profile: NumpyArray, cut_left: float, cut_right: float
):
    """Smooth slice method.

    Args:
        dt (NumpyArray): _description_
        profile (NumpyArray): _description_
        cut_left (float): _description_
        cut_right (float): _description_
    """
    # Constants init
    n_slices = len(profile)
    bin_width = (cut_right - cut_left) / n_slices
    inv_bin_width = 1.0 / bin_width
    const1 = cut_left + bin_width * 0.5
    const2 = cut_right - bin_width * 0.5

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
        profile[ffbin] += 0.5 - distToCenter
        profile[fffbin] += 0.5 + distToCenter


# ---------------------------------------------------


# --------------- Similar to linear_interp_kick.cpp -----------------
def linear_interp_kick(
    dt: NumpyArray,
    dE: NumpyArray,
    voltage: NumpyArray,
    bin_centers: NumpyArray,
    charge: float,
    acceleration_kick: float,
):
    """Interpolated kick method.

    Args:
        dt (NumpyArray): _description_
        dE (NumpyArray): _description_
        voltage (NumpyArray): _description_
        bin_centers (NumpyArray): _description_
        charge (float): _description_
        acceleration_kick (float): _description_
    """
    n_slices = len(bin_centers)
    inv_bin_width = (n_slices - 1) / (bin_centers[-1] - bin_centers[0])

    fbin = np.floor((dt - bin_centers[0]) * inv_bin_width).astype(np.int32)

    helper1 = charge * (voltage[1:] - voltage[:-1]) * inv_bin_width
    helper2 = (
        charge * voltage[:-1] - bin_centers[:-1] * helper1
    ) + acceleration_kick

    for i in range(len(dt)):
        # fbin = int(np.floor((dt[i]-bin_centers[0])*inv_bin_width))
        if (fbin[i] >= 0) and (fbin[i] < n_slices - 1):
            dE[i] += dt[i] * helper1[fbin[i]] + helper2[fbin[i]]


# ---------------------------------------------------


# --------------- Similar to synchrotron_radiation.cpp -----------------
def synchrotron_radiation(
    dE: NumpyArray, U0: float, n_kicks: int, tau_z: float
):
    """Apply SR

    Args:
        dE (NumpyArray): _description_
        U0 (float): _description_
        n_kicks (int): _description_
        tau_z (float): _description_
    """
    for _ in range(n_kicks):
        dE += -(2.0 / tau_z / n_kicks * dE + U0 / n_kicks)


def synchrotron_radiation_full(
    dE: NumpyArray,
    U0: float,
    n_kicks: int,
    tau_z: float,
    sigma_dE: float,
    energy: float,
):
    """Apply SR with quantum excitation

    Args:
        dE (NumpyArray): _description_
        U0 (float): _description_
        n_kicks (int): _description_
        tau_z (float): _description_
        sigma_dE (float): _description_
        energy (float): _description_
    """

    for _ in range(n_kicks):
        dE += -(
            2.0 / tau_z / n_kicks * dE
            + U0 / n_kicks
            - 2.0
            * sigma_dE
            / np.sqrt(tau_z * n_kicks)
            * energy
            * RNG.standard_normal(len(dE))
        )


def set_random_seed(seed: int):
    """Set the seed of the RNG used in synchrotron radiation

    Args:
        seed (int): _description_
    """
    global RNG
    # Re-initialize the RNG with new seed
    RNG = np.random.default_rng(seed)
    np.random.seed(seed)


# ---------------------------------------------------


# --------------- Similar to music_track.cpp -----------------
def music_track(
    dt: NumpyArray,
    dE: NumpyArray,
    induced_voltage: NumpyArray,
    array_parameters: NumpyArray,
    alpha: float,
    omega_bar: float,
    const: float,
    coeff1: float,
    coeff2: float,
    coeff3: float,
    coeff4: float,
):
    """
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
    """

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

        product_first_component = exp_term * (
            (cos_term + coeff1 * sin_term) * input_first_component
            + coeff2 * sin_term * input_second_component
        )
        product_second_component = exp_term * (
            coeff3 * sin_term * input_first_component
            + (cos_term + coeff4 * sin_term) * input_second_component
        )

        induced_voltage[i + 1] = const * (0.5 + product_first_component)
        dE[i + 1] += induced_voltage[i + 1]

        input_first_component = product_first_component + 1.0
        input_second_component = product_second_component

    array_parameters[0] = input_first_component
    array_parameters[1] = input_second_component
    array_parameters[3] = dt[-1]


def music_track_multiturn(
    dt: NumpyArray,
    dE: NumpyArray,
    induced_voltage: NumpyArray,
    array_parameters: NumpyArray,
    alpha: float,
    omega_bar: float,
    const: float,
    coeff1: float,
    coeff2: float,
    coeff3: float,
    coeff4: float,
):
    """This function calculates the multi-turn induced voltage and updates the
    energies of the particles.
    Parameters and Returns as for music_track.

    Args:
        dt (NumpyArray): _description_
        dE (NumpyArray): _description_
        induced_voltage (NumpyArray): _description_
        array_parameters (NumpyArray): _description_
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
        (cos_term + coeff1 * sin_term) * array_parameters[0]
        + coeff2 * sin_term * array_parameters[1]
    )

    product_second_component = exp_term * (
        coeff3 * sin_term * array_parameters[0]
        + (cos_term + coeff4 * sin_term) * array_parameters[1]
    )

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

        product_first_component = exp_term * (
            (cos_term + coeff1 * sin_term) * input_first_component
            + coeff2 * sin_term * input_second_component
        )
        product_second_component = exp_term * (
            coeff3 * sin_term * input_first_component
            + (cos_term + coeff4 * sin_term) * input_second_component
        )

        induced_voltage[i + 1] = const * (0.5 + product_first_component)
        dE[i + 1] += induced_voltage[i + 1]

        input_first_component = product_first_component + 1.0
        input_second_component = product_second_component

    array_parameters[0] = input_first_component
    array_parameters[1] = input_second_component
    array_parameters[3] = dt[-1]


# ---------------------------------------------------


# --------------- Similar to fast_resonator.cpp -----------------
def fast_resonator(
    R_S: NumpyArray,
    Q: NumpyArray,
    frequency_array: NumpyArray,
    frequency_R: NumpyArray,
    impedance: Optional[NumpyArray] = None,
) -> NumpyArray:
    """
    This function takes as an input a list of resonators parameters and
    computes the impedance in an optimised way.

    Parameters
    ----------
    frequencies: float array
        array of frequency in Hz
    shunt_impedances: float array
        array of shunt impedances in Ohm
    Q_values: float array
        array of quality factors
    resonant_frequencies: float array
        array of resonant frequency in Hz
    n_resonators: int
        number of resonantors
    n_frequencies: int
        length of the array 'frequencies'

    Returns
    -------
    impedanceReal: float array
        real part of the impedance
    impedanceImag: float array
        imaginary part of the impedance
    """
    if impedance is None:
        impedance = np.zeros(len(frequency_array), dtype=complex)

    for i in range(len(R_S)):
        impedance[1:] += R_S[i] / (
            1
            + 1j
            * Q[i]
            * (
                frequency_array[1:] / frequency_R[i]
                - frequency_R[i] / frequency_array[1:]
            )
        )

    return impedance


# ---------------------------------------------------


# --------------- Similar to beam_phase.cpp -----------------
def beam_phase(
    bin_centers: NumpyArray,
    profile: NumpyArray,
    alpha: float,
    omega_rf: float,
    phi_rf: float,
    bin_size: float,
) -> float:
    scoeff = np.trapezoid(
        np.exp(alpha * (bin_centers))
        * np.sin(omega_rf * bin_centers + phi_rf)
        * profile,
        dx=bin_size,
    )
    ccoeff = np.trapezoid(
        np.exp(alpha * (bin_centers))
        * np.cos(omega_rf * bin_centers + phi_rf)
        * profile,
        dx=bin_size,
    )

    return scoeff / ccoeff


def beam_phase_fast(
    bin_centers: NumpyArray,
    profile: NumpyArray,
    omega_rf: float,
    phi_rf: float,
    bin_size: float,
) -> float:
    scoeff = np.trapezoid(
        profile * np.sin(omega_rf * bin_centers + phi_rf), dx=bin_size
    )
    ccoeff = np.trapezoid(
        profile * np.cos(omega_rf * bin_centers + phi_rf), dx=bin_size
    )

    return scoeff / ccoeff


# ---------------------------------------------------


# --------------- Similar to sparse_histogram.cpp -----------------
def sparse_histogram(
    dt: NumpyArray,
    profile: NumpyArray,
    cut_left: NumpyArray,
    cut_right: NumpyArray,
    bunch_indexes: NumpyArray,
    n_slices_bucket: int,
):
    """
    Optimised routine that calculates the histogram for a sparse beam
    Author: Juan F. Esteban Mueller, Danilo Quartullo, Alexandre Lasheen, Markus Schwarz
    """
    # Only valid for cut_edges = edges
    inv_bucket_length = 1.0 / (cut_right[0] - cut_left[0])
    inv_bin_width = inv_bucket_length * n_slices_bucket

    profile[:] = 0.0

    for a in dt:
        if (a < cut_left[0]) or (a > cut_right[-1]):
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
def distribution_from_tomoscope(
    dt: np.ndarray,
    dE: np.ndarray,
    probDistr: np.ndarray,
    seed: int,
    profLen: int,
    cutoff: float,
    x0: float,
    y0: float,
    dtBin: float,
    dEBin: float,
):
    """
    Generation of particle distribution from probability density
    Author: Helga Timko
    """

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


def resonator_induced_voltage_1_turn(
    kappa1: NumpyArray,
    n_macroparticles: NumpyArray,
    bin_centers: NumpyArray,
    bin_size: float,
    n_time: int,
    deltaT: NumpyArray,
    tArray: NumpyArray,
    reOmegaP: NumpyArray,
    imOmegaP: NumpyArray,
    Qtilde: NumpyArray,
    n_resonators: int,
    omega_r: NumpyArray,
    Q: NumpyArray,
    tmp_matrix: NumpyArray,
    charge: float,
    beam_n_macroparticles: int,
    ratio: float,
    R: NumpyArray,
    induced_voltage: NumpyArray,
    float_precision: type,
):
    r"""
    Method to calculate the induced voltage through linearly
    interpolating the line density and applying the analytic equation
    to the result.

    Parameters
    ----------
    kappa1: NumpyArray
        For ``InducedVoltageResonator``:  np.zeros(int(profile.n_slices - 1), dtype=bm.precision.real_t, order='C')
    n_macroparticles: NumpyArray
        ``Profile`` options
    bin_centers: NumpyArray
        ``Profile`` options
    bin_size: float
        ``Profile`` options
    n_time: int
        length of ``tArray``
    deltaT: NumpyArray
        For ``InducedVoltageResonator``: np.zeros((n_time, profile.n_slices), dtype=bm.precision.real_t, order='C')
    tArray: NumpyArray
        Array of time values where the induced voltage is calculated.
        If left out, the induced voltage is calculated at the times of the
        line density
    reOmegaP: NumpyArray
        For``InducedVoltageResonator``:  omega_r * Qtilde / Q
    imOmegaP: NumpyArray
        For``InducedVoltageResonator``: omega_r / (2. * Q)
    Qtilde: NumpyArray
        For ``InducedVoltageResonator``:  Q * np.sqrt(1. - 1. / (4. * Q**2.))
    n_resonators: int
        Number of resonators
    omega_r: NumpyArray
        The resonant frequencies of the Resonators [1/s]
    Q: NumpyArray
        Resonators parameters: Quality factors of the resonators
    tmp_matrix: NumpyArray
        For ``InducedVoltageResonator``: np.ones((n_resonators, n_time), dtype=bm.precision.real_t, order='C')
    charge: float
        ``Beam`` parameter
    beam_n_macroparticles: int
        ``Beam`` parameter
    ratio: float
        ``Beam`` parameter
    R: NumpyArray
        Resonators parameters: Shunt impedances of the Resonators [:math:`\Omega`]
    induced_voltage: NumpyArray
        Computed induced voltage [V]
    float_precision: type
        Digital precision of calculation
    """
    from blond.utils import (
        bmath as bm,
    )  # local import to prevent cyclic import

    # Compute the slopes of the line sections of the linearly interpolated
    # (normalized) line density.
    kappa1[:] = (
        bm.diff(n_macroparticles)
        / bm.diff(bin_centers)
        / (beam_n_macroparticles * bin_size)
    )
    # [:] makes kappa pass by reference

    for t in range(n_time):
        deltaT[t] = tArray[t] - bin_centers

    # For each cavity compute the induced voltage and store in the r-th row
    for r in range(n_resonators):
        tmp_sum = (
            (
                (
                    (
                        2 * bm.cos(reOmegaP[r] * deltaT)
                        + bm.sin(reOmegaP[r] * deltaT) / Qtilde[r]
                    )
                    * bm.exp(-imOmegaP[r] * deltaT)
                )
                * 0.5
                * (bm.sign(deltaT) + 1)
            )  # Heaviside
            - bm.sign(deltaT)
        )
        # bm.sum performs the sum over the points of the line density
        tmp_matrix[r] = (
            R[r]
            / (2 * omega_r[r] * Q[r])
            * bm.sum(kappa1 * bm.diff(tmp_sum), axis=1)
        )

    # To obtain the voltage, sum the contribution of each cavity...
    induced_voltage = tmp_matrix.sum(axis=0)
    # ... and multiply with bunch charge
    induced_voltage *= -charge * e * beam_n_macroparticles * ratio
    induced_voltage = induced_voltage.astype(
        dtype=float_precision, order="C", copy=False
    )

    return induced_voltage, deltaT
