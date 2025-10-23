# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Filters and methods for control loops**

:Authors: **Birk Emil Karlsen-Bæck**, **Helga Timko**
"""

from __future__ import annotations

# Set up logging
import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npla
from scipy import signal as sgn
from scipy.constants import e
from scipy.special import comb

from ..utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import NDArray as NumpyArray

    from .impulse_response import TravellingWaveCavity
    from ..beam.profile import Profile

logger = logging.getLogger(__name__)


def polar_to_cartesian(
    amplitude: float | NumpyArray, phase: float | NumpyArray
) -> NumpyArray | complex:
    """Convert data from polar to cartesian (I,Q) coordinates.

    Parameters
    ----------
    amplitude : float array
        Amplitude of signal
    phase : float array
        Phase of signal

    Returns
    -------
    complex array
        Signal with in-phase and quadrature (I,Q) components
    """

    logger.debug("Converting from polar to Cartesian")

    return amplitude * (np.cos(phase) + 1j * np.sin(phase))


def cartesian_to_polar(IQ_vector: NumpyArray) -> tuple[NumpyArray, NumpyArray]:
    """Convert data from Cartesian (I,Q) to polar coordinates.

    Parameters
    ----------
    IQ_vector : complex array
        Signal with in-phase and quadrature (I,Q) components

    Returns
    -------
    float array
        Amplitude of signal
    float array
        Phase of signal

    """

    logger.debug("Converting from Cartesian to polar")

    return np.absolute(IQ_vector), np.angle(IQ_vector)


def get_power_gen_i(I_gen_per_cav: NumpyArray, Z_0: float) -> float:
    """RF generator power from generator current (physical, in [A]), for any
    f_r (and thus any tau)

    Parameters
    ----------
    I_gen_per_cav : complex array
        Generator current for a single cavity
    Z_0 : float

    Returns
    -------
    float array
        Absolute value of the generator power

    """
    return 0.5 * Z_0 * np.abs(I_gen_per_cav) ** 2


def modulator(
    signal: NumpyArray,
    omega_i: float,
    omega_f: float,
    T_sampling: float,
    phi_0: float = 0,
    dt: float = 0,
) -> NumpyArray:
    """Demodulate a signal from initial frequency to final frequency. The two
    frequencies should be close.

    Parameters
    ----------
    signal : float array
        Signal to be demodulated
    omega_i : float
        Initial revolution frequency [1/s] of signal (before demodulation)
    omega_f : float
        Final revolution frequency [1/s] of signal (after demodulation)
    T_sampling : float
        Sampling period (temporal bin size) [s] of the signal
    phi_0 : float
        # todo
    dt: float
        # todo
    Returns
    -------
    float array
        Demodulated signal at f_final

    """

    if len(signal) < 2:
        # TypeError
        raise RuntimeError(
            "ERROR in filters.py/demodulator: signal should" + " be an array!"
        )
    delta_phi = (omega_i - omega_f) * (
        T_sampling * np.arange(len(signal)) + dt
    )
    # Precompute sine and cosine for speed up
    cs = np.cos(delta_phi + phi_0)
    sn = np.sin(delta_phi + phi_0)
    I_new = cs * signal.real - sn * signal.imag
    Q_new = sn * signal.real + cs * signal.imag

    return I_new + 1j * Q_new


@handle_legacy_kwargs
def rf_beam_current(
    profile: Profile,
    omega_c: float,
    T_rev: float,
    lpf: bool = True,
    downsample: Optional[dict] = None,
    external_reference: bool = True,
    dT: float = 0,
) -> NumpyArray | tuple[NumpyArray, NumpyArray]:
    r"""Function calculating the beam charge at the (RF) frequency, slice by
    slice. The charge distribution [C] of the beam is determined from the beam
    profile :math:`\lambda_i`, the particle charge :math:`q_p` and the real vs.
    macro-particle ratio :math:`N_{\mathsf{real}}/N_{\mathsf{macro}}`

    .. math::
        Q_i = \frac{N_{\mathsf{real}}}{N_{\mathsf{macro}}} q_p \lambda_i

    The total charge [C] in the beam is then

    .. math::
        Q_{\mathsf{tot}} = \sum_i{Q_i}

    The DC beam current [A] is the total number of charges per turn :math:`T_0`

    .. math:: I_{\mathsf{DC}} = \frac{Q_{\mathsf{tot}}}{T_0}

    The RF beam charge distribution [C] at a revolution frequency
    :math:`\omega_c` is the complex quantity

    .. math::
        \left( \begin{matrix} I_{rf,i} \\
        Q_{rf,i} \end{matrix} \right)
        = 2 Q_i \left( \begin{matrix} \cos(\omega_c t_i) \\
        \sin(\omega_c t_i)\end{matrix} \right) \, ,

    where :math:`t_i` are the time coordinates of the beam profile. After de-
    modulation, a low-pass filter at 20 MHz is applied.

    For multi-bunch cases, make sure that the real beam intensity is the total
    number of charges in the ring.

    Parameters
    ----------
    profile : class
        A Profile type class
    omega_c : float
        Revolution frequency [1/s] at which the current should be calculated
    T_rev : float
        Revolution period [s] of the machine
    lpf : bool
        Apply low-pass filter; default is True
    downsample : dict
        Dictionary containing float value for 'Ts' sampling time and int value
        for 'points'. Will downsample the RF beam charge onto a coarse time
        grid with 'Ts' sampling time and 'points' points.
    external_reference : bool
        Option to include the changing external reference of the time-grid
    dT : float
        The shift in time due to shifting reference frames

    Returns
    -------
    complex array
        RF beam charge array [C] at 'frequency' omega_c, with the sampling time
        of the Profile object. To obtain current, divide by the sampling time
    (complex array)
        If time_coarse is specified, returns also the RF beam charge array [C]
        on the coarse time grid

    """

    # Convert from dimensionless to Coulomb/Ampères
    # Take into account macro-particle charge with real-to-macro-particle ratio
    charges = (
        profile.beam.ratio
        * profile.beam.particle.charge
        * e
        * np.copy(profile.n_macroparticles)
    )
    logger.debug(
        "Sum of particles: %d, total charge: %.4e C",
        np.sum(profile.n_macroparticles),
        np.sum(charges),
    )
    logger.debug("DC current is %.4e A", np.sum(charges) / T_rev)

    # Mix with frequency of interest; remember factor 2 demodulation
    I_f = 2.0 * charges * np.cos(omega_c * profile.bin_centers)
    Q_f = -2.0 * charges * np.sin(omega_c * profile.bin_centers)

    # Pass through a low-pass filter
    if lpf is True:
        # Nyquist frequency 0.5*f_slices; cutoff at 20 MHz
        cutoff = 20.0e6 * 2.0 * profile.bin_size
        I_f = low_pass_filter(I_f, cutoff_frequency=cutoff)
        Q_f = low_pass_filter(Q_f, cutoff_frequency=cutoff)
    logger.debug("RF total current is %.4e A", np.fabs(np.sum(I_f)) / T_rev)

    charges_fine = I_f + 1j * Q_f
    if external_reference:
        # slippage in phase due to a non-integer harmonic number
        dphi = dT * omega_c
        # Total phase correction
        phase = dphi
        charges_fine = charges_fine * np.exp(1j * phase)

    if downsample:
        try:
            T_s = float(downsample["Ts"])
            n_points = int(downsample["points"])
        except Exception:
            raise RuntimeError(
                "Downsampling input erroneous in rf_beam_current"
            )

        # Find which index in fine grid matches index in coarse grid
        ind_fine = np.round((profile.bin_centers + dT - np.pi / omega_c) / T_s)
        ind_fine = np.array(ind_fine, dtype=int)
        indices = np.where((ind_fine[1:] - ind_fine[:-1]) == 1)[0]

        # Pick total current within one coarse grid
        charges_coarse = np.zeros(n_points, dtype=complex)
        charges_coarse[ind_fine[0]] = np.sum(
            charges_fine[np.arange(indices[0])]
        )
        for i in range(1, len(indices)):
            charges_coarse[i + ind_fine[0]] = np.sum(
                charges_fine[np.arange(indices[i - 1], indices[i])]
            )

        return charges_fine, charges_coarse

    else:
        return charges_fine


def comb_filter(y: NumpyArray, x: NumpyArray, a: float) -> NumpyArray:
    """Feedback comb filter."""

    return a * y + (1 - a) * x


def fir_filter_coefficients(
    n_taps: int, sampling_freq: float, cutoff_freq: float
) -> NumpyArray:
    """Band-stop type FIR filter from scipy
    http://docs.scipy.org

    Parameters
    ----------
    n_taps : int
        Number of taps, should be impair
    sampling_freq : float
        Sampling frequency [Hz]
    cutoff_freq : float
        Cutoff frequency [Hz]


    Returns
    -------
    ndarray
        FIR filter coefficients of length n_taps

    """
    fPass = cutoff_freq / sampling_freq

    return sgn.firwin(n_taps, [fPass], pass_zero=True)


def fir_filter_lhc_otfb_coeff(n_taps: int = 63) -> list[float]:
    """FIR filter designed for the LHC OTFB, for a sampling frequency of
    40 MS/s, with 63 taps.

    Parameters
    ----------
    n_taps : int
        Number of taps. 63 for 40 MS/s or 15 for 10 MS/s

    Returns
    -------
    double array
        Coefficients of LHC-type FIR filter
    """
    # todo might return arrays?
    if n_taps == 15:
        coeff = [
            -0.0469,
            -0.016,
            0.001,
            0.0321,
            0.0724,
            0.1127,
            0.1425,
            0.1534,
            0.1425,
            0.1127,
            0.0724,
            0.0321,
            0.001,
            -0.016,
            -0.0469,
        ]
    elif n_taps == 63:
        coeff = [
            -0.038636,
            -0.00687283,
            -0.00719296,
            -0.00733319,
            -0.00726159,
            -0.00694037,
            -0.00634775,
            -0.00548098,
            -0.00432789,
            -0.00288188,
            -0.0011339,
            0.00090253,
            0.00321323,
            0.00577238,
            0.00856464,
            0.0115605,
            0.0147307,
            0.0180265,
            0.0214057,
            0.0248156,
            0.0282116,
            0.0315334,
            0.0347311,
            0.0377502,
            0.0405575,
            0.0431076,
            0.0453585,
            0.047243,
            0.0487253,
            0.049782,
            0.0504816,
            0.0507121,
            0.0504816,
            0.049782,
            0.0487253,
            0.047243,
            0.0453585,
            0.0431076,
            0.0405575,
            0.0377502,
            0.0347311,
            0.0315334,
            0.0282116,
            0.0248156,
            0.0214057,
            0.0180265,
            0.0147307,
            0.0115605,
            0.00856464,
            0.00577238,
            0.00321323,
            0.00090253,
            -0.0011339,
            -0.00288188,
            -0.00432789,
            -0.00548098,
            -0.00634775,
            -0.00694037,
            -0.00726159,
            -0.00733319,
            -0.00719296,
            -0.00687283,
            -0.038636,
        ]
    else:
        raise ValueError(
            "In LHC FIR filter, number of taps has to be 15 or 63"
        )

    return coeff


def fir_filter(coeff: NumpyArray, signal: NumpyArray):
    """Apply FIR filter on discrete time signal.

    Parameters
    ---------
    coeff : double array
        Coefficients of FIR filter with length of number of taps
    signal : complex or double array
        Input signal to be filtered

    Returns
    -------
    complex or double array
        Filtered signal of length len(signal) - len(coeff)
    """

    n_taps = len(coeff)
    filtered_signal = np.zeros(len(signal) - n_taps)
    for i in range(n_taps, len(signal)):
        for k in range(n_taps):
            filtered_signal[i - n_taps] += coeff[k] * signal[i - k]

    return filtered_signal


def low_pass_filter(
    signal: NumpyArray, cutoff_frequency: float = 0.5
) -> NumpyArray:
    """Low-pass filter based on Butterworth 5th order digital filter from
    scipy,
    http://docs.scipy.org

    Parameters
    ----------
    signal : float array
        Signal to be filtered
    cutoff_frequency : float
        Cutoff frequency [1] corresponding to a 3 dB gain drop, relative to the
        Nyquist frequency of 1; default is 0.5

    Returns
    -------
    float array
        Low-pass filtered signal

    """

    b, a = sgn.butter(5, cutoff_frequency, "low", analog=False)

    return sgn.filtfilt(b, a, signal)


def moving_average(
    x: NumpyArray, N: int, x_prev: Optional[NumpyArray] = None
) -> NumpyArray:
    """Function to calculate the moving average (or running mean) of the input
    data.

    Parameters
    ----------
    x : float array
        Data to be smoothed
    N : int
        Window size in points
    x_prev : float array
        Data to pad with in front

    Returns
    -------
    float array
        Smoothed data array of size
            * len(x) - N + 1, if x_prev = None
            * len(x) + len(x_prev) - N + 1, if x_prev given

    """

    if x_prev is not None:
        # Pad in front with x_prev signal
        x = np.concatenate((x_prev, x))

    # based on https://stackoverflow.com/a/14314054
    mov_avg = np.cumsum(x)
    mov_avg[N:] = mov_avg[N:] - mov_avg[:-N]
    return mov_avg[N - 1 :] / N


def moving_average_improved(
    x: NumpyArray, N: int, x_prev: Optional[NumpyArray] = None
):
    if x_prev is not None:
        x = np.concatenate((x_prev, x))

    mov_avg = sgn.fftconvolve(x, (1 / N) * np.ones(N), mode="full")[
        -x.shape[0] :
    ]

    return mov_avg[: x.shape[0] - N + 1]


def H_cav(x: NumpyArray, n_sections: int, x_prev: Optional[NumpyArray] = None):
    if x_prev is not None:
        x = np.concatenate((x_prev, x))

    if n_sections == 3:
        h = np.array(
            [
                -0.04120219,
                -0.00765499,
                -0.00724786,
                -0.00600952,
                -0.00380694,
                -0.00067663,
                0.00343537,
                0.0084533,
                0.01421418,
                0.02071802,
                0.02764441,
                0.03476114,
                0.04193753,
                0.04882965,
                0.05522681,
                0.06083675,
                0.0654471,
                0.06887487,
                0.07100091,
                0.09043617,
                0.07100091,
                0.06887487,
                0.0654471,
                0.06083675,
                0.05522681,
                0.04882965,
                0.04193753,
                0.03476114,
                0.02764441,
                0.02071802,
                0.01421418,
                0.0084533,
                0.00343537,
                -0.00067663,
                -0.00380694,
                -0.00600952,
                -0.00724786,
                -0.00765499,
                -0.04120219,
            ]
        )
    else:
        h = np.array(
            [
                -0.0671217,
                0.01355402,
                0.01365686,
                0.01444814,
                0.01571424,
                0.01766679,
                0.01996413,
                0.02251791,
                0.02529718,
                0.02817416,
                0.03113348,
                0.03398052,
                0.03674144,
                0.03924433,
                0.04153931,
                0.04344182,
                0.04502165,
                0.04612467,
                0.04685122,
                0.06409968,
                0.04685122,
                0.04612467,
                0.04502165,
                0.04344182,
                0.04153931,
                0.03924433,
                0.03674144,
                0.03398052,
                0.03113348,
                0.02817416,
                0.02529718,
                0.02251791,
                0.01996413,
                0.01766679,
                0.01571424,
                0.01444814,
                0.01365686,
                0.01355402,
                -0.0671217,
            ]
        )

    resp = sgn.fftconvolve(x, h, mode="full")[-x.shape[0] :]

    return resp[: x.shape[0] - h.shape[0] + 1]


def smooth_step(x: NumpyArray, x_min: float = 0, x_max: float = 1, N: int = 1):
    """Function to make a smooth step.

    Parameters
    ----------
    x : float array
        Data to be smoothed
    x_min : float
        Minimum output value of step
    x_max : float
        Maximum output value of step
    N : int
        Order of smoothness

    Returns
    -------
    float array
        Smooth step of input signal
    Taken from: https://stackoverflow.com/questions/45165452/how-to-implement-a-smooth-clamp-function-in-python
    """
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result


def feedforward_filter(
    TWC: TravellingWaveCavity,
    T_s: float,
    taps: Optional[int] = None,
    opt_output: bool = False,
) -> tuple[NumpyArray, int, int, int]:
    """Function to design n-tap FIR filter for SPS TravellingWaveCavity.

    Parameters
    ----------
    TWC : TravellingWaveCavity
        TravellingWaveCavity type class
    T_s : float
        Sampling time [s]
    taps : int
        User-defined number of taps; default is None and number of taps is
        calculated from the filling time
    opt_output : bool
        When True, activates optional output; default is False

    Returns
    -------
    float array
        FIR filter coefficients
    int
        Optional output: Number of FIR filter taps
    int
        Optional output: Filling time in samples
    int
        Optional output: Fitting time in samples, n_filling, n_fit
    """

    # Filling time in samples
    n_filling = round(TWC.tau / T_s)
    logger.debug("Filling time in samples: %d", n_filling)

    # Number of FIR filter taps
    if taps is not None:
        n_taps = int(taps)
    else:
        n_taps = 2 * int(0.5 * n_filling) + 13  # 31

    if n_taps % 2 == 0:
        raise RuntimeError("Number of taps in feedforward filter must be odd!")
    logger.debug("Number of taps: %d", n_taps)

    # Fitting samples
    n_fit = int(n_taps + n_filling)
    logger.debug("Fitting samples: %d", n_fit)

    def V_real(t, tauf):
        output = np.zeros(t.shape)
        for i in range(len(t)):
            if t[i] < -tauf:
                output[i] = 0
            elif t[i] < 0:
                output[i] = (t[i] / tauf + 1) ** 2 / 2
            elif t[i] < tauf:
                output[i] = -1 / 2 * (t[i] / tauf) ** 2 + t[i] / tauf + 1 / 2
            else:
                output[i] = 1
        return output

    # Imaginary part
    def V_imag(t, tauf):
        output = np.zeros(t.shape)
        for i in range(len(t)):
            if t[i] < -tauf:
                output[i] = 0
            elif t[i] < 0:
                output[i] = -((t[i] / tauf + 1) ** 2) / 2
            elif t[i] < tauf:
                output[i] = -1 / 2 * (t[i] / tauf) ** 2 + t[i] / tauf - 1 / 2
            else:
                output[i] = 0
        return output

    Pfit_ = np.linspace(-(n_fit - 1) / 2, (n_fit - 1) / 2, n_fit)

    # Even symmetric part of beam loading
    Dvectoreven = V_real(Pfit_, n_filling)

    # Odd symmetric part of beam loading
    Dvectorodd = V_imag(Pfit_, n_filling)

    # Step response of FIR. (M1, N1) must be odd
    def Smatrix(M1, N1):
        output = np.zeros((M1, N1))
        for i in range(M1):
            for j in range(N1):
                if i - j >= (M1 - N1) / 2:
                    output[i, j] = 1
        return output

    # Response of symmetrix rectangle of length L. P must be odd
    def Rmatrix(P, L):
        output = np.zeros((P, P + L - 1))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if i - j <= 0 and i - j > -L:
                    output[i, j] = 1
        return output

    def uniform_weighting(n, Nt):
        return 1

    def Weigthing(Nt, weighting_function):
        output = np.zeros((Nt, Nt))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if i == j:
                    output[i, j] = weighting_function(i, Nt)
        return output

    def EvenMatrix(Nt):
        output = np.zeros((Nt, (Nt + 1) // 2))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if i + j == (Nt + 0) // 2:
                    output[i, j] = 1
                elif i - j == (Nt - 1) // 2:
                    output[i, j] = 1
        return output

    def OddMatrix(Nt):
        output = np.zeros((Nt, (Nt - 1) // 2))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if i + j == (Nt - 2) // 2:
                    output[i, j] = -1
                elif i - j == (Nt + 1) // 2:
                    output[i, j] = 1
        return output

    def Hoptreal(Nt, L, P, Dvectoreven):
        output = (
            EvenMatrix(Nt).T
            @ Smatrix(P + L - 1, Nt).T
            @ Rmatrix(P, L).T
            @ Dvectoreven
        )

        matrix1 = EvenMatrix(Nt)
        matrix1 = (
            Rmatrix(P, L).T
            @ Weigthing(P, uniform_weighting)
            @ Rmatrix(P, L)
            @ Smatrix(P + L - 1, Nt)
            @ matrix1
        )
        matrix1 = EvenMatrix(Nt).T @ Smatrix(P + L - 1, Nt).T @ matrix1
        matrix1 = npla.inv(matrix1)

        return matrix1 @ output

    def Hoptimag(Nt, L, P, Dvectorodd):
        output = (
            OddMatrix(Nt).T
            @ Smatrix(P + L - 1, Nt).T
            @ Rmatrix(P, L).T
            @ Weigthing(P, uniform_weighting)
            @ Dvectorodd
        )
        matrix1 = OddMatrix(Nt)
        matrix1 = (
            Rmatrix(P, L).T
            @ Weigthing(P, uniform_weighting)
            @ Rmatrix(P, L)
            @ Smatrix(P + L - 1, Nt)
            @ matrix1
        )
        matrix1 = OddMatrix(Nt).T @ Smatrix(P + L - 1, Nt).T @ matrix1
        matrix1 = npla.inv(matrix1)

        return matrix1 @ output

    def Hopteven(Nt, L, P, Dvectoreven):
        return np.concatenate(
            [
                Hoptreal(Nt, L, P, Dvectoreven)[1:][::-1],
                Hoptreal(Nt, L, P, Dvectoreven),
            ]
        )

    def Hoptodd(Nt, L, P, Dvectorodd):
        output = np.concatenate(
            [-Hoptimag(Nt, L, P, Dvectorodd)[::-1], np.array([0])]
        )
        return np.concatenate([output, Hoptimag(Nt, L, P, Dvectorodd)])

    h_ff = Hopteven(n_taps, n_filling, n_fit, Dvectoreven) + Hoptodd(
        n_taps, n_filling, n_fit, Dvectorodd
    )

    if opt_output:
        return h_ff, n_taps, n_filling, n_fit
    else:
        return h_ff


feedforward_filter_TWC3 = np.array(
    [
        -0.00760838,
        0.01686764,
        0.00205761,
        0.00205761,
        0.00205761,
        0.00205761,
        -0.03497942,
        0.00205761,
        0.00205761,
        0.00205761,
        0.00205761,
        -0.0053474,
        0.00689061,
        0.00308642,
        0.00308642,
        0.00308642,
        0.00308642,
        0.00308642,
        -0.00071777,
        0.01152024,
        0.00411523,
        0.00411523,
        0.00411523,
        0.00411523,
        0.03806584,
        -0.00205761,
        -0.00205761,
        -0.00205761,
        -0.00205761,
        -0.01686764,
        0.00760838,
    ]
)

feedforward_filter_TWC4 = np.array(
    [
        0.01050256,
        -0.0014359,
        0.00106667,
        0.00106667,
        0.00106667,
        -0.01226667,
        -0.01226667,
        0.00106667,
        0.00106667,
        0.00106667,
        0.00231795,
        -0.00365128,
        0.0016,
        0.0016,
        0.0016,
        0.0016,
        0.0016,
        0.0016,
        0.0016,
        0.0016,
        0.0016,
        0.0016,
        0.0016,
        0.0016,
        0.0016,
        0.00685128,
        0.00088205,
        0.00213333,
        0.00213333,
        0.00213333,
        0.01506667,
        0.01266667,
        -0.00106667,
        -0.00106667,
        -0.00106667,
        0.0014359,
        -0.01050256,
    ]
)

feedforward_filter_TWC5 = np.array(
    [
        0.01802423,
        -0.01004643,
        0.00069372,
        0.00069372,
        0.00069372,
        -0.01005897,
        -0.01005897,
        0.00069372,
        0.00069372,
        0.00069372,
        0.0060638,
        -0.00797153,
        0.00104058,
        0.00104058,
        0.00104058,
        0.00104058,
        0.00104058,
        0.00104058,
        0.00104058,
        0.00104058,
        0.00104058,
        0.00104058,
        0.00104058,
        0.00104058,
        0.00104058,
        0.00104058,
        0.00104058,
        0.00104058,
        0.00104058,
        0.00104058,
        0.00104058,
        0.0100527,
        -0.00398263,
        0.00138744,
        0.00138744,
        0.00138744,
        0.01187999,
        0.01031911,
        -0.00069372,
        -0.00069372,
        -0.00069372,
        0.01004643,
        -0.01802423,
    ]
)


def plot_frequency_response(b: NumpyArray, a=1):
    """Plotting the frequency response of a filter with coefficients a, b."""

    w, H = sgn.freqz(b, a)

    plt.subplot(211)
    plt.plot(2 * w / np.max(w), np.absolute(H))
    plt.ylabel("Amplitude [linear]")
    plt.xlabel(r"Frequency w.r.t. sampling frequency")
    plt.title(r"Frequency response")

    plt.subplot(212)
    phase = np.unwrap(np.angle(H))
    plt.plot(w / max(w), phase)
    plt.ylabel("Phase [radians]")
    plt.xlabel(r"Frequency w.r.t. sampling frequency")
    plt.title(r"Phase response")
    plt.subplots_adjust(hspace=0.5)

    plt.show()
