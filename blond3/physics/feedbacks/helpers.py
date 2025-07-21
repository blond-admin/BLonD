from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import scipy
from scipy.constants import e
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.special import comb

logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional
    from typing import Optional

    from numpy.typing import NDArray as NumpyArray

    from numpy.typing import NDArray as NumpyArray

    from ...physics.profiles import StaticProfile

logger = logging.getLogger(__name__)

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
    # TODO MOVE
    # TODO TESTCASEW
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result


def cavity_response_sparse_matrix(
        I_beam,
        I_gen,
        n_samples,
        V_ant_init,
        I_gen_init,
        samples_per_rf,
        R_over_Q,
        Q_L,
        detuning,
):
    """Solving the ACS cavity response model as a sparse matrix problem
    for a given set of initial conditions, resonator parameters and
    generator and RF beam currents.

    Parameters
    ----------
    I_beam : complex array
        RF beam current
    I_gen : complex array
        Generator current
    n_samples : int
        Number of samples of the result array - 1
    V_ant_init : complex float
        Initial condition for the antenna voltage
    I_gen_init : complex float
        Initial condition of the generator current, i.e. one sample before the I_gen array
    samples_per_rf : int
        Number of samples per RF period
    R_over_Q : float
        The R over Q of the cavity
    Q_L : float
        The loaded quality factor of the cavity
    detuning : float
        The detuning of the cavity in frequency divided by the rf frequency

    Returns
    -------
    complex array
        The antenna voltage evaluated for the same period as I_beam and I_gen of length n_samples + 1

    """

    # TODO MOVE
    # TODO TESTCASE

    # Add a zero at the start of RF beam current
    if len(I_beam) != n_samples + 1:
        I_beam = np.concatenate((np.zeros(1, dtype=complex), I_beam))

    # Check length of the generator current array
    if len(I_gen) != n_samples + 1:
        I_gen = np.concatenate((I_gen_init * np.ones(1, dtype=complex), I_gen))

    # Compute matrix elements
    A = 0.5 * R_over_Q * samples_per_rf
    B = 1 - 0.5 * samples_per_rf / Q_L + 1j * detuning * samples_per_rf

    # Initialize the two sparse matrices needed to find antenna voltage
    B_matrix = diags(
        [-B, 1], [-1, 0], (n_samples + 1, n_samples + 1), dtype=complex, format="csc"
    )
    I_matrix = diags([A], [-1], (n_samples + 1, n_samples + 1), dtype=complex)

    # Find vector on the "current" side of the equation
    b = I_matrix.dot(2 * I_gen - I_beam)
    b[0] = V_ant_init

    # Solve the sparse linear system of equations and return
    return spsolve(B_matrix, b)

# TODO MOVE
def fir_filter_lhc_otfb_coeff(n_taps: int = 63) -> list[float]:  # pragma: no cover
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
        raise ValueError("In LHC FIR filter, number of taps has to be 15 or 63")

    return coeff


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


def comb_filter(
    y: NumpyArray,
    x: NumpyArray,
    a: float,
) -> NumpyArray:
    """
    Feedback comb filter.

    Notes
    -----
    See https://en.wikipedia.org/wiki/Comb_filter

    Parameters
    ----------
    y
        # TODO
    x
        # TODO
    a
        scaling factor
    """

    return a * y + (1 - a) * x


def cartesian_to_polar(
    IQ_vector: NumpyArray,
) -> tuple[NumpyArray, NumpyArray]:
    """Convert data from Cartesian (I,Q) to polar coordinates.

    Parameters
    ----------
    IQ_vector : complex array
        Signal with in-phase and quadrature (I,Q) components

    Returns
    -------
    amplitude
        Amplitude of signal
    phase
        Phase of signal, in [rad]

    """
    return np.absolute(IQ_vector), np.angle(IQ_vector)


def polar_to_cartesian(
    amplitude: float | NumpyArray,
    phase: float | NumpyArray,
) -> NumpyArray | complex:
    """Convert data from polar to cartesian (I,Q) coordinates.

    Parameters
    ----------
    amplitude
        Amplitude of signal
    phase
        Phase of signal, in [rad]

    Returns
    -------
    complex array
        Signal with in-phase and quadrature (I,Q) components
    """

    return amplitude * (np.cos(phase) + 1j * np.sin(phase))


def low_pass_filter(signal: NumpyArray, cutoff_frequency: float = 0.5) -> NumpyArray:
    """
    Low-pass filter based on Butterworth 5th order digital filter

    Notes
    -----
    See `scipy`, https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html

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

    b, a = scipy.signal.butter(5, cutoff_frequency, "low", analog=False)

    return scipy.signal.filtfilt(b, a, signal)


def rf_beam_current(
    profile: StaticProfile,
    omega_c: float,
    T_rev: float,
    use_lowpass_filter: bool = True,
    downsample: Optional[dict] = None,
    external_reference: bool = True,
    dT: float = 0,
) -> NumpyArray | tuple[NumpyArray, NumpyArray]:
    r"""Calculates the beam charge at the (RF) frequency slice by slice

    Function calculating the beam charge at the (RF) frequency, slice by
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

    where :math:`t_i` are the time coordinates of the beam profile.
    After demodulation, a low-pass filter at 20 MHz is applied.

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
    use_lowpass_filter : bool
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
    if use_lowpass_filter is True:
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
            raise RuntimeError("Downsampling input erroneous in rf_beam_current")

        # Find which index in fine grid matches index in coarse grid
        ind_fine = np.round((profile.bin_centers + dT - np.pi / omega_c) / T_s)
        ind_fine = np.array(ind_fine, dtype=int)
        indices = np.where((ind_fine[1:] - ind_fine[:-1]) == 1)[0]

        # Pick total current within one coarse grid
        charges_coarse = np.zeros(n_points, dtype=complex)
        charges_coarse[ind_fine[0]] = np.sum(charges_fine[np.arange(indices[0])])
        for i in range(1, len(indices)):
            charges_coarse[i + ind_fine[0]] = np.sum(
                charges_fine[np.arange(indices[i - 1], indices[i])]
            )

        return charges_fine, charges_coarse

    else:
        return charges_fine


def modulator(
    signal: NumpyArray,
    omega_i: float,
    omega_f: float,
    T_sampling: float,
    phi_0: float = 0.0,
    dt: float = 0.0,
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
    delta_phi = (omega_i - omega_f) * (T_sampling * np.arange(len(signal)) + dt)
    # Precompute sine and cosine for speed up
    cs = np.cos(delta_phi + phi_0)
    sn = np.sin(delta_phi + phi_0)
    I_new = cs * signal.real - sn * signal.imag
    Q_new = sn * signal.real + cs * signal.imag

    return I_new + 1j * Q_new
