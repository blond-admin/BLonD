from __future__ import annotations

from typing import Optional

import numpy as np
from numpy._typing import NDArray as NumpyArray


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
