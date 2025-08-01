from __future__ import annotations

import numpy as np
from numpy._typing import NDArray as NumpyArray
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.special import comb


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
