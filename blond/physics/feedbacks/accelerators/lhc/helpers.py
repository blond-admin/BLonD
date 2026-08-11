# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Helper functions for LHC feedback models.

Notes
-----
Authors:
Birk Emil Karlsen-Bæck
Helga Timko
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


def _resolve_span(onset, zero_gain_current=None, maximum_current=None):
    """
    Resolve the span of the output of the klystron model.

    Parameters
    ----------
    onset
        The current value [A] beyond which the gains starts to decrease.
    zero_gain_current
        The input current [A] where the gain is exactly zero.
    maximum_current
        The maximum generator current [A] available from the klystron.

    Returns
    -------
    span
        The span of the klystron output.
    """
    if (zero_gain_current is None) == (maximum_current is None):
        raise ValueError(
            "Specify exactly one of zero_gain_point or max_output"
        )

    if zero_gain_current is not None:
        if zero_gain_current <= onset:
            raise ValueError("zero_gain_point must be greater than onset")
        span = zero_gain_current - onset
    else:
        if maximum_current <= onset:
            raise ValueError("max_output must be greater than onset")
        span = 1.5 * (
            maximum_current - onset
        )  # peak output = onset + (2/3)*span

    return span


def klystron_saturation_curve(
    predrive: NumpyArray | complex | float,
    onset: float = 0.80,
    maximum_current: float | None = 1.0,
    zero_gain_current: float | None = None,
):
    """
    Model the saturation curve of a klystron.

    Parameters
    ----------
    predrive
        Input signal to the klystron [A].
    onset
        The current value [A] beyond which the gains starts to decrease.
    maximum_current
        The maximum generator current [A] available from the klystron.
    zero_gain_current
        The input current [A] where the gain is exactly zero.

    Returns
    -------
    output_current
        The output generator current [A].
    """
    predrive = np.asarray(predrive, dtype=float)

    span = _resolve_span(onset, zero_gain_current, maximum_current)
    k = 1.0 / span**2
    s = np.sign(predrive)
    ax = np.abs(predrive)
    overshoot = np.maximum(ax - onset, 0.0)

    inner = np.minimum(ax, onset)  # linear part, slope 1
    outer = overshoot - k * overshoot**3 / 3.0  # rolled-off part

    return s * (inner + outer)


def ideal_switch_and_limit(signal: NumpyArray | float, limit: float = 1.0):
    """
    Limit a signal to a certain value.

    Parameters
    ----------
    signal
        Input signal to limit.
    limit
        Limit value of the signal.

    Returns
    -------
    output_signal
        Limited output signal.
    """
    return np.clip(signal, -limit, limit)


def cavity_response_sparse_matrix(
    i_beam: NumpyArray,
    i_gen: NumpyArray,
    n_samples: int,
    v_ant_init: float,
    i_gen_init: float,
    samples_per_rf: float,  # TODO: is this float or int
    r_over_q: float,
    q_l: float,
    detuning: float,
):
    """
    Solving the ACS cavity response model as a sparse matrix problem.

    The calculation is done for a given set of initial conditions, resonator parameters and
    generator and RF beam currents.

    Parameters
    ----------
    i_beam
        RF beam current.
    i_gen
        Generator current.
    n_samples
        Number of samples of the result array - 1.
    v_ant_init
        Initial condition for the antenna voltage.
    i_gen_init
        Initial condition of the generator current, i.e.
        one sample before the I_gen array.
    samples_per_rf
        Number of samples per RF period.
    r_over_q
        The R over Q of the cavity.
    q_l
        The loaded quality factor of the cavity.
    detuning
        The detuning of the cavity in frequency divided by the rf frequency.

    Returns
    -------
    complex array
        The antenna voltage evaluated for the same period as I_beam and I_gen of length n_samples + 1.
    """
    # TODO MOVE

    # Add a zero at the start of RF beam current
    if len(i_beam) != n_samples + 1:
        i_beam = np.concatenate((np.zeros(1, dtype=complex), i_beam))

    # Check length of the generator current array
    if len(i_gen) != n_samples + 1:
        i_gen = np.concatenate((i_gen_init * np.ones(1, dtype=complex), i_gen))

    # Compute matrix elements
    A = 0.5 * r_over_q * samples_per_rf
    B = 1 - 0.5 * samples_per_rf / q_l + 1j * detuning * samples_per_rf

    # Initialize the two sparse matrices needed to find antenna voltage
    B_matrix = diags(
        [-B, 1],
        [-1, 0],
        (n_samples + 1, n_samples + 1),
        dtype=complex,
        format="csc",
    )
    I_matrix = diags([A], [-1], (n_samples + 1, n_samples + 1), dtype=complex)

    # Find vector on the "current" side of the equation
    b = I_matrix.dot(2 * i_gen - i_beam)
    b[0] = v_ant_init

    # Solve the sparse linear system of equations and return
    return spsolve(B_matrix, b)[-n_samples:]


def fir_filter_lhc_otfb_coeff(
    n_taps: int = 63,
) -> list[float]:  # pragma: no cover
    """
    FIR filter designed for the LHC OTFB, for a sampling frequency of 40 MS/s, with 63 taps.

    Parameters
    ----------
    n_taps
        Number of taps. 63 for 40 MS/s or 15 for 10 MS/s.

    Returns
    -------
    double array
        Coefficients of LHC-type FIR filter.
    """
    n_taps_otfb_short = 15
    n_taps_otfb_long = 63

    if n_taps == n_taps_otfb_short:
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
    elif n_taps == n_taps_otfb_long:
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
