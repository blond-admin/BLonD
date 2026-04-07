# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Fix for machines without Intel SVML."""
# pragma: no cover

import math
import struct

import mpmath
from numba import njit
from numpy import ndarray as NumpyArray

# ---------------------------------------------------------------------------
# Cody-Waite 3-part split of π/2.
# π/2 is split so that each part has trailing zeros in its IEEE 754
# representation, making x - k*_PI2_1 exact in floating point.
# mpmath at 50 decimal digits provides π/2 well beyond double precision,
# avoiding any dependency on the limited accuracy of math.pi/2 for the
# higher-order remainders.
# ---------------------------------------------------------------------------
dps_org = mpmath.mp.dps
mpmath.mp.dps = 50
_half_pi = mpmath.pi / 2
_2_OVER_PI = float(2.0 / mpmath.pi)
mpmath.mp.dps = dps_org


def _zero_low20(x: float | NumpyArray):  # pragma: no cover
    """
    Return the double nearest to x with its low 20 mantissa bits zeroed.

    Parameters
    ----------
    x
        Input value.

    Returns
    -------
    x_zero_low20
        Output value.
    """
    bits = struct.unpack("Q", struct.pack("d", float(x)))[0]
    bits &= ~((1 << 20) - 1)
    return struct.unpack("d", struct.pack("Q", bits))[0]


_PI2_1 = _zero_low20(_half_pi)  # high 33 bits of π/2
_PI2_1T = _zero_low20(_half_pi - _PI2_1)  # next 33 bits
_PI2_2 = _zero_low20(_half_pi - _PI2_1 - _PI2_1T)  # next 33 bits
_PI2_2T = float(_half_pi - _PI2_1 - _PI2_1T - _PI2_2)  # remainder (~20 bits)

# ---------------------------------------------------------------------------
# Taylor series coefficients for sin(r), r ∈ [-π/4, π/4]
# sin(r) ≈ r + r³·(S1 + r²·(S2 + r²·(S3 + r²·(S4 + r²·(S5 + r²·S6)))))
# S_n = (-1)^n / (2n+1)!  for n = 1..6
# ---------------------------------------------------------------------------
_S1 = -1.0 / math.factorial(3)  # -1/6
_S2 = 1.0 / math.factorial(5)  #  1/120
_S3 = -1.0 / math.factorial(7)  # -1/5040
_S4 = 1.0 / math.factorial(9)  #  1/362880
_S5 = -1.0 / math.factorial(11)  # -1/39916800
_S6 = 1.0 / math.factorial(13)  #  1/6227020800

# ---------------------------------------------------------------------------
# Taylor series coefficients for cos(r), r ∈ [-π/4, π/4]
# cos(r) ≈ 1 + r²·(C1 + r²·(C2 + r²·(C3 + r²·(C4 + r²·(C5 + r²·C6)))))
# C_n = (-1)^n / (2n)!  for n = 1..6
# ---------------------------------------------------------------------------
_C1 = -1.0 / math.factorial(2)  # -1/2
_C2 = 1.0 / math.factorial(4)  #  1/24
_C3 = -1.0 / math.factorial(6)  # -1/720
_C4 = 1.0 / math.factorial(8)  #  1/40320
_C5 = -1.0 / math.factorial(10)  # -1/3628800
_C6 = 1.0 / math.factorial(12)  #  1/479001600


@njit()
def fast_sin(x: float) -> float:  # pragma: no cover
    """
    Fast float64 sin equivalent to Intel SVML vdsin.

    Parameters
    ----------
    x
        Input value.

    Methods
    -------
    1. Cody-Waite range reduction (3-part, ~99-bit π/2):
         k  = round(x · 2/π)
         r  = x - k·(π/2)     |r| ≤ π/4

    2. Select sin or cos polynomial based on k mod 4:
         k%4 == 0  →  sin(r)
         k%4 == 1  →  cos(r)
         k%4 == 2  → -sin(r)
         k%4 == 3  → -cos(r)

    Accuracy: < 2 ULP for |x| < 2²⁰.  For larger arguments the
    3-part Cody-Waite constants lose precision; use math.sin there.

    Returns
    -------
    sin_x
        Sin(x) calculated with fast-math.

    References
    ----------
    .. [1] W. J. Cody, Jr. and W. Waite, "Software Manual for the Elementary Functions",
    SIAM Review, vol. 24, issue 1, 1982.
    .. [2] Gal, Shmuel, "An accurate elementary mathematical library
    for the IEEE floating point standard", ACM Transactions on Mathematical
    Soware (TOMS), vol. 17, issue, 1991.
    """
    # --- range reduction ---
    k = int(math.floor(x * _2_OVER_PI + 0.5))
    r = ((x - k * _PI2_1) - k * _PI2_1T) - k * _PI2_2 - k * _PI2_2T

    r2 = r * r

    q = k & 3  # k mod 4 (works for negative k via two's-complement)

    if q == 0:  # NOQA
        return r + r * r2 * (
            _S1 + r2 * (_S2 + r2 * (_S3 + r2 * (_S4 + r2 * (_S5 + r2 * _S6))))
        )
    elif q == 1:  # NOQA
        return 1.0 + r2 * (
            _C1 + r2 * (_C2 + r2 * (_C3 + r2 * (_C4 + r2 * (_C5 + r2 * _C6))))
        )
    elif q == 2:  # NOQA
        return -(
            r
            + r
            * r2
            * (
                _S1
                + r2 * (_S2 + r2 * (_S3 + r2 * (_S4 + r2 * (_S5 + r2 * _S6))))
            )
        )
    else:
        return -(
            1.0
            + r2
            * (
                _C1
                + r2 * (_C2 + r2 * (_C3 + r2 * (_C4 + r2 * (_C5 + r2 * _C6))))
            )
        )
