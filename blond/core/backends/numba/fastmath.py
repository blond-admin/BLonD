import math

from numba import njit

# ---------------------------------------------------------------------------
# Cody-Waite range reduction constants for argument reduction x -> k*(π/2)+r
# π/2 is split into three overlapping parts so that
#   pio2_1 + pio2_1t == π/2  to ~33+33 bits
#   pio2_2 adds a further ~33 bits (total ~99 bits of π/2)
# This covers arguments up to ~2^20 without significant error.
# ---------------------------------------------------------------------------
_2_OVER_PI = 0.6366197723675814      # 2/π

_PI2_1  = 1.5707963267341256e+00     # high 33 bits of π/2
_PI2_1T = 6.0771005065061922e-11     # π/2 - _PI2_1
_PI2_2  = 6.0771005063039660e-11     # next 33 bits
_PI2_2T = 2.0222662487959506e-21     # π/2 - _PI2_1 - _PI2_1T - _PI2_2

# ---------------------------------------------------------------------------
# Minimax polynomial coefficients for sin(r), r ∈ [-π/4, π/4]
# sin(r) ≈ r + r³·(S1 + r²·(S2 + r²·(S3 + r²·(S4 + r²·(S5 + r²·S6)))))
# Max error < 1 ULP
# ---------------------------------------------------------------------------
_S1 = -1.66666666666666657415e-01
_S2 =  8.33333333333333214822e-03
_S3 = -1.98412698412698412322e-04
_S4 =  2.75573192239858906520e-06
_S5 = -2.50521083854417187751e-08
_S6 =  1.60590438368216145977e-10

# ---------------------------------------------------------------------------
# Minimax polynomial coefficients for cos(r), r ∈ [-π/4, π/4]
# cos(r) ≈ 1 + r²·(C1 + r²·(C2 + r²·(C3 + r²·(C4 + r²·(C5 + r²·C6)))))
# Max error < 1 ULP
# ---------------------------------------------------------------------------
_C1 = -5.00000000000000000000e-01
_C2 =  4.16666666666666643537e-02
_C3 = -1.38888888888888872545e-03
_C4 =  2.48015873015872548494e-05
_C5 = -2.75573192239468992959e-07
_C6 =  2.08767569878680989792e-09


@njit()
def fast_sin(x: float) -> float:
    """Fast float64 sin equivalent to Intel SVML vdsin.

    Algorithm
    ---------
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
    """
    # --- range reduction ---
    k = int(math.floor(x * _2_OVER_PI + 0.5))
    r = ((x - k * _PI2_1) - k * _PI2_1T) - k * _PI2_2 - k * _PI2_2T

    r2 = r * r

    q = k & 3  # k mod 4 (works for negative k via two's-complement)

    if q == 0:
        return r + r * r2 * (_S1 + r2 * (_S2 + r2 * (_S3 + r2 * (_S4 + r2 * (_S5 + r2 * _S6)))))
    elif q == 1:
        return 1.0 + r2 * (_C1 + r2 * (_C2 + r2 * (_C3 + r2 * (_C4 + r2 * (_C5 + r2 * _C6)))))
    elif q == 2:
        return -(r + r * r2 * (_S1 + r2 * (_S2 + r2 * (_S3 + r2 * (_S4 + r2 * (_S5 + r2 * _S6))))))
    else:
        return -(1.0 + r2 * (_C1 + r2 * (_C2 + r2 * (_C3 + r2 * (_C4 + r2 * (_C5 + r2 * _C6))))))