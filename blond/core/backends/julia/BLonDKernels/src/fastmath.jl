# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# Fast double-precision sine for the vectorised CPU particle loops.
#
# A port of `fast_sin` from the vdt library (D. Piparo, T. Hauth,
# V. Innocente; LGPL), which the C++ backend uses through
# `blond/core/backends/cpp/sincos.h`: same constants, same range reduction,
# same polynomials. Unlike `Base.sin` it has no data-dependent branches, so
# a loop calling it vectorises.

# Three-part split of pi/4 for the extended-precision range reduction.
const PI_OVER_4_PART_1 = 7.853981554508209228515625E-1
const PI_OVER_4_PART_2 = 7.94662735614792836714E-9
const PI_OVER_4_PART_3 = 3.06161699786838294307E-17

const SINE_COEFFICIENT_1 = 1.58962301576546568060E-10
const SINE_COEFFICIENT_2 = -2.50507477628578072866E-8
const SINE_COEFFICIENT_3 = 2.75573136213857245213E-6
const SINE_COEFFICIENT_4 = -1.98412698295895385996E-4
const SINE_COEFFICIENT_5 = 8.33333333332211858878E-3
const SINE_COEFFICIENT_6 = -1.66666666666666307295E-1

const COSINE_COEFFICIENT_1 = -1.13585365213876817300E-11
const COSINE_COEFFICIENT_2 = 2.08757008419747316778E-9
const COSINE_COEFFICIENT_3 = -2.75573141792967388112E-7
const COSINE_COEFFICIENT_4 = 2.48015872888517045348E-5
const COSINE_COEFFICIENT_5 = -1.38888888888730564116E-3
const COSINE_COEFFICIENT_6 = 4.16666666666665929218E-2

"""
    fast_sin(x::Float64) -> Float64

Sine of `x` (in radians), within one `eps()` of `Base.sin` for the RF
phases BLonD tracks, written branch-free so that loops calling it
vectorise.

Like its vdt original, the range reduction truncates `4|x|/pi` to an
integer, so it is meant for phases, not for arbitrarily large arguments.
"""
@inline function fast_sin(x::Float64)::Float64
    absolute_x = abs(x)
    # Nearest even multiple of pi/4 at or below |x| (rounded up to even),
    # so that the remainder lies within [-pi/4, pi/4].
    octant = unsafe_trunc(Int64, (4 / pi) * absolute_x)
    octant = (octant + 1) & ~1
    octant_float = Float64(octant)
    remainder =
        ((absolute_x - octant_float * PI_OVER_4_PART_1) -
         octant_float * PI_OVER_4_PART_2) - octant_float * PI_OVER_4_PART_3

    # `@fastmath` (contraction into fused multiply-adds) only for the
    # polynomials: in the range reduction above it would merge the three
    # parts of pi/4 and lose the extended precision (errors of 4e-14
    # instead of 1e-16 at |x| = 1e3). Kick of 1e5 particles on one thread:
    # 1.52 instead of 2.17 ns per particle.
    sine, cosine = @fastmath begin
        remainder_squared = remainder * remainder
        sine_polynomial =
            (((((SINE_COEFFICIENT_1 * remainder_squared +
                 SINE_COEFFICIENT_2) * remainder_squared +
                SINE_COEFFICIENT_3) * remainder_squared +
               SINE_COEFFICIENT_4) * remainder_squared +
              SINE_COEFFICIENT_5) * remainder_squared + SINE_COEFFICIENT_6)
        cosine_polynomial =
            (((((COSINE_COEFFICIENT_1 * remainder_squared +
                 COSINE_COEFFICIENT_2) * remainder_squared +
                COSINE_COEFFICIENT_3) * remainder_squared +
               COSINE_COEFFICIENT_4) * remainder_squared +
              COSINE_COEFFICIENT_5) * remainder_squared +
             COSINE_COEFFICIENT_6)
        (
            remainder + remainder * remainder_squared * sine_polynomial,
            1.0 - remainder_squared * 0.5 +
            remainder_squared * remainder_squared * cosine_polynomial,
        )
    end

    # Odd octant pairs swap to the cosine; octants 4..7 flip the sign.
    uses_cosine = ((octant - 2) & 2) == 0
    value = ifelse(uses_cosine, cosine, sine)
    value = ifelse((octant & 4) != 0, -value, value)
    return ifelse(x < 0.0, -value, value)
end

"""
    fast_sin_cos_nonnegative(x::Float64) -> (sin(x), cos(x))

Sine and cosine of a non-negative angle `x` (in radians) from the single
range reduction and the polynomials of [`fast_sin`], branch-free. Meant
for angles of at most a few thousand radians, like `fast_sin`.
"""
@inline function fast_sin_cos_nonnegative(x::Float64)::NTuple{2, Float64}
    # Nearest even multiple of pi/4, so that the remainder lies within
    # [-pi/4, pi/4].
    octant = unsafe_trunc(Int64, (4 / pi) * x)
    octant = (octant + 1) & ~1
    octant_float = Float64(octant)
    remainder =
        ((x - octant_float * PI_OVER_4_PART_1) -
         octant_float * PI_OVER_4_PART_2) - octant_float * PI_OVER_4_PART_3

    remainder_squared = remainder * remainder
    sine_polynomial =
        (((((SINE_COEFFICIENT_1 * remainder_squared + SINE_COEFFICIENT_2) *
            remainder_squared + SINE_COEFFICIENT_3) * remainder_squared +
           SINE_COEFFICIENT_4) * remainder_squared + SINE_COEFFICIENT_5) *
         remainder_squared + SINE_COEFFICIENT_6)
    cosine_polynomial =
        (((((COSINE_COEFFICIENT_1 * remainder_squared +
             COSINE_COEFFICIENT_2) * remainder_squared +
            COSINE_COEFFICIENT_3) * remainder_squared +
           COSINE_COEFFICIENT_4) * remainder_squared +
          COSINE_COEFFICIENT_5) * remainder_squared + COSINE_COEFFICIENT_6)
    remainder_sine =
        remainder + remainder * remainder_squared * sine_polynomial
    remainder_cosine =
        1.0 - remainder_squared * 0.5 +
        remainder_squared * remainder_squared * cosine_polynomial

    # x = remainder + octant * pi/4 with octant mod 8 in {0, 2, 4, 6}:
    # octants 2 and 6 swap sine and cosine, the sine is negative in
    # octants 4 and 6, the cosine in octants 2 and 4.
    swaps = (octant & 2) != 0
    sine = ifelse(swaps, remainder_cosine, remainder_sine)
    cosine = ifelse(swaps, remainder_sine, remainder_cosine)
    sine = ifelse((octant & 4) != 0, -sine, sine)
    cosine = ifelse(((octant + 2) & 4) != 0, -cosine, cosine)
    return sine, cosine
end

"""
    fast_sin_cos(x::Float64) -> (sin(x), cos(x))

[`fast_sin_cos_nonnegative`] for angles of either sign.
"""
@inline function fast_sin_cos(x::Float64)::NTuple{2, Float64}
    sine, cosine = fast_sin_cos_nonnegative(abs(x))
    return ifelse(x < 0.0, -sine, sine), cosine
end

# Fast exponential for the vectorised CPU beam phase, written in the same
# spirit as `fast_sin`: no data-dependent branch and no library call, so
# that loops calling it vectorise. The reduction uses the fdlibm split of
# ln 2; the remainder |r| <= ln(2) / 2 is expanded to degree 14, whose
# truncation error (< 5e-18) is below one ulp.

const LN2_PART_1 = 6.93147180369123816490e-01
const LN2_PART_2 = 1.90821492927058770002e-10
const INVERSE_LN2 = 1.44269504088896338700e+00

"""
    fast_exp(x::Float64) -> Float64

Exponential of `x`, within two ulps of `Base.exp` for ``|x| <= 700``,
written branch-free so that loops calling it vectorise.

Like `fast_sin`, it is meant for the arguments BLonD evaluates (the
beam-phase weights), not for arbitrary ones: beyond ``|x| = 700`` the power
of two it scales with is no longer representable, and the result is
meaningless.
"""
@inline function fast_exp(x::Float64)::Float64
    # Nearest integer multiple of ln 2.
    multiple = unsafe_trunc(Int64, x * INVERSE_LN2 + copysign(0.5, x))
    multiple_float = Float64(multiple)
    remainder =
        (x - multiple_float * LN2_PART_1) - multiple_float * LN2_PART_2
    polynomial = @fastmath begin
        1.0 + remainder * (1.0 + remainder * (1.0 / 2 + remainder * (
            1.0 / 6 + remainder * (1.0 / 24 + remainder * (1.0 / 120 +
            remainder * (1.0 / 720 + remainder * (1.0 / 5040 + remainder * (
                1.0 / 40320 + remainder * (1.0 / 362880 + remainder * (
                    1.0 / 3628800 + remainder * (1.0 / 39916800 +
                    remainder * (1.0 / 479001600 +
                    remainder * (1.0 / 6227020800)))))))))))))
    end
    # 2^multiple from its bits; the unchecked conversion keeps the loop
    # vectorisable.
    scale = reinterpret(Float64, ((1023 + multiple) % UInt64) << 52)
    return polynomial * scale
end
