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
    sine = remainder + remainder * remainder_squared * sine_polynomial
    cosine =
        1.0 - remainder_squared * 0.5 +
        remainder_squared * remainder_squared * cosine_polynomial

    # Odd octant pairs swap to the cosine; octants 4..7 flip the sign.
    uses_cosine = ((octant - 2) & 2) == 0
    value = ifelse(uses_cosine, cosine, sine)
    value = ifelse((octant & 4) != 0, -value, value)
    return ifelse(x < 0.0, -value, value)
end
