// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

/**
BLonD common header file
@Author: Konstantinos Iliakis
@Date: 12.12.2023
*/

// Precondition for every kernel including this header: coordinates are
// finite. The beam coordinates (dt, dE) and the profile coordinates
// (bin_centers, cut edges) must contain neither NaN nor +/-Inf. Nothing
// checks for it -- the check would not be free in a per-particle loop.
// Note that the guards protecting the conversion of a bin index to
// `int` are written as `index < lo || index >= hi`: a NaN index
// compares false against both bounds, passes the guard and reaches the
// conversion, which is undefined behaviour. The caller must not produce
// non-finite coordinates. See `Specials` in blond/core/backends/backend.py.

#pragma once

#include "cos.h"
#include "exp.h"
#include "sin.h"
#include <complex>
#include <cstdint>

#ifdef USEFLOAT

typedef float real_t;
#define FAST_SIN vdt::fast_sinf
#define FAST_COS vdt::fast_cosf
#define FAST_EXP vdt::fast_expf

#else

typedef double real_t;
#define FAST_SIN vdt::fast_sin
#define FAST_COS vdt::fast_cos
#define FAST_EXP vdt::fast_exp

#endif

typedef std::complex<real_t> complex_t;

// Integer type of macro-particle counts, particle loop counters and particle
// ids, so a single process can hold more than 2^31 - 1 macro-particles.
// Must match `INDEX_DTYPE` in blond/core/backends/backend.py.
typedef std::int64_t index_t;
