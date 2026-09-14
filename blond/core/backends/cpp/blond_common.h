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
