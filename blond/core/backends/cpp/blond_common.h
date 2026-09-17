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

// Per-kernel vector width. GCC tunes x86-64 to 256-bit vectors
// (-mprefer-vector-width=256 is the default for every current Intel
// -mtune), which is right for memory-bound loops but leaves ~1.4x on the
// table for compute-bound per-particle kernels. Marking an individual
// kernel with this macro raises just that kernel to 512-bit, so the wider
// vectors -- and the core downclocking they cause -- stay out of the loops
// that gain nothing from them.
//
// Measured on an i5-11500 (single core, min ns/particle, 1e5 / 1e6
// particles): kick_single_harmonic and kick_multi_harmonic ~1.44-1.49x,
// beam_phase ~1.46x. drift_exact, histogram and loss_box are not
// vectorized to 512 bits by GCC at all, so they cannot benefit.
//
// Placement matters: the attribute must follow `extern "C"`, not precede
// it. In front of `extern "C"` GCC rejects it with a mere warning and
// silently emits 256-bit code -- a no-op that looks just like success.
// tests/unittests/core/backends/cpp/test_vector_width.py guards this.
// Definable from the command line: compiling with
// `-DBLOND_PREFER_VECTOR_WIDTH_512=` turns every use back into nothing, so
// the whole feature can be switched off for a build on a CPU whose AVX-512
// downclocking makes it a loss:
//     blond-compile-cpp --flags="-DBLOND_PREFER_VECTOR_WIDTH_512="
// (the `--flags=` form is required; a separate argument starting with `-D`
// is parsed as an option and rejected). The screening script
// dev_tools/performance_blond3/vector_width_mca.py relies on this to
// compare a widened kernel against its own 256-bit code.
#ifndef BLOND_PREFER_VECTOR_WIDTH_512
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#define BLOND_PREFER_VECTOR_WIDTH_512                                          \
  __attribute__((target("prefer-vector-width=512")))
#else
#define BLOND_PREFER_VECTOR_WIDTH_512
#endif
#endif

typedef std::complex<real_t> complex_t;

// Integer type of macro-particle counts, particle loop counters and particle
// ids, so a single process can hold more than 2^31 - 1 macro-particles.
// Must match `INDEX_DTYPE` in blond/core/backends/backend.py.
typedef std::int64_t index_t;
