// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Optimised C++ routine that calculates the drift.
// Author: Danilo Quartullo, Helga Timko, Alexandre Lasheen, Elleanor Lamb

#include "blond_common.h"
#include "particle_kernels.h"
#include <array>
#include <cmath>

namespace {

// The number of higher-order momentum compaction factors is a compile-time
// parameter, and `drift_exact` below dispatches to the instantiation that
// matches the `n_alpha` it is called with. The point is the alpha loop: with
// a run-time trip count GCC keeps it a real loop inside the particle loop,
// which stops the particle loop from vectorizing at all -- the whole kernel
// stays scalar, and its cost grows linearly with `n_alpha`. Fixing the count
// lets GCC unroll the alpha loop into straight-line FMAs, after which the
// particle loop vectorizes.
//
// Measured on an i5-11500 (single core, pinned, 1e6 particles, cycles per
// particle via `perf`) for n_alpha = 0 / 2 / 4: 12.96 / 15.51 / 19.57
// before, 7.10 / 7.22 / 7.40 after -- 1.8x / 2.2x / 2.7x, and nearly flat in
// `n_alpha` where the scalar version grew with it. llvm-mca puts the
// vectorized loop at ~5.0 cycles per particle, so the remainder is memory
// traffic rather than compute.
//
// Instantiating 0..4 covers every real machine; a longer
// expansion falls back to the generic loop below, which is the original
// kernel and still correct, just scalar. Widening these loops to 512 bits
// (BLOND_PREFER_VECTOR_WIDTH_512) was measured at 0.98-1.02x and llvm-mca
// agrees (5.14 vs 5.03 cycles per particle), so the kernel stays at 256.
template <int N_ALPHA>
inline void drift_exact_unrolled(real_t *__restrict__ beam_dt,
                                 const real_t *__restrict__ beam_dE,
                                 const real_t T, const real_t alpha_zero,
                                 const real_t *__restrict__ higher_alpha,
                                 const real_t beta, const real_t energy,
                                 const index_t begin, const index_t end) {
  const real_t inv_beta_sq = 1.0 / (beta * beta);
  const real_t inv_energy = 1.0 / energy;
  const real_t inv_energy_sq = inv_energy * inv_energy;

  // Copied out of the caller's buffer once, so the compiler knows these do
  // not alias `beam_dt` and can keep them in registers across the loop.
  std::array<real_t, N_ALPHA> alpha{};
  for (int k = 0; k < N_ALPHA; ++k) {
    alpha[k] = higher_alpha[k];
  }

  for (index_t i = begin; i < end; i++) {

    const real_t dE = beam_dE[i];

    const real_t delta =
        std::sqrt(1.0 + inv_beta_sq *
                            (dE * dE * inv_energy_sq + 2.0 * dE * inv_energy)) -
        1.0;

    real_t poly = 1.0 + alpha_zero * delta;
    real_t delta_power = delta * delta; // starts at δ²

    for (int k = 0; k < N_ALPHA; ++k) { // unrolled: N_ALPHA is a constant
      poly += alpha[k] * delta_power;
      delta_power *= delta; // next power
    }

    beam_dt[i] += T * (poly * (1.0 + dE * inv_energy) / (1.0 + delta) - 1.0);
  }
}

// Generic fallback for more than four higher-order factors. Scalar,
// because the alpha loop keeps its run-time trip count here.
void drift_exact_generic(real_t *__restrict__ beam_dt,
                         const real_t *__restrict__ beam_dE, const real_t T,
                         const real_t alpha_zero,
                         const real_t *__restrict__ higher_alpha,
                         const int n_alpha, const real_t beta,
                         const real_t energy, const index_t begin,
                         const index_t end) {
  const real_t inv_beta_sq = 1.0 / (beta * beta);
  const real_t inv_energy = 1.0 / energy;
  const real_t inv_energy_sq = inv_energy * inv_energy;

  for (index_t i = begin; i < end; i++) {

    const real_t dE = beam_dE[i];

    const real_t delta =
        std::sqrt(1.0 + inv_beta_sq *
                            (dE * dE * inv_energy_sq + 2.0 * dE * inv_energy)) -
        1.0;

    real_t poly = 1.0 + alpha_zero * delta;
    real_t delta_power = delta * delta; // starts at δ²

    for (int k = 0; k < n_alpha; ++k) {
      poly += higher_alpha[k] * delta_power;
      delta_power *= delta; // next power
    }

    beam_dt[i] += T * (poly * (1.0 + dE * inv_energy) / (1.0 + delta) - 1.0);
  }
}

} // namespace

// Deferred and eager: dispatch on the number of higher-order factors to the
// unrolled instantiation that matches it (see above).
void apply_to_chunk(const DriftExactArgs &args, real_t *beam_dt,
                    const real_t *beam_dE, const index_t begin,
                    const index_t end) {
  const real_t *higher_alpha = &args.higher_alpha[0];
  switch (args.n_alpha) {
  case 0:
    drift_exact_unrolled<0>(beam_dt, beam_dE, args.T, args.alpha_0,
                            higher_alpha, args.beta, args.energy, begin, end);
    return;
  case 1:
    drift_exact_unrolled<1>(beam_dt, beam_dE, args.T, args.alpha_0,
                            higher_alpha, args.beta, args.energy, begin, end);
    return;
  case 2:
    drift_exact_unrolled<2>(beam_dt, beam_dE, args.T, args.alpha_0,
                            higher_alpha, args.beta, args.energy, begin, end);
    return;
  case 3:
    drift_exact_unrolled<3>(beam_dt, beam_dE, args.T, args.alpha_0,
                            higher_alpha, args.beta, args.energy, begin, end);
    return;
  case 4:
    drift_exact_unrolled<4>(beam_dt, beam_dE, args.T, args.alpha_0,
                            higher_alpha, args.beta, args.energy, begin, end);
    return;
  default:
    drift_exact_generic(beam_dt, beam_dE, args.T, args.alpha_0, higher_alpha,
                        args.n_alpha, args.beta, args.energy, begin, end);
    return;
  }
}

extern "C" void drift_exact(real_t *__restrict__ beam_dt,
                            const real_t *__restrict__ beam_dE, const real_t T,
                            const real_t alpha_zero,
                            const real_t *__restrict__ higher_alpha,
                            const int n_alpha, const real_t beta,
                            const real_t energy,
                            const index_t n_macroparticles) {
  // A null pointer carries no coefficients, whatever `n_alpha` claims.
  const int n_used = (higher_alpha == nullptr) ? 0 : n_alpha;
  const int capacity = sizeof(DriftExactArgs::higher_alpha) / sizeof(real_t);

  if (n_used > capacity) {
    // More factors than a kernel call record holds: the generic loop,
    // straight from the caller's buffer.
#pragma omp parallel
    {
      index_t begin = 0;
      index_t end = 0;
      this_thread_range(n_macroparticles, begin, end);
      drift_exact_generic(beam_dt, beam_dE, T, alpha_zero, higher_alpha, n_used,
                          beta, energy, begin, end);
    }
    return;
  }
  DriftExactArgs args{};
  args.T = T;
  args.alpha_0 = alpha_zero;
  args.beta = beta;
  args.energy = energy;
  args.n_alpha = n_used;
  for (int k = 0; k < n_used; k++) {
    args.higher_alpha[k] = higher_alpha[k];
  }
  run_on_all_particles(args, beam_dt, beam_dE, n_macroparticles);
}
