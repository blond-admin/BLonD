// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Per-particle kernel bodies, one overload of `apply_to_chunk` per kernel
// call record (kernel_call_records.h). Each overload takes `const real_t *`
// for the coordinate it only reads; the executor's `real_t *` converts. Two drivers use them:
//  - the eager extern "C" kernels, via `run_on_all_particles`, which
//    gives every OpenMP thread its share of the whole beam;
//  - the deferred executor (deferred.cpp), which applies all queued
//    records to one cache-sized chunk before moving to the next.
// A formula therefore exists once on the CPU.
//
// Compute-bound overloads carry BLOND_PREFER_VECTOR_WIDTH_512 together
// with `noinline`: inlined into an OpenMP region without the attribute,
// GCC silently falls back to 256-bit code.

#pragma once

#include <cmath>

#include "blond_common.h"
#include "kernel_call_records.h"
#include "openmp.h"

#define BLOND_NOINLINE __attribute__((noinline))

// Split [0, n) evenly over the threads, in whole `granule`s of one
// 64-byte cache line of coordinates, so on a 64-byte aligned array no
// two threads write the same line. numpy does not guarantee that
// alignment, so a boundary may still split one line; a coarser granule
// would not avoid that either, and only lets the slowest thread do up
// to a granule more (128 gave it 7.5% more than an even split at 1e4
// particles over 12 threads).
inline void thread_range(const index_t n, const int thread_id,
                         const int n_threads, index_t &begin, index_t &end,
                         const index_t granule = 64 / sizeof(real_t)) {
  const index_t n_granules = (n + granule - 1) / granule;
  const index_t per_thread = n_granules / n_threads;
  const index_t rest = n_granules % n_threads;
  const index_t first =
      thread_id * per_thread + (thread_id < rest ? thread_id : rest);
  const index_t count = per_thread + (thread_id < rest ? 1 : 0);
  begin = first * granule < n ? first * granule : n;
  end = (first + count) * granule < n ? (first + count) * granule : n;
}

// The calling thread's share of [0, n); call inside a parallel region.
inline void this_thread_range(const index_t n, index_t &begin, index_t &end) {
  thread_range(n, omp_get_thread_num(), omp_get_num_threads(), begin, end);
}

BLOND_PREFER_VECTOR_WIDTH_512 BLOND_NOINLINE inline void
apply_to_chunk(const KickSingleHarmonicArgs &args,
               const real_t *__restrict__ beam_dt, real_t *__restrict__ beam_dE,
               const index_t begin, const index_t end) {
  const real_t charge = args.charge;
  const real_t voltage = args.voltage;
  const real_t omega_RF = args.omega_rf;
  const real_t phi_RF = args.phi_rf;
  const real_t acc_kick = args.acceleration_kick;
  for (index_t i = begin; i < end; i++) {
    beam_dE[i] +=
        charge * voltage * FAST_SIN(omega_RF * beam_dt[i] + phi_RF) + acc_kick;
  }
}

// Sum of `N` RF harmonics at one particle's `dt`. With `N` a compile-time
// constant GCC unrolls the sum and vectorizes the particle loop around
// it. With a run-time count it vectorizes this loop instead, as a
// reduction over the harmonics, and leaves the particles scalar: few
// harmonics then run almost entirely in the reduction's scalar remainder
// (5 harmonics took 9x as long as 4; i5-11500, 1e6 particles).
template <int N>
inline real_t rf_harmonics_sum(const real_t *__restrict__ voltage,
                               const real_t *__restrict__ omega_RF,
                               const real_t *__restrict__ phi_RF,
                               const real_t dt) {
  real_t sum = 0.0;
  for (int j = 0; j < N; j++) {
    // The columns trail a kernel call record's Args (harmonics_of); the
    // analyzer takes the end of the Args for the end of the object.
    // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
    sum += voltage[j] * FAST_SIN(omega_RF[j] * dt + phi_RF[j]);
  }
  return sum;
}

// One pass of `N` harmonics over the particles. `noinline` gives every
// group size its own function, so its code does not depend on the other
// group sizes: GCC schedules the same instructions differently next to
// different neighbours, which moved single-harmonic timings by up to 9%.
template <int N>
BLOND_PREFER_VECTOR_WIDTH_512 BLOND_NOINLINE void kick_harmonic_group(
    const real_t *__restrict__ voltage, const real_t *__restrict__ omega_RF,
    const real_t *__restrict__ phi_RF, const real_t charge,
    const real_t acc_kick, const real_t *__restrict__ beam_dt,
    real_t *__restrict__ beam_dE, const index_t begin, const index_t end) {
  for (index_t i = begin; i < end; i++) {
    const real_t dE_sum =
        rf_harmonics_sum<N>(voltage, omega_RF, phi_RF, beam_dt[i]);
    beam_dE[i] += charge * dE_sum + acc_kick;
  }
}

// All `n_rf` harmonics on [begin, end): one pass per group of four, then
// one for the remaining 1-3. `acc_kick` goes into the last pass only;
// n_rf == 0 still makes one pass, for `acc_kick` alone.
inline void kick_harmonic_groups(
    const real_t *__restrict__ voltage, const real_t *__restrict__ omega_RF,
    const real_t *__restrict__ phi_RF, const int n_rf, const real_t charge,
    const real_t acc_kick, const real_t *__restrict__ beam_dt,
    real_t *__restrict__ beam_dE, const index_t begin, const index_t end) {
  const int n_groups_of_four = n_rf / 4;
  const int n_remaining = n_rf % 4;
  for (int group = 0; group < n_groups_of_four; group++) {
    const int first = 4 * group;
    const bool is_last_pass = n_remaining == 0 && group == n_groups_of_four - 1;
    kick_harmonic_group<4>(voltage + first, omega_RF + first, phi_RF + first,
                           charge,
                           is_last_pass ? acc_kick : static_cast<real_t>(0),
                           beam_dt, beam_dE, begin, end);
  }
  const int first = 4 * n_groups_of_four;
  switch (n_remaining) {
  case 1:
    kick_harmonic_group<1>(voltage + first, omega_RF + first, phi_RF + first,
                           charge, acc_kick, beam_dt, beam_dE, begin, end);
    break;
  case 2:
    kick_harmonic_group<2>(voltage + first, omega_RF + first, phi_RF + first,
                           charge, acc_kick, beam_dt, beam_dE, begin, end);
    break;
  case 3:
    kick_harmonic_group<3>(voltage + first, omega_RF + first, phi_RF + first,
                           charge, acc_kick, beam_dt, beam_dE, begin, end);
    break;
  default:
    if (n_groups_of_four == 0) {
      kick_harmonic_group<0>(voltage, omega_RF, phi_RF, charge, acc_kick,
                             beam_dt, beam_dE, begin, end);
    }
    break;
  }
}

// Particles per block when more than four harmonics take several passes:
// 16 KiB of dt/dE pairs, the budget of DEFERRED_CHUNK_SIZE (callables.py),
// so every pass after the first reads and writes L1d. Without the blocks
// each pass streams the whole range through memory again, which 12
// threads on 1e6 particles already showed (5 harmonics ~1.3x slower;
// threaded timings on the i5-11500 are noisy, so take it as direction).
constexpr index_t KICK_HARMONICS_BLOCK = 16384 / (2 * sizeof(real_t));

inline void apply_to_chunk(const KickMultiHarmonicArgs &args,
                           const real_t *__restrict__ beam_dt,
                           real_t *__restrict__ beam_dE, const index_t begin,
                           const index_t end) {
  const RfHarmonics harmonics = harmonics_of(args);
  const int n_rf = args.n_rf;
  const index_t block = (n_rf <= 4) ? end - begin : KICK_HARMONICS_BLOCK;
  for (index_t block_begin = begin; block_begin < end; block_begin += block) {
    const index_t block_end =
        (end - block_begin < block) ? end : block_begin + block;
    kick_harmonic_groups(harmonics.voltage, harmonics.omega_rf,
                         harmonics.phi_rf, n_rf, args.charge,
                         args.acceleration_kick, beam_dt, beam_dE, block_begin,
                         block_end);
  }
}

inline void apply_to_chunk(const DriftSimpleArgs &args,
                           real_t *__restrict__ beam_dt,
                           const real_t *__restrict__ beam_dE,
                           const index_t begin, const index_t end) {
  const real_t coeff =
      args.T * args.eta_0 / (args.beta * args.beta * args.energy);
  for (index_t i = begin; i < end; i++) {
    beam_dt[i] += coeff * beam_dE[i];
  }
}

inline void apply_to_chunk(const DriftLikeLineSegmentArgs &args,
                           real_t *__restrict__ beam_dt,
                           const real_t *__restrict__ beam_dE,
                           const index_t begin, const index_t end) {
  const real_t inv_beta_sq = 1.0 / (args.beta * args.beta);
  const real_t inv_energy = 1.0 / args.energy;
  const real_t inv_energy_sq = inv_energy * inv_energy;
  for (index_t i = begin; i < end; i++) {
    const real_t dE = beam_dE[i];
    const real_t delta =
        std::sqrt(1.0 + inv_beta_sq *
                            (dE * dE * inv_energy_sq + 2.0 * dE * inv_energy)) -
        1.0;
    beam_dt[i] += args.T * args.eta_0 * delta;
  }
}

// Defined in drift_exact.cpp, which keeps its unrolled
// `drift_exact_unrolled<N>` instantiations and dispatches on n_alpha.
void apply_to_chunk(const DriftExactArgs &args, real_t *beam_dt,
                    const real_t *beam_dE, index_t begin, index_t end);

// Defined in linear_interp_kick.cpp.
void apply_to_chunk(const KickInterpolatedArgs &args, const real_t *beam_dt,
                    real_t *beam_dE, index_t begin, index_t end);

// Table read by the interpolated kick: `2 * n_slices` entries,
// [bin_centers[0], inverse bin width, (slope, offset) per bin], with
// `charge` and `acc_kick` folded into the pairs.
extern "C" void linear_interp_kick_table(const real_t *voltage,
                                         const real_t *bin_centers,
                                         real_t charge, int n_slices,
                                         real_t acc_kick, real_t *table);

// Eager driver: `args` on all particles, each thread on its own share.
// `Dt`/`DE` are `const real_t *` for the coordinate the kernel only reads.
template <class Args, class Dt, class DE>
inline void run_on_all_particles(const Args &args, Dt beam_dt, DE beam_dE,
                                 const index_t n_macroparticles) {
#pragma omp parallel
  {
    index_t begin = 0;
    index_t end = 0;
    this_thread_range(n_macroparticles, begin, end);
    apply_to_chunk(args, beam_dt, beam_dE, begin, end);
  }
}
