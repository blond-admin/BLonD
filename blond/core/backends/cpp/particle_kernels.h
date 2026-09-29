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

// Split [0, n) evenly over the threads, in whole `granule`s, so two
// threads never share a cache line of particles.
inline void thread_range(const index_t n, const int thread_id,
                         const int n_threads, index_t &begin, index_t &end,
                         const index_t granule = 128) {
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

BLOND_PREFER_VECTOR_WIDTH_512 BLOND_NOINLINE inline void
apply_to_chunk(const KickMultiHarmonicArgs &args,
               const real_t *__restrict__ beam_dt, real_t *__restrict__ beam_dE,
               const index_t begin, const index_t end) {
  const real_t *__restrict__ voltage = &args.voltage[0];
  const real_t *__restrict__ omega_RF = &args.omega_rf[0];
  const real_t *__restrict__ phi_RF = &args.phi_rf[0];
  const real_t charge = args.charge;
  const real_t acc_kick = args.acceleration_kick;
  const int n_rf = args.n_rf;

  // Unroll loop for up to 4 RF harmonics for speedup. The branches differ;
  // clang-tidy only sees near-identical bodies.
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (n_rf == 1) {
    for (index_t i = begin; i < end; i++) {
      const real_t dE_sum =
          voltage[0] * FAST_SIN(omega_RF[0] * beam_dt[i] + phi_RF[0]);
      beam_dE[i] += charge * dE_sum + acc_kick;
    }
  } else if (n_rf == 2) {
    for (index_t i = begin; i < end; i++) {
      const real_t dE_sum =
          voltage[0] * FAST_SIN(omega_RF[0] * beam_dt[i] + phi_RF[0]) +
          voltage[1] * FAST_SIN(omega_RF[1] * beam_dt[i] + phi_RF[1]);
      beam_dE[i] += charge * dE_sum + acc_kick;
    }
  } else if (n_rf == 3) {
    for (index_t i = begin; i < end; i++) {
      const real_t dE_sum =
          voltage[0] * FAST_SIN(omega_RF[0] * beam_dt[i] + phi_RF[0]) +
          voltage[1] * FAST_SIN(omega_RF[1] * beam_dt[i] + phi_RF[1]) +
          voltage[2] * FAST_SIN(omega_RF[2] * beam_dt[i] + phi_RF[2]);
      beam_dE[i] += charge * dE_sum + acc_kick;
    }
  } else if (n_rf == 4) {
    for (index_t i = begin; i < end; i++) {
      const real_t dE_sum =
          voltage[0] * FAST_SIN(omega_RF[0] * beam_dt[i] + phi_RF[0]) +
          voltage[1] * FAST_SIN(omega_RF[1] * beam_dt[i] + phi_RF[1]) +
          voltage[2] * FAST_SIN(omega_RF[2] * beam_dt[i] + phi_RF[2]) +
          voltage[3] * FAST_SIN(omega_RF[3] * beam_dt[i] + phi_RF[3]);
      beam_dE[i] += charge * dE_sum + acc_kick;
    }
  } else {
    for (index_t i = begin; i < end; i++) {
      real_t dE_sum = 0.0;
      // fallback to loop for n_rf > 4 (and n_rf == 0)
      for (int j = 0; j < n_rf; j++) {
        dE_sum += voltage[j] * FAST_SIN(omega_RF[j] * beam_dt[i] + phi_RF[j]);
      }
      beam_dE[i] += charge * dE_sum + acc_kick;
    }
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
