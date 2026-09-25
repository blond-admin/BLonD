// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Per-particle kernels ("particle ops"), each written exactly once.
//
// Every op is a struct holding its parameters (`Args`) and an `apply`
// that updates the particles in [begin, end). Two drivers run them:
//  - the eager extern "C" kernels (kick.cpp, drift.cpp, ...) hand every
//    OpenMP thread its share of the whole beam (`run_parallel`), one
//    kernel after the other, as before;
//  - the deferred executor (deferred.cpp) queues several ops and runs all
//    of them on one cache-sized chunk of the beam before moving on to the
//    next chunk, so the particles are read from DRAM once, not per op.
// A fix to a formula therefore goes into `apply` here and reaches both.
//
// `apply` of a compute-bound op carries BLOND_PREFER_VECTOR_WIDTH_512
// (see blond_common.h) together with `noinline`: inlined into a caller
// without the attribute -- the OpenMP region of `run_parallel` or the
// deferred executor -- GCC would silently fall back to 256-bit code.
//
// Queueing an op: give it `unpack`, which reads its `Args` from the
// packed scalars `s` and arrays `p`, and `scalars()`/`arrays()`, the
// names of what `unpack` reads, in its order. The names are those of the
// `Specials` method's arguments: Python packs by name, so the order is
// only ever written down here. An array written by the op is marked
// `:out` (Python copies the others when queueing). Then list the op in
// BLOND_PARTICLE_OPS (deferred.cpp) and give `DeferredCppSpecials`
// (deferred.py) a method that queues it.

#pragma once

#include <cmath>

#include "blond_common.h"
#include "openmp.h"

#define BLOND_NOINLINE __attribute__((noinline))

// Split [0, n) evenly over the threads, in whole `granule`s. The default
// keeps two threads' particles off a shared cache line and lines them up
// with the histogram tiles.
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
inline void this_thread_range(const index_t n, index_t &begin, index_t &end,
                              const index_t granule = 128) {
  thread_range(n, omp_get_thread_num(), omp_get_num_threads(), begin, end,
               granule);
}

// Eager driver: `Op` on all particles, every thread on its own share.
// `Dt`/`DE` are `const real_t *` for the coordinate the op only reads.
template <class Op, class Dt, class DE>
inline void run_parallel(const typename Op::Args &args, Dt beam_dt, DE beam_dE,
                         const index_t n_macroparticles) {
#pragma omp parallel
  {
    index_t begin = 0;
    index_t end = 0;
    this_thread_range(n_macroparticles, begin, end);
    Op::apply(args, beam_dt, beam_dE, begin, end);
  }
}

struct KickSingleHarmonic {
  struct Args {
    real_t charge, voltage, omega_rf, phi_rf, acc_kick;
  };
  static const char *scalars() {
    return "charge voltage omega_rf phi_rf acceleration_kick";
  }
  static const char *arrays() { return ""; }
  static Args unpack(const double *s, void *const * /*p*/) {
    const Args args = {s[0], s[1], s[2], s[3], s[4]};
    return args;
  }

  BLOND_PREFER_VECTOR_WIDTH_512 BLOND_NOINLINE static void
  apply(const Args &a, const real_t *__restrict__ beam_dt,
        real_t *__restrict__ beam_dE, const index_t begin, const index_t end) {
    for (index_t i = begin; i < end; i++) {
      beam_dE[i] +=
          a.charge * a.voltage * FAST_SIN(a.omega_rf * beam_dt[i] + a.phi_rf) +
          a.acc_kick;
    }
  }
};

struct KickMultiHarmonic {
  struct Args {
    int n_rf;
    real_t charge;
    const real_t *voltage, *omega_rf, *phi_rf;
    real_t acc_kick;
  };
  static const char *scalars() { return "n_rf charge acceleration_kick"; }
  static const char *arrays() { return "voltage omega_rf phi_rf"; }
  static Args unpack(const double *s, void *const *p) {
    const Args args = {static_cast<int>(s[0]),
                       s[1],
                       static_cast<const real_t *>(p[0]),
                       static_cast<const real_t *>(p[1]),
                       static_cast<const real_t *>(p[2]),
                       s[2]};
    return args;
  }

  BLOND_PREFER_VECTOR_WIDTH_512 BLOND_NOINLINE static void
  apply(const Args &a, const real_t *__restrict__ beam_dt,
        real_t *__restrict__ beam_dE, const index_t begin, const index_t end) {
    const real_t *__restrict__ voltage = a.voltage;
    const real_t *__restrict__ omega_RF = a.omega_rf;
    const real_t *__restrict__ phi_RF = a.phi_rf;
    const real_t charge = a.charge;
    const real_t acc_kick = a.acc_kick;
    const int n_rf = a.n_rf;
    // Unroll loop for up to 4 RF harmonics for speedup. The branches
    // differ; clang-tidy only sees identical loop bodies.
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
        // fallback to loop for n_rf > 4
        for (int j = 0; j < n_rf; j++) {
          dE_sum += voltage[j] * FAST_SIN(omega_RF[j] * beam_dt[i] + phi_RF[j]);
        }
        beam_dE[i] += charge * dE_sum + acc_kick;
      }
    }
  }
};

struct DriftSimple {
  struct Args {
    real_t T, eta_zero, beta, energy;
  };
  static const char *scalars() { return "T eta_0 beta energy"; }
  static const char *arrays() { return ""; }
  static Args unpack(const double *s, void *const * /*p*/) {
    const Args args = {s[0], s[1], s[2], s[3]};
    return args;
  }

  static void apply(const Args &a, real_t *__restrict__ beam_dt,
                    const real_t *__restrict__ beam_dE, const index_t begin,
                    const index_t end) {
    const real_t coeff = a.T * a.eta_zero / (a.beta * a.beta * a.energy);
    for (index_t i = begin; i < end; i++) {
      beam_dt[i] += coeff * beam_dE[i];
    }
  }
};

// Drift with the linear slip factor but the exact relativistic delta;
// reproduces the longitudinal drift of an xsuite LineSegmentMap.
struct DriftLikeLineSegment {
  struct Args {
    real_t T, eta_zero, beta, energy;
  };
  static const char *scalars() { return "T eta_0 beta energy"; }
  static const char *arrays() { return ""; }
  static Args unpack(const double *s, void *const * /*p*/) {
    const Args args = {s[0], s[1], s[2], s[3]};
    return args;
  }

  static void apply(const Args &a, real_t *__restrict__ beam_dt,
                    const real_t *__restrict__ beam_dE, const index_t begin,
                    const index_t end) {
    const real_t inv_beta_sq = 1.0 / (a.beta * a.beta);
    const real_t inv_energy = 1.0 / a.energy;
    const real_t inv_energy_sq = inv_energy * inv_energy;
    for (index_t i = begin; i < end; i++) {
      const real_t dE = beam_dE[i];
      const real_t delta =
          std::sqrt(1.0 + inv_beta_sq * (dE * dE * inv_energy_sq +
                                         2.0 * dE * inv_energy)) -
          1.0;
      beam_dt[i] += a.T * a.eta_zero * delta;
    }
  }
};

// Kick by a voltage sampled at equidistant `bin_centers`, linearly
// interpolated. Needs per-bin tables (slope `voltage_kick` and offset
// `factor`) which `fill_table` computes before `apply` may run. The tables
// hold one entry more than there are bins, the *trash entry* (slope 0,
// offset acc_kick) that every particle outside the bins is pointed to, so
// the particle loop needs no range check.
struct LinearInterpKick {
  struct Args {
    const real_t *voltage;
    const real_t *bin_centers;
    real_t charge;
    int n_slices;
    real_t acc_kick;
    real_t *voltage_kick; // n_slices entries, filled by `fill_table`
    real_t *factor;       // n_slices entries, filled by `fill_table`
  };
  static const char *scalars() { return "charge n_bins acceleration_kick"; }
  static const char *arrays() { return "voltage bin_centers"; }
  // The tables are not packed: the deferred executor owns them.
  static Args unpack(const double *s, void *const *p) {
    const Args args = {static_cast<const real_t *>(p[0]),
                       static_cast<const real_t *>(p[1]),
                       s[0],
                       static_cast<int>(s[1]),
                       s[2],
                       nullptr,
                       nullptr};
    return args;
  }

  // Particles per tile of `apply`'s two-pass loop.
  static const int STEP = 64;

  static real_t inv_bin_width(const Args &a) {
    return (a.n_slices - 1) /
           (a.bin_centers[a.n_slices - 1] - a.bin_centers[0]);
  }

  // Table entries of the bins [bin_begin, bin_end) within [0, n_slices - 1).
  static void fill_table(const Args &a, const real_t inv_bin_width,
                         const int bin_begin, const int bin_end) {
    const real_t *__restrict__ voltage = a.voltage;
    const real_t *__restrict__ bin_centers = a.bin_centers;
    real_t *__restrict__ voltage_kick = a.voltage_kick;
    real_t *__restrict__ factor = a.factor;
    for (int i = bin_begin; i < bin_end; i++) {
      voltage_kick[i] =
          a.charge * (voltage[i + 1] - voltage[i]) * inv_bin_width;
      factor[i] = (a.charge * voltage[i] - bin_centers[i] * voltage_kick[i]) +
                  a.acc_kick;
    }
  }

  // The trash entry at index n_slices - 1: out of range only the
  // interpolated voltage is undefined; acc_kick carries the reference
  // energy change, which applies to the whole beam.
  static void fill_trash_entry(const Args &a) {
    a.voltage_kick[a.n_slices - 1] = 0.0;
    a.factor[a.n_slices - 1] = a.acc_kick;
  }

  // Two passes per tile: first every bin index (vectorised), then the
  // table lookups, branch-free. Measured 23% faster per particle than a
  // range check per particle (i5-11500, bunch in the profile). AVX-512
  // gathers for the lookups were slower still: since the "Downfall"
  // microcode fix, gathers are slow on Ice/Tiger/Rocket Lake.
  static void apply(const Args &a, const real_t *__restrict__ beam_dt,
                    real_t *__restrict__ beam_dE, const index_t begin,
                    const index_t end) {
    const real_t inv_bin_width_ = inv_bin_width(a);
    const real_t *__restrict__ voltage_kick = a.voltage_kick;
    const real_t *__restrict__ factor = a.factor;
    const real_t bin_center_0 = a.bin_centers[0];
    const int trash = a.n_slices - 1;
    const double last = static_cast<double>(trash);
    // Scratch, fully written before it is read: a C array, as in the
    // original kernel, so it is not zeroed on every call.
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    int bins[STEP];

    for (index_t i = begin; i < end; i += STEP) {
      const int count = end - i > STEP ? STEP : static_cast<int>(end - i);

      for (int j = 0; j < count; j++) {
        const double fbin =
            std::floor((beam_dt[i + j] - bin_center_0) * inv_bin_width_);
        // Select before converting: converting an out-of-range double to
        // int is undefined behaviour.
        const bool in_range = (fbin >= 0.0) && (fbin < last);
        bins[j] = static_cast<int>(in_range ? fbin : last);
      }

      for (int j = 0; j < count; j++) {
        beam_dE[i + j] +=
            std::fma(beam_dt[i + j], voltage_kick[bins[j]], factor[bins[j]]);
      }
    }
  }
};

// Histogram of `dt` (or `dE`), counted into one private row of
// `n_slices + 1` integer counters per thread; the last counter is the
// trash bin that swallows out-of-range particles (see bin_index_of in
// histogram.cpp).
struct Histogram {
  static size_t row_size(const int n_slices) { return (size_t)n_slices + 1; }
  struct Args {
    real_t cut_left, cut_right;
    int n_slices;
    bool reads_dE; // deferred only: histogram dE instead of dt
    real_t *output;
  };
  static const char *scalars() { return "start stop n_bins reads_dE"; }
  static const char *arrays() { return "array_write:out"; }
  static Args unpack(const double *s, void *const *p) {
    const Args args = {s[0], s[1], static_cast<int>(s[2]), s[3] != 0.0,
                       static_cast<real_t *>(p[0])};
    return args;
  }

  // Count input[begin, end) into the thread's row `counts`. Defined in
  // histogram.cpp, next to the bin index helpers it uses.
  static void count(const Args &a, index_t *__restrict__ counts,
                    const real_t *__restrict__ input, index_t begin,
                    index_t end);

  // Sum the per-thread rows of bins [bin_begin, bin_end) into `output`.
  // The trash bin past the last slice is left out, which is what drops
  // the out-of-range particles.
  static void reduce(const Args &a, const index_t *__restrict__ counts,
                     const int n_threads, const int bin_begin,
                     const int bin_end) {
    const size_t row = row_size(a.n_slices);
    for (int i = bin_begin; i < bin_end; i++) {
      index_t total = 0;
      for (int t = 0; t < n_threads; t++) {
        total += counts[(size_t)t * row + i];
      }
      // exact while a bin holds fewer than 2^53 particles
      a.output[i] = static_cast<real_t>(total);
    }
  }
};
