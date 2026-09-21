// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Optimised C++ routine that calculates the histogram
// Author: Danilo Quartullo, Alexandre Lasheen, Konstantinos Iliakis

#include <math.h>
#include <stdlib.h> // mmalloc()
#include <string.h> // memset()

#include "blond_common.h"
#include "openmp.h"

// Particles `histogram` handles per tile. Counting cannot be
// vectorised, so it works one tile at a time: a SIMD pass computes
// every bin index of the tile into a scratch buffer, then a scalar pass
// increments one bin per index. Measured on Rocket Lake, 128 is the
// flattest point: smaller tiles do not amortise the loop overhead,
// larger ones push the scratch buffer out of the store buffer.
#define HISTOGRAM_TILE 128

// Lanes the bin index is computed in: one full SIMD register of the
// widest instruction set this build targets. Affects speed only -- gcc
// emulates any width, so the two-lane fallback is correct everywhere.
#if defined(__AVX512F__)
#define BIN_LANES 8
#elif defined(__AVX__)
#define BIN_LANES 4
#else
#define BIN_LANES 2
#endif

// Bin indices are computed in double even in a float build: a float
// cannot represent indices above 2^24 exactly. The `aligned(1)` types
// are for reading the coordinates and writing the scratch buffer, which
// are addressed at arbitrary offsets.
typedef double bin_vec_t __attribute__((vector_size(BIN_LANES * 8)));
typedef int bin_index_vec_t __attribute__((vector_size(BIN_LANES * 4)));
typedef double unaligned_bin_vec_t
    __attribute__((vector_size(BIN_LANES * 8), aligned(1)));
typedef int unaligned_bin_index_vec_t
    __attribute__((vector_size(BIN_LANES * 4), aligned(1)));
#ifdef USEFLOAT
typedef float unaligned_coordinate_vec_t
    __attribute__((vector_size(BIN_LANES * 4), aligned(1)));
#endif

/** Read `BIN_LANES` consecutive coordinates, widened to double. */
static inline bin_vec_t load_coordinates(const real_t *coordinates) {
#ifdef USEFLOAT
  return __builtin_convertvector(
      *(const unaligned_coordinate_vec_t *)coordinates, bin_vec_t);
#else
  return *(const unaligned_bin_vec_t *)coordinates;
#endif
}

/**
 * Bin index of one particle, or `n_slices` if it falls outside the range.
 *
 * `n_slices` doubles as the index of a *trash bin*: `histogram`
 * allocates one slot past the profile and counts every out-of-range
 * particle into it, so that neither this function nor its vector
 * counterpart needs a branch. That slot is never read back, which is
 * what drops those particles.
 */
static inline int bin_index_of(const real_t coordinate, const double cut_left,
                               const double cut_right,
                               const double inv_bin_width, const int n_slices) {
  const double value = (double)coordinate;
  const double scaled = (value - cut_left) * inv_bin_width;
  // Scaling is not exact: a value at or just below cut_right can scale
  // to n_slices. Fold it back into the last bin, as np.histogram does,
  // instead of dropping the particle.
  const bool folds_back = (scaled >= n_slices) && (value <= cut_right);
  const double folded = folds_back ? (double)(n_slices - 1) : scaled;
  // Select before converting, not after: converting an out-of-range
  // double to int is undefined behaviour.
  const bool inside = (folded >= 0.0) && (folded < n_slices);
  // Truncation equals floor here, `folded` being non-negative.
  return (int)(inside ? folded : (double)n_slices);
}

/** `bin_index_of` for a whole tile, step for step and branch-free. */
static inline void bin_indices_of_tile(const real_t *__restrict__ coordinates,
                                       const double cut_left,
                                       const double cut_right,
                                       const double inv_bin_width,
                                       const int n_slices,
                                       int *__restrict__ bin_indices) {
  const bin_vec_t zero = (bin_vec_t){};
  const bin_vec_t vec_cut_left = zero + cut_left;
  const bin_vec_t vec_cut_right = zero + cut_right;
  const bin_vec_t vec_inv_bin_width = zero + inv_bin_width;
  const bin_vec_t vec_n_slices = zero + (double)n_slices;
  const bin_vec_t vec_last_bin = zero + (double)(n_slices - 1);

  for (int lane = 0; lane < HISTOGRAM_TILE; lane += BIN_LANES) {
    const bin_vec_t value = load_coordinates(coordinates + lane);
    const bin_vec_t scaled = (value - vec_cut_left) * vec_inv_bin_width;
    const bin_vec_t folded =
        ((scaled >= vec_n_slices) & (value <= vec_cut_right)) ? vec_last_bin
                                                              : scaled;
    const bin_vec_t inside_or_trash =
        ((folded >= zero) & (folded < vec_n_slices)) ? folded : vec_n_slices;
    *(unaligned_bin_index_vec_t *)(bin_indices + lane) =
        __builtin_convertvector(inside_or_trash, bin_index_vec_t);
  }
}

extern "C" void histogram(const real_t *__restrict__ input,
                          real_t *__restrict__ output, const real_t cut_left,
                          const real_t cut_right, const int n_slices,
                          const index_t n_macroparticles, const int n_threads) {
  const double inv_bin_width = n_slices / ((double)cut_right - cut_left);

  // One private histogram per thread, plus the trash bin each of them
  // needs. index_t counters, so one bin can hold more than 2^31 - 1
  // particles. One flat block rather than a table of per-thread
  // pointers: that indirection would sit inside the counting loop's
  // read-modify-write and roughly doubles its cost.
  //
  // `n_threads` is chosen by the caller (`histogram_n_threads.py`):
  // every thread is woken up and zeroes a private histogram, which with
  // few particles or many bins costs more than the counting it takes
  // over. The counters are integers, so the result does not depend on
  // it.
  const size_t bins_per_thread = (size_t)n_slices + 1;
  index_t *histo =
      (index_t *)malloc((size_t)n_threads * bins_per_thread * sizeof(index_t));

#pragma omp parallel num_threads(n_threads)
  {
    const int id = omp_get_thread_num();
    const int threads = omp_get_num_threads();
    index_t *__restrict__ h = histo + (size_t)id * bins_per_thread;
    memset(h, 0, bins_per_thread * sizeof(index_t));

    alignas(64) int bin_indices[HISTOGRAM_TILE];

#pragma omp for
    for (index_t i = 0; i < n_macroparticles; i += HISTOGRAM_TILE) {
      if (n_macroparticles - i < HISTOGRAM_TILE) {
        // Last, partial tile: not worth a masked vector pass.
        for (index_t j = i; j < n_macroparticles; j++)
          h[bin_index_of(input[j], cut_left, cut_right, inv_bin_width,
                         n_slices)] += 1;
        continue;
      }
      bin_indices_of_tile(input + i, cut_left, cut_right, inv_bin_width,
                          n_slices, bin_indices);
      for (int j = 0; j < HISTOGRAM_TILE; j++)
        h[bin_indices[j]] += 1;
    }

// Reduce to a single histogram. The trash bin past the last slice is
// left out, which is what drops the out-of-range particles.
#pragma omp for
    for (int i = 0; i < n_slices; i++) {
      index_t total = 0;
      for (int t = 0; t < threads; t++)
        total += histo[(size_t)t * bins_per_thread + i];
      output[i] = (real_t)total;
    }
  }

  // free memory
  free(histo);
}

extern "C" void smooth_histogram(const real_t *__restrict__ input,
                                 real_t *__restrict__ output,
                                 const real_t cut_left, const real_t cut_right,
                                 const int n_slices,
                                 const index_t n_macroparticles) {
  // Constants init
  const real_t inv_bin_width = n_slices / (cut_right - cut_left);
  const real_t bin_width = (cut_right - cut_left) / n_slices;
  const real_t const1 = (cut_left + bin_width * 0.5);
  const real_t const2 = (cut_right - bin_width * 0.5);

  // memory alloc for per thread histo
  real_t **histo = (real_t **)malloc(omp_get_max_threads() * sizeof(real_t *));
  histo[0] =
      (real_t *)malloc(omp_get_max_threads() * n_slices * sizeof(real_t));
  for (int i = 0; i < omp_get_max_threads(); i++)
    histo[i] = (*histo + n_slices * i);

#pragma omp parallel
  {
    const int id = omp_get_thread_num();
    const int threads = omp_get_num_threads();
    memset(histo[id], 0, n_slices * sizeof(real_t));

// main caclulation
#pragma omp for
    for (index_t i = 0; i < n_macroparticles; i++) {
      int fffbin = 0;
      real_t a = input[i];
      if ((a < const1) || (a > const2))
        continue;
      real_t fbin = (a - cut_left) * inv_bin_width;
      int ffbin = (int)(fbin);
      real_t distToCenter = fbin - (real_t)(ffbin);
      if (distToCenter > 0.5)
        fffbin = (int)(fbin + 1.0);
      else
        fffbin = (int)(fbin - 1.0);

      // Bounds check to prevent buffer overrun
      if (ffbin >= 0 && ffbin < n_slices)
        histo[id][ffbin] += 0.5 - distToCenter;
      if (fffbin >= 0 && fffbin < n_slices)
        histo[id][fffbin] += 0.5 + distToCenter;
    }

// Reduce to a single histogram
#pragma omp for
    for (int i = 0; i < n_slices; i++) {
      output[i] = 0.;
      for (int t = 0; t < threads; t++)
        output[i] += histo[t][i];
    }
  }
  // free memory
  free(histo[0]);
  free(histo);
}

/***** serial histogram

extern "C" void histogram(const double *__restrict__ input,
                          double *__restrict__ output,
                          const double cut_left, const double cut_right,
                          const int n_slices,
                          const index_t n_macroparticles)
{
    // Number of Iterations of the inner loop
    const int STEP = 16;
    const double inv_bin_width = n_slices / (cut_right - cut_left);
    float fbin[STEP];

    memset(output, 0., n_slices * sizeof(double));
    for (index_t i = 0; i < n_macroparticles; i += STEP) {

        const index_t loop_count = n_macroparticles - i > STEP ?
                               STEP : n_macroparticles - i;

        // First calculate the index to update
        for (index_t j = 0; j < loop_count; j++) {
            fbin[j] = floor((input[i + j] - cut_left) * inv_bin_width);
        }
        // Then update the corresponding bins
        for (index_t j = 0; j < loop_count; j++) {
            const index_t bin  = (index_t) fbin[j];
            if (bin < 0 || bin >= n_slices) continue;
            output[bin] += 1.;
        }
    }

}

*******/
