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

extern "C" void histogram(const real_t *__restrict__ input,
                          real_t *__restrict__ output, const real_t cut_left,
                          const real_t cut_right, const int n_slices,
                          const index_t n_macroparticles) {
  // Number of Iterations of the inner loop
  const int STEP = 16;
  const real_t inv_bin_width = n_slices / (cut_right - cut_left);
#ifdef PARALLEL
  // Shared, not per-thread: as privates inside the parallel region these
  // roughly double the cost of the branch-free loop below.
  const real_t n_slices_real = (real_t)n_slices;
  const real_t last_bin = (real_t)(n_slices - 1);
#endif

  // index_t counters, so one bin can hold more than 2^31 - 1 particles.
  // One flat block rather than a table of per-thread pointers: that
  // indirection would sit inside the branch-free loop's
  // read-modify-write and roughly doubles its cost.
  const int max_threads = omp_get_max_threads();
  index_t *histo =
      (index_t *)malloc((size_t)max_threads * n_slices * sizeof(index_t));

#pragma omp parallel
  {
    const int id = omp_get_thread_num();
    const int threads = omp_get_num_threads();
    index_t *__restrict__ h = histo + (size_t)id * n_slices;
    memset(h, 0, n_slices * sizeof(index_t));

    // The two libraries are compiled separately, so each takes the loop
    // shape that is faster for it: single-threaded the range check is
    // well predicted and the selects below are pure overhead, while with
    // many threads the branch-free form does less work on large
    // profiles. Both must produce identical counts.
    //
    // Measure in CPU cycles, not wall time: under many threads the
    // package downclocks within seconds, which moves nanosecond timings
    // by more than the loop shape does.
#ifdef PARALLEL
    int bin_index[STEP];
    // 0/1 weight rather than a branch: out-of-range particles add
    // nothing instead of being skipped.
    index_t is_inside[STEP];
#else
    // Keep the bin index in double until it is range-checked: a float
    // cannot represent indices above 2^24 exactly, and converting an
    // out-of-range double to int is undefined behaviour (on x86 it
    // yields INT_MIN, i.e. a wild write).
    double fbin[STEP] = {-1};
#endif
#pragma omp for
    for (index_t i = 0; i < n_macroparticles; i += STEP) {

      const index_t loop_count =
          n_macroparticles - i > STEP ? STEP : (index_t)(n_macroparticles - i);

#ifdef PARALLEL
      for (index_t j = 0; j < loop_count; j++) {
        const real_t value = input[i + j];
        const real_t bin = floor((value - cut_left) * inv_bin_width);

        // Scaling is not exact: a value at or just below cut_right can
        // land on n_slices. Fold it back into the last bin, as
        // np.histogram does, instead of dropping the particle.
        const bool folds_back = (bin >= n_slices_real) && (value <= cut_right);
        const real_t folded = folds_back ? last_bin : bin;

        const bool inside = (folded >= (real_t)0) && (folded < n_slices_real);
        // Clamp before the conversion, not after: converting an
        // out-of-range value to int is undefined behaviour, and a
        // select evaluates both arms.
        bin_index[j] = (int)(inside ? folded : (real_t)0);
        is_inside[j] = inside ? 1 : 0;
      }
      for (index_t j = 0; j < loop_count; j++) {
        h[bin_index[j]] += is_inside[j];
      }
#else
      for (index_t j = 0; j < loop_count; j++) {
        fbin[j] = floor((input[i + j] - cut_left) * inv_bin_width);

        // Scaling is not exact: a value at or just below cut_right can
        // land on n_slices. Fold it back into the last bin, as
        // np.histogram does, instead of dropping the particle.
        if (fbin[j] >= (double)n_slices && input[i + j] <= cut_right) {
          fbin[j] = n_slices - 1;
        }
      }
      for (index_t j = 0; j < loop_count; j++) {
        if (fbin[j] < 0.0 || fbin[j] >= (double)n_slices)
          continue;
        h[(int)fbin[j]] += 1;
      }
#endif
    }

// Reduce to a single histogram
#pragma omp for
    for (int i = 0; i < n_slices; i++) {
      index_t total = 0;
      for (int t = 0; t < threads; t++)
        total += histo[(size_t)t * n_slices + i];
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
