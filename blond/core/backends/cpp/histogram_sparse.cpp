// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

#include <cstdio>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "blond_common.h"
#include "openmp.h"

extern "C" void
histogram_sparse(const real_t *__restrict__ input, real_t *__restrict__ output,
                 const real_t first_left_cut, const real_t left_cut_distance,
                 const real_t cut_width, const int bins_per_profile,
                 const int n_active_profiles, const int n_buckets,
                 const int n_macroparticles,
                 const bool *__restrict__ filling_pattern,
                 const int *__restrict__ bucket_index_to_memory_index) {
  const real_t cut_left0 = first_left_cut;
  const real_t inv_hist_dist = real_t(1) / left_cut_distance;
  const real_t inv_bin_width = real_t(bins_per_profile) / cut_width;
  const int n_out = bins_per_profile * n_active_profiles;

  // Per-thread private histograms, reduced at the end -- this avoids a
  // `#pragma omp atomic` on every particle (an atomic is a locked memory op
  // even when uncontended, and dominated this kernel: ~6x slower than the
  // dense `histogram`, which already uses this pattern).
  //
  // The scratch buffer is re-used across calls (the profile size is constant
  // turn after turn), reallocating only when a larger one is needed -- this
  // removes a malloc+free every call, which matters for large histograms.
  // Safe because BLonD drives the kernels from a single Python thread (the
  // OpenMP parallelism is internal); it is not re-entrant.
  const int nthreads = omp_get_max_threads();
  const size_t need = (size_t)nthreads * (size_t)n_out;
  static real_t *histo = nullptr;
  static size_t histo_cap = 0;
  if (need > histo_cap) {
    free(histo);
    histo = (real_t *)malloc(need * sizeof(real_t));
    histo_cap = need;
  }

#pragma omp parallel
  {
    const int id = omp_get_thread_num();
    const int threads = omp_get_num_threads();
    real_t *__restrict__ h = histo + (size_t)id * n_out;
    memset(h, 0, n_out * sizeof(real_t));

// ---------------------------------
// Particle loop (into the private histogram, no atomics)
// ---------------------------------
#pragma omp for schedule(static)
    for (int i = 0; i < n_macroparticles; ++i) {
      const real_t dt = input[i];

      const int bucket_i = (int)((dt - cut_left0) * inv_hist_dist);
      if (bucket_i >= n_buckets || bucket_i < 0)
        continue;
      if (!filling_pattern[bucket_i]) {
        continue;
      }
      const real_t cut_left = cut_left0 + bucket_i * left_cut_distance;
      const real_t cut_right = cut_left + cut_width;

      // Check if the value is within the cut range
      if (dt == cut_right) {
        h[bucket_index_to_memory_index[bucket_i] + bins_per_profile - 1] += 1;
        continue;
      }
      if (dt < cut_left || dt >= cut_right)
        continue;

      // Calculate the bin index
      const int bin = (int)((dt - cut_left) * inv_bin_width);
      if ((unsigned)bin < (unsigned)bins_per_profile) {
        h[bucket_index_to_memory_index[bucket_i] + bin] += 1;
      }
    }

// ---------------------------------
// Reduce the private histograms into the output
// ---------------------------------
#pragma omp for schedule(static)
    for (int k = 0; k < n_out; ++k) {
      real_t s = 0;
      for (int t = 0; t < threads; ++t)
        s += histo[(size_t)t * n_out + k];
      output[k] = s;
    }
  }
  // `histo` is intentionally NOT freed: it persists and is re-used next call.
}
