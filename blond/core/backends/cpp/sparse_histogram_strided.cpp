// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENCE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_ALLOC(size, align) _aligned_malloc(size, align)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#define ALIGNED_ALLOC(size, align) aligned_alloc(align, size)
#define ALIGNED_FREE(ptr) free(ptr)
#endif

#include "blond_common.h"
#include "openmp.h"

extern "C" void sparse_histogram_strided(
    const real_t *__restrict__ input, real_t *__restrict__ output,
    const real_t first_left_cut, const real_t left_cut_distance,
    const real_t cut_width, const int bins_per_profile,
    const int n_active_profiles, const int n_buckets,
    const int n_macroparticles, const bool *__restrict__ filling_pattern,
    const int *__restrict__ bucket_index_to_memory_index) {
  const real_t cut_left0 = first_left_cut;
  const real_t inv_hist_dist = real_t(1) / left_cut_distance;
  const real_t inv_bin_width = real_t(bins_per_profile) / cut_width;

  const int compact_size = n_active_profiles * bins_per_profile;

  // -------------------------------------
  // Zero global output
  // -------------------------------------
  // #pragma omp parallel for schedule(static)
  for (int i = 0; i < compact_size; ++i)
    output[i] = real_t(0);

// -------------------------------------
// Parallel region
// -------------------------------------
#pragma omp parallel
  {
    // Thread-local persistent buffer
    static thread_local real_t *local_output = nullptr;
    static thread_local int local_size = 0;

    // Allocate only if size changed or first use
    if (local_size < compact_size) {
      ALIGNED_FREE(local_output);
      local_output = (real_t *)ALIGNED_ALLOC(sizeof(real_t) * compact_size, 64);
      local_size = compact_size;
    }

    // Clear only used portion
    memset(local_output, 0, sizeof(real_t) * compact_size);
// ---------------------------------
// Particle loop
// ---------------------------------
#pragma omp for schedule(static)
    for (int i = 0; i < n_macroparticles; ++i) {
      const real_t a = input[i];

      const int bucket_i = (int)((a - cut_left0) * inv_hist_dist);
      if (bucket_i >= n_buckets || bucket_i < 0)
        continue;
      if (!filling_pattern[bucket_i]) {
        continue;
      }
      const real_t cut_left = cut_left0 + bucket_i * left_cut_distance;
      const real_t cut_right = cut_left + cut_width;

      // Check if the value is within the cut range
      if (a == cut_right) {
        local_output[bucket_index_to_memory_index[bucket_i] + bins_per_profile -
                     1] += 1;
        continue;
      }
      if (a < cut_left || a >= cut_right)
        continue;

      // Calculate the bin index
      const int bin = (int)((a - cut_left) * inv_bin_width);
      if ((unsigned)bin < (unsigned)bins_per_profile) {
        local_output[bucket_index_to_memory_index[bucket_i] + bin] += 1;
      }
    }

// ---------------------------------
// Reduction step
// ---------------------------------
#pragma omp critical
    {
      for (int i = 0; i < compact_size; ++i) {
        output[i] += local_output[i];
      }
    }
  }
}
