// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENCE.txt.
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

  // -------------------------------------
  // Zero global output
  // -------------------------------------
#pragma omp parallel for schedule(static)
  for (int i = 0; i < (bins_per_profile * n_active_profiles); ++i)
    output[i] = real_t(0);

// -------------------------------------
// Parallel region
// -------------------------------------
#pragma omp parallel
  {
// ---------------------------------
// Particle loop
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
#pragma omp atomic // the array is big and it is rare that the same index is
                   // written
        output[bucket_index_to_memory_index[bucket_i] + bins_per_profile - 1] +=
            1;
        continue;
      }
        if (dt < cut_left || dt >= cut_right)
            continue;

      // Calculate the bin index
      const int bin = (int)((dt - cut_left) * inv_bin_width);
      if ((unsigned)bin < (unsigned)bins_per_profile) {
#pragma omp atomic // the array is big and it is rare that the same index is
                   // written
        output[bucket_index_to_memory_index[bucket_i] + bin] += 1;
      }
    }
  }
}
