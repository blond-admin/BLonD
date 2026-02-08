/*
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3),
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities
granted to it by virtue of its status as an Intergovernmental Organization or
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/

// Optimised C++ routine that calculates the histogram for a sparse beam
// with STRIDED memory layout (empty space between profiles)
// Author: Juan F. Esteban Mueller, Danilo Quartullo, Alexandre Lasheen, Markus Schwarz
// Modified for strided layout: 2026-02-07

#include <stdio.h>
#include <string.h>     // memset()
#include <stdlib.h>     // mmalloc()
#include <math.h>
#include <vector>

#include "blond_common.h"
#include "openmp.h"

extern "C" void sparse_histogram_strided(const real_t * __restrict__ input,
                real_t * __restrict__ output,
               const real_t * __restrict__ cut_left_array,
               const real_t * __restrict__ cut_right_array,
               const real_t * __restrict__ bunch_indexes,
               const int n_slices_bucket,
               const int n_filled_buckets,
               const int n_macroparticles,
               const int stride){
  // Constants init
  const real_t inv_hist_dist = 1 / (cut_left_array[1] - cut_left_array[0]);
  const real_t inv_bin_width = n_slices_bucket / (cut_right_array[0] - cut_left_array[0]);
  const int total = n_filled_buckets * stride;
  // memory alloc for per thread histo
  real_t **histo = (real_t **)malloc(omp_get_max_threads() * sizeof(real_t *));
  histo[0] =
      (real_t *)malloc(omp_get_max_threads() * total * sizeof(real_t));
  for (int i = 0; i < omp_get_max_threads(); i++)
    histo[i] = (*histo + total * i);

#pragma omp parallel
  {
    const int id = omp_get_thread_num();
    const int threads = omp_get_num_threads();
    memset(histo[id], 0, total * sizeof(real_t));

// main calculation
#pragma omp for
    for (int i = 0; i < n_macroparticles; i++) {
      real_t a = input[i];
      int hist_i = (int)((a - cut_left_array[0]) * inv_hist_dist);

      real_t cut_left_loc = (cut_left_array[hist_i]);
      real_t cut_right_loc = (cut_right_array[hist_i]);
      if ((a < cut_left_loc) || (a > cut_right_loc))
        continue;
      int bin_idx_loc = (a - cut_left_array[hist_i]) * inv_bin_width;
      if (bin_idx_loc <n_slices_bucket){
          histo[id][hist_i * stride + bin_idx_loc] += 1;
      }
    }

// Reduce to a single histogram
#pragma omp for
    for (int i = 0; i < total; i++) {
      output[i] = 0.;
      for (int t = 0; t < threads; t++)
        output[i] += histo[t][i];
    }
  }
  // free memory
  free(histo[0]);
  free(histo);
}
