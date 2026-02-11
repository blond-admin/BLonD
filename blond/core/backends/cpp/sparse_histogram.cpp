// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENCE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Optimised C++ routine that calculates the histogram for a sparse beam
// with STRIDED memory layout (empty space between profiles)
// Author: Juan F. Esteban Mueller, Danilo Quartullo, Alexandre Lasheen, Markus
// Schwarz Modified for strided layout: 2026-02-07

#include <math.h>
#include <stdio.h>
#include <stdlib.h> // mmalloc()
#include <string.h> // memset()

#include "blond_common.h"
#include "openmp.h"

extern "C" void sparse_histogram(const real_t *__restrict__ input,
                                 real_t *__restrict__ output,
                                 const real_t *__restrict__ left_cuts,
                                 const real_t *__restrict__ right_cuts,
                                 const int *__restrict__ bins_per_profile,
                                 const int *__restrict__ start_indices,

                                 const int n_profiles, const int total_bins,
                                 const int n_macroparticles) {

  // Persistent storage (int, compact layout without stride gaps)
  static int *histo_all = nullptr;
  static int histo_threads = 0;
  static int histo_local = 0;

  const int max_threads = omp_get_max_threads();

  if (histo_all == nullptr || histo_threads < max_threads ||
      histo_local < total_bins) {
    free(histo_all);

    histo_threads = max_threads;
    histo_local = total_bins;

    histo_all = (int *)malloc(sizeof(int) * histo_threads * histo_local);

    if (!histo_all)
      return;
  }

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int threads = omp_get_num_threads();
    int *histo = histo_all + tid * histo_local;

    memset(histo, 0, histo_local * sizeof(int));

#pragma omp for schedule(static)
    for (int i = 0; i < n_macroparticles; ++i) {
      const real_t a = input[i];

      int hist_i = -1;
      real_t cut_left;
      real_t cut_right;
      for (int j = 0; j < n_profiles; j++) {
        cut_right = right_cuts[j];
        if (a <= cut_right) {
          cut_left = left_cuts[j];
          hist_i = j;
          break; // stop at first match
        }
      }
      if (hist_i == -1)
        continue;
      // no interval found
      if (a < cut_left)
        continue;

      if (a == cut_right) {
        histo[start_indices[hist_i] + bins_per_profile[hist_i] - 1] += 1;
        continue;
      }
      const real_t inv_bin_width =
          bins_per_profile[hist_i] / (cut_right - cut_left);

      const int bin = (int)((a - cut_left) * inv_bin_width);
      histo[start_indices[hist_i] + bin] += 1;
    }

// Reduce compact histogram into output (same compact indexing)
#pragma omp for schedule(static)
    for (int p = 0; p < n_profiles; ++p) {

      const int base = start_indices[p];
      const int nb = bins_per_profile[p];

      for (int b = 0; b < nb; ++b) {
        int sum = 0;

#pragma omp simd reduction(+ : sum)
        for (int t = 0; t < threads; ++t)
          sum += histo_all[t * histo_local + base + b];

        output[base + b] = static_cast<real_t>(sum);
      }
    }
  }
}
