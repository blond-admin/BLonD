// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENCE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

#include "blond_common.h" // must define real_t #include <cstdlib> #include <cstring> #include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <omp.h>

extern "C" void sparse_histogram(const real_t *__restrict__ input,
                                 real_t *__restrict__ output,
                                 const real_t *__restrict__ left_cuts,
                                 const real_t *__restrict__ right_cuts,
                                 const int *__restrict__ bins_per_profile,
                                 const int *__restrict__ start_indices,
                                 const int n_profiles, const int total_bins,
                                 const int n_macroparticles) {

  // Precompute per-profile constants
  // Allocate small arrays on heap if n_profiles can be large
  real_t *inv_bin_width = (real_t *)malloc(sizeof(real_t) * n_profiles);
  int *last_bin = (int *)malloc(sizeof(int) * n_profiles);

  for (int p = 0; p < n_profiles; ++p) {
    last_bin[p] = bins_per_profile[p] - 1;
    const real_t width = right_cuts[p] - left_cuts[p];
    inv_bin_width[p] =
        (width > 0) ? (static_cast<real_t>(bins_per_profile[p]) / width) : 0.0;
  }

  const int max_threads = omp_get_max_threads();

  static int *histo_all = nullptr;
  static int histo_threads = 0;
  static int histo_local = 0;

  if (histo_all == nullptr || histo_threads < max_threads ||
      histo_local < total_bins) {
    free(histo_all);
    histo_threads = max_threads;
    histo_local = total_bins;
    // align to 64 bytes to reduce cache-line issues
    posix_memalign((void **)&histo_all, 64,
                   sizeof(int) * histo_threads * histo_local);
    if (!histo_all) {
      free(inv_bin_width);
      free(last_bin);
      return;
    }
  }

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int threads = omp_get_num_threads();
    int *histo = histo_all + tid * histo_local;
    memset(histo, 0, histo_local * sizeof(int));

    // Particle loop: use binary search on right_cuts (assumed sorted ascending)
#pragma omp for schedule(static)
    for (int i = 0; i < n_macroparticles; ++i) {
      const real_t a = input[i];

      // find first right_cut >= a  -> upper_bound gives first > a, so use
      // lower_bound on right_cuts
      int lo = 0, hi = n_profiles;
      while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (right_cuts[mid] < a)
          lo = mid + 1;
        else
          hi = mid;
      }
      if (lo == n_profiles)
        continue; // a > all right_cuts
      const int p = lo;
      const real_t left = left_cuts[p];
      if (a < left)
        continue; // outside left bound

      if (a == right_cuts[p]) {
        histo[start_indices[p] + last_bin[p]] += 1;
      } else {
        // compute bin index using precomputed inv_bin_width
        int bin = static_cast<int>((a - left) * inv_bin_width[p]);
        // clamp safety
        if (bin < 0)
          bin = 0;
        else if (bin > last_bin[p])
          bin = last_bin[p];
        histo[start_indices[p] + bin] += 1;
      }
    }

    // Reduction: iterate profiles and bins; sum across threads for each bin
#pragma omp for schedule(static)
    for (int p = 0; p < n_profiles; ++p) {
      const int base = start_indices[p];
      const int nb = bins_per_profile[p];
      for (int b = 0; b < nb; ++b) {
        int sum = 0;
        // manual unroll could help; keep simple for portability
        for (int t = 0; t < threads; ++t)
          sum += histo_all[t * histo_local + base + b];
        output[base + b] = static_cast<real_t>(sum);
      }
    }
  } // omp parallel

  free(inv_bin_width);
  free(last_bin);
}
