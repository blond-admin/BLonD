// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENCE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Optimised C++ routine that calculates the histogram for a sparse beam
// with STRIDED memory layout (empty space between profiles)
// Author: Juan F. Esteban Mueller, Danilo Quartullo, Alexandre Lasheen, Markus Schwarz
// Modified for strided layout: 2026-02-07

#include <stdio.h>
#include <string.h>     // memset()
#include <stdlib.h>     // mmalloc()
#include <math.h>

#include "blond_common.h"
#include "openmp.h"

extern "C" void sparse_histogram_strided(
    const real_t *__restrict__ input,
    real_t *__restrict__ output,
    const real_t first_left_cut,
    const real_t left_cut_distance,
    const real_t cut_width,
    const int bins_per_profile,
    const int n_profiles,
    const int n_macroparticles,
    const int stride)
{
    const real_t cut_left0 = first_left_cut;
    const real_t inv_hist_dist = real_t(1) / (left_cut_distance);
    const real_t inv_bin_width =
        real_t(bins_per_profile) / (cut_width);

    const int compact_size = n_profiles * bins_per_profile;

    // Persistent storage (int, compact layout without stride gaps)
    static int *histo_all = nullptr;
    static int histo_threads = 0;
    static int histo_compact = 0;

    const int max_threads = omp_get_max_threads();

    if (histo_all == nullptr || histo_threads < max_threads || histo_compact < compact_size) {
        free(histo_all);

        histo_threads = max_threads;
        histo_compact = compact_size;

        histo_all = (int *)malloc(
            sizeof(int) * histo_threads * histo_compact);

        if (!histo_all)
            return;
    }

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        int *histo = histo_all + tid * histo_compact;

        memset(histo, 0, histo_compact * sizeof(int));

#pragma omp for schedule(static)
        for (int i = 0; i < n_macroparticles; ++i) {
            const real_t a = input[i];

            const int hist_i = (int)((a - cut_left0) * inv_hist_dist);
            if ((unsigned)hist_i >= (unsigned)n_profiles)
                continue;

            const real_t cut_left = cut_left0 + hist_i * left_cut_distance;
            const real_t cut_right = cut_left + cut_width;
            if (a == cut_right){
                histo[hist_i * bins_per_profile + bins_per_profile - 1] += 1;
                continue;
            }
            if (a < cut_left || a >= cut_right)
                continue;

            const int bin = (int)((a - cut_left) * inv_bin_width);
            if ((unsigned)bin < (unsigned)bins_per_profile)
                histo[hist_i * bins_per_profile + bin] += 1;
        }

        // Reduce compact histogram into strided output
#pragma omp for schedule(static)
        for (int p = 0; p < n_profiles; ++p) {
            const int out_base = p * stride;
            const int compact_base = p * bins_per_profile;

            for (int b = 0; b < bins_per_profile; ++b) {
                int sum = 0;
                #pragma omp simd reduction(+:sum)
                for (int t = 0; t < threads; ++t)
                    sum += histo_all[t * histo_compact + compact_base + b];
                output[out_base + b] = (real_t)sum;
            }

            // Zero the gap region
            memset(&output[out_base + bins_per_profile], 0,
                   (stride - bins_per_profile) * sizeof(real_t));
        }
    }
}
