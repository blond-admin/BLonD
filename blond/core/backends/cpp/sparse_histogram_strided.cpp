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

extern "C" void sparse_histogram_strided(
    const real_t *__restrict__ input,
    real_t *__restrict__ output,
    const real_t *__restrict__ cut_left_array,
    const real_t *__restrict__ cut_right_array,
    const real_t *__restrict__ bunch_indexes,
    const int n_slices_bucket,
    const int n_filled_buckets,
    const int n_macroparticles,
    const int stride)
{
    const real_t cut_left0 = cut_left_array[0];
    const real_t inv_hist_dist = real_t(1) / (cut_left_array[1] - cut_left0);
    const real_t inv_bin_width =
        real_t(n_slices_bucket) / (cut_right_array[0] - cut_left0);

    const int compact_size = n_filled_buckets * n_slices_bucket;

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
            if ((unsigned)hist_i >= (unsigned)n_filled_buckets)
                continue;

            const real_t cut_left = cut_left_array[hist_i];
            if (a < cut_left || a >= cut_right_array[hist_i])
                continue;

            const int bin = (int)((a - cut_left) * inv_bin_width);
            if ((unsigned)bin < (unsigned)n_slices_bucket)
                histo[hist_i * n_slices_bucket + bin] += 1;
        }

        // Reduce compact histogram into strided output
#pragma omp for schedule(static)
        for (int p = 0; p < n_filled_buckets; ++p) {
            const int out_base = p * stride;
            const int compact_base = p * n_slices_bucket;

            for (int b = 0; b < n_slices_bucket; ++b) {
                int sum = 0;
                #pragma omp simd reduction(+:sum)
                for (int t = 0; t < threads; ++t)
                    sum += histo_all[t * histo_compact + compact_base + b];
                output[out_base + b] = (real_t)sum;
            }

            // Zero the gap region
            memset(&output[out_base + n_slices_bucket], 0,
                   (stride - n_slices_bucket) * sizeof(real_t));
        }
    }
}
