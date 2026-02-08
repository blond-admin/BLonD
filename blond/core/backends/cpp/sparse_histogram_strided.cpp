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

    const int total = n_filled_buckets * stride;

    // Persistent storage
    static real_t *histo_all = nullptr;
    static int histo_threads = 0;
    static int histo_size = 0;

    const int max_threads = omp_get_max_threads();

    if (histo_all == nullptr || histo_threads < max_threads || histo_size < total) {
        free(histo_all);

        histo_threads = max_threads;
        histo_size = total;

        histo_all = (real_t *)malloc(
            sizeof(real_t) * histo_threads * histo_size);

        if (!histo_all)
            return;
    }

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        real_t *histo = histo_all + tid * histo_size;

        // zero local histogram
        memset(histo, 0, histo_size * sizeof(real_t));

#pragma omp for schedule(static)
        for (int i = 0; i < n_macroparticles; ++i) {
            const real_t a = input[i];

            const int hist_i = (int)((a - cut_left0) * inv_hist_dist);
            if ((unsigned)hist_i >= (unsigned)n_filled_buckets)
                continue;

            const real_t cl = cut_left_array[hist_i];
            if (a < cl || a >= cut_right_array[hist_i])
                continue;

            const int bin = (int)((a - cl) * inv_bin_width);
            if ((unsigned)bin < (unsigned)n_slices_bucket)
                histo[hist_i * stride + bin] += real_t(1);
        }

#pragma omp for schedule(static)
        for (int i = 0; i < total; ++i) {
            real_t sum = 0;
            for (int t = 0; t < omp_get_num_threads(); ++t)
                sum += histo_all[t * histo_size + i];
            output[i] = sum;
        }
    }
}
