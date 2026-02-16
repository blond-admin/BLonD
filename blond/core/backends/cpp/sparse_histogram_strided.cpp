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
    const int n_buckets,
    const int n_macroparticles,
    const bool *__restrict__ filling_pattern,
    const int *__restrict__ bucket_index_to_memory_index)
{
    const real_t cut_left0     = first_left_cut;
    const real_t inv_hist_dist = real_t(1) / left_cut_distance;
    const real_t inv_bin_width = real_t(bins_per_profile) / cut_width;

    const int compact_size = n_profiles * bins_per_profile;

    // -------------------------------------
    // Zero global output
    // -------------------------------------
#pragma omp parallel for schedule(static)
    for (int i = 0; i < compact_size; ++i)
        output[i] = real_t(0);

    // -------------------------------------
    // Parallel region
    // -------------------------------------
#pragma omp parallel
    {
        // Thread-local persistent buffer
        static thread_local real_t* local_output = nullptr;
        static thread_local int     local_size   = 0;

        // Allocate only if size changed or first use
        if (local_size < compact_size)
        {
            free(local_output);
            local_output = (real_t*) aligned_alloc(64, sizeof(real_t) * compact_size);
            local_size   = compact_size;
        }

        // Clear only used portion
        memset(local_output, 0, sizeof(real_t) * compact_size);

        // ---------------------------------
        // Particle loop
        // ---------------------------------
#pragma omp for schedule(static)
        for (int i = 0; i < n_macroparticles; ++i)
        {
            const real_t shifted = input[i] - cut_left0;

            const int bucket_i = (int)(shifted * inv_hist_dist);
            if ((unsigned)bucket_i >= (unsigned)n_buckets)
                continue;

            if (!filling_pattern[bucket_i])
                continue;

            const real_t local =
                shifted - bucket_i * left_cut_distance;

            if (local >= cut_width)
                continue;

            const int bin = (int)(local * inv_bin_width);
            if ((unsigned)bin >= (unsigned)bins_per_profile)
                continue;

            const int base  = bucket_index_to_memory_index[bucket_i];
            const int index = base + bin;

            local_output[index] += real_t(1);
        }

        // ---------------------------------
        // Reduction step
        // ---------------------------------
#pragma omp for schedule(static) reduction(+:output[:compact_size])
        for (int i = 0; i < compact_size; ++i)
        {
            output[i] += local_output[i];
        }
    }
}
