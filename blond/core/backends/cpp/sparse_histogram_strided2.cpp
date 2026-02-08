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
    /**
     * Sparse histogram with strided memory layout.
     *
     * This function fills histograms for multiple non-contiguous buckets,
     * with a specified stride between consecutive profiles. This allows
     * for empty space between profiles for extensions or smooth transitions.
     *
     * Memory layout example with stride = 2 * n_slices_bucket:
     *   Profile 0: output[0:n_slices_bucket]
     *   Empty:     output[n_slices_bucket:2*n_slices_bucket]
     *   Profile 1: output[2*n_slices_bucket:3*n_slices_bucket]
     *   Empty:     output[3*n_slices_bucket:4*n_slices_bucket]
     *   ...
     *
     * @param input: Particle dt values (n_macroparticles)
     * @param output: Output histogram array (n_filled_buckets * stride)
     * @param cut_left_array: Left edge of each bucket (n_filled_buckets)
     * @param cut_right_array: Right edge of each bucket (n_filled_buckets)
     * @param bunch_indexes: Mapping from bucket to profile index, -1 if empty (variable size)
     * @param n_slices_bucket: Number of bins per bucket/profile
     * @param n_filled_buckets: Number of non-empty buckets
     * @param n_macroparticles: Number of particles
     * @param stride: Memory stride between consecutive profiles (typically 2*n_slices_bucket)
     */

    int j;
    int i_bucket;
    real_t a;
    real_t fbin;
    real_t fbunch;
    int ffbin;
    int ffbunch;

    // Only valid for cut_edges = edges
    const real_t inv_bucket_length = 1.0 / (cut_right_array[0] - cut_left_array[0]);
    const real_t inv_bin_width = inv_bucket_length * (real_t) n_slices_bucket;

    // Total histogram size with stride
    const int total_bins = n_filled_buckets * stride;

    // ====================================================================
    // OPTIMIZATION 1: Static storage for thread-local histograms
    // Avoids repeated allocation/deallocation overhead (~15MB per call)
    // ====================================================================
    static std::vector<std::vector<real_t>> thread_local_histograms;
    static int cached_max_threads = 0;
    static int cached_total_bins = 0;
    static bool storage_initialized = false;

    const int num_threads = omp_get_max_threads();

    // Only allocate if size changed or first call
    if (!storage_initialized || cached_max_threads != num_threads || cached_total_bins != total_bins) {
        thread_local_histograms.clear();
        thread_local_histograms.resize(num_threads);
        for (int i = 0; i < num_threads; i++) {
            thread_local_histograms[i].resize(total_bins);
        }
        cached_max_threads = num_threads;
        cached_total_bins = total_bins;
        storage_initialized = true;
    }

    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();

        // Zero-initialize thread-local histogram
        // Use memset for faster clearing (optimized by compiler)
        memset(thread_local_histograms[id].data(), 0, total_bins * sizeof(real_t));

        // ====================================================================
        // Histogram filling loop
        // ====================================================================
        #pragma omp for schedule(static)
        for (j = 0; j < n_macroparticles; j++){
            a = input[j];   // Particle dt
            if ((a < cut_left_array[0]) || (a > cut_right_array[n_filled_buckets-1]))
                continue;

            // Find bucket in which the particle is and its index
            fbunch = (a - cut_left_array[0]) * inv_bucket_length;
            ffbunch = (int) fbunch;

            // CRITICAL FIX: Bounds check on bucket index before array access
            // Prevents buffer overrun when particle is at boundary or calculation overflow
            if (ffbunch < 0 || ffbunch >= n_filled_buckets)
                continue;

            i_bucket = (int) bunch_indexes[ffbunch];

            // Check if bucket is empty or invalid
            if (i_bucket == -1 || i_bucket < 0 || i_bucket >= n_filled_buckets)
                continue;

            // Find the bin inside the corresponding bucket
            fbin = (a - cut_left_array[i_bucket]) * inv_bin_width;

            // KEY DIFFERENCE: Use stride instead of n_slices_bucket
            // This allows for gaps between profiles
            ffbin = i_bucket * stride + (int) fbin;

            // CRITICAL FIX: Bounds check before array access
            // Prevents buffer overrun when bin calculation gives negative or out-of-range values
            // Also ensure we're writing within the active region of the profile (not in the gap)
            if (ffbin >= 0 && ffbin < total_bins) {
                // Additional check: make sure we're not in a gap region
                // The bin should be within [i_bucket*stride, i_bucket*stride+n_slices_bucket)
                const int profile_start = i_bucket * stride;
                const int profile_end = profile_start + n_slices_bucket;
                if (ffbin >= profile_start && ffbin < profile_end) {
                    thread_local_histograms[id][ffbin] += 1.0;
                }
            }
        }

        // ====================================================================
        // OPTIMIZATION 2: Cache-friendly reduction
        // Each thread processes a chunk of bins, improving cache locality
        // ====================================================================
        #pragma omp for schedule(static)
        for (int i = 0; i < total_bins; i++) {
            real_t sum = 0.0;

            // Vectorization hint: this inner loop can vectorize
            #pragma omp simd reduction(+:sum)
            for (int t = 0; t < threads; t++) {
                sum += thread_local_histograms[t][i];
            }

            output[i] = sum;
        }
    }

    // Note: Static storage persists between calls (intentional for performance)
}
