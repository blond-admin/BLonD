/*
 Copyright 2016 CERN. This software is distributed under the
 terms of the GNU General Public Licence version 3 (GPL Version 3),
 copied verbatim in the file LICENCE.md.
 In applying this licence, CERN does not waive the privileges and immunities
 granted to it by virtue of its status as an Intergovernmental Organization or
 submit itself to any jurisdiction.
 Project website: http://blond.web.cern.ch/
 */

// Optimised C++ routine that calculates the histogram
// Author: Danilo Quartullo, Alexandre Lasheen, Konstantinos Iliakis

#include <string.h>     // memset()
#include <stdlib.h>     // mmalloc()
#include <math.h>

#include "blond_common.h"
#include "openmp.h"


extern "C" void histogram(const real_t *__restrict__ input,
                          real_t *__restrict__ output, const real_t cut_left,
                          const real_t cut_right, const int n_slices,
                          const int n_macroparticles)
{
    // Number of Iterations of the inner loop
    const int STEP = 16;
    const real_t inv_bin_width = n_slices / (cut_right - cut_left);

    // allocate memory for the thread_private histogram
    int **histo = (int **) malloc(omp_get_max_threads() * sizeof(int *));
    histo[0] = (int *) malloc (omp_get_max_threads() * n_slices * sizeof(int));
    for (int i = 0; i < omp_get_max_threads(); i++)
        histo[i] = (*histo + n_slices * i);

    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        memset(histo[id], 0, n_slices * sizeof(int));
        float fbin[STEP] = { -1 };
        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;

            // First calculate the index to update
            for (int j = 0; j < loop_count; j++) {
                fbin[j] = floor((input[i + j] - cut_left) * inv_bin_width);
            }
            // Then update the corresponding bins
            for (int j = 0; j < loop_count; j++) {
                const int bin  = (int) fbin[j];
                if (bin < 0 || bin >= n_slices) continue;
                histo[id][bin] += 1.;
            }
        }

        // Reduce to a single histogram
        #pragma omp for
        for (int i = 0; i < n_slices; i++) {
            output[i] = 0.;
            for (int t = 0; t < threads; t++)
                output[i] += histo[t][i];
        }
    }

    // free memory
    free(histo[0]);
    free(histo);
}

extern "C" void smooth_histogram(const real_t *__restrict__ input,
                                 real_t *__restrict__ output, const real_t cut_left,
                                 const real_t cut_right, const int n_slices,
                                 const int n_macroparticles)
{
    // Constants init
    const real_t inv_bin_width = n_slices / (cut_right - cut_left);
    const real_t bin_width = (cut_right - cut_left) / n_slices;
    const real_t const1 = (cut_left + bin_width * 0.5);
    const real_t const2 = (cut_right - bin_width * 0.5);

    // memory alloc for per thread histo
    real_t **histo = (real_t **) malloc(omp_get_max_threads() * sizeof(real_t *));
    histo[0] = (real_t *) malloc (omp_get_max_threads() * n_slices * sizeof(real_t));
    for (int i = 0; i < omp_get_max_threads(); i++)
        histo[i] = (*histo + n_slices * i);


    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        memset(histo[id], 0, n_slices * sizeof(real_t));

        // main caclulation
        #pragma omp for
        for (int i = 0; i < n_macroparticles; i++) {
            int fffbin = 0;
            real_t a = input[i];
            if ((a < const1) || (a > const2))
                continue;
            real_t fbin = (a - cut_left) * inv_bin_width;
            int ffbin = (int)(fbin);
            real_t distToCenter = fbin - (real_t)(ffbin);
            if (distToCenter > 0.5)
                fffbin = (int)(fbin + 1.0);
            else
                fffbin = (int)(fbin - 1.0);

            histo[id][ffbin] = histo[id][ffbin] + 0.5 - distToCenter;
            histo[id][fffbin] = histo[id][fffbin] + 0.5 + distToCenter;
        }

        // Reduce to a single histogram
        #pragma omp for
        for (int i = 0; i < n_slices; i++) {
            output[i] = 0.;
            for (int t = 0; t < threads; t++)
                output[i] += histo[t][i];
        }


    }
    // free memory
    free(histo[0]);
    free(histo);

}


/***** serial histogram

extern "C" void histogram(const double *__restrict__ input,
                          double *__restrict__ output,
                          const double cut_left, const double cut_right,
                          const int n_slices, const int n_macroparticles)
{
    // Number of Iterations of the inner loop
    const int STEP = 16;
    const double inv_bin_width = n_slices / (cut_right - cut_left);
    float fbin[STEP];

    memset(output, 0., n_slices * sizeof(double));
    for (int i = 0; i < n_macroparticles; i += STEP) {

        const int loop_count = n_macroparticles - i > STEP ?
                               STEP : n_macroparticles - i;

        // First calculate the index to update
        for (int j = 0; j < loop_count; j++) {
            fbin[j] = floor((input[i + j] - cut_left) * inv_bin_width);
        }
        // Then update the corresponding bins
        for (int j = 0; j < loop_count; j++) {
            const int bin  = (int) fbin[j];
            if (bin < 0 || bin >= n_slices) continue;
            output[bin] += 1.;
        }
    }

}

*******/
