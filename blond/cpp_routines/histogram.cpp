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
#include "openmp.h"


extern "C" void histogram(const double *__restrict__ input,
                          double *__restrict__ output, const double cut_left,
                          const double cut_right, const int n_slices,
                          const int n_macroparticles)
{
    // Number of Iterations of the inner loop
    const int STEP = 16;
    const double inv_bin_width = n_slices / (cut_right - cut_left);

    // allocate memory for the thread_private histogram
    double **histo = (double **) malloc(omp_get_max_threads() * sizeof(double *));
    histo[0] = (double *) malloc (omp_get_max_threads() * n_slices * sizeof(double));
    for (int i = 0; i < omp_get_max_threads(); i++)
        histo[i] = (*histo + n_slices * i);

    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        memset(histo[id], 0., n_slices * sizeof(double));
        float fbin[STEP];
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

extern "C" void smooth_histogram(const double *__restrict__ input,
                                 double *__restrict__ output, const double cut_left,
                                 const double cut_right, const int n_slices,
                                 const int n_macroparticles)
{
    // Constants init
    const double inv_bin_width = n_slices / (cut_right - cut_left);
    const double bin_width = (cut_right - cut_left) / n_slices;
    const double const1 = (cut_left + bin_width * 0.5);
    const double const2 = (cut_right - bin_width * 0.5);

    // memory alloc for per thread histo
    double **histo = (double **) malloc(omp_get_max_threads() * sizeof(double *));
    histo[0] = (double *) malloc (omp_get_max_threads() * n_slices * sizeof(double));
    for (int i = 0; i < omp_get_max_threads(); i++)
        histo[i] = (*histo + n_slices * i);


    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        memset(histo[id], 0., n_slices * sizeof(double));

        // main caclulation
        #pragma omp for
        for (int i = 0; i < n_macroparticles; i++) {
            int fffbin = 0;
            double a = input[i];
            if ((a < const1) || (a > const2))
                continue;
            double fbin = (a - cut_left) * inv_bin_width;
            int ffbin = (int)(fbin);
            double distToCenter = fbin - (double)(ffbin);
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


extern "C" void histogramf(const float *__restrict__ input,
                           float *__restrict__ output, const float cut_left,
                           const float cut_right, const int n_slices,
                           const int n_macroparticles)
{
    // Number of Iterations of the inner loop
    const int STEP = 16;
    const float inv_bin_width = n_slices / (cut_right - cut_left);

    // allocate memory for the thread_private histogram
    static float **histo = nullptr;

    if (!histo) {
        histo = (float **) malloc(omp_get_max_threads() * sizeof(float *));
        histo[0] = (float *) malloc (omp_get_max_threads() * n_slices * sizeof(float));
        for (int i = 0; i < omp_get_max_threads(); i++)
            histo[i] = (*histo + n_slices * i);
    }

    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        memset(histo[id], 0., n_slices * sizeof(float));
        float fbin[STEP];
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
    // free(histo[0]);
    // free(histo);
}


extern "C" void smooth_histogramf(const float *__restrict__ input,
                                  float *__restrict__ output, const float cut_left,
                                  const float cut_right, const int n_slices,
                                  const int n_macroparticles)
{
    // Constants init
    const float inv_bin_width = n_slices / (cut_right - cut_left);
    const float bin_width = (cut_right - cut_left) / n_slices;
    const float const1 = (cut_left + bin_width * 0.5);
    const float const2 = (cut_right - bin_width * 0.5);

    // memory alloc for per thread histo
    float **histo = (float **) malloc(omp_get_max_threads() * sizeof(float *));
    histo[0] = (float *) malloc (omp_get_max_threads() * n_slices * sizeof(float));
    for (int i = 0; i < omp_get_max_threads(); i++)
        histo[i] = (*histo + n_slices * i);


    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        memset(histo[id], 0., n_slices * sizeof(float));

        // main caclulation
        #pragma omp for
        for (int i = 0; i < n_macroparticles; i++) {
            int fffbin = 0;
            float a = input[i];
            if ((a < const1) || (a > const2))
                continue;
            float fbin = (a - cut_left) * inv_bin_width;
            int ffbin = (int)(fbin);
            float distToCenter = fbin - (float)(ffbin);
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
