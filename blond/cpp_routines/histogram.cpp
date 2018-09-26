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

#ifdef PARALLEL
#include <omp.h>        // omp_get_thread_num(), omp_get_num_threads()
#else
int omp_get_max_threads() {return 1;}
int omp_get_num_threads() {return 1;}
int omp_get_thread_num() {return 0;}
#endif

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

    int i;
    double a;
    double fbin;
    double ratioffbin;
    double ratiofffbin;
    double distToCenter;
    int ffbin;
    int fffbin;
    const double inv_bin_width = n_slices / (cut_right - cut_left);
    const double bin_width = (cut_right - cut_left) / n_slices;

    for (i = 0; i < n_slices; i++) {
        output[i] = 0.0;
    }

    for (i = 0; i < n_macroparticles; i++) {
        a = input[i];
        if ((a < (cut_left + bin_width * 0.5))
                || (a > (cut_right - bin_width * 0.5)))
            continue;
        fbin = (a - cut_left) * inv_bin_width;
        ffbin = (int)(fbin);
        distToCenter = fbin - (double)(ffbin);
        if (distToCenter > 0.5)
            fffbin = (int)(fbin + 1.0);
        ratioffbin = 1.5 - distToCenter;
        ratiofffbin = 1 - ratioffbin;
        if (distToCenter < 0.5)
            fffbin = (int)(fbin - 1.0);
        ratioffbin = 0.5 - distToCenter;
        ratiofffbin = 1 - ratioffbin;
        output[ffbin] = output[ffbin] + ratioffbin;
        output[fffbin] = output[fffbin] + ratiofffbin;
    }
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