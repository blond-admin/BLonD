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

#include <omp.h>        // omp_get_thread_num(), omp_get_num_threads()
#include <string.h>     // memset()

const int MAX_SLICES = 100000;
const int MAX_THREADS = 56;
static double hist[MAX_THREADS][MAX_SLICES];

extern "C" void histogram(const double *__restrict__ input,
                          double *__restrict__ output, const double cut_left,
                          const double cut_right, const int n_slices,
                          const int n_macroparticles)
{

    const double inv_bin_width = n_slices / (cut_right - cut_left);
    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        memset(hist[id], 0., n_slices * sizeof(double));

        #pragma omp for
        for (int i = 0; i < n_macroparticles; ++i) {
            if (input[i] < cut_left || input[i] > cut_right) continue;
            hist[id][(int)((input[i] - cut_left)*inv_bin_width)] += 1.;
        }

        #pragma omp for
        for (int i = 0; i < n_slices; i++) {
            output[i] = 0.;
            for (int t = 0; t < threads; t++)
                output[i] += hist[t][i];
        }
    }
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
