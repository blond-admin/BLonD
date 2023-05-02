/*
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3),
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities
granted to it by virtue of its status as an Intergovernmental Organization or
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/

// Authors: Juan F. Esteban Mueller, Alexandre Lasheen, D. Quartullo, K. Iliakis

// Optimised C++ routine that calculates the kick of a voltage array on particles

#include <stdlib.h>
#include <math.h>
#include <cmath>


extern "C" void linear_interp_kick(double * __restrict__ beam_dt,
                                   double * __restrict__ beam_dE,
                                   const double * __restrict__ voltage_array,
                                   const double * __restrict__ bin_centers,
                                   const double charge,
                                   const int n_slices,
                                   const int n_macroparticles,
                                   const double acc_kick)
{


    const int STEP = 64;
    const double inv_bin_width = (n_slices - 1)
                                 / (bin_centers[n_slices - 1]
                                    - bin_centers[0]);

    double *voltageKick = (double *) malloc ((n_slices - 1) * sizeof(double));
    double *factor = (double *) malloc ((n_slices - 1) * sizeof(double));

    #pragma omp parallel
    {
        unsigned fbin[STEP];

        #pragma omp for
        for (int i = 0; i < n_slices - 1; i++) {
            voltageKick[i] =  charge * (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
            factor[i] = (charge * voltage_array[i] - bin_centers[i] * voltageKick[i]) + acc_kick;
        }

        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;

            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (unsigned) std::floor((beam_dt[i + j] - bin_centers[0])
                                                * inv_bin_width);
            }

            for (int j = 0; j < loop_count; j++) {
                if (fbin[j] < (unsigned) (n_slices - 1)) {
                    beam_dE[i + j] += beam_dt[i + j] * voltageKick[fbin[j]] + factor[fbin[j]];
                }
            }

        }
    }
    free(voltageKick);
    free(factor);
}

// Optimised C++ routine that interpolates the induced voltage
// assuming constant slice width and a shift of the time array by a constant.
// Only right extrapolation is assumed; it gives zero values.
// This routine contributes to the computation of multi-turn wake with acceleration
extern "C" void linear_interp_time_translation(
    double * __restrict__ xp,
    double * __restrict__ yp,
    double * __restrict__ x,
    double * __restrict__ y,
    const int len_xp) {

    const double inv_bin_width = (len_xp - 1) / (xp[len_xp - 1] - xp[0]);

    const int ffbin0 = (int)((x[0] - xp[0]) * inv_bin_width);
    const int diff = len_xp - ffbin0;

    #pragma omp parallel for
    for (int i = 0; i < diff - 1; i++) {
        int ffbin;
        ffbin = ffbin0 + i;
        y[i] = yp[ffbin] + (x[i] - xp[ffbin]) * (yp[ffbin + 1] - yp[ffbin]) * inv_bin_width;
    }

}


extern "C" void linear_interp_kickf(float * __restrict__ beam_dt,
                                    float * __restrict__ beam_dE,
                                    const float * __restrict__ voltage_array,
                                    const float * __restrict__ bin_centers,
                                    const float charge,
                                    const int n_slices,
                                    const int n_macroparticles,
                                    const float acc_kick)
{


    const int STEP = 64;
    const float inv_bin_width = (n_slices - 1)
                                / (bin_centers[n_slices - 1]
                                   - bin_centers[0]);

    float *voltageKick = (float *) malloc ((n_slices - 1) * sizeof(float));
    float *factor = (float *) malloc ((n_slices - 1) * sizeof(float));

    #pragma omp parallel
    {
        unsigned fbin[STEP];

        #pragma omp for
        for (int i = 0; i < n_slices - 1; i++) {
            voltageKick[i] =  charge * (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
            factor[i] = (charge * voltage_array[i] - bin_centers[i] * voltageKick[i]) + acc_kick;
        }

        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;

            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (unsigned) std::floor((beam_dt[i + j] - bin_centers[0])
                                                * inv_bin_width);
            }

            for (int j = 0; j < loop_count; j++) {
                if (fbin[j] < (unsigned) (n_slices - 1)) {
                    beam_dE[i + j] += beam_dt[i + j] * voltageKick[fbin[j]] + factor[fbin[j]];
                }
            }

        }
    }
    free(voltageKick);
    free(factor);
}

// Optimised C++ routine that interpolates the induced voltage
// assuming constant slice width and a shift of the time array by a constant.
// Only right extrapolation is assumed; it gives zero values.
// This routine contributes to the computation of multi-turn wake with acceleration
extern "C" void linear_interp_time_translationf(
    float * __restrict__ xp,
    float * __restrict__ yp,
    float * __restrict__ x,
    float * __restrict__ y,
    const int len_xp) {

    const float inv_bin_width = (len_xp - 1) / (xp[len_xp - 1] - xp[0]);

    const int ffbin0 = (int)((x[0] - xp[0]) * inv_bin_width);
    const int diff = len_xp - ffbin0;

    #pragma omp parallel for
    for (int i = 0; i < diff - 1; i++) {
        int ffbin;
        ffbin = ffbin0 + i;
        y[i] = yp[ffbin] + (x[i] - xp[ffbin]) * (yp[ffbin + 1] - yp[ffbin]) * inv_bin_width;
    }

}
