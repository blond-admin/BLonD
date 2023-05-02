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
// Author: Juan F. Esteban Mueller, Danilo Quartullo, Alexandre Lasheen, Markus Schwarz

#include <stdio.h>
#include <string.h>     // memset()
#include <stdlib.h>     // mmalloc()
#include <math.h>
#include "../cpp_routines/openmp.h"

extern "C" void sparse_histogram(const double * __restrict__ input,
                double * __restrict__ output,
               const double * __restrict__ cut_left_array,
               const double * __restrict__ cut_right_array,
               const double * __restrict__ bunch_indexes,
               const int n_slices_bucket,
               const int n_filled_buckets,
               const int n_macroparticles){
	
    int j;
    int i_bucket;
    double a;
    double fbin;
    double fbunch;
    int ffbin;
    int ffbunch;
    
    // Only valid for cut_edges = edges
    const double inv_bucket_length = 1.0 / (cut_right_array[0] - cut_left_array[0]);
    const double inv_bin_width = inv_bucket_length * (double) n_slices_bucket;
    
    // allocate memory for the thread_private histogram
    double **histo = (double **) malloc(omp_get_max_threads() * sizeof(double *));
    histo[0] = (double *) malloc (omp_get_max_threads() * n_filled_buckets * n_slices_bucket * sizeof(double));
    for (int i = 0; i < omp_get_max_threads(); i++)
        histo[i] = (*histo + n_filled_buckets * n_slices_bucket * i);


    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        memset(histo[id], 0., n_filled_buckets * n_slices_bucket * sizeof(double));
    
        // Histogram loop
        #pragma omp for
        for (j = 0; j < n_macroparticles; j++){
            a = input[j];   // Particle dt
            if ((a < cut_left_array[0])||(a > cut_right_array[n_filled_buckets-1]))
                continue;
            // Find bucket in which the particle is and its index
            fbunch = (a - cut_left_array[0]) * inv_bucket_length;
            ffbunch = (int) fbunch;
            i_bucket = (int) bunch_indexes[ffbunch];
            if (i_bucket == -1)
                continue;
            // Find the bin inside the corresponding bucket
            fbin = (a - cut_left_array[i_bucket]) * inv_bin_width;
            ffbin = i_bucket*n_slices_bucket + (int) fbin;
            histo[id][ffbin] += 1.0;
        }
        
        // Reduce to a single histogram
        #pragma omp for
        for (int i = 0; i < n_filled_buckets * n_slices_bucket; i++) {
            output[i] = 0.;
            for (int t = 0; t < threads; t++)
                output[i] += histo[t][i];
        }

    }
    
    // free memory
    free(histo[0]);
    free(histo);
}


extern "C" void sparse_histogramf(const float * __restrict__ input,
            float * __restrict__ output,
               const float * __restrict__ cut_left_array,
               const float * __restrict__ cut_right_array,
               const float * __restrict__ bunch_indexes,
               const int n_slices_bucket,
               const int n_filled_buckets,
               const int n_macroparticles){
    
    int j;
    int i_bucket;
    float a;
    float fbin;
    float fbunch;
    int ffbin;
    int ffbunch;
    
    // Only valid for cut_edges = edges
    const float inv_bucket_length = 1.0 / (cut_right_array[0] - cut_left_array[0]);
    const float inv_bin_width = inv_bucket_length * (float) n_slices_bucket;
    
    // allocate memory for the thread_private histogram
    float **histo = (float **) malloc(omp_get_max_threads() * sizeof(float *));
    histo[0] = (float *) malloc (omp_get_max_threads() * n_filled_buckets * n_slices_bucket * sizeof(float));
    for (int i = 0; i < omp_get_max_threads(); i++)
        histo[i] = (*histo + n_filled_buckets * n_slices_bucket * i);


    #pragma omp parallel
    {

        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        memset(histo[id], 0., n_filled_buckets * n_slices_bucket * sizeof(float));
    
        // Histogram loop
        #pragma omp for
        for (j = 0; j < n_macroparticles; j++){
            a = input[j];   // Particle dt
            if ((a < cut_left_array[0])||(a > cut_right_array[n_filled_buckets-1]))
                continue;
            // Find bucket in which the particle is and its index
            fbunch = (a - cut_left_array[0]) * inv_bucket_length;
            ffbunch = (int) fbunch;
            i_bucket = (int) bunch_indexes[ffbunch];
            if (i_bucket == -1)
                continue;
            // Find the bin inside the corresponding bucket
            fbin = (a - cut_left_array[i_bucket]) * inv_bin_width;
            ffbin = i_bucket*n_slices_bucket + (int) fbin;
            histo[id][ffbin] += 1.0;
        }
        
        // Reduce to a single histogram
        #pragma omp for
        for (int i = 0; i < n_filled_buckets * n_slices_bucket; i++) {
            output[i] = 0.;
            for (int t = 0; t < threads; t++)
                output[i] += histo[t][i];
        }

    }
    
    // free memory
    free(histo[0]);
    free(histo);
}
