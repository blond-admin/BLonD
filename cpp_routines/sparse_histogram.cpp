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
// Author: Juan F. Esteban Mueller

//using uint = unsigned int;

#include <stdio.h>

extern "C" void sparse_histogram(const double * __restrict__ input,
		    double * __restrict__ output,
               const double * __restrict__ cut_left_array,
               const double * __restrict__ cut_right_array,
               const double * __restrict__ bunch_indexes,
               const int n_slices,
               const int n_filled_buckets,
               const int n_macroparticles){
	
    int i;
    int j;
    int i_bucket;
    double a;
    double fbin;
    double fbunch;
    double cut_left, cut_right;
    int ffbin;
    int ffbunch;
    const double inv_bin_width = n_slices / (cut_right_array[0] - cut_left_array[0]);
    const double inv_bucket_length = 1 / (cut_right_array[0] - cut_left_array[0]);
   
    // Initialises all slicing arrays to zero
    for (i = 0; i < n_filled_buckets*n_slices; i++){
        output[i] = 0.0;
    }

    // Histogram loop
    for (i = 0; j < n_macroparticles; j++){
        a = input[j];   // Particle dt
        // Find bucket in which the particle is and its index
        fbunch = (a - cut_left_array[0]) * inv_bucket_length;
        ffbunch = (int)(fbunch);
        if ((a < cut_left_array[0])||(a > cut_right_array[n_filled_buckets-1]))
            continue;
        i_bucket = bunch_indexes[ffbunch];
        if (i_bucket == -1)
            continue;
        // Find the bin inside the corresponding bucket
        fbin = (a - cut_left_array[i_bucket]) * inv_bin_width;
        ffbin = (int)(fbin);
        output[i_bucket*n_slices+ffbin] = output[i_bucket+ffbin] + 1.0;
    }

}
