/*
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3),
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities
granted to it by virtue of its status as an Intergovernmental Organization or
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/

// Author: Simon Lauber

#include <math.h>
#include <string.h>

#include "blond_common.h"

extern "C" int flagged_to_end(
    const int flag,
    int* __restrict__ flags,
    real_t* __restrict__ dt,
    real_t* __restrict__ dE,
    int_t* __restrict__ ids,
    const int n_macroparticles
) {
    // Set j to the end of the array.
    // Later every entry matching the flag is put to the end of the array
    // and j is moved to one position left.
    // Like that all particles that match the flag will be transferred to the
    // end of the array.

    int global_j = n_macroparticles - 1; // common for all threads

    // Temporary variables for swapping
    int flags_tmp;
    real_t dt_tmp;
    real_t dE_tmp;
    int_t ids_tmp;

    // Parallel loop
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < n_macroparticles; ++i) {
            while ((i <= global_j) && (flags[i] == flag)) {
                int j;

                // Atomically decrement global_j and get old value
                #pragma omp atomic capture
                j = global_j--;

                if (i >= j) break; // Prevent swapping beyond j

                // Swap all fields between i and j
                dt_tmp = dt[i]; dt[i] = dt[j]; dt[j] = dt_tmp;
                dE_tmp = dE[i]; dE[i] = dE[j]; dE[j] = dE_tmp;
                flags_tmp = flags[i]; flags[i] = flags[j]; flags[j] = flags_tmp;
                ids_tmp = ids[i]; ids[i] = ids[j]; ids[j] = ids_tmp;
            }
        }
    }

    return global_j + 1;
}
