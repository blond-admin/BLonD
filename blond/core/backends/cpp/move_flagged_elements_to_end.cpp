// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENCE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Author: Simon Lauber

#include <math.h>
#include <string.h>

#include "blond_common.h"

extern "C" int move_flagged_elements_to_end(
    const int flag,
    int* __restrict__ flags,
    real_t* __restrict__ dt,
    real_t* __restrict__ dE,
    int_t* __restrict__ ids,
    const int n_macroparticles
) {
    // Use sequential two-pointer approach for correctness
    // Parallelizing in-place partition with swaps causes data races
    int i = 0;  // scan from front
    int j = n_macroparticles - 1;  // scan from back

    while (i < j) {
        // Find next flagged element from front
        while (i < j && flags[i] != flag) {
            ++i;
        }

        // Find next non-flagged element from back
        while (i < j && flags[j] == flag) {
            --j;
        }

        // If pointers haven't crossed, swap elements
        if (i < j) {
            // Swap all fields between i and j
            real_t dt_tmp = dt[i];
            dt[i] = dt[j];
            dt[j] = dt_tmp;

            real_t dE_tmp = dE[i];
            dE[i] = dE[j];
            dE[j] = dE_tmp;

            int flags_tmp = flags[i];
            flags[i] = flags[j];
            flags[j] = flags_tmp;

            int_t ids_tmp = ids[i];
            ids[i] = ids[j];
            ids[j] = ids_tmp;

            ++i;
            --j;
        }
    }

    // Return index of first flagged particle
    // All particles at indices >= return value have the flag
    if (i < n_macroparticles && flags[i] == flag) {
        return i;
    }
    return i + 1;
}
