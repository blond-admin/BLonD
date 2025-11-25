// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENCE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Optimised C++ routine that calculates the drift.
// Author: Danilo Quartullo, Helga Timko, Alexandre Lasheen

#include <math.h>
#include <string.h>

#include "blond_common.h"

extern "C" void loss_box(
                     const real_t top,
                     const real_t bottom,
                     const real_t left,
                     const real_t right,
                     const real_t * dt,
                     const real_t * dE,
                     int * __restrict__ flags,
                     const int n_macroparticles
                     )
{

    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++) {
        const bool outside = (dE[i] > top) || (dE[i] < bottom) || (dt[i] < left) || (dt[i] > right);
        if (outside){
            flags[i] =  -500; // assume (BeamFlags.LOST.value)
        }
    }
}
