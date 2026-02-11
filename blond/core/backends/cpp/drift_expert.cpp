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

extern "C" void drift_exact(real_t * __restrict__ beam_dt,
                            const real_t * __restrict__ beam_dE,
                            const real_t T0,
                            const real_t length_ratio,
                            const real_t alpha_zero,
                            const real_t alpha_one,
                            const real_t alpha_two,
                            const real_t beta,
                            const real_t energy,
                            const int n_macroparticles)
{
    const real_t T = T0 * length_ratio;
    const real_t inv_beta_sq = 1.0 / (beta * beta);
    const real_t inv_energy  = 1.0 / energy;
    const real_t inv_energy_sq = inv_energy * inv_energy;

    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++) {

        const real_t delta =
            sqrt(1.0 + inv_beta_sq *
                 (beam_dE[i] * beam_dE[i] * inv_energy_sq +
                  2.0 * beam_dE[i] * inv_energy)) - 1.0;

        beam_dt[i] += T * (
            (1.0
             + alpha_zero * delta
             + alpha_one  * delta * delta
             + alpha_two  * delta * delta * delta)
            * (1.0 + beam_dE[i] * inv_energy)
            / (1.0 + delta)
            - 1.0
        );
    }
}
