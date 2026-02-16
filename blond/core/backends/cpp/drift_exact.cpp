// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENCE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Optimised C++ routine that calculates the drift.
// Author: Danilo Quartullo, Helga Timko, Alexandre Lasheen, Elleanor Lamb

#include <math.h>
#include <string.h>
#include "blond_common.h"

extern "C" void drift_exact(real_t * __restrict__ beam_dt,
                            const real_t * __restrict__ beam_dE,
                            const real_t t_rev,
                            const real_t alpha_zero,
                            const real_t* __restrict__ higher_alpha,
                            const int n_alpha,
                            const real_t beta,
                            const real_t energy,
                            const int n_macroparticles)
{
    const real_t inv_beta_sq = 1.0 / (beta * beta);
    const real_t inv_energy  = 1.0 / energy;
    const real_t inv_energy_sq = inv_energy * inv_energy;

#pragma omp parallel for
for (int i = 0; i < n_macroparticles; i++) {

    const real_t dE = beam_dE[i];

    const real_t delta =
        sqrt(1.0 + inv_beta_sq *
             (dE * dE * inv_energy_sq +
              2.0 * dE * inv_energy)) - 1.0;

    real_t poly = 1.0 + alpha_zero * delta;

    if (n_alpha > 0 && higher_alpha != nullptr) {
        real_t delta_power = delta * delta;  // starts at δ²

        for (int k = 0; k < n_alpha; ++k) {
            poly += higher_alpha[k] * delta_power;
            delta_power *= delta;  // next power
        }
    }

    beam_dt[i] += t_rev * (
        poly
        * (1.0 + dE * inv_energy)
        / (1.0 + delta)
        - 1.0
    );
}
