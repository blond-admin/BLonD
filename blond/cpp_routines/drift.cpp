/*
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3),
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities
granted to it by virtue of its status as an Intergovernmental Organization or
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/

// Optimised C++ routine that calculates the drift.
// Author: Danilo Quartullo, Helga Timko, Alexandre Lasheen

#include <string.h>
#include <math.h>

#include "blond_common.h"

extern "C" void drift(real_t * __restrict__ beam_dt,
                      const real_t * __restrict__ beam_dE,
                      const int solver,
                      const real_t T0, const real_t length_ratio,
                      const real_t alpha_order, const real_t eta_zero,
                      const real_t eta_one, const real_t eta_two,
                      const real_t alpha_zero, const real_t alpha_one,
                      const real_t alpha_two,
                      const real_t beta, const real_t energy,
                      const int n_macroparticles) {

    int i;
    real_t T = T0 * length_ratio;

    if ( solver == 0 ) {
        real_t coeff = T * eta_zero / (beta * beta * energy);
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++)
        beam_dt[i] += coeff * beam_dE[i];
    } else if ( solver == 1 ) {
        const real_t coeff = 1. / (beta * beta * energy);
        const real_t eta0 = eta_zero * coeff;
        const real_t eta1 = eta_one * coeff * coeff;
        const real_t eta2 = eta_two * coeff * coeff * coeff;

        if (alpha_order == 0)
            for ( i = 0; i < n_macroparticles; i++ )
                beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]) - 1.);
        else if (alpha_order == 1)
            for ( i = 0; i < n_macroparticles; i++ )
                beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]
                                         - eta1 * beam_dE[i] * beam_dE[i]) - 1.);
        else
            for ( i = 0; i < n_macroparticles; i++ )
                beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]
                                         - eta1 * beam_dE[i] * beam_dE[i]
                                         - eta2 * beam_dE[i] * beam_dE[i] * beam_dE[i]) - 1.);
    } else {

        const real_t invbetasq = 1 / (beta * beta);
        const real_t invenesq = 1 / (energy * energy);
        // real_t beam_delta;

        #pragma omp parallel for
        for ( i = 0; i < n_macroparticles; i++ ) {

            real_t beam_delta = sqrt(1. + invbetasq *
                              (beam_dE[i] * beam_dE[i] * invenesq + 2.*beam_dE[i] / energy)) - 1.;

            beam_dt[i] += T * (
                              (1. + alpha_zero * beam_delta +
                               alpha_one * (beam_delta * beam_delta) +
                               alpha_two * (beam_delta * beam_delta * beam_delta)) *
                              (1. + beam_dE[i] / energy) / (1. + beam_delta) - 1.);

        }

    }

}
