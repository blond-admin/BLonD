// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Optimised C++ routine that calculates the drift.
// Author: Danilo Quartullo, Helga Timko, Alexandre Lasheen

#include <math.h>
#include <string.h>

#include "blond_common.h"

extern "C" void drift_simple(real_t *__restrict__ beam_dt,
                             const real_t *__restrict__ beam_dE, const real_t T,
                             const real_t eta_zero, const real_t beta,
                             const real_t energy, const int n_macroparticles) {

  real_t coeff = T * eta_zero / (beta * beta * energy);
#pragma omp parallel for
  for (int i = 0; i < n_macroparticles; i++) {
    beam_dt[i] += coeff * beam_dE[i];
  }
}

// Drift with the linear slip factor but the exact relativistic delta;
// reproduces the longitudinal drift of an xsuite LineSegmentMap.
extern "C" void drift_like_line_segment(
    real_t *__restrict__ beam_dt, const real_t *__restrict__ beam_dE,
    const real_t T, const real_t eta_zero, const real_t beta,
    const real_t energy, const int n_macroparticles) {

  const real_t inv_beta_sq = 1.0 / (beta * beta);
  const real_t inv_energy = 1.0 / energy;
  const real_t inv_energy_sq = inv_energy * inv_energy;
#pragma omp parallel for
  for (int i = 0; i < n_macroparticles; i++) {
    const real_t dE = beam_dE[i];
    const real_t delta =
        sqrt(1.0 + inv_beta_sq * (dE * dE * inv_energy_sq +
                                  2.0 * dE * inv_energy)) -
        1.0;
    beam_dt[i] += T * eta_zero * delta;
  }
}
