// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Optimised C++ routine that calculates the drift.
// Author: Danilo Quartullo, Helga Timko, Alexandre Lasheen

#include "blond_common.h"
#include "particle_kernels.h"

extern "C" void drift_simple(real_t *__restrict__ beam_dt,
                             const real_t *__restrict__ beam_dE, const real_t T,
                             const real_t eta_zero, const real_t beta,
                             const real_t energy,
                             const index_t n_macroparticles) {
  DriftSimpleArgs args{};
  args.T = T;
  args.eta_0 = eta_zero;
  args.beta = beta;
  args.energy = energy;
  run_on_all_particles(args, beam_dt, beam_dE, n_macroparticles);
}

extern "C" void drift_like_line_segment(real_t *__restrict__ beam_dt,
                                        const real_t *__restrict__ beam_dE,
                                        const real_t T, const real_t eta_zero,
                                        const real_t beta, const real_t energy,
                                        const index_t n_macroparticles) {
  DriftLikeLineSegmentArgs args{};
  args.T = T;
  args.eta_0 = eta_zero;
  args.beta = beta;
  args.energy = energy;
  run_on_all_particles(args, beam_dt, beam_dE, n_macroparticles);
}
