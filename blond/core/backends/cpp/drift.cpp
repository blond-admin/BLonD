// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENCE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Optimised C++ routine that calculates the drift.
// Author: Danilo Quartullo, Helga Timko, Alexandre Lasheen

#include "blond_common.h"

extern "C" void drift_simple(real_t *__restrict__ beam_dt,
                             const real_t *__restrict__ beam_dE, const real_t T,
                             const real_t eta_zero, const real_t beta,
                             const real_t energy, const int n_macroparticles) {

  const real_t coeff = T * eta_zero / (beta * beta * energy);
  beam_dt = (real_t *)__builtin_assume_aligned(beam_dt, 64);
  beam_dE = (const real_t *)__builtin_assume_aligned(beam_dE, 64);
// Non-temporal stores bypass the cache: avoids polluting L3 for large beams
// where the array won't be re-read soon (array size >> L3 cache).
#pragma omp parallel for simd nontemporal(beam_dt)
  for (int i = 0; i < n_macroparticles; i++) {
    beam_dt[i] += coeff * beam_dE[i];
  }
}
