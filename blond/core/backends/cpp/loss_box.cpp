// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// C++ routine that calculates loss_box.
// Author: Simon Lauber

#include <math.h>
#include <string.h>

#include "blond_common.h"

// `flag_lost` is `BeamFlags.LOST` (blond/core/beam/flags.py), passed in by
// the Python wrapper so the enum stays the single source of truth.
extern "C" void loss_box(const real_t e_max, const real_t e_min,
                         const real_t t_min, const real_t t_max,
                         const real_t *dt, const real_t *dE,
                         int *__restrict__ flags, const int flag_lost,
                         const index_t n_macroparticles) {

#pragma omp parallel for
  for (index_t i = 0; i < n_macroparticles; i++) {
    const bool outside = (dE[i] > e_max) || (dE[i] < e_min) ||
                         (dt[i] < t_min) || (dt[i] > t_max);
    if (outside) {
      flags[i] = flag_lost;
    }
  }
}
