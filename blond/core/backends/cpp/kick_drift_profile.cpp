// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENCE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

#include "blond_common.h"

#include <math.h>
#include <stdlib.h> // mmalloc()
#include <string.h> // memset()

#include "blond_common.h"
#include "openmp.h"

extern "C" void kick_drift_profile(
    real_t *__restrict__ beam_dt, real_t *__restrict__ beam_dE,
    const real_t charge, const real_t voltage, const real_t omega_RF,
    const real_t phi_RF, const real_t T, const real_t eta_zero,
    const real_t beta, const real_t energy,
    const int n_macroparticles, const real_t acc_kick) {

  const real_t coeff = T * eta_zero / (beta * beta * energy);



#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_macroparticles; i += 1) {

        real_t dEij = beam_dE[i];
        real_t dtij = beam_dt[i];
        dEij += charge * voltage * FAST_SIN(omega_RF * dtij + phi_RF) + acc_kick;
        dtij += coeff * dEij;
        beam_dE[i] = dEij;
        beam_dt[i] = dtij;
    }


}
