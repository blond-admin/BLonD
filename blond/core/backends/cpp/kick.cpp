// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Optimised C++ routine that calculates the kicks
// Author: Danilo Quartullo, Helga Timko, Alexandre Lasheen

#include "blond_common.h"
#include "particle_ops.h"

extern "C" void kick_multi_harmonic(
    const real_t *__restrict__ beam_dt, real_t *__restrict__ beam_dE,
    const int n_rf, const real_t charge, const real_t *__restrict__ voltage,
    const real_t *__restrict__ omega_RF, const real_t *__restrict__ phi_RF,
    const index_t n_macroparticles, const real_t acc_kick) {
  const KickMultiHarmonic::Args args = {n_rf,     charge, voltage,
                                        omega_RF, phi_RF, acc_kick};
  run_parallel<KickMultiHarmonic>(args, beam_dt, beam_dE, n_macroparticles);
}

extern "C" void kick_single_harmonic(const real_t *__restrict__ beam_dt,
                                     real_t *__restrict__ beam_dE,
                                     const real_t charge, const real_t voltage,
                                     const real_t omega_RF, const real_t phi_RF,
                                     const index_t n_macroparticles,
                                     const real_t acc_kick) {
  const KickSingleHarmonic::Args args = {charge, voltage, omega_RF, phi_RF,
                                         acc_kick};
  run_parallel<KickSingleHarmonic>(args, beam_dt, beam_dE, n_macroparticles);
}

extern "C" void rf_volt_comp(const real_t *__restrict__ voltage,
                             const real_t *__restrict__ omega_RF,
                             const real_t *__restrict__ phi_RF,
                             const real_t *__restrict__ bin_centers,
                             const int n_rf, const int n_bins,
                             real_t *__restrict__ rf_voltage) {
#pragma omp parallel for
  for (int i = 0; i < n_bins; i++) {
    for (int j = 0; j < n_rf; j++) {
      rf_voltage[i] +=
          voltage[j] * FAST_SIN(omega_RF[j] * bin_centers[i] + phi_RF[j]);
    }
  }
}
