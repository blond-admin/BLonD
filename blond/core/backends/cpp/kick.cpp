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
#include "particle_kernels.h"

extern "C" void kick_multi_harmonic(
    const real_t *__restrict__ beam_dt, real_t *__restrict__ beam_dE,
    const int n_rf, const real_t charge, const real_t *__restrict__ voltage,
    const real_t *__restrict__ omega_RF, const real_t *__restrict__ phi_RF,
    const index_t n_macroparticles, const real_t acc_kick) {
  // The harmonics travel inline in `KickMultiHarmonicArgs`, at most
  // MAX_RF_HARMONICS_PER_RECORD per pass; `acc_kick` goes into the last
  // pass only, and one pass always runs so that n_rf == 0 still applies it.
  const int per_pass = sizeof(KickMultiHarmonicArgs::voltage) / sizeof(real_t);
  const int n_passes = (n_rf > per_pass) ? (n_rf + per_pass - 1) / per_pass : 1;
  for (int pass = 0; pass < n_passes; pass++) {
    const int first = pass * per_pass;
    const int count = (n_rf - first < per_pass) ? n_rf - first : per_pass;
    KickMultiHarmonicArgs args{};
    args.n_rf = count;
    for (int j = 0; j < count; j++) {
      args.voltage[j] = voltage[first + j];
      args.omega_rf[j] = omega_RF[first + j];
      args.phi_rf[j] = phi_RF[first + j];
    }
    args.charge = charge;
    args.acceleration_kick = (pass == n_passes - 1) ? acc_kick : real_t(0);
    run_on_all_particles(args, beam_dt, beam_dE, n_macroparticles);
  }
}

extern "C" void kick_single_harmonic(const real_t *__restrict__ beam_dt,
                                     real_t *__restrict__ beam_dE,
                                     const real_t charge, const real_t voltage,
                                     const real_t omega_RF, const real_t phi_RF,
                                     const index_t n_macroparticles,
                                     const real_t acc_kick) {
  KickSingleHarmonicArgs args{};
  args.voltage = voltage;
  args.omega_rf = omega_RF;
  args.phi_rf = phi_RF;
  args.charge = charge;
  args.acceleration_kick = acc_kick;
  run_on_all_particles(args, beam_dt, beam_dE, n_macroparticles);
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
