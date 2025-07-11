/*
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3), 
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities 
granted to it by virtue of its status as an Intergovernmental Organization or 
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/

// Optimised C++ routine that calculates the kicks
// Author: Danilo Quartullo, Helga Timko, Alexandre Lasheen

#include "blond_common.h"

extern "C" void kick_multi_harmonic(const real_t * __restrict__ beam_dt,
                        real_t * __restrict__ beam_dE, const int n_rf,
                        const real_t charge,
                        const real_t * __restrict__ voltage,
                        const real_t * __restrict__ omega_RF,
                        const real_t * __restrict__ phi_RF,
                        const int n_macroparticles,
                        const real_t acc_kick) {



    // Unroll loop for up to 4 RF harmonics for speedup
    if (n_rf == 1) {
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++){
            real_t dE_sum = voltage[0] * FAST_SIN(omega_RF[0] * beam_dt[i] + phi_RF[0]);
            beam_dE[i] += charge * dE_sum + acc_kick;
        }

    } else if (n_rf == 2) {
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++){
            real_t dE_sum =
                     voltage[0] * FAST_SIN(omega_RF[0] * beam_dt[i] + phi_RF[0])
                   + voltage[1] * FAST_SIN(omega_RF[1] * beam_dt[i] + phi_RF[1]);
            beam_dE[i] += charge * dE_sum + acc_kick;
        }
    } else if (n_rf == 3) {
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++){
            real_t dE_sum =
                     voltage[0] * FAST_SIN(omega_RF[0] * beam_dt[i] + phi_RF[0])
                   + voltage[1] * FAST_SIN(omega_RF[1] * beam_dt[i] + phi_RF[1])
                   + voltage[2] * FAST_SIN(omega_RF[2] * beam_dt[i] + phi_RF[2]);
            beam_dE[i] += charge * dE_sum + acc_kick;
        }
    } else if (n_rf == 4) {
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++){
            real_t dE_sum =
                     voltage[0] * FAST_SIN(omega_RF[0] * beam_dt[i] + phi_RF[0])
                   + voltage[1] * FAST_SIN(omega_RF[1] * beam_dt[i] + phi_RF[1])
                   + voltage[2] * FAST_SIN(omega_RF[2] * beam_dt[i] + phi_RF[2])
                   + voltage[3] * FAST_SIN(omega_RF[3] * beam_dt[i] + phi_RF[3]);
            beam_dE[i] += charge * dE_sum + acc_kick;
        }

    } else {
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++) {
            real_t dE_sum = 0.0;
            // fallback to loop for n_rf > 4
            for (int j = 0; j < n_rf; j++) {
                dE_sum += voltage[j] * FAST_SIN(omega_RF[j] * beam_dt[i] + phi_RF[j]);
            }
            beam_dE[i] += charge * dE_sum + acc_kick;

        }
    }

}



extern "C" void kick_single_harmonic(
                        const real_t * __restrict__ beam_dt,
                        real_t * __restrict__ beam_dE,
                        const real_t charge,
                        const real_t  voltage,
                        const real_t omega_RF,
                        const real_t phi_RF,
                        const int n_macroparticles,
                        const real_t acc_kick) {

    // KICK
    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++){
        beam_dE[i] += charge * voltage * FAST_SIN(omega_RF * beam_dt[i] + phi_RF) + acc_kick;
    }
}


extern "C" void rf_volt_comp(const real_t * __restrict__ voltage,
                             const real_t * __restrict__ omega_RF,
                             const real_t * __restrict__ phi_RF,
                             const real_t * __restrict__ bin_centers,
                             const int n_rf,
                             const int n_bins,
                             real_t *__restrict__ rf_voltage) {
    for (int j = 0; j < n_rf; j++) {
        #pragma omp parallel for
        for (int i = 0; i < n_bins; i++) {
            rf_voltage[i] += voltage[j]
                             * FAST_SIN(omega_RF[j] * bin_centers[i] + phi_RF[j]);
        }
    }
}
