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

extern "C" void kick(const real_t * __restrict__ beam_dt,
                        real_t * __restrict__ beam_dE, const int n_rf,
                        const real_t charge,
                        const real_t * __restrict__ voltage,
                        const real_t * __restrict__ omega_RF,
                        const real_t * __restrict__ phi_RF,
                        const int n_macroparticles,
                        const real_t acc_kick) {

    // KICK
    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++){
        real_t dE_sum = 0.0;
        const real_t dti = beam_dt[i];
        for (int j = 0; j < n_rf; j++){
            dE_sum += voltage[j] * FAST_SIN(omega_RF[j] * dti + phi_RF[j]);
        }
        beam_dE[i] += charge * dE_sum + acc_kick;
    }
}

extern "C" void rf_volt_comp(const real_t * __restrict__ voltage,
                             const real_t * __restrict__ omega_RF,
                             const real_t * __restrict__ phi_RF,
                             const real_t * __restrict__ bin_centers,
                             const int n_rf,
                             const int n_bins,
                             real_t *__restrict__ rf_voltage) {
    #pragma omp parallel for
    for (int i = 0; i < n_bins; i++) {
        const real_t bin_center = bin_centers[i];

        if (n_rf == 1) {
            rf_voltage[i] = voltage[0] * FAST_SIN(omega_RF[0] * bin_center + phi_RF[0]);
        } else {
            real_t rf_voltage_i = 0.0; // rf_voltage = np.empty in wrapper function
            for (int j = 0; j < n_rf; j++) {
                rf_voltage_i += voltage[j] * FAST_SIN(omega_RF[j] * bin_center + phi_RF[j]);
            }
            rf_voltage[i] = rf_voltage_i;

        }

    }
}

