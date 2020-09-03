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

#include "sin.h"

using namespace vdt;

extern "C" void kick(const double * __restrict__ beam_dt, 
					 double * __restrict__ beam_dE, const int n_rf, 
					 const double * __restrict__ voltage, 
					 const double * __restrict__ omega_RF, 
					 const double * __restrict__ phi_RF,
					 const int n_macroparticles,
					 const double acc_kick){
int j;

// KICK
for (j = 0; j < n_rf; j++)
#pragma omp parallel for
		for (int i = 0; i < n_macroparticles; i++)
				beam_dE[i] = beam_dE[i] + voltage[j]
						   * fast_sin(omega_RF[j] * beam_dt[i] + phi_RF[j]);

// SYNCHRONOUS ENERGY CHANGE
#pragma omp parallel for
	for (int i = 0; i < n_macroparticles; i++)
		beam_dE[i] = beam_dE[i] + acc_kick;

}

extern "C" void rf_volt_comp(const double * __restrict__ voltage,
                             const double * __restrict__ omega_RF,
                             const double * __restrict__ phi_RF,
                             const double * __restrict__ bin_centers,
                             const int n_rf,
                             const int n_bins,
                             double *__restrict__ rf_voltage)
{
    for (int j = 0; j < n_rf; j++) {
        #pragma omp parallel for
        for (int i = 0; i < n_bins; i++) {
            rf_voltage[i] += voltage[j]
                             * fast_sin(omega_RF[j] * bin_centers[i] + phi_RF[j]);
        }
    }
}


extern "C" void kickf(const float * __restrict__ beam_dt,
                      float * __restrict__ beam_dE, const int n_rf,
                      const float * __restrict__ voltage,
                      const float * __restrict__ omega_RF,
                      const float * __restrict__ phi_RF,
                      const int n_macroparticles,
                      const float acc_kick) {
    int j;

// KICK
    for (j = 0; j < n_rf; j++)
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++)
            beam_dE[i] = beam_dE[i] + voltage[j]
                         * fast_sinf(omega_RF[j] * beam_dt[i] + phi_RF[j]);

// SYNCHRONOUS ENERGY CHANGE
    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++)
        beam_dE[i] = beam_dE[i] + acc_kick;

}

extern "C" void rf_volt_compf(const float * __restrict__ voltage,
                              const float * __restrict__ omega_RF,
                              const float * __restrict__ phi_RF,
                              const float * __restrict__ bin_centers,
                              const int n_rf,
                              const int n_bins,
                              float *__restrict__ rf_voltage)
{
    for (int j = 0; j < n_rf; j++) {
        #pragma omp parallel for
        for (int i = 0; i < n_bins; i++) {
            rf_voltage[i] += voltage[j]
                             * fast_sinf(omega_RF[j] * bin_centers[i] + phi_RF[j]);
        }
    }
}

