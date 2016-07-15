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

#include <math.h>
#include <stdlib.h>
#include <random>

extern "C" void SR_full(double * __restrict__ beam_dE, const double U0, 
					 const int n_macroparticles,
					 const double sigma_dE,
					 const double tau_z, const double energy){
    // SR term due to energy spread
    const double const_synch_rad = 2.0 / tau_z;
    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++)
        beam_dE[i] -= const_synch_rad * beam_dE[i];

    // Average energy change due to SR
    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++)
        beam_dE[i] -= U0;

    // Quantum excitation term    
	std::random_device rd;
    std::minstd_rand gen(rd()); 
    std::normal_distribution<> d(0,1);
	
    const double const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;
    for (int i = 0; i < n_macroparticles; i++){
        beam_dE[i] += const_quantum_exc * d(gen);
	}
}

extern "C" void SR(double * __restrict__ beam_dE, const double U0, 
					 const int n_macroparticles,
					 const double tau_z){
    // SR term due to energy spread
    const double const_synch_rad = 2.0 / tau_z;
    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++)
        beam_dE[i] -= const_synch_rad * beam_dE[i];

    // Average energy change due to SR
    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++)
        beam_dE[i] -= U0;
}
