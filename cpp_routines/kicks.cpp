/*
Copyright 2015 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3), 
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities 
granted to it by virtue of its status as an Intergovernmental Organization or 
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/

// Optimised C++ routine that calculates the kicks
// Author: Danilo Quartullo

#include "sin.h"
using namespace vdt;
using uint = unsigned int;


extern "C" void kicks(const double * __restrict__ beam_theta, double * __restrict__ beam_dE, const int n_rf, 
						const double * __restrict__ voltage, const double * __restrict__ harmonic, 
						const double * __restrict__ phi_offset, const uint n_macroparticles, const double acc_kick){
int j;
uint i;  

// KICK
for (j = 0; j < n_rf; j++) 
 	for (i = 0; i < n_macroparticles; i++) 
            beam_dE[i] = beam_dE[i] + voltage[j] * fast_sin(harmonic[j] * beam_theta[i] + phi_offset[j]);

// KICK ACCELERATION
for (i = 0; i < n_macroparticles; i++) 
    beam_dE[i] = beam_dE[i] + acc_kick;
}
