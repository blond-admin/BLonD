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
int i;

// KICK
for (j = 0; j < n_rf; j++)
		for (int i = 0; i < n_macroparticles; i++)
				beam_dE[i] = beam_dE[i] + voltage[j]
						   * fast_sin(omega_RF[j] * beam_dt[i] + phi_RF[j]);

// SYNCHRONOUS ENERGY CHANGE
	for (int i = 0; i < n_macroparticles; i++)
		beam_dE[i] = beam_dE[i] + acc_kick;

}

