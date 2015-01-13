/*
Copyright 2015 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3), 
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities 
granted to it by virtue of its status as an Intergovernmental Organization or 
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/

// Optimised C++ routine that calculates the drift_simple
// Author: Danilo Quartullo

using uint = unsigned int;
const double M_PI = 3.14159265358979323846;

extern "C" void drift_simple(double * __restrict__ beam_theta, 
							const double * __restrict__ beam_dE, 
							const double omega_1, const double omega_2,
                             const double omega_3, const double length_ratio, 
                             const double eta_zero, const double beta_r,
                             const double energy, const uint n_macroparticles){

uint i;
const double coeff = 2.0 * M_PI * length_ratio; 
const double coeff2 = (omega_3 * eta_zero * coeff) / (beta_r * beta_r * energy);
const double coeff3 = coeff * omega_2;

for (i = 0; i < n_macroparticles; i++) 
    beam_theta[i] = omega_1 * beam_theta[i] + coeff2 * beam_dE[i] + coeff3;

}
