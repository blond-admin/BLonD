/*
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3),
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities
granted to it by virtue of its status as an Intergovernmental Organization or
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/

// Optimised C++ routine that calculates and applies synchrotron radiation (SR)
// damping and quantum excitation terms
// The random number generator from C++ (>=C++11) is used
// This routine is not optimized for parallel computation
// Author: Juan F. Esteban Mueller

#include <math.h>
#include <stdlib.h>
#include <random>
#include "synchrotron_radiation.h"

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::normal_distribution<> d(0.0,1.0);
// This function calculates and applies synchrotron radiation damping and
// quantum excitation terms
extern "C" void synchrotron_radiation_full(double * __restrict__ beam_dE, const double U0,
                                        const int n_macroparticles, const double sigma_dE,
                                        const double tau_z,const double energy,
                                        double * __restrict__ random_array,
                                        const int n_kicks){
    
    // Quantum excitation constant
    const double const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;

    // Random number generator for the quantum excitation term
    
    for (int j=0; j<n_kicks; j++){
        // Compute synchrotron radiation damping term
        synchrotron_radiation(beam_dE, U0, n_macroparticles, tau_z, 1);
    
        // Re-calculate the random (Gaussian) number array
        for (int i = 0; i < n_macroparticles; i++){
            random_array[i] = d(gen);
        }
        
        // Applies the quantum excitation term
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++){
            beam_dE[i] += const_quantum_exc * random_array[i];
        }
    }
}
