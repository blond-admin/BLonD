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
// damping term
// Author: Juan F. Esteban Mueller

#include <math.h>
#include <stdlib.h>
#include <random>

#ifdef BOOST
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

std::random_device rd;
boost::mt19937_64 gen(rd());
boost::normal_distribution<> dist(0.0, 1.0);
// gen.seed(std::random_device{}());
// boost::variate_generator< boost::mt19937_64, boost::normal_distribution<> > dist(gen, distribution);
    
#else
std::random_device rd;
std::mt19937_64 gen(rd());
std::normal_distribution<> dist(0.0,1.0);
#endif


// This function calculates and applies only the synchrotron radiation damping term
extern "C" void synchrotron_radiation(double * __restrict__ beam_dE, const double U0, 
                            const int n_macroparticles, const double tau_z, 
                            const int n_kicks){

    // SR damping constant
    const double const_synch_rad = 2.0 / tau_z;

    for (int j=0; j<n_kicks; j++){
        // SR damping term due to energy spread
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++)
            beam_dE[i] -= const_synch_rad * beam_dE[i];
    
        // Average energy change due to SR
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++)
            beam_dE[i] -= U0;
    }
}


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
            random_array[i] = dist(gen);
        }
        
        // Applies the quantum excitation term
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++){
            beam_dE[i] += const_quantum_exc * random_array[i];
        }
    }
}