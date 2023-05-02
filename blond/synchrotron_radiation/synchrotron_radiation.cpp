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
// Author: Juan F. Esteban Mueller, Konstantinos Iliakis

#include <math.h>
#include <stdlib.h>
#include <random>
#include <thread>
#include <chrono>
#include "../cpp_routines/openmp.h"

#ifdef BOOST
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
using namespace boost;

#else
using namespace std;

#endif

long unsigned int seed = 1234;

// This function calculates and applies only the synchrotron radiation damping term
extern "C" void synchrotron_radiation(double * __restrict__ beam_dE, const double U0,
                                      const int n_macroparticles, const double tau_z,
                                      const int n_kicks) {

    // SR damping constant, adjusted for better performance
    const double const_synch_rad = 1.0 - 2.0 / tau_z;

    for (int j = 0; j < n_kicks; j++) {
        // SR damping term due to energy spread and
        // Average energy change due to SR
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++)
            beam_dE[i] = beam_dE[i] * const_synch_rad - U0;

        // #pragma omp parallel for
        // for (int i = 0; i < n_macroparticles; i++)
        //     beam_dE[i] -= U0;
    }
}


// This function calculates and applies synchrotron radiation damping and
// quantum excitation terms
extern "C" void synchrotron_radiation_full(double * __restrict__ beam_dE, const double U0,
        const int n_macroparticles, const double sigma_dE,
        const double tau_z, const double energy,
        const int n_kicks)
{

    // std::hash<std::thread::id> hash;

    // Quantum excitation constant
    const double const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;

    // Adjusted SR damping constant
    const double const_synch_rad = 1.0 - 2.0 / tau_z;

    // Random number generator for the quantum excitation term

    for (int j = 0; j < n_kicks; j++) {
        // Compute synchrotron radiation damping term and
        // Applies the quantum excitation term
        #pragma omp parallel
        {
            static __thread mt19937_64 *gen = nullptr;
            if (!gen) gen = new mt19937_64(seed + omp_get_thread_num());
            static thread_local normal_distribution<> dist(0.0, 1.0);
            #pragma omp for
            for (int i = 0; i < n_macroparticles; i++) {
                beam_dE[i] = beam_dE[i] * const_synch_rad
                             + const_quantum_exc * dist(*gen)
                             - U0;
            }
        }
    }
}


// This function calculates and applies only the synchrotron radiation damping term
extern "C" void synchrotron_radiationf(float * __restrict__ beam_dE, const float U0,
                                      const int n_macroparticles, const float tau_z,
                                      const int n_kicks) {

    // SR damping constant, adjusted for better performance
    const float const_synch_rad = 1.0 - 2.0 / tau_z;

    for (int j = 0; j < n_kicks; j++) {
        // SR damping term due to energy spread and
        // Average energy change due to SR
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++)
            beam_dE[i] = beam_dE[i] * const_synch_rad - U0;

        // #pragma omp parallel for
        // for (int i = 0; i < n_macroparticles; i++)
        //     beam_dE[i] -= U0;
    }
}


// This function calculates and applies synchrotron radiation damping and
// quantum excitation terms
extern "C" void synchrotron_radiation_fullf(float * __restrict__ beam_dE, const float U0,
        const int n_macroparticles, const float sigma_dE,
        const float tau_z, const float energy,
        const int n_kicks)
{

    // std::hash<std::thread::id> hash;

    // Quantum excitation constant
    const float const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;

    // Adjusted SR damping constant
    const float const_synch_rad = 1.0 - 2.0 / tau_z;

    // Random number generator for the quantum excitation term

    for (int j = 0; j < n_kicks; j++) {
        // Compute synchrotron radiation damping term and
        // Applies the quantum excitation term
        #pragma omp parallel
        {
            static __thread mt19937_64 *gen = nullptr;
            if (!gen) gen = new mt19937_64(seed + omp_get_thread_num());
            static thread_local normal_distribution<> dist(0.0, 1.0);
            #pragma omp for
            for (int i = 0; i < n_macroparticles; i++) {
                beam_dE[i] = beam_dE[i] * const_synch_rad
                             + const_quantum_exc * dist(*gen)
                             - U0;
            }
        }
    }
}


extern "C" void set_random_seed(const int _seed) {
    seed = _seed;
}
