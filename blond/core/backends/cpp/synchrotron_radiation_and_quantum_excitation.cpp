// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Fused synchrotron-radiation damping + (optional) quantum-excitation kick.
//
// The Gaussian noise uses the C++ standard library generator so we do not
// maintain (or have to justify) a hand-rolled PRNG:
//
//   * `std::mt19937_64` for the underlying stream.
//   * `std::normal_distribution<real_t>` for the N(0, 1) transform.
//   * seeded once from `std::random_device` XORed with the OpenMP thread id
//     so every run and every thread gets an independent stream.
//
// The loop runs under OpenMP. `std::normal_distribution` is stateful (it
// caches the second value of the Box-Muller pair internally), so both the
// generator and the distribution are kept in `thread_local` storage: one
// instance per thread, never shared between OpenMP threads.
//
// Author: Simon Lauber

#include "blond_common.h"
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" void apply_synchrotron_radiation_no_excitation(
    real_t* __restrict__ beam_dE,
    const real_t damping_factor,
    const real_t energy_lost,
    const int n_macroparticles
) {
#pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++) {
        beam_dE[i] = damping_factor * beam_dE[i] - energy_lost;
    }
}

extern "C" void apply_synchrotron_radiation_and_quantum_excitation(
    real_t* __restrict__ beam_dE,
    const real_t damping_factor,
    const real_t energy_lost,
    const real_t noise_scale,
    const int n_macroparticles
) {
#pragma omp parallel
    {
        // One standard-library generator and Gaussian distribution per
        // thread, seeded once on first use with a thread-distinct value.
        static thread_local std::mt19937_64 generator;
        static thread_local std::normal_distribution<real_t> standard_normal(
            real_t(0.0), real_t(1.0)
        );
        static thread_local bool seeded = false;
        if (!seeded) {
            unsigned int thread_id = 0;
#ifdef _OPENMP
            thread_id = static_cast<unsigned int>(omp_get_thread_num());
#endif
            std::random_device entropy_source;
            generator.seed(entropy_source() ^ thread_id);
            seeded = true;
        }

#pragma omp for
        for (int i = 0; i < n_macroparticles; i++) {
            beam_dE[i] = damping_factor * beam_dE[i] - energy_lost
                       + noise_scale * standard_normal(generator);
        }
    }
}
