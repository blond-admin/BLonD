// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Fused synchrotron-radiation damping + (optional) quantum-excitation kick.
//
// The Gaussian noise is generated inline (no external buffer, no extra
// allocations in the hot loop) using a modern PRNG:
//
//   * `xoshiro256+` for the underlying uniform stream — small state
//     (4 x uint64), 6-cycle output, much faster than libstdc++'s
//     `std::mt19937_64`.
//   * Marsaglia polar Box-Muller for the Gaussian transform — 1 sqrt + 1
//     log per *two* samples (cached), no transcendentals on ~78% of draws.
//
// Each OpenMP thread keeps its own state in `thread_local` storage so the
// streams are uncorrelated and there is zero contention.
//
// Author: Simon Lauber

#include "blond_common.h"
#include <chrono>
#include <cmath>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// ---------- xoshiro256+ uniform PRNG ----------
// xoshiro256+ and splitmix64 by Blackman & Vigna (https://prng.di.unimi.it/, CC0/public domain).
// See https://prng.di.unimi.it/xoshiro256starstar.c
// See https://prng.di.unimi.it/splitmix64.c
struct Xoshiro256p {
    uint64_t s[4];
};

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoshiro_next(Xoshiro256p& rng) {
    const uint64_t result = rng.s[0] + rng.s[3];
    const uint64_t t = rng.s[1] << 17;
    rng.s[2] ^= rng.s[0];
    rng.s[3] ^= rng.s[1];
    rng.s[1] ^= rng.s[2];
    rng.s[0] ^= rng.s[3];
    rng.s[2] ^= t;
    rng.s[3] = rotl(rng.s[3], 45);
    return result;
}

// Seed via splitmix64 (the recommended xoshiro initializer).
static inline void xoshiro_seed(Xoshiro256p& rng, uint64_t seed_val) {
    uint64_t z = seed_val;
    for (int i = 0; i < 4; ++i) {
        z += 0x9E3779B97F4A7C15ULL;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        x = x ^ (x >> 31);
        rng.s[i] = x;
    }
}

// Uniform in (0, 1), uses the top 53 bits of a uint64.
static inline real_t uniform01(Xoshiro256p& rng) {
    return real_t(xoshiro_next(rng) >> 11) * (real_t(1.0) / real_t(1ULL << 53));
}

// ---------- Gaussian generator (Marsaglia polar Box-Muller with cache) ----------
struct GaussianState {
    Xoshiro256p rng;
    real_t spare;
    bool has_spare;
    bool seeded;
};

static inline real_t standard_normal(GaussianState& gaussian_state) {
    if (gaussian_state.has_spare) {
        gaussian_state.has_spare = false;
        return gaussian_state.spare;
    }
    // Polar form: draw points in the unit disc, reject ~21%.
    real_t u, v, s;
    do {
        u = real_t(2.0) * uniform01(gaussian_state.rng) - real_t(1.0);
        v = real_t(2.0) * uniform01(gaussian_state.rng) - real_t(1.0);
        s = u * u + v * v;
    } while (s >= real_t(1.0) || s == real_t(0.0));
    const real_t factor = std::sqrt(real_t(-2.0) * std::log(s) / s);
    gaussian_state.spare = v * factor;
    gaussian_state.has_spare = true;
    return u * factor;
}

}  // namespace

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
        // One persistent generator per thread, seeded once on first use
        // with a thread-distinct value mixed from a wall-clock sample.
        static thread_local GaussianState gaussian_state{};
        if (!gaussian_state.seeded) {
            const uint64_t seed_val = static_cast<uint64_t>(
                std::chrono::high_resolution_clock::now()
                    .time_since_epoch()
                    .count()
            )
#ifdef _OPENMP
                ^ (static_cast<uint64_t>(omp_get_thread_num() + 1) *
                   0x9E3779B97F4A7C15ULL)
#endif
                ;
            xoshiro_seed(gaussian_state.rng, seed_val);
            gaussian_state.has_spare = false;
            gaussian_state.seeded = true;
        }

#pragma omp for
        for (int i = 0; i < n_macroparticles; i++) {
            beam_dE[i] = damping_factor * beam_dE[i] - energy_lost
                       + noise_scale * standard_normal(gaussian_state);
        }
    }
}
