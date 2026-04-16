// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENCE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// C++ implementation of induced voltage calculation using pole-residue
// (vector fitting) models, parallelized with OpenMP over poles.

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "blond_common.h"
#include "openmp.h"

// Complex exponential: exp(a + bi) = exp(a) * (cos(b) + i*sin(b))
static inline void fast_cexp(const real_t re, const real_t im,
                             real_t &out_re, real_t &out_im) {
    const real_t e = FAST_EXP(re);
    out_re = e * FAST_COS(im);
    out_im = e * FAST_SIN(im);
}

// Complex multiply: (a + bi) * (c + di)
static inline void cmul(const real_t a_re, const real_t a_im,
                        const real_t b_re, const real_t b_im,
                        real_t &out_re, real_t &out_im) {
    out_re = a_re * b_re - a_im * b_im;
    out_im = a_re * b_im + a_im * b_re;
}

/**
 * Apply poles based on the profile to generate voltage.
 *
 * Complex arrays (poles, residues, states) are interleaved:
 *   [re0, im0, re1, im1, ...]
 *
 * Parameters
 * ----------
 * profile        : Beam profile histogram, length n_bins.
 * profile_dts    : Time step base, length n_profile_dts (>= n_bins + 1).
 * poles          : Complex poles, interleaved, length 2 * n_poles.
 * residues       : Complex residues, interleaved, length 2 * n_poles.
 * states         : Complex state vector, interleaved, length 2 * (n_poles + 1).
 *                  Last complex element stores t_start (real part only).
 * voltage        : Output voltage [V], length n_bins.
 * voltage_threaded : Per-thread voltage buffer, length n_threads * n_bins.
 * update_on_bin  : Bin indices triggering dt update, length n_updates.
 * factor         : Conversion factor (profile to current per bin [A]).
 * n_bins         : Number of bins in profile.
 * n_poles        : Number of poles.
 * n_threads      : Size of first dimension of voltage_threaded (>= omp_get_max_threads()).
 * n_updates      : Length of update_on_bin.
 * n_profile_dts  : Length of profile_dts.
 */
extern "C" void apply_poles(
    const real_t *__restrict__ profile,
    const real_t *__restrict__ profile_dts,
    const real_t *__restrict__ poles,
    const real_t *__restrict__ residues,
    const bool beam_counter_rotation_flag,
    const real_t *__restrict__ cr_pole_flip_flags,
    real_t *__restrict__ states,
    real_t *__restrict__ voltage,
    real_t *__restrict__ voltage_threaded,
    const int *__restrict__ update_on_bin,
    const real_t factor,
    const int n_bins,
    const int n_poles,
    const int n_threads,
    const int n_updates,
    const int n_profile_dts)
{
    const real_t two_factor = real_t(2) * factor;

    // Zero voltage and voltage_threaded from previous call
    memset(voltage, 0, n_bins * sizeof(real_t));
    memset(voltage_threaded, 0, (size_t)n_threads * n_bins * sizeof(real_t));

    // t_start from states[-1] (real part of last complex element)
    const real_t t_start = states[2 * n_poles];

    // Parallel over poles: each pole carries sequential state across bins,
    // but different poles are fully independent.
#pragma omp parallel for schedule(static)
    for (int pole_i = 0; pole_i < n_poles; pole_i++) {
        const int thread_i = omp_get_thread_num();

        real_t cr_pole_flip = 1;
        if (beam_counter_rotation_flag && cr_pole_flip_flags[pole_i]) {
            cr_pole_flip = -1;
        }

        const real_t pole_re = poles[2 * pole_i];
        const real_t pole_im = poles[2 * pole_i + 1];
        const real_t res_re = residues[2 * pole_i];
        const real_t res_im = residues[2 * pole_i + 1];

        real_t state_re = states[2 * pole_i];
        real_t state_im = states[2 * pole_i + 1];

        int i_update = 0;
        int update_on_bin_i = (n_updates > 0) ? update_on_bin[0] : -1;

        real_t decay_re = 0, decay_im = 0;
        real_t *__restrict__ vt = voltage_threaded + (size_t)thread_i * n_bins;

        for (int bin_i = 0; bin_i < n_bins; bin_i++) {
            const real_t profile_i_ = real_t(0.5) * profile[bin_i];

            if (bin_i == update_on_bin_i) {
                // Compute t_jump (real scalar)
                real_t t_jump;
                if (bin_i == 0) {
                    t_jump = profile_dts[0] - t_start;
                } else {
                    t_jump = profile_dts[bin_i] - profile_dts[bin_i - 1];
                }

                // state *= exp(pole * t_jump)
                real_t e_re, e_im;
                fast_cexp(pole_re * t_jump, pole_im * t_jump, e_re, e_im);

                real_t new_re, new_im;
                cmul(state_re, state_im, e_re, e_im, new_re, new_im);
                state_re = new_re;
                state_im = new_im;

                // decay = exp(pole * dt)
                const real_t dt = profile_dts[bin_i + 1] - profile_dts[bin_i];
                fast_cexp(pole_re * dt, pole_im * dt, decay_re, decay_im);

                i_update++;
                if (i_update < n_updates) {
                    update_on_bin_i = update_on_bin[i_update];
                }
            } else {
                // state *= decay
                real_t new_re, new_im;
                cmul(state_re, state_im, decay_re, decay_im, new_re, new_im);
                state_re = new_re;
                state_im = new_im;
            }

            // state += profile_i_ (real part only, imag part is zero)
            state_re += cr_pole_flip * profile_i_;

            // amp = Re(residue * state)
            const real_t amp = res_re * state_re - res_im * state_im;
            vt[bin_i] += cr_pole_flip * two_factor * amp;

            // state += profile_i_ (second half of trapezoidal rule)
            state_re += cr_pole_flip * profile_i_;
        }

        // Store state back
        states[2 * pole_i] = state_re;
        states[2 * pole_i + 1] = state_im;
    }

    // Reduce voltage_threaded into voltage (parallel over bins)
#pragma omp parallel for schedule(static)
    for (int bin_i = 0; bin_i < n_bins; bin_i++) {
        real_t sum = 0;
        for (int t = 0; t < n_threads; t++) {
            sum += voltage_threaded[(size_t)t * n_bins + bin_i];
        }
        voltage[bin_i] = sum;
    }

    // Store last profile_dts value into states[-1] for next call
    states[2 * n_poles] = profile_dts[n_profile_dts - 1];
    states[2 * n_poles + 1] = 0;
}
