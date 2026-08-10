// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// C++ implementation of the travelling-wave-cavity wake via a phasor FIR
// recursion, parallelized with OpenMP over cavity modes.
//
// Per mode, one rotating phasor (pole on the imaginary axis) carries the
// cosine wake; a sliding-window subtraction (the FIR part) builds the
// linear (1 - t/a_tilde) taper and terminates the wake after the filling
// time a_tilde. Assumes an equidistant profile grid (bin_dt).

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "blond_common.h"
#include "openmp.h"

// Complex multiply: (a + bi) * (c + di)
static inline void twc_cmul(const real_t a_re, const real_t a_im,
                            const real_t b_re, const real_t b_im,
                            real_t &out_re, real_t &out_im) {
    out_re = a_re * b_re - a_im * b_im;
    out_im = a_re * b_im + a_im * b_re;
}

/**
 * Travelling-wave-cavity wake via a phasor FIR recursion.
 *
 * Parameters
 * ----------
 * profile        : Beam profile histogram, length n_bins (equidistant grid).
 * r_shunt        : Shunt impedance per mode [Ohm], length n_modes.
 * a_tilde        : Wake support (filling) time per mode [s], length n_modes.
 * omega_r        : Angular resonant frequency per mode [rad/s], length n_modes.
 * bin_dt         : Bin width of the profile grid [s].
 * factor         : Conversion factor (profile to current per bin [A]).
 * voltage        : Output voltage [V], length n_bins. Overwritten.
 * voltage_threaded : Per-thread voltage buffer, length n_threads * n_bins.
 * n_bins         : Number of bins in profile.
 * n_modes        : Number of cavity modes.
 * n_threads      : Size of first dimension of voltage_threaded.
 */
extern "C" void wake_from_twc_fir(
    const real_t *__restrict__ profile,
    const real_t *__restrict__ r_shunt,
    const real_t *__restrict__ a_tilde,
    const real_t *__restrict__ omega_r,
    const real_t bin_dt,
    const real_t factor,
    real_t *__restrict__ voltage,
    real_t *__restrict__ voltage_threaded,
    const int n_bins,
    const int n_modes,
    const int n_threads)
{
    const real_t two_factor = real_t(2) * factor;

    // Zero voltage and voltage_threaded from previous call
    memset(voltage, 0, n_bins * sizeof(real_t));
    memset(voltage_threaded, 0, (size_t)n_threads * n_bins * sizeof(real_t));

    // Parallel over modes: each mode carries sequential state across bins,
    // but different modes are fully independent.
#pragma omp parallel for schedule(static)
    for (int mode_i = 0; mode_i < n_modes; mode_i++) {
        const int thread_i = omp_get_thread_num();
        real_t *__restrict__ vt = voltage_threaded + (size_t)thread_i * n_bins;

        // W(0) amplitude of the single-cosine kernel (no conjugate pair);
        // the trapezoidal half/half injection below supplies the
        // (sign(t) + 1) factor of the analytic wake
        const real_t wake_amplitude =
            real_t(2) * r_shunt[mode_i] / a_tilde[mode_i];
        // taper length in bins; ceil quantizes the wake support to the
        // grid (relative error ~ bin_dt / a_tilde)
        const int n_taper = (int)ceil((double)(a_tilde[mode_i] / bin_dt));
        const real_t inv_taper = two_factor / (real_t)n_taper;

        const real_t phase = omega_r[mode_i] * bin_dt;
        const real_t rot_re = (real_t)cos((double)phase);
        const real_t rot_im = (real_t)sin((double)phase);
        // phase accumulated by a taper term over its full lifetime
        const real_t removal_phase = phase * (real_t)n_taper;
        const real_t removal_re = (real_t)cos((double)removal_phase);
        const real_t removal_im = (real_t)sin((double)removal_phase);

        real_t state_re = 0, state_im = 0;
        real_t taper_re = 0, taper_im = 0;

        for (int bin_i = 0; bin_i < n_bins; bin_i++) {
            if (bin_i > 0) {
                taper_re += profile[bin_i - 1] * inv_taper;
                const int oldest = bin_i - 1 - n_taper;
                if (oldest >= 0) {
                    // fully decayed: remove with its accumulated phase
                    const real_t old_amp = profile[oldest] * inv_taper;
                    taper_re -= old_amp * removal_re;
                    taper_im -= old_amp * removal_im;
                }
                real_t new_re, new_im;
                twc_cmul(taper_re, taper_im, rot_re, rot_im, new_re, new_im);
                taper_re = new_re;
                taper_im = new_im;

                state_re -= taper_re;
                state_im -= taper_im;
            }

            const real_t profile_i_half =
                real_t(0.5) * profile[bin_i] * two_factor;

            state_re += profile_i_half;
            vt[bin_i] += wake_amplitude * state_re;
            state_re += profile_i_half;

            real_t new_re, new_im;
            twc_cmul(state_re, state_im, rot_re, rot_im, new_re, new_im);
            state_re = new_re;
            state_im = new_im;
        }
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
}
