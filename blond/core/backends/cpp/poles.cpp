// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
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
    const real_t cmplx_res = FAST_EXP(re);
    out_re = cmplx_res * FAST_COS(im);
    out_im = cmplx_res * FAST_SIN(im);
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
 * Each pole's state is read, advanced by one bin and then given the bin's
 * charge, in that order. So the state a bin reads holds the history
 * referenced one bin back and NOT the bin's own charge: this kernel covers
 * lags of one bin or more, and the caller adds the self-bin term. That is
 * what lets the residues carry the triangle bin-average
 * ((exp(p*dt) - 1) / (p*dt))^2, whose lag likewise starts at dt and which
 * stays bounded by one at any binning.
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
 * is_counterrotating_beam : If true, the current beam is counter-rotating.
 * counterrotating_pole_signs :  Array per pole, -1 if the sign of the
 *                               impedance is flipped for a counter-rotating beam.
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
extern "C" void wake_from_pole_residue(
    const real_t *__restrict__ profile,
    const real_t *__restrict__ profile_dts,
    const real_t *__restrict__ poles,
    const real_t *__restrict__ residues,
    const bool is_counterrotating_beam,
    const real_t *__restrict__ counterrotating_pole_signs,
    const int *__restrict__ update_on_bin,
    const real_t factor,
    real_t *__restrict__ states,
    real_t *__restrict__ voltage,
    real_t *__restrict__ voltage_threaded,
    const int n_bins,
    const int n_poles,
    const int n_threads,
    const int n_updates,
    const int n_profile_dts)
{
    const real_t two_factor = real_t(2) * factor;

    // Only the first `n_used_threads` rows of `voltage_threaded` are ever written:
    // the pole loop is parallelised over poles, so at most one row per pole
    // (and never more than `n_threads`) is touched. Zeroing/reducing all
    // `n_threads` rows when `n_poles` is small wastes O(n_threads * n_bins)
    // of memory bandwidth, which dominates the (cheap) recursion for a few
    // poles. Size the work to what is actually used.
    const int n_used_threads = (n_poles < n_threads) ? n_poles : n_threads;

    // Zero voltage and the used rows of voltage_threaded from previous call
    memset(voltage, 0, n_bins * sizeof(real_t));
    memset(voltage_threaded, 0, (size_t)n_used_threads * n_bins * sizeof(real_t));

    // t_start from states[-1] (real part of last complex element)
    const real_t t_start = states[2 * n_poles];

    // The bin-averaged wake reaches one bin further back than the state's own
    // reference (the residues carry the triangle factor
    // ((exp(p*dt) - 1) / (p*dt))^2, whose lag starts at dt). Every bin is
    // bin_dt wide -- a sparse profile's gaps are whole numbers of bins -- so
    // that lag is the same everywhere.
    const real_t bin_dt = profile_dts[1] - profile_dts[0];

    // Parallel over poles: each pole carries sequential state across bins,
    // but different poles are fully independent. With schedule(static) and
    // n_poles < n_threads, only threads [0, n_poles) receive iterations, so
    // the rows written are exactly [0, n_used_threads) -- the rows we zero and reduce.
    // (We deliberately do NOT use num_threads(n_used_threads): resizing the team each
    // call thrashes OpenMP's thread pool and is far slower than the savings.)
#pragma omp parallel for schedule(static)
    for (int pole_i = 0; pole_i < n_poles; pole_i++) {
        const int thread_i = omp_get_thread_num();

        // `cr_pole_flip` is intentionally applied to BOTH the state injection
        // and the output amplitude: for the counter-rotating beam's own wake
        // the two factors cancel (flip * flip == 1); only contributions of
        // the other beam, accumulated in the shared `states`, see a net
        // sign flip.
        real_t cr_pole_flip = 1;
        if (is_counterrotating_beam) {
            if (counterrotating_pole_signs[pole_i] == -1) {
                cr_pole_flip = -1;
            }
        }
        const int pole_n = 2 * pole_i;
        const real_t pole_re = poles[pole_n];
        const real_t pole_im = poles[pole_n + 1];
        const real_t res_re = residues[pole_n];
        const real_t res_im = residues[pole_n + 1];

        real_t state_re = states[pole_n];
        real_t state_im = states[pole_n + 1];

        int i_update = 0;
        int update_on_bin_i = (n_updates > 0) ? update_on_bin[0] : -1;

        // `state_prev` lags `state` by one bin. At a call boundary the two
        // coincide: everything the previous call saw is at least two bins in
        // the past by the time this one starts.
        real_t state_prev_re = state_re;
        real_t state_prev_im = state_im;

        real_t decay_re = 0, decay_im = 0;
        real_t advance_re = 0, advance_im = 0;
        real_t res_lb_re = res_re, res_lb_im = res_im;
        real_t chunk_dt = 0;
        real_t jump_prev = 0;
        int bins_since_jump = 2;  // force the lag factor on the first two bins
        real_t *__restrict__ vt = voltage_threaded + (size_t)thread_i * n_bins;

        for (int bin_i = 0; bin_i < n_bins; bin_i++) {

            real_t t_jump;
            if (bin_i == update_on_bin_i) {
                // decay = exp(pole * chunk_dt)
                chunk_dt = profile_dts[bin_i + 1] - profile_dts[bin_i];
                fast_cexp(pole_re * chunk_dt, pole_im * chunk_dt, decay_re, decay_im);

                if (bin_i == 0) {
                    t_jump = profile_dts[0] - t_start;
                } else {
                    t_jump = profile_dts[bin_i] - profile_dts[bin_i - 1];
                }
                // advance = exp(pole * t_jump)
                fast_cexp(pole_re * t_jump, pole_im * t_jump, advance_re, advance_im);
                bins_since_jump = 0;

                i_update++;
                if (i_update < n_updates) {
                    update_on_bin_i = update_on_bin[i_update];
                }
            } else {
                t_jump = chunk_dt;
                advance_re = decay_re;
                advance_im = decay_im;
            }

            if (bins_since_jump < 2) {
                // `state_prev` is referenced two bins back only when the last
                // two steps were both one bin wide; otherwise reach across
                // whatever they actually were. The lag is clamped at zero so
                // the exponent keeps a non-positive real part and cannot
                // overflow -- a caller handing in a state less than two bins
                // old has nothing to reach back to, and only a zero state may
                // do so.
                const real_t lag = t_jump + jump_prev - real_t(2) * bin_dt;
                if (lag > real_t(0)) {
                    real_t lb_re, lb_im;
                    fast_cexp(pole_re * lag, pole_im * lag, lb_re, lb_im);
                    cmul(res_re, res_im, lb_re, lb_im, res_lb_re, res_lb_im);
                } else {
                    res_lb_re = res_re;
                    res_lb_im = res_im;
                }
                bins_since_jump++;
            } else {
                res_lb_re = res_re;
                res_lb_im = res_im;
            }

            // Read the state that lags by two bins: the bin-averaged wake
            // starts three half-bins back, so this recursion covers lags of
            // two bins and more. The nearer three lags -- the previous bin,
            // this one and the next -- are added by the solver.
            const real_t amp = res_lb_re * state_prev_re - res_lb_im * state_prev_im;
            vt[bin_i] += cr_pole_flip * amp;

            state_prev_re = state_re;
            state_prev_im = state_im;

            // state *= advance
            {
                real_t new_re, new_im;
                cmul(state_re, state_im, advance_re, advance_im, new_re, new_im);
                state_re = new_re;
                state_im = new_im;
            }

            // Inject this bin's charge (real part only, imag part is zero).
            state_re += cr_pole_flip * profile[bin_i] * two_factor;
            jump_prev = t_jump;
        }

        // Store state back
        states[2 * pole_i] = state_re;
        states[2 * pole_i + 1] = state_im;
    }

    // Reduce the used rows of voltage_threaded into voltage (parallel over bins)
#pragma omp parallel for schedule(static)
    for (int bin_i = 0; bin_i < n_bins; bin_i++) {
        real_t sum = 0;
        for (int t = 0; t < n_used_threads; t++) {
            sum += voltage_threaded[(size_t)t * n_bins + bin_i];
        }
        voltage[bin_i] = sum;
    }

    // Store last profile_dts value into states[-1] for next call
    states[2 * n_poles] = profile_dts[n_profile_dts - 1];
    states[2 * n_poles + 1] = 0;
}
