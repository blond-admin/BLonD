// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENCE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Induced-voltage calculation via pole-residue (vector-fitting) models.
// Parallelised with OpenMP: each pole is independent and owns its state.

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "blond_common.h"
#include "openmp.h"

static inline void complex_exp(const real_t exponent_real,
                                const real_t exponent_imag,
                                real_t &out_real, real_t &out_imag)
{
    const real_t magnitude = FAST_EXP(exponent_real);
    out_real = magnitude * FAST_COS(exponent_imag);
    out_imag = magnitude * FAST_SIN(exponent_imag);
}

// ---------------------------------------------------------------------------

/**
 * Compute the induced voltage from a pole-residue impedance model.
 *
 * Physical model
 * --------------
 * Each circuit pole p_k with residue c_k satisfies
 *
 *   ds_k/dt = p_k * s_k(t) + I(t),    V(t) = 2 * factor * Re(sum_k c_k * s_k(t))
 *
 * where I(t) is the beam current (particle counts; physical conversion via `factor`).
 *
 * Per-bin integration
 * -------------------
 * The current is assumed uniform over each bin: I = profile[i] / dt.
 * The ODE then has the exact solution
 *
 *   s(t_left + tau) = exp(pole*tau) * s_left + (profile[i]/dt/pole) * (exp(pole*tau) - 1)
 *
 * Three per-bunch constants are precomputed from this:
 *
 *   propagator = exp(pole * dt)                      // free decay over one bin
 *   kernel     = (propagator - 1) / (pole * dt)      // maps bin_count → state advance
 *   int2_kernel = (kernel - 1)     / (pole * dt)      // (propagator - 1 - pole*dt) / (pole*dt)²
 *                                                      // = (1/dt²) integral_0^dt integral_0^tau exp(p*u) du dtau
 *
 * kernel appears in both formulas because it is both the charge-injection coefficient
 * for the state advance and the decay coefficient for the bin-average:
 *
 *   s_right = propagator * s_left + bin_count * kernel      // state at right edge
 *   <s>     = kernel * s_left     + bin_count * int2_kernel  // bin-average state
 *
 * V[i] = 2 * factor * Re(residue * <s>[i])
 *
 * Using the bin-average (rather than a point sample at the bin centre) ensures that
 * resonators faster than the profile's Nyquist frequency contribute zero net voltage,
 * consistent with a frequency-domain solver.
 *
 * Inter-bunch gaps are bridged by a free-decay step exp(pole * t_jump).
 *
 * State semantics
 * ---------------
 * states[-1] encodes the reference time:
 *   on entry  — left edge of the first bin to process
 *   on return — right edge of the last bin processed
 * Python initialises it to profile_dts[0] - dt/2 and subtracts the inter-turn
 * elapsed time before each subsequent call.
 *
 * Array layout: all complex arrays are interleaved [re_0, im_0, re_1, im_1, ...].
 */
extern "C" void apply_poles(
    const real_t *__restrict__ profile,
    const real_t *__restrict__ profile_dts,
    const real_t *__restrict__ poles,
    const real_t *__restrict__ residues,
    real_t       *__restrict__ states,
    real_t       *__restrict__ voltage,
    real_t       *__restrict__ voltage_threaded,
    const int    *__restrict__ update_on_bin,
    const real_t factor,
    const int    n_bins,
    const int    n_poles,
    const int    n_threads,
    const int    n_updates,
    const int    n_profile_dts)
{
    const real_t voltage_scale = real_t(2) * factor;

    memset(voltage_threaded, 0, (size_t)n_threads * n_bins * sizeof(real_t));

    const real_t reference_time = states[2 * n_poles];  // left edge of first bin

#pragma omp parallel for schedule(static)
    for (int pole_index = 0; pole_index < n_poles; pole_index++) {
        const int thread_index = omp_get_thread_num();

        const real_t pole_real    = poles    [2 * pole_index    ];
        const real_t pole_imag    = poles    [2 * pole_index + 1];
        const real_t residue_real = residues [2 * pole_index    ];
        const real_t residue_imag = residues [2 * pole_index + 1];

        real_t state_real = states[2 * pole_index    ];
        real_t state_imag = states[2 * pole_index + 1];

        // Precomputed per-bunch constants (refreshed at each bunch boundary).
        real_t propagator_real = 0, propagator_imag = 0;  // exp(pole * dt)
        real_t kernel_real     = 0, kernel_imag     = 0;  // (propagator - 1) / (pole * dt)
        real_t int2_kernel_real = 0, int2_kernel_imag = 0;  // (kernel - 1) / (pole * dt)  — see Step 1 for derivation
        real_t half_dt = 0;  // dt/2, used only to compute bin-edge timestamps

        real_t prev_right_edge = reference_time;
        int    update_index    = 0;
        int    next_update_bin = (n_updates > 0) ? update_on_bin[0] : -1;

        real_t *__restrict__ thread_voltage = voltage_threaded + (size_t)thread_index * n_bins;

        for (int bin_index = 0; bin_index < n_bins; bin_index++) {

            // ---- Bunch boundary: recompute propagators and bridge the inter-bunch gap ----
            if (bin_index == next_update_bin) {

                const real_t dt = profile_dts[bin_index + 1] - profile_dts[bin_index];
                half_dt = dt * real_t(0.5);

                complex_exp(pole_real * dt, pole_imag * dt, propagator_real, propagator_imag);

                // Both kernel and int2_kernel have the form (z - 1) / (pole * dt).
                // Complex division: Re(z/w) = (Re(z)*Re(w) + Im(z)*Im(w)) / |w|^2
                //                   Im(z/w) = (Im(z)*Re(w) - Re(z)*Im(w)) / |w|^2
                // Here w = pole * dt, so one dt cancels: denominator = pole_norm_squared * dt.
                const real_t pole_norm_squared = pole_real * pole_real + pole_imag * pole_imag;

                if (pole_norm_squared > real_t(0)) {

                    const real_t pm1_real = propagator_real - real_t(1);
                    const real_t pm1_imag = propagator_imag;
                    kernel_real = (pm1_real * pole_real + pm1_imag * pole_imag) / (pole_norm_squared * dt);
                    kernel_imag = (pm1_imag * pole_real - pm1_real * pole_imag) / (pole_norm_squared * dt);

                    const real_t km1_real = kernel_real - real_t(1);
                    const real_t km1_imag = kernel_imag;
                    int2_kernel_real = (km1_real * pole_real + km1_imag * pole_imag) / (pole_norm_squared * dt);
                    int2_kernel_imag = (km1_imag * pole_real - km1_real * pole_imag) / (pole_norm_squared * dt);

                } else {
                    // pole = 0: L'Hopital gives kernel -> 1, int2_kernel -> 1/2 (uniform average)
                    kernel_real     = real_t(1);   kernel_imag     = real_t(0);
                    int2_kernel_real = real_t(0.5); int2_kernel_imag = real_t(0);
                }

                // Free-decay over the inter-bunch gap (zero for equidistant bins).
                const real_t t_jump = (profile_dts[bin_index] - half_dt) - prev_right_edge;
                if (t_jump > real_t(0)) {
                    real_t jump_real, jump_imag;
                    complex_exp(pole_real * t_jump, pole_imag * t_jump, jump_real, jump_imag);
                    const real_t decayed_real = jump_real * state_real - jump_imag * state_imag;
                    const real_t decayed_imag = jump_real * state_imag + jump_imag * state_real;
                    state_real = decayed_real;
                    state_imag = decayed_imag;
                }

                update_index++;
                if (update_index < n_updates)
                    next_update_bin = update_on_bin[update_index];
            }

            // ---- Per-bin: compute bin-average voltage, then advance state ----
            const real_t amplitude = profile[bin_index];

            // Step 1 — bin-average state:  <s> = kernel * s_left + amplitude * int2_kernel
            //
            // Derivation: s(tau) = exp(p*tau)*s_left + (q/dt) * integral_0^tau exp(p*(tau-u)) du
            //             <s>    = (1/dt) * integral_0^dt s(tau) dtau
            //
            // The amplitude term picks up two integrals:
            //   inner: integral_0^tau exp(p*(tau-u)) du  = (exp(p*tau) - 1) / p
            //   outer: (1/dt^2) * integral_0^dt (exp(p*tau) - 1) / p dtau  = int2_kernel
            //
            // int2_kernel = (kernel - 1) / (pole * dt)  = (propagator - 1 - pole*dt) / (pole*dt)^2
            const real_t avg_state_real = kernel_real * state_real - kernel_imag * state_imag
                                        + amplitude * int2_kernel_real;
            const real_t avg_state_imag = kernel_real * state_imag + kernel_imag * state_real
                                        + amplitude * int2_kernel_imag;

            // Step 2 — voltage:  V[i] = 2 * factor * Re(residue * <s>)
            thread_voltage[bin_index] += voltage_scale
                * (residue_real * avg_state_real - residue_imag * avg_state_imag);

            // Step 3 — advance to right edge:  s_right = propagator * s_left + amplitude * kernel
            const real_t next_state_real = propagator_real * state_real - propagator_imag * state_imag
                                         + amplitude * kernel_real;
            const real_t next_state_imag = propagator_real * state_imag + propagator_imag * state_real
                                         + amplitude * kernel_imag;
            state_real = next_state_real;
            state_imag = next_state_imag;

            prev_right_edge = profile_dts[bin_index] + half_dt;  // right edge of this bin
        }

        states[2 * pole_index    ] = state_real;
        states[2 * pole_index + 1] = state_imag;
    }

    // Reduce per-thread voltage buffers into the output array.
#pragma omp parallel for schedule(static)
    for (int bin_index = 0; bin_index < n_bins; bin_index++) {
        real_t voltage_sum = real_t(0);
        for (int thread_index = 0; thread_index < n_threads; thread_index++)
            voltage_sum += voltage_threaded[(size_t)thread_index * n_bins + bin_index];
        voltage[bin_index] = voltage_sum;
    }

    // Store the right edge of the last bin for the next call.
    const real_t last_half_dt =
        (profile_dts[n_profile_dts - 1] - profile_dts[n_profile_dts - 2]) * real_t(0.5);
    states[2 * n_poles    ] = profile_dts[n_profile_dts - 1] + last_half_dt;
    states[2 * n_poles + 1] = real_t(0);
}
