// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// C++ implementation of the far field of a pole-residue (vector fitting)
// wake, parallelized with OpenMP over poles. Mirrors
// `PythonSpecials.wake_from_pole_residue`; see
// `Specials.wake_from_pole_residue` in blond/core/backends/backend.py for
// the contract.

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "blond_common.h"
#include "openmp.h"

// Complex exponential: exp(a + bi) = exp(a) * (cos(b) + i*sin(b))
static inline void fast_cexp(const real_t re, const real_t im, real_t &out_re,
                             real_t &out_im) {
  const real_t cmplx_res = FAST_EXP(re);
  out_re = cmplx_res * FAST_COS(im);
  out_im = cmplx_res * FAST_SIN(im);
}

// Complex multiply: (a + bi) * (c + di)
static inline void cmul(const real_t a_re, const real_t a_im, const real_t b_re,
                        const real_t b_im, real_t &out_re, real_t &out_im) {
  out_re = a_re * b_re - a_im * b_im;
  out_im = a_re * b_im + a_im * b_re;
}

// Decay a pole's state from `clock` to `to_time`, never backwards. A step of
// exactly one bin uses the precomputed `decay_*`.
static inline void advance_state(real_t &state_re, real_t &state_im,
                                 real_t &clock, const real_t to_time,
                                 const real_t pole_re, const real_t pole_im,
                                 const real_t bin_dt, const real_t decay_re,
                                 const real_t decay_im,
                                 const real_t tolerance) {
  const real_t step = to_time - clock;
  if (step <= tolerance) {
    return;
  }
  real_t e_re, e_im;
  if (fabs(step - bin_dt) <= tolerance) {
    e_re = decay_re;
    e_im = decay_im;
  } else {
    fast_cexp(pole_re * step, pole_im * step, e_re, e_im);
  }
  real_t new_re, new_im;
  cmul(state_re, state_im, e_re, e_im, new_re, new_im);
  state_re = new_re;
  state_im = new_im;
  clock = to_time;
}

/**
 * Far field of a pole-residue wake, one complex state per pole.
 *
 * Parameters: see Specials.wake_from_pole_residue in backend.py.
 *
 * C-side memory layout:
 * - Complex arrays (poles, residues, states) are interleaved
 *   [re0, im0, re1, im1, ...]; states has n_poles elements.
 * - voltage_threaded is n_threads * n_bins, n_threads >=
 *   omp_get_max_threads().
 */
extern "C" void wake_from_pole_residue(
    const real_t *__restrict__ profile_time, const real_t *__restrict__ profile,
    const real_t *__restrict__ carried_charge,
    const bool carried_is_counterrotating, const real_t state_lag_dt,
    const real_t carried_lag_dt, const real_t *__restrict__ poles,
    const real_t *__restrict__ residues, const bool is_counterrotating_beam,
    const real_t *__restrict__ counterrotating_pole_signs, const real_t factor,
    const real_t bin_dt, real_t *__restrict__ states,
    real_t *__restrict__ voltage, real_t *__restrict__ voltage_threaded,
    const int n_bins, const int n_poles, const int n_threads) {
  // Only the first `n_used_threads` rows are ever written (one row per
  // pole at most); zeroing and reducing more wastes bandwidth.
  const int n_used_threads = (n_poles < n_threads) ? n_poles : n_threads;
  memset(voltage, 0, n_bins * sizeof(real_t));
  memset(voltage_threaded, 0, (size_t)n_used_threads * n_bins * sizeof(real_t));

  // Times relative to the first bin. A bin is read out two bins behind
  // its own time; the carried charge was emitted before t_0.
  const real_t t_0 = profile_time[0];
  const real_t read_lag = real_t(2) * bin_dt;
  const real_t tolerance = real_t(1e-6) * bin_dt;
  const real_t t_carried = -carried_lag_dt;
  const real_t t_handover = (profile_time[n_bins - 1] - t_0) - bin_dt;

#pragma omp parallel for schedule(static)
  for (int pole_i = 0; pole_i < n_poles; pole_i++) {
    const int thread_i = omp_get_thread_num();
    const int pole_n = 2 * pole_i;
    const real_t pole_re = poles[pole_n];
    const real_t pole_im = poles[pole_n + 1];
    const real_t res_re = residues[pole_n];
    const real_t res_im = residues[pole_n + 1];

    // The flip is applied to both the injection and the read-out, so a
    // beam's own wake never flips; a charge carried from the other beam
    // keeps that beam's flip.
    const bool flipped = counterrotating_pole_signs[pole_i] == real_t(-1);
    const real_t flip = (is_counterrotating_beam && flipped) ? -1 : 1;
    const real_t carried_flip =
        (carried_is_counterrotating && flipped) ? -1 : 1;
    // A complex pole stands in for its unstored conjugate partner.
    const real_t pair = (pole_im == real_t(0)) ? real_t(1) : real_t(2);
    real_t decay_re, decay_im;
    fast_cexp(pole_re * bin_dt, pole_im * bin_dt, decay_re, decay_im);

    real_t state_re = states[pole_n];
    real_t state_im = states[pole_n + 1];
    real_t clock = -read_lag - state_lag_dt;
    int next_bin = 0;
    real_t *__restrict__ vt = voltage_threaded + (size_t)thread_i * n_bins;

    for (int bin_i = 0; bin_i < n_bins; bin_i++) {
      const real_t read_time = (profile_time[bin_i] - t_0) - read_lag;
      // every bin but the last enters the state when it is due
      while (next_bin < n_bins - 1 &&
             (profile_time[next_bin] - t_0) <= read_time + tolerance) {
        advance_state(state_re, state_im, clock, profile_time[next_bin] - t_0,
                      pole_re, pole_im, bin_dt, decay_re, decay_im, tolerance);
        state_re += flip * pair * factor * profile[next_bin];
        next_bin++;
      }
      advance_state(state_re, state_im, clock, read_time, pole_re, pole_im,
                    bin_dt, decay_re, decay_im, tolerance);
      vt[bin_i] += flip * (res_re * state_re - res_im * state_im);
      if (bin_i == 0) {
        // The carried charge enters after the first read-out; older than
        // the clock it is decayed to it, newer it moves the clock.
        if (t_carried <= clock) {
          real_t e_re, e_im;
          fast_cexp(pole_re * (clock - t_carried),
                    pole_im * (clock - t_carried), e_re, e_im);
          const real_t charge = carried_flip * pair * carried_charge[0];
          state_re += charge * e_re;
          state_im += charge * e_im;
        } else {
          advance_state(state_re, state_im, clock, t_carried, pole_re, pole_im,
                        bin_dt, decay_re, decay_im, tolerance);
          state_re += carried_flip * pair * carried_charge[0];
        }
      }
    }
    // Hand over one bin before the last bin, every other bin in.
    while (next_bin < n_bins - 1) {
      advance_state(state_re, state_im, clock, profile_time[next_bin] - t_0,
                    pole_re, pole_im, bin_dt, decay_re, decay_im, tolerance);
      state_re += flip * pair * factor * profile[next_bin];
      next_bin++;
    }
    advance_state(state_re, state_im, clock, t_handover, pole_re, pole_im,
                  bin_dt, decay_re, decay_im, tolerance);

    states[pole_n] = state_re;
    states[pole_n + 1] = state_im;
  }

#pragma omp parallel for schedule(static)
  for (int bin_i = 0; bin_i < n_bins; bin_i++) {
    real_t sum = 0;
    for (int t = 0; t < n_used_threads; t++) {
      sum += voltage_threaded[(size_t)t * n_bins + bin_i];
    }
    voltage[bin_i] = sum;
  }
}
