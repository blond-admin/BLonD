// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// C++ implementation of the far field of a pole-residue (vector fitting)
// wake, parallelized with OpenMP over poles and vectorized over blocks of
// poles within each thread. Mirrors
// `PythonSpecials.wake_from_pole_residue`; see
// `Specials.wake_from_pole_residue` in blond/core/backends/backend.py for
// the contract.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>

#include "blond_common.h"
#include "openmp.h"

namespace {

// Complex exponential: exp(a + bi) = exp(a) * (cos(b) + i*sin(b))
inline void fast_cexp(const real_t re, const real_t im, real_t &out_re,
                      real_t &out_im) {
  const real_t cmplx_res = FAST_EXP(re);
  out_re = cmplx_res * FAST_COS(im);
  out_im = cmplx_res * FAST_SIN(im);
}

// Complex exponential for an arbitrary step. The fast sine and cosine
// lose accuracy above ~1e8 rad and overflow their quadrant above ~1.7e9
// rad, which a lag of many RF periods reaches; such steps are rare.
inline void exact_cexp(const real_t re, const real_t im, real_t &out_re,
                       real_t &out_im) {
  const real_t cmplx_res = std::exp(re);
  out_re = cmplx_res * std::cos(im);
  out_im = cmplx_res * std::sin(im);
}

// Complex multiply: (a + bi) * (c + di)
inline void cmul(const real_t a_re, const real_t a_im, const real_t b_re,
                 const real_t b_im, real_t &out_re, real_t &out_im) {
  out_re = a_re * b_re - a_im * b_im;
  out_im = a_re * b_im + a_im * b_re;
}

// Arguments shared by every block of one call.
struct PoleKernelArgs {
  const real_t *profile_time;
  const real_t *profile;
  real_t carried_charge;
  bool carried_is_counterrotating;
  real_t state_lag_dt;
  real_t carried_lag_dt;
  const real_t *poles;
  const real_t *residues;
  bool is_counterrotating_beam;
  const real_t *counterrotating_pole_signs;
  real_t factor;
  real_t bin_dt;
  real_t *states;
  int n_bins;
};

// Index of the real part of complex element `i` in an interleaved array.
inline std::size_t re_index(const int i) {
  return static_cast<std::size_t>(2) * static_cast<std::size_t>(i);
}

// B poles advanced together by one pass over the bins. All control flow
// (clock, step, which bin enters) depends only on the time axis, never on
// the pole, so it is shared by the block and paid once per bin instead of
// once per pole; the per-pole arithmetic runs in SIMD lanes, and the
// independent recursions hide the latency of the complex multiply.
template <int B> class PoleBlock {
public:
  // Load poles [first_pole, first_pole + B) and their states.
  PoleBlock(const PoleKernelArgs &args, const int first_pole) {
    for (int k = 0; k < B; k++) {
      const std::size_t pole_n = re_index(first_pole + k);
      pole_re[k] = args.poles[pole_n];
      pole_im[k] = args.poles[pole_n + 1];
      res_re[k] = args.residues[pole_n];
      res_im[k] = args.residues[pole_n + 1];
      state_re[k] = args.states[pole_n];
      state_im[k] = args.states[pole_n + 1];
      // The flip is applied to both the injection and the read-out, so a
      // beam's own wake never flips; a charge carried from the other beam
      // keeps that beam's flip.
      const bool flipped =
          args.counterrotating_pole_signs[first_pole + k] == real_t(-1);
      flip[k] = (args.is_counterrotating_beam && flipped) ? -1 : 1;
      const real_t carried_flip =
          (args.carried_is_counterrotating && flipped) ? -1 : 1;
      // A complex pole stands in for its unstored conjugate partner.
      const real_t pair = (pole_im[k] == real_t(0)) ? real_t(1) : real_t(2);
      injection[k] = flip[k] * pair * args.factor;
      carried_injection[k] = carried_flip * pair * args.carried_charge;
      fast_cexp(pole_re[k] * args.bin_dt, pole_im[k] * args.bin_dt, decay_re[k],
                decay_im[k]);
    }
  }

  // Write the states of poles [first_pole, first_pole + B) back.
  void store(const PoleKernelArgs &args, const int first_pole) const {
    for (int k = 0; k < B; k++) {
      const std::size_t pole_n = re_index(first_pole + k);
      args.states[pole_n] = state_re[k];
      args.states[pole_n + 1] = state_im[k];
    }
  }

  // Multiply every state by exp(pole * step).
  void decay_by(const real_t step) {
    for (int k = 0; k < B; k++) {
      real_t e_re = 0;
      real_t e_im = 0;
      exact_cexp(pole_re[k] * step, pole_im[k] * step, e_re, e_im);
      real_t new_re = 0;
      real_t new_im = 0;
      cmul(state_re[k], state_im[k], e_re, e_im, new_re, new_im);
      state_re[k] = new_re;
      state_im[k] = new_im;
    }
  }

  // Decay the states from `clock` to `to_time`, never backwards. A step of
  // exactly one bin uses the precomputed `decay_*`.
  void advance(real_t &clock, const real_t to_time, const real_t bin_dt,
               const real_t tolerance) {
    const real_t step = to_time - clock;
    if (step <= tolerance) {
      return;
    }
    if (std::fabs(step - bin_dt) <= tolerance) {
#pragma omp simd
      for (int k = 0; k < B; k++) {
        real_t new_re = 0;
        real_t new_im = 0;
        cmul(state_re[k], state_im[k], decay_re[k], decay_im[k], new_re,
             new_im);
        state_re[k] = new_re;
        state_im[k] = new_im;
      }
    } else {
      decay_by(step);
    }
    clock = to_time;
  }

  // Add one bin's charge.
  void inject(const real_t charge) {
#pragma omp simd
    for (int k = 0; k < B; k++) {
      state_re[k] += injection[k] * charge;
    }
  }

  // The block's voltage for the current states.
  real_t read_out() const {
    real_t voltage_sum = 0;
#pragma omp simd reduction(+ : voltage_sum)
    for (int k = 0; k < B; k++) {
      voltage_sum +=
          flip[k] * (res_re[k] * state_re[k] - res_im[k] * state_im[k]);
    }
    return voltage_sum;
  }

  // The carried charge enters after the first read-out; older than the
  // clock it is decayed to it, newer it moves the clock.
  void inject_carried(real_t &clock, const real_t t_carried,
                      const real_t bin_dt, const real_t tolerance) {
    if (t_carried <= clock) {
      for (int k = 0; k < B; k++) {
        real_t e_re = 0;
        real_t e_im = 0;
        exact_cexp(pole_re[k] * (clock - t_carried),
                   pole_im[k] * (clock - t_carried), e_re, e_im);
        state_re[k] += carried_injection[k] * e_re;
        state_im[k] += carried_injection[k] * e_im;
      }
    } else {
      advance(clock, t_carried, bin_dt, tolerance);
      for (int k = 0; k < B; k++) {
        state_re[k] += carried_injection[k];
      }
    }
  }

private:
  alignas(64) std::array<real_t, B> pole_re;
  alignas(64) std::array<real_t, B> pole_im;
  alignas(64) std::array<real_t, B> decay_re;
  alignas(64) std::array<real_t, B> decay_im;
  alignas(64) std::array<real_t, B> res_re;
  alignas(64) std::array<real_t, B> res_im;
  alignas(64) std::array<real_t, B> flip;
  alignas(64) std::array<real_t, B> injection;
  alignas(64) std::array<real_t, B> carried_injection;
  alignas(64) std::array<real_t, B> state_re;
  alignas(64) std::array<real_t, B> state_im;
};

// The far field of poles [first_pole, first_pole + B), added to `vt`.
template <int B>
void wake_from_pole_block(const PoleKernelArgs &args, const int first_pole,
                          real_t *__restrict__ vt) {
  const real_t *__restrict__ profile_time = args.profile_time;
  const real_t *__restrict__ profile = args.profile;
  const int n_bins = args.n_bins;
  const real_t bin_dt = args.bin_dt;
  PoleBlock<B> block(args, first_pole);

  // Times relative to the first bin. A bin is read out two bins behind
  // its own time; the carried charge was emitted before t_0.
  const real_t t_0 = profile_time[0];
  const real_t read_lag = real_t(2) * bin_dt;
  const real_t tolerance = real_t(1e-6) * bin_dt;
  const real_t t_carried = -args.carried_lag_dt;
  const real_t t_handover = (profile_time[n_bins - 1] - t_0) - bin_dt;
  real_t clock = -read_lag - args.state_lag_dt;
  int next_bin = 0;

  for (int bin_i = 0; bin_i < n_bins; bin_i++) {
    const real_t read_time = (profile_time[bin_i] - t_0) - read_lag;
    // every bin but the last enters the state when it is due
    while (next_bin < n_bins - 1 &&
           (profile_time[next_bin] - t_0) <= read_time + tolerance) {
      block.advance(clock, profile_time[next_bin] - t_0, bin_dt, tolerance);
      block.inject(profile[next_bin]);
      next_bin++;
    }
    block.advance(clock, read_time, bin_dt, tolerance);
    vt[bin_i] += block.read_out();
    if (bin_i == 0) {
      block.inject_carried(clock, t_carried, bin_dt, tolerance);
    }
  }
  // Hand over one bin before the last bin, every other bin in.
  for (; next_bin < n_bins - 1; next_bin++) {
    block.advance(clock, profile_time[next_bin] - t_0, bin_dt, tolerance);
    block.inject(profile[next_bin]);
  }
  block.advance(clock, t_handover, bin_dt, tolerance);
  block.store(args, first_pole);
}

// Poles [begin, end) in blocks of 16, the rest in 8, 4, 2, 1.
void wake_from_pole_range(const PoleKernelArgs &args, const int begin,
                          const int end, real_t *__restrict__ vt) {
  int pole_i = begin;
  for (; end - pole_i >= 16; pole_i += 16) {
    wake_from_pole_block<16>(args, pole_i, vt);
  }
  if (end - pole_i >= 8) {
    wake_from_pole_block<8>(args, pole_i, vt);
    pole_i += 8;
  }
  if (end - pole_i >= 4) {
    wake_from_pole_block<4>(args, pole_i, vt);
    pole_i += 4;
  }
  if (end - pole_i >= 2) {
    wake_from_pole_block<2>(args, pole_i, vt);
    pole_i += 2;
  }
  if (end - pole_i >= 1) {
    wake_from_pole_block<1>(args, pole_i, vt);
  }
}

} // namespace

/**
 * Far field of a pole-residue wake, one complex state per pole.
 *
 * Parameters: see Specials.wake_from_pole_residue in backend.py.
 *
 * C-side memory layout:
 * - Complex arrays (poles, residues, states) are interleaved
 *   [re0, im0, re1, im1, ...]; states has n_poles elements.
 * - voltage_threaded is n_threads * n_bins, n_threads >=
 *   omp_get_max_threads(); the wrapper asserts this, so the kernel itself
 *   only needs the actual team size.
 *
 * Every thread owns one contiguous, balanced range of poles (as many
 * threads work as there are poles) and walks it in blocks.
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
    const int n_bins, const int n_poles, const int /*n_threads*/) {
  const PoleKernelArgs args = {profile_time,
                               profile,
                               carried_charge[0],
                               carried_is_counterrotating,
                               state_lag_dt,
                               carried_lag_dt,
                               poles,
                               residues,
                               is_counterrotating_beam,
                               counterrotating_pole_signs,
                               factor,
                               bin_dt,
                               states,
                               n_bins};
  const std::size_t row_size = static_cast<std::size_t>(n_bins);

#pragma omp parallel
  {
    // Only the first `n_used_threads` rows are ever written (one row per
    // pole at most); zeroing and reducing more wastes bandwidth.
    const int team_size = omp_get_num_threads();
    const int n_used_threads = (n_poles < team_size) ? n_poles : team_size;
    const int thread_i = omp_get_thread_num();
    if (thread_i < n_used_threads) {
      real_t *__restrict__ vt =
          voltage_threaded + static_cast<std::size_t>(thread_i) * row_size;
      std::memset(vt, 0, row_size * sizeof(real_t));
      const int begin = static_cast<int>(static_cast<long long>(thread_i) *
                                         n_poles / n_used_threads);
      const int end = static_cast<int>(static_cast<long long>(thread_i + 1) *
                                       n_poles / n_used_threads);
      wake_from_pole_range(args, begin, end, vt);
    }
#pragma omp barrier
#pragma omp for schedule(static)
    for (int bin_i = 0; bin_i < n_bins; bin_i++) {
      real_t sum = 0;
      for (int t = 0; t < n_used_threads; t++) {
        sum += voltage_threaded[static_cast<std::size_t>(t) * row_size +
                                static_cast<std::size_t>(bin_i)];
      }
      voltage[bin_i] = sum;
    }
  }
}
