// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Precondition for every kernel in this file: coordinates are finite.
// The beam coordinates (beam_dt, beam_dE) and the profile coordinates
// (bin_centers, cut edges) must contain neither NaN nor +/-Inf. Nothing
// here checks for it -- the check would not be free in a per-particle
// loop. Note that the guards protecting a C++ conversion of a bin index
// to `int` are written as `index < lo || index >= hi`: a NaN index
// compares false against both bounds, passes the guard and reaches the
// conversion, which is undefined behaviour (`floor_to_int` avoids the
// C++ conversion and instead files a NaN in bin 0). The caller must not
// produce non-finite coordinates. See `Specials` in
// blond/core/backends/backend.py.

#ifdef USEFLOAT
typedef float real_t;
#else
typedef double real_t;
#endif

// Integer type of macro-particle counts and particle loop counters.
// Must match `INDEX_DTYPE` in blond/core/backends/backend.py.
typedef long long index_t;

extern "C" __global__ void drift_simple(real_t *__restrict__ beam_dt,
                                        real_t *__restrict__ beam_dE,
                                        const real_t T, const real_t eta_zero,
                                        const real_t beta, const real_t energy,
                                        const index_t n_macroparticles) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  real_t coeff = T * eta_zero / (beta * beta * energy);
  for (index_t i = tid; i < n_macroparticles; i = i + blockDim.x * gridDim.x)
    beam_dt[i] += coeff * beam_dE[i];
}

// Drift with the linear slip factor but the exact relativistic delta;
// reproduces the longitudinal drift of an xsuite LineSegmentMap.
extern "C" __global__ void
drift_like_line_segment(real_t *__restrict__ beam_dt,
                        real_t *__restrict__ beam_dE, const real_t T,
                        const real_t eta_zero, const real_t beta,
                        const real_t energy, const index_t n_macroparticles) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const real_t inv_beta_sq = 1.0 / (beta * beta);
  const real_t inv_energy = 1.0 / energy;
  for (index_t i = tid; i < n_macroparticles; i = i + blockDim.x * gridDim.x) {
    const real_t dE = beam_dE[i];
    const real_t delta =
        sqrt(1.0 + inv_beta_sq * (dE * dE * inv_energy * inv_energy +
                                  2.0 * dE * inv_energy)) -
        1.0;
    beam_dt[i] += T * eta_zero * delta;
  }
}

extern "C" __global__ void
kick_single_harmonic(real_t *__restrict__ beam_dt, real_t *__restrict__ beam_dE,
                     const real_t charge, const real_t voltage,
                     const real_t omega_RF, const real_t phi_RF,
                     const index_t n_macroparticles, const real_t acc_kick) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for (index_t i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
    beam_dE[i] +=
        charge * voltage * sin(omega_RF * beam_dt[i] + phi_RF) + acc_kick;
  }
}

// Per-harmonic RF parameters, passed to `kick_multi_harmonic` by value.
// They arrive in the kernel's parameter space with the launch itself, so
// the per-turn kick needs no host-to-device copy of three tiny arrays --
// those copies used to cost more than the kick itself for small beams.
// Must match `MAX_RF_HARMONICS_PER_LAUNCH` and `_RF_PARAMS_BATCH_DTYPE` in
// blond/core/backends/cuda/callables.py, which splits more harmonics
// over several launches. The 32 is not the warp size -- see callables.py
// for why it was chosen.
#define MAX_RF_HARMONICS_PER_LAUNCH 32
struct RFParamsBatch {
  real_t voltage[MAX_RF_HARMONICS_PER_LAUNCH];
  real_t omega_rf[MAX_RF_HARMONICS_PER_LAUNCH];
  real_t phi_rf[MAX_RF_HARMONICS_PER_LAUNCH];
};

extern "C" __global__ void
kick_multi_harmonic(const real_t *__restrict__ beam_dt,
                    real_t *__restrict__ beam_dE,
                    const RFParamsBatch rf_params_batch,
                    const int n_rf_in_batch, const real_t charge,
                    const index_t n_macroparticles, const real_t acc_kick) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for (index_t i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
    const real_t dt = beam_dt[i];
    // Starting from acc_kick rather than zero saves an FP64 add per
    // particle, measurable on GPUs with low FP64 throughput.
    real_t dE_sum = acc_kick;
    for (int j = 0; j < n_rf_in_batch; j++)
      dE_sum +=
          charge * rf_params_batch.voltage[j] *
          sin(rf_params_batch.omega_rf[j] * dt + rf_params_batch.phi_rf[j]);
    beam_dE[i] += dE_sum;
  }
}

extern "C" __global__ void beam_phase(const real_t *__restrict__ hist_x,
                                      const real_t *__restrict__ hist_y,
                                      real_t *result, real_t alpha,
                                      real_t omega_rf, real_t phi_rf,
                                      real_t bin_size, int n_bins) {
  extern __shared__ real_t shared[];

  real_t *sin_partial = shared;
  real_t *cos_partial = shared + blockDim.x;

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  real_t sin_val = 0.0;
  real_t cos_val = 0.0;

  if (i < n_bins) {
    real_t x = hist_x[i];
    real_t prof = hist_y[i];
    real_t phase = omega_rf * x + phi_rf;
    real_t base = exp(alpha * x) * prof;

    real_t coeff = ((i == 0) || (i == n_bins - 1)) ? 1.0 : 2.0;

    sin_val = coeff * base * sin(phase);
    cos_val = coeff * base * cos(phase);
  }

  sin_partial[threadIdx.x] = sin_val;
  cos_partial[threadIdx.x] = cos_val;

  __syncthreads();

  // Parallel reduction within block. Halving from the next power of two
  // (slots beyond `blockDim.x` count as zero) keeps every thread slot in
  // the sum also when `GPU_THREADS` is not a power of two.
  int reduction_width = 1;
  while (reduction_width < blockDim.x)
    reduction_width <<= 1;
  for (int s = reduction_width / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s && threadIdx.x + s < blockDim.x) {
      sin_partial[threadIdx.x] += sin_partial[threadIdx.x + s];
      cos_partial[threadIdx.x] += cos_partial[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Only thread 0 adds to global memory
  if (threadIdx.x == 0) {
    atomicAdd(&result[0], sin_partial[0]);
    atomicAdd(&result[1], cos_partial[0]);
  }
}

// floor(x) as an `int`, in one saturating instruction (cvt.rmi): an
// out-of-range result clamps to INT_MIN/INT_MAX instead of being
// undefined behaviour, so callers range-check the integer afterwards.
// That spares a floor and the FP64 range compares per particle, which is
// what bounds the per-particle binning kernels on GPUs with low FP64
// throughput (1/32 rate on consumer cards). A NaN converts to 0.
__device__ __forceinline__ int floor_to_int(const real_t x) {
#ifdef USEFLOAT
  return __float2int_rd(x);
#else
  return __double2int_rd(x);
#endif
}

// Bin index of `value` in a histogram of `n_slices` bins over
// [cut_left, cut_right]; any index outside [0, n_slices) means the value
// lies outside the cut (a NaN lands in bin 0, see `floor_to_int`).
__device__ __forceinline__ int
histogram_bin(const real_t value, const real_t cut_left, const real_t cut_right,
              const real_t inv_bin_width, const int n_slices) {
  int bin = floor_to_int((value - cut_left) * inv_bin_width);
  // Scaling is not exact: a value at or just below cut_right can land
  // on n_slices. Fold it back into the last bin, as np.histogram does,
  // instead of dropping the particle.
  if (bin == n_slices && value <= cut_right)
    bin = n_slices - 1;
  return bin;
}

extern "C" __global__ void
hybrid_histogram(const real_t *__restrict__ input, real_t *__restrict__ output,
                 const real_t cut_left, const real_t cut_right,
                 const unsigned int n_slices, const index_t n_macroparticles,
                 const int capacity) {
  extern __shared__ int block_hist[];
  //reset shared memory
  for (int i = threadIdx.x; i < capacity; i += blockDim.x)
    block_hist[i] = 0;
  __syncthreads();
  int const tid = threadIdx.x + blockDim.x * blockIdx.x;
  real_t const inv_bin_width = n_slices / (cut_right - cut_left);

  const int low_tbin = (n_slices / 2) - (capacity / 2);
  const int high_tbin = low_tbin + capacity;

  for (index_t i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
    const int target_bin =
        histogram_bin(input[i], cut_left, cut_right, inv_bin_width, n_slices);
    if ((unsigned int)target_bin >= n_slices)
      continue;
    if (target_bin >= low_tbin && target_bin < high_tbin)
      atomicAdd(&(block_hist[target_bin - low_tbin]), 1);
    else
      atomicAdd(&(output[target_bin]), 1);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < capacity; i += blockDim.x)
    atomicAdd(&output[low_tbin + i], (real_t)block_hist[i]);
}

extern "C" __global__ void
sm_histogram(const real_t *__restrict__ input, real_t *__restrict__ output,
             const real_t cut_left, const real_t cut_right,
             const unsigned int n_slices, const index_t n_macroparticles) {
  extern __shared__ int block_hist[];
  for (int i = threadIdx.x; i < n_slices; i += blockDim.x)
    block_hist[i] = 0;
  __syncthreads();
  int const tid = threadIdx.x + blockDim.x * blockIdx.x;
  real_t const inv_bin_width = n_slices / (cut_right - cut_left);
  for (index_t i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
    const int target_bin =
        histogram_bin(input[i], cut_left, cut_right, inv_bin_width, n_slices);
    if ((unsigned int)target_bin < n_slices)
      atomicAdd(&(block_hist[target_bin]), 1);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n_slices; i += blockDim.x)
    atomicAdd(&output[i], (real_t)block_hist[i]);
}

extern "C" __global__ void
lik_only_gm_copy(real_t *__restrict__ beam_dt, real_t *__restrict__ beam_dE,
                 const real_t *__restrict__ voltage_array,
                 const real_t *__restrict__ bin_centers, const real_t charge,
                 const int n_slices, const index_t n_macroparticles,
                 const real_t acc_kick,
                 real_t *__restrict__ glob_vkick_factor) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  real_t const inv_bin_width =
      (n_slices - 1) / (bin_centers[n_slices - 1] - bin_centers[0]);

  for (int i = tid; i < n_slices - 1; i += gridDim.x * blockDim.x) {
    glob_vkick_factor[2 * i] =
        charge * (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
    glob_vkick_factor[2 * i + 1] = (charge * voltage_array[i] -
                                    bin_centers[i] * glob_vkick_factor[2 * i]) +
                                   acc_kick;
  }
}

extern "C" __global__ void
lik_only_gm_comp(real_t *__restrict__ beam_dt, real_t *__restrict__ beam_dE,
                 const real_t *__restrict__ voltage_array,
                 const real_t *__restrict__ bin_centers, const real_t charge,
                 const int n_slices, const index_t n_macroparticles,
                 const real_t acc_kick,
                 real_t *__restrict__ glob_vkick_factor) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  real_t const inv_bin_width =
      (n_slices - 1) / (bin_centers[n_slices - 1] - bin_centers[0]);
  const real_t bin0 = bin_centers[0];
  for (index_t i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
    const real_t dt = beam_dt[i];
    const int fbin = floor_to_int((dt - bin0) * inv_bin_width);
    if ((unsigned int)fbin < (unsigned int)(n_slices - 1)) {
      beam_dE[i] +=
          dt * glob_vkick_factor[2 * fbin] + glob_vkick_factor[2 * fbin + 1];
    } else {
      // Out of range only the interpolated voltage is undefined; acc_kick
      // carries the reference energy change and applies to the whole beam
      // (glob_vkick_factor already folds it in for in-range particles).
      beam_dE[i] += acc_kick;
    }
  }
}

// Sparse variants of lik_only_gm_copy/lik_only_gm_comp: bin_centers/voltage
// are a concatenation of one dense island per active RF bucket (gaps
// between islands whenever the filling pattern skips a bucket). Unlike the
// dense kernels, inv_bin_width is derived from bins_per_profile/cut_width
// (constant per bucket) instead of the array's global endpoints, and each
// particle is first resolved to its bucket (mirroring histogram_sparse)
// before indexing into glob_vkick_factor.
extern "C" __global__ void
lik_sparse_gm_copy(const real_t *__restrict__ voltage_array,
                   const real_t *__restrict__ bin_centers, const real_t charge,
                   const int n_slices_total, const real_t acc_kick,
                   const real_t cut_width, const int bins_per_profile,
                   real_t *__restrict__ glob_vkick_factor) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const real_t inv_bin_width = real_t(bins_per_profile) / cut_width;

  for (int i = tid; i < n_slices_total - 1; i += gridDim.x * blockDim.x) {
    glob_vkick_factor[2 * i] =
        charge * (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
    glob_vkick_factor[2 * i + 1] = (charge * voltage_array[i] -
                                    bin_centers[i] * glob_vkick_factor[2 * i]) +
                                   acc_kick;
  }
}

extern "C" __global__ void
lik_sparse_gm_comp(real_t *__restrict__ beam_dt, real_t *__restrict__ beam_dE,
                   const index_t n_macroparticles, const real_t first_left_cut,
                   const real_t left_cut_distance, const real_t cut_width,
                   const int bins_per_profile, const int n_buckets,
                   const bool *__restrict__ filling_pattern,
                   const int *__restrict__ bucket_index_to_memory_index,
                   const real_t acc_kick,
                   real_t *__restrict__ glob_vkick_factor) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const real_t inv_hist_dist = real_t(1) / left_cut_distance;
  const real_t inv_bin_width = real_t(bins_per_profile) / cut_width;
  const real_t bin_width = cut_width / real_t(bins_per_profile);

  for (index_t i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
    const real_t dt = beam_dt[i];
    // Range-check in floating point before the conversion to `int`:
    // converting an out-of-range value is undefined behaviour.
    const real_t bucket_real = floor((dt - first_left_cut) * inv_hist_dist);
    // A particle that gets no interpolated voltage still receives
    // acc_kick -- notably one in an *unfilled* bucket, which is fully
    // inside the turn.
    if (bucket_real < real_t(0) || bucket_real >= real_t(n_buckets)) {
      beam_dE[i] += acc_kick;
      continue;
    }
    const int bucket_i = (int)bucket_real;
    if (!filling_pattern[bucket_i]) {
      beam_dE[i] += acc_kick;
      continue;
    }

    const real_t cut_left = first_left_cut + bucket_i * left_cut_distance;
    const real_t bucket_bin_center0 = cut_left + bin_width / real_t(2);
    const real_t local_bin_real =
        floor((dt - bucket_bin_center0) * inv_bin_width);
    if (local_bin_real < real_t(0) ||
        local_bin_real >= real_t(bins_per_profile - 1)) {
      beam_dE[i] += acc_kick;
      continue;
    }
    const int local_bin = (int)local_bin_real;

    const int fbin = bucket_index_to_memory_index[bucket_i] + local_bin;
    beam_dE[i] +=
        dt * glob_vkick_factor[2 * fbin] + glob_vkick_factor[2 * fbin + 1];
  }
}

// `flag_lost` is `BeamFlags.LOST` (blond/core/beam/flags.py), passed in by
// the Python wrapper so the enum stays the single source of truth.
extern "C" __global__ void loss_box(const real_t e_max, const real_t e_min,
                                    const real_t t_min, const real_t t_max,
                                    const real_t *dt, const real_t *dE,
                                    int *__restrict__ flags,
                                    const int flag_lost,
                                    const index_t n_macroparticles) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for (index_t i = tid; i < n_macroparticles; i = i + blockDim.x * gridDim.x) {
    const bool outside = (dE[i] > e_max) || (dE[i] < e_min) ||
                         (dt[i] < t_min) || (dt[i] > t_max);
    if (outside) {
      flags[i] = flag_lost;
    }
  }
}

// =================================================================
// Synchrotron-radiation + quantum-excitation energy kick
//
// Fused: beam_dE[i] = damping_factor * beam_dE[i] - energy_lost
//                   + noise_scale * N(0, 1)
//
// The Gaussian noise is drawn with NVIDIA's cuRAND device library
// (curand_kernel.h) so we do not maintain (or have to justify) a
// hand-rolled PRNG. Each thread keeps its own cuRAND state, seeded
// from a call-unique `base_seed` with the global thread index as the
// cuRAND subsequence, giving every thread on every launch an
// independent, well-decorrelated stream.
// =================================================================

#include <curand_kernel.h>

// N(0, 1) draw at the backend's real_t precision.
__device__ __forceinline__ real_t
curand_standard_normal(curandStatePhilox4_32_10_t *state) {
#ifdef USEFLOAT
  return curand_normal(state);
#else
  return curand_normal_double(state);
#endif
}

extern "C" __global__ void apply_sr_without_quantum_excitation(
    real_t *__restrict__ beam_dE, const real_t damping_factor,
    const real_t energy_lost, const index_t n_macroparticles) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (index_t i = tid; i < n_macroparticles; i += stride) {
    beam_dE[i] = damping_factor * beam_dE[i] - energy_lost;
  }
}

extern "C" __global__ void apply_sr_with_quantum_excitation(
    real_t *__restrict__ beam_dE, const real_t damping_factor,
    const real_t energy_lost, const real_t noise_scale,
    const unsigned long long base_seed, const index_t n_macroparticles) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;

  // One cuRAND state per thread. `base_seed` is unique per launch and
  // `tid` selects the cuRAND subsequence, so the streams are independent
  // across threads and across launches.
  curandStatePhilox4_32_10_t state;
  curand_init(base_seed, tid, 0, &state);

  for (index_t i = tid; i < n_macroparticles; i += stride) {
    beam_dE[i] = damping_factor * beam_dE[i] - energy_lost +
                 noise_scale * curand_standard_normal(&state);
  }
}

extern "C" __global__ void drift_exact(real_t *__restrict__ beam_dt,
                                       const real_t *__restrict__ beam_dE,
                                       const real_t T, const real_t alpha_zero,
                                       const real_t *__restrict__ higher_alpha,
                                       const int n_alpha, const real_t beta,
                                       const real_t energy,
                                       const index_t n_macroparticles) {
  const real_t inv_beta_sq = 1.0 / (beta * beta);
  const real_t inv_energy = 1.0 / energy;
  const real_t inv_energy_sq = inv_energy * inv_energy;

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for (index_t i = tid; i < n_macroparticles; i = i + blockDim.x * gridDim.x) {

    const real_t dE = beam_dE[i];

    const real_t delta = sqrt(1.0 + inv_beta_sq * (dE * dE * inv_energy_sq +
                                                   2.0 * dE * inv_energy)) -
                         1.0;

    real_t poly = 1.0 + alpha_zero * delta;

    if (n_alpha > 0 && higher_alpha != nullptr) {
      real_t delta_power = delta * delta; // starts at δ²

      for (int k = 0; k < n_alpha; ++k) {
        poly += higher_alpha[k] * delta_power;
        delta_power *= delta; // next power
      }
    }

    beam_dt[i] += T * (poly * (1.0 + dE * inv_energy) / (1.0 + delta) - 1.0);
  }
}

extern "C" __global__ void
histogram_sparse(const real_t *__restrict__ input, real_t *__restrict__ output,
                 const real_t first_left_cut, const real_t left_cut_distance,
                 const real_t cut_width, const int bins_per_profile,
                 const int n_buckets, const index_t n_macroparticles,
                 const bool *__restrict__ filling_pattern,
                 const int *__restrict__ bucket_index_to_memory_index) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  const real_t cut_left0 = first_left_cut;
  const real_t inv_hist_dist = real_t(1) / left_cut_distance;
  const real_t inv_bin_width = real_t(bins_per_profile) / cut_width;

  // Loop through input particles and update histograms in shared memory
  for (index_t i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
    const real_t dt = input[i];

    // Range-check in floating point before the conversion to `int`:
    // converting an out-of-range value is undefined behaviour.
    const real_t bucket_real = (dt - cut_left0) * inv_hist_dist;
    if (bucket_real < real_t(0) || bucket_real >= real_t(n_buckets))
      continue;
    const int bucket_i = (int)bucket_real;
    if (!filling_pattern[bucket_i]) {
      continue;
    }
    const real_t cut_left = cut_left0 + bucket_i * left_cut_distance;
    const real_t cut_right = cut_left + cut_width;

    // Check if the value is within the cut range
    if (dt == cut_right) {
      atomicAdd(&output[bucket_index_to_memory_index[bucket_i] +
                        bins_per_profile - 1],
                1);
      continue;
    }
    if (dt < cut_left || dt >= cut_right)
      continue;

    // Calculate the bin index
    const int bin = (int)((dt - cut_left) * inv_bin_width);
    if ((unsigned)bin < (unsigned)bins_per_profile) {
      atomicAdd(&output[bucket_index_to_memory_index[bucket_i] + bin], 1);
    }
  }
  __syncthreads();
}

// Far field of a pole-residue (vector fitting) wake, one thread per pole.
// Mirrors cpp/poles.cpp; see `Specials.wake_from_pole_residue` in
// blond/core/backends/backend.py for the contract. Pole contributions are
// reduced into `voltage` via atomicAdd. Complex arrays (poles, residues,
// states) are stored as interleaved real/imag: [re0, im0, re1, im1, ...].

// Decay a pole's state from `clock` to `to_time`, never backwards. A step of
// exactly one bin uses the precomputed `decay_*`.
__device__ static inline void advance_pole_state(
    real_t &state_re, real_t &state_im, real_t &clock, const real_t to_time,
    const real_t pole_re, const real_t pole_im, const real_t bin_dt,
    const real_t decay_re, const real_t decay_im, const real_t tolerance) {
  const real_t step = to_time - clock;
  if (step <= tolerance) {
    return;
  }
  real_t e_re, e_im;
  if (fabs(step - bin_dt) <= tolerance) {
    e_re = decay_re;
    e_im = decay_im;
  } else {
    const real_t e_abs = exp(pole_re * step);
    e_re = e_abs * cos(pole_im * step);
    e_im = e_abs * sin(pole_im * step);
  }
  const real_t new_re = state_re * e_re - state_im * e_im;
  const real_t new_im = state_re * e_im + state_im * e_re;
  state_re = new_re;
  state_im = new_im;
  clock = to_time;
}

extern "C" __global__ void wake_from_pole_residue(
    const real_t *__restrict__ profile_time, const real_t *__restrict__ profile,
    const real_t *__restrict__ carried_charge,
    const bool carried_is_counterrotating, const real_t state_lag_dt,
    const real_t carried_lag_dt, const real_t *__restrict__ poles,
    const real_t *__restrict__ residues, const bool is_counterrotating_beam,
    const real_t *__restrict__ counterrotating_pole_signs, const real_t factor,
    const real_t bin_dt, real_t *__restrict__ states,
    real_t *__restrict__ voltage, const int n_bins, const int n_poles) {
  const int pole_i = blockIdx.x * blockDim.x + threadIdx.x;
  if (pole_i >= n_poles)
    return;

  // Times relative to the first bin. A bin is read out two bins behind
  // its own time; the carried charge was emitted before t_0.
  const real_t t_0 = profile_time[0];
  const real_t read_lag = real_t(2) * bin_dt;
  const real_t tolerance = real_t(1e-6) * bin_dt;
  const real_t t_carried = -carried_lag_dt;
  const real_t t_handover = (profile_time[n_bins - 1] - t_0) - bin_dt;

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
  const real_t carried_flip = (carried_is_counterrotating && flipped) ? -1 : 1;
  // A complex pole stands in for its unstored conjugate partner.
  const real_t pair = (pole_im == real_t(0)) ? real_t(1) : real_t(2);
  const real_t decay_abs = exp(pole_re * bin_dt);
  const real_t decay_re = decay_abs * cos(pole_im * bin_dt);
  const real_t decay_im = decay_abs * sin(pole_im * bin_dt);

  real_t state_re = states[pole_n];
  real_t state_im = states[pole_n + 1];
  real_t clock = -read_lag - state_lag_dt;
  int next_bin = 0;

  for (int bin_i = 0; bin_i < n_bins; ++bin_i) {
    const real_t read_time = (profile_time[bin_i] - t_0) - read_lag;
    // every bin but the last enters the state when it is due
    while (next_bin < n_bins - 1 &&
           (profile_time[next_bin] - t_0) <= read_time + tolerance) {
      advance_pole_state(state_re, state_im, clock,
                         profile_time[next_bin] - t_0, pole_re, pole_im, bin_dt,
                         decay_re, decay_im, tolerance);
      state_re += flip * pair * factor * profile[next_bin];
      ++next_bin;
    }
    advance_pole_state(state_re, state_im, clock, read_time, pole_re, pole_im,
                       bin_dt, decay_re, decay_im, tolerance);
    atomicAdd(&voltage[bin_i], flip * (res_re * state_re - res_im * state_im));
    if (bin_i == 0) {
      // The carried charge enters after the first read-out; older than
      // the clock it is decayed to it, newer it moves the clock.
      if (t_carried <= clock) {
        const real_t e_abs = exp(pole_re * (clock - t_carried));
        const real_t charge = carried_flip * pair * carried_charge[0];
        state_re += charge * e_abs * cos(pole_im * (clock - t_carried));
        state_im += charge * e_abs * sin(pole_im * (clock - t_carried));
      } else {
        advance_pole_state(state_re, state_im, clock, t_carried, pole_re,
                           pole_im, bin_dt, decay_re, decay_im, tolerance);
        state_re += carried_flip * pair * carried_charge[0];
      }
    }
  }
  // Hand over one bin before the last bin, every other bin in.
  while (next_bin < n_bins - 1) {
    advance_pole_state(state_re, state_im, clock, profile_time[next_bin] - t_0,
                       pole_re, pole_im, bin_dt, decay_re, decay_im, tolerance);
    state_re += flip * pair * factor * profile[next_bin];
    ++next_bin;
  }
  advance_pole_state(state_re, state_im, clock, t_handover, pole_re, pole_im,
                     bin_dt, decay_re, decay_im, tolerance);

  states[pole_n] = state_re;
  states[pole_n + 1] = state_im;
}

// Travelling-wave-cavity wake; see `Specials.wake_from_twc_fir` in
// blond/core/backends/backend.py for the contract. The phasor FIR
// recursion of the other backends is an exact lattice convolution: the
// charge q_j of the bin at site s_j adds q_j rot^m (1 - m / n) to the bin
// m = s_i - s_j sites later, for 0 < m < n (n = ceil(a_tilde / bin_dt)
// sites, rot = exp(i omega_r bin_dt)), and half of itself at m = 0. The
// phasor is undamped, so with the window sums over the bins j < i with
// s_j > s_i - n,
//   S0_i = sum q_j rot^(-s_j),   S1_i = sum q_j s_j rot^(-s_j),
// the wake of a mode at bin i is
//   Re(rot^(s_i) ((1 - s_i / n) S0_i + S1_i / n)) + q_i / 2,
// every bin and mode in parallel. A window sum is the difference of two
// prefix sums over the bins: `twc_scan_chunks` scans each chunk of
// TWC_CHUNK bins of a mode and keeps the chunk's total, `twc_scan_totals`
// turns the totals into each chunk's offset, and `wake_from_twc_fir`
// takes the differences. Prefix sums and totals are [modes][...][4]:
// S0 re, S0 im, S1 re, S1 im; the scan also keeps rot^(-s_j) per mode and
// bin, [modes][bins][2], whose conjugate is the read-out's rot^(s_i).

#define TWC_THREADS 256
#define TWC_PER_THREAD 4
#define TWC_CHUNK (TWC_THREADS * TWC_PER_THREAD)
#define TWC_WARPS (TWC_THREADS / 32)

struct TwcSums {
  real_t s0_re, s0_im, s1_re, s1_im;
};

__device__ static inline TwcSums twc_zero() { return {0, 0, 0, 0}; }

__device__ static inline TwcSums twc_add(const TwcSums x, const TwcSums y) {
  return {x.s0_re + y.s0_re, x.s0_im + y.s0_im, x.s1_re + y.s1_re,
          x.s1_im + y.s1_im};
}

__device__ static inline TwcSums twc_sub(const TwcSums x, const TwcSums y) {
  return {x.s0_re - y.s0_re, x.s0_im - y.s0_im, x.s1_re - y.s1_re,
          x.s1_im - y.s1_im};
}

__device__ static inline TwcSums twc_shfl_up(const TwcSums x, const int delta) {
  return {__shfl_up_sync(0xffffffffu, x.s0_re, delta),
          __shfl_up_sync(0xffffffffu, x.s0_im, delta),
          __shfl_up_sync(0xffffffffu, x.s1_re, delta),
          __shfl_up_sync(0xffffffffu, x.s1_im, delta)};
}

__device__ static inline TwcSums twc_load(const real_t *__restrict__ sums,
                                          const size_t i) {
  return {sums[4 * i], sums[4 * i + 1], sums[4 * i + 2], sums[4 * i + 3]};
}

__device__ static inline void twc_store(real_t *__restrict__ sums,
                                        const size_t i, const TwcSums x) {
  sums[4 * i] = x.s0_re;
  sums[4 * i + 1] = x.s0_im;
  sums[4 * i + 2] = x.s1_re;
  sums[4 * i + 3] = x.s1_im;
}

// Exclusive scan of one value per thread over the block, and the block's
// total. Every thread of the block calls it.
__device__ static TwcSums twc_block_scan(const TwcSums value,
                                         TwcSums &block_total) {
  __shared__ TwcSums warp_totals[TWC_WARPS];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  TwcSums inclusive = value;
  for (int delta = 1; delta < 32; delta *= 2) {
    const TwcSums up = twc_shfl_up(inclusive, delta);
    if (lane >= delta)
      inclusive = twc_add(inclusive, up);
  }
  const TwcSums previous = twc_shfl_up(inclusive, 1);
  const TwcSums in_warp = lane > 0 ? previous : twc_zero();
  if (lane == 31)
    warp_totals[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    TwcSums total = lane < TWC_WARPS ? warp_totals[lane] : twc_zero();
    for (int delta = 1; delta < TWC_WARPS; delta *= 2) {
      const TwcSums up = twc_shfl_up(total, delta);
      if (lane >= delta)
        total = twc_add(total, up);
    }
    if (lane < TWC_WARPS)
      warp_totals[lane] = total;
  }
  __syncthreads();
  const TwcSums before = warp > 0 ? warp_totals[warp - 1] : twc_zero();
  block_total = warp_totals[TWC_WARPS - 1];
  __syncthreads(); // warp_totals is reused by the next call
  return twc_add(before, in_warp);
}

// Per mode (blockIdx.y) and chunk of TWC_CHUNK bins (blockIdx.x): the
// prefix sums of the chunk's terms from its first bin on, and its total.
extern "C" __global__ void __launch_bounds__(TWC_THREADS)
    twc_scan_chunks(const real_t *__restrict__ profile,
                    const int *__restrict__ grid_index,
                    const real_t *__restrict__ omega_r, const real_t bin_dt,
                    const int n_bins, real_t *__restrict__ prefix,
                    real_t *__restrict__ chunk_totals,
                    real_t *__restrict__ phasors) {
  const int mode = blockIdx.y;
  const real_t phase = omega_r[mode] * bin_dt;
  const long long first =
      (long long)blockIdx.x * TWC_CHUNK + threadIdx.x * TWC_PER_THREAD;
  const size_t row = (size_t)mode * n_bins;
  TwcSums local[TWC_PER_THREAD];
  TwcSums running = twc_zero();
  for (int u = 0; u < TWC_PER_THREAD; ++u) {
    const long long j = first + u;
    if (j < n_bins) {
      const real_t site = (real_t)grid_index[j];
      real_t sine, cosine;
      sincos(-phase * site, &sine, &cosine);
      phasors[2 * (row + j)] = cosine;
      phasors[2 * (row + j) + 1] = sine;
      const real_t charge = profile[j];
      const TwcSums term = {charge * cosine, charge * sine,
                            charge * site * cosine, charge * site * sine};
      running = twc_add(running, term);
    }
    local[u] = running;
  }
  TwcSums chunk_total;
  const TwcSums before = twc_block_scan(running, chunk_total);
  for (int u = 0; u < TWC_PER_THREAD; ++u)
    if (first + u < n_bins)
      twc_store(prefix, row + first + u, twc_add(before, local[u]));
  if (threadIdx.x == 0)
    twc_store(chunk_totals, (size_t)mode * gridDim.x + blockIdx.x, chunk_total);
}

// Per mode (blockIdx.x): the chunk totals into each chunk's offset, the
// sum of the chunks before it.
extern "C" __global__ void __launch_bounds__(TWC_THREADS)
    twc_scan_totals(const real_t *__restrict__ chunk_totals, const int n_chunks,
                    real_t *__restrict__ offsets) {
  const size_t row = (size_t)blockIdx.x * n_chunks;
  TwcSums carry = twc_zero();
  for (int first = 0; first < n_chunks; first += TWC_THREADS) {
    const int chunk = first + threadIdx.x;
    const TwcSums total =
        chunk < n_chunks ? twc_load(chunk_totals, row + chunk) : twc_zero();
    TwcSums tile_total;
    const TwcSums before = twc_block_scan(total, tile_total);
    if (chunk < n_chunks)
      twc_store(offsets, row + chunk, twc_add(carry, before));
    carry = twc_add(carry, tile_total);
  }
}

// Prefix sum of bins [a + 1, b] of one mode, a < b: the offsets of their
// chunks and the prefix sums within, each difference taken first.
__device__ static inline TwcSums
twc_window(const real_t *__restrict__ prefix,
           const real_t *__restrict__ offsets, const size_t row,
           const size_t offset_row, const long long a, const long long b) {
  const TwcSums upper = twc_load(prefix, row + b);
  const TwcSums upper_offset = twc_load(offsets, offset_row + b / TWC_CHUNK);
  if (a < 0)
    return twc_add(upper_offset, upper);
  const TwcSums lower = twc_load(prefix, row + a);
  const TwcSums lower_offset = twc_load(offsets, offset_row + a / TWC_CHUNK);
  return twc_add(twc_sub(upper_offset, lower_offset), twc_sub(upper, lower));
}

// One thread per bin, every mode: the window sums into the wake; overwrites
// `voltage`.
extern "C" __global__ void __launch_bounds__(TWC_THREADS) wake_from_twc_fir(
    const real_t *__restrict__ profile, const int *__restrict__ grid_index,
    const real_t *__restrict__ r_shunt, const real_t *__restrict__ a_tilde,
    const real_t *__restrict__ omega_r, const real_t bin_dt,
    const real_t factor, const int n_bins, const int n_modes,
    const int n_chunks, const real_t *__restrict__ prefix,
    const real_t *__restrict__ offsets, const real_t *__restrict__ phasors,
    real_t *__restrict__ voltage) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_bins)
    return;
  const long long site = grid_index[i];
  const real_t half_charge = 0.5 * profile[i];
  real_t total = 0;
  for (int mode = 0; mode < n_modes; ++mode) {
    const real_t n_taper = ceil(a_tilde[mode] / bin_dt);
    real_t wake = half_charge;
    // the first bin of the window: grid_index >= site - n_taper + 1
    const long long lowest = site - (long long)n_taper + 1;
    int low = 0, high = i;
    while (low < high) {
      const int middle = (low + high) / 2;
      if (grid_index[middle] < lowest)
        low = middle + 1;
      else
        high = middle;
    }
    if (low < i) {
      const TwcSums sums =
          twc_window(prefix, offsets, (size_t)mode * n_bins,
                     (size_t)mode * n_chunks, (long long)low - 1, i - 1);
      const real_t taper = 1 - (real_t)site / n_taper;
      const real_t re = taper * sums.s0_re + sums.s1_re / n_taper;
      const real_t im = taper * sums.s0_im + sums.s1_im / n_taper;
      // rot^(s_i), the conjugate of the scan's rot^(-s_i)
      const size_t k = 2 * ((size_t)mode * n_bins + i);
      wake += phasors[k] * re + phasors[k + 1] * im;
    }
    // W(0) of the cosine is 2 r_shunt / a_tilde; the 2 of 2 factor
    total += 2 * r_shunt[mode] / a_tilde[mode] * 2 * factor * wake;
  }
  voltage[i] = total;
}
