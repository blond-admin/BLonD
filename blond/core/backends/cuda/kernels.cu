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

// `flag_lost` is `BeamFlags.LOST` (blond/core/backends/flags.py), passed in by
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

// ---------------------------------------------------------------------------
// Far field of a pole-residue (vector fitting) wake, fused and tiled.
// Mirrors cpp/poles.cpp; see `Specials.wake_from_pole_residue` in
// blond/core/backends/backend.py for the contract. Positions are in whole
// bins from the first bin, `offsets` (the wrapper converts the times);
// complex arrays are interleaved [re0, im0, re1, im1, ...].
//
// The poles are a diagonal linear state-space model. The bins are walked in
// tiles: up to POLE_TILE contiguous bins, starting at every run start and
// every POLE_TILE bins within a run. For a tile starting at position a with
// L bins, the state Y (at a - 2) gives the read-outs at a - 2 + k,
//   V[k] = Re sum_p r_p d_p^k Y_p   +   sum_s w(k - s - 1) h[s],
// where h[0] is the previous bin if it is adjacent (a halo, else 0) and
// h[s] the tile's bins, w the wake summed over the poles, and the tile's
// own sum, at a + L - 2 and without its last bin,
//   Z_p = sum_{s < L} inj_p d_p^(L - 1 - s) h[s].
// To the next tile at a' (gap g = a' - a - L >= 0):
//   Y' = d^(a' - a) Y + d^g Z (+ inj d^(g - 1) h[L] if g >= 1).
// Every exponent is >= 0. A block holds POLE_BLOCK poles and walks one
// time chunk of bins; the poles' powers d^0..d^POLE_TILE live in shared
// memory, the states in shared memory, and each read-out is summed over
// the block's poles before one atomicAdd per bin and block. Chunks run in
// parallel: pass 1 (`pole_residue_chunk_maps`) stores each chunk's state from zero,
// L_c, and its decay P_c = d^(span); pass 2 (`wake_from_pole_residue`) folds
// them, Y_c = P_{c-1} Y_{c-1} + L_{c-1}, and reads out. Memory: the maps,
// [n_chunks][n_poles][4] values, besides the output.
// ---------------------------------------------------------------------------
#define POLE_TILE 64
#define POLE_BLOCK 16
#define POLE_THREADS 64 // == POLE_TILE: one thread per read-out of a tile

struct PoleTileShared {
  real_t pole_re[POLE_BLOCK], pole_im[POLE_BLOCK]; // pole * bin_dt
  real_t injection[POLE_BLOCK];                    // flip * pair
  real_t power_re[POLE_TILE + 1][POLE_BLOCK];      // d^m
  real_t power_im[POLE_TILE + 1][POLE_BLOCK];
  real_t readout_re[POLE_BLOCK], readout_im[POLE_BLOCK];       // r = flip * rho
  real_t state_re[POLE_BLOCK], state_im[POLE_BLOCK];           // Y, at a - 2
  real_t read_state_re[POLE_BLOCK], read_state_im[POLE_BLOCK]; // r Y
  real_t read_carried_re[POLE_BLOCK], read_carried_im[POLE_BLOCK]; // r C
  real_t own_re[POLE_BLOCK], own_im[POLE_BLOCK];                   // Z
  real_t carried_re[POLE_BLOCK], carried_im[POLE_BLOCK];           // at -1
  real_t charges[POLE_TILE + 1];                                   // h
  int tile_length;
};

__device__ static inline void pole_cexp(const real_t re, const real_t im,
                                        real_t &out_re, real_t &out_im) {
  const real_t e = exp(re);
  out_re = e * cos(im);
  out_im = e * sin(im);
}

// Load the block's poles: powers, read-out weights, injection, carried
// charge (decayed to -1) and, if `with_state`, the incoming state at -2.
__device__ static void pole_tile_load(
    PoleTileShared &sh, const real_t *__restrict__ poles,
    const real_t *__restrict__ residues, const real_t *__restrict__ signs,
    const real_t *__restrict__ states, const bool is_cr, const bool carried_cr,
    const real_t carried_charge, const real_t carried_lag_dt,
    const real_t state_lag_dt, const real_t bin_dt, const int first_pole,
    const int n_poles, const bool with_state) {
  const int tid = threadIdx.x;
  if (tid < POLE_BLOCK) {
    const int p = first_pole + tid;
    const bool live = p < n_poles;
    const real_t pr = live ? poles[2 * p] : real_t(0);
    const real_t pi = live ? poles[2 * p + 1] : real_t(0);
    sh.pole_re[tid] = pr * bin_dt;
    sh.pole_im[tid] = pi * bin_dt;
    const bool flipped = live && signs[p] == real_t(-1);
    const real_t flip = (is_cr && flipped) ? -1 : 1;
    const real_t carried_flip = (carried_cr && flipped) ? -1 : 1;
    const real_t pair = (pi == real_t(0)) ? real_t(1) : real_t(2);
    sh.injection[tid] = live ? flip * pair : real_t(0);
    // carried charge, emitted carried_lag_dt before t_0, decayed to -1
    real_t e_re, e_im;
    pole_cexp(pr * (carried_lag_dt - bin_dt), pi * (carried_lag_dt - bin_dt),
              e_re, e_im);
    const real_t q = live ? carried_flip * pair * carried_charge : real_t(0);
    sh.carried_re[tid] = q * e_re;
    sh.carried_im[tid] = q * e_im;
    real_t s_re = 0, s_im = 0;
    if (with_state && live) {
      pole_cexp(pr * state_lag_dt, pi * state_lag_dt, e_re, e_im);
      const real_t in_re = states[2 * p], in_im = states[2 * p + 1];
      s_re = in_re * e_re - in_im * e_im;
      s_im = in_re * e_im + in_im * e_re;
    }
    sh.state_re[tid] = s_re;
    sh.state_im[tid] = s_im;
    const real_t r_re = live ? flip * residues[2 * p] : real_t(0);
    const real_t r_im = live ? flip * residues[2 * p + 1] : real_t(0);
    sh.readout_re[tid] = r_re;
    sh.readout_im[tid] = r_im;
    sh.read_carried_re[tid] =
        r_re * sh.carried_re[tid] - r_im * sh.carried_im[tid];
    sh.read_carried_im[tid] =
        r_re * sh.carried_im[tid] + r_im * sh.carried_re[tid];
  }
  __syncthreads();
  // powers d^m, m = 0..POLE_TILE
  for (int i = tid; i < (POLE_TILE + 1) * POLE_BLOCK; i += blockDim.x) {
    const int m = i / POLE_BLOCK, j = i % POLE_BLOCK;
    real_t e_re, e_im;
    pole_cexp(sh.pole_re[j] * m, sh.pole_im[j] * m, e_re, e_im);
    sh.power_re[m][j] = e_re;
    sh.power_im[m][j] = e_im;
  }
  __syncthreads();
}

// r Y of the block's poles, for the read-outs of the next tile.
__device__ static inline void pole_tile_read_state(PoleTileShared &sh) {
  const int j = threadIdx.x;
  if (j < POLE_BLOCK) {
    const real_t r_re = sh.readout_re[j], r_im = sh.readout_im[j];
    const real_t y_re = sh.state_re[j], y_im = sh.state_im[j];
    sh.read_state_re[j] = r_re * y_re - r_im * y_im;
    sh.read_state_im[j] = r_re * y_im + r_im * y_re;
  }
}

// The tile starting at bin `first`: its length (into sh.tile_length) and
// charges (into sh.charges). Every block computes the same tiles.
__device__ static void pole_tile_fetch(PoleTileShared &sh,
                                       const index_t *__restrict__ offsets,
                                       const real_t *__restrict__ profile,
                                       const real_t factor, const int first,
                                       const int stop) {
  const int tid = threadIdx.x;
  // the tile ends at the first bin k that is past `stop` or not adjacent to
  // its predecessor; every thread checks one bin
  if (tid == 0)
    sh.tile_length = POLE_TILE;
  __syncthreads();
  if (tid >= 1 && tid < POLE_TILE &&
      (first + tid >= stop ||
       offsets[first + tid] != offsets[first + tid - 1] + 1))
    atomicMin(&sh.tile_length, tid);
  __syncthreads();
  const int length = sh.tile_length;
  // h[0]: the previous bin if adjacent; h[s], s = 1..length: the tile's
  // bins (h[length] is the last one, not part of the tile's own sum)
  for (int s = tid; s <= POLE_TILE; s += blockDim.x) {
    real_t h = 0;
    if (s == 0) {
      if (first > 0 && offsets[first - 1] + 1 == offsets[first])
        h = factor * profile[first - 1];
    } else if (s <= length) {
      h = factor * profile[first + s - 1];
    }
    sh.charges[s] = h;
  }
  __syncthreads();
}

// The tile's own sum Z, one pole per POLE_THREADS / POLE_BLOCK threads.
__device__ static void pole_tile_own(PoleTileShared &sh) {
  const int tid = threadIdx.x;
  const int lanes = POLE_THREADS / POLE_BLOCK; // threads per pole
  const int j = tid / lanes, part = tid % lanes;
  const int length = sh.tile_length;
  real_t z_re = 0, z_im = 0;
  for (int s = part; s < length; s += lanes) {
    const real_t h = sh.charges[s];
    z_re += h * sh.power_re[length - 1 - s][j];
    z_im += h * sh.power_im[length - 1 - s][j];
  }
  for (int offset = lanes / 2; offset > 0; offset /= 2) {
    z_re += __shfl_down_sync(0xffffffffu, z_re, offset, lanes);
    z_im += __shfl_down_sync(0xffffffffu, z_im, offset, lanes);
  }
  if (part == 0) {
    sh.own_re[j] = sh.injection[j] * z_re;
    sh.own_im[j] = sh.injection[j] * z_im;
  }
}

// The tile's read-outs, one per thread, added to `voltage`.
__device__ static void pole_tile_read(PoleTileShared &sh,
                                      const real_t *__restrict__ wake,
                                      const bool add_wake,
                                      const bool add_carried, const int first,
                                      real_t *__restrict__ voltage) {
  const int k = threadIdx.x;
  if (k >= sh.tile_length)
    return;
  real_t v = 0;
  for (int j = 0; j < POLE_BLOCK; ++j)
    v += sh.power_re[k][j] * sh.read_state_re[j] -
         sh.power_im[k][j] * sh.read_state_im[j];
  if (add_carried && k >= 1) // the carried charge sits at -1
    for (int j = 0; j < POLE_BLOCK; ++j)
      v += sh.power_re[k - 1][j] * sh.read_carried_re[j] -
           sh.power_im[k - 1][j] * sh.read_carried_im[j];
  if (add_wake) // the poles' wake within the tile, summed over all poles
    for (int s = 0; s < k; ++s)
      v += wake[k - s - 1] * sh.charges[s];
  atomicAdd(&voltage[first + k], v);
}

// Y -> Y' at the next tile (start position `next_position`), per pole.
__device__ static void pole_tile_step(PoleTileShared &sh,
                                      const index_t position,
                                      const index_t next_position,
                                      const bool add_carried) {
  const int j = threadIdx.x;
  if (j >= POLE_BLOCK)
    return;
  const int length = sh.tile_length;
  const index_t gap = next_position - position - length;
  real_t y_re = sh.state_re[j], y_im = sh.state_im[j];
  real_t z_re = sh.own_re[j], z_im = sh.own_im[j];
  // decays: over the tile and the gap, and over the gap alone
  real_t step_re = sh.power_re[length][j], step_im = sh.power_im[length][j];
  if (gap > 0) {
    real_t g_re, g_im; // d^(gap - 1)
    pole_cexp(sh.pole_re[j] * (gap - 1), sh.pole_im[j] * (gap - 1), g_re, g_im);
    // Z d^gap + inj h_last d^(gap - 1) = d^(gap - 1) (Z d + inj h_last)
    const real_t d_re = sh.power_re[1][j], d_im = sh.power_im[1][j];
    const real_t a_re =
        z_re * d_re - z_im * d_im + sh.injection[j] * sh.charges[length];
    const real_t a_im = z_re * d_im + z_im * d_re;
    z_re = g_re * a_re - g_im * a_im;
    z_im = g_re * a_im + g_im * a_re;
    // d^(L + gap) = d^L d d^(gap - 1)
    const real_t t_re = step_re * d_re - step_im * d_im;
    const real_t t_im = step_re * d_im + step_im * d_re;
    step_re = t_re * g_re - t_im * g_im;
    step_im = t_re * g_im + t_im * g_re;
  }
  real_t n_re = step_re * y_re - step_im * y_im + z_re;
  real_t n_im = step_re * y_im + step_im * y_re + z_im;
  if (add_carried) { // the carried charge, at -1, to next_position - 2
    real_t c_re, c_im;
    pole_cexp(sh.pole_re[j] * (next_position - 1),
              sh.pole_im[j] * (next_position - 1), c_re, c_im);
    n_re += c_re * sh.carried_re[j] - c_im * sh.carried_im[j];
    n_im += c_re * sh.carried_im[j] + c_im * sh.carried_re[j];
  }
  sh.state_re[j] = n_re;
  sh.state_im[j] = n_im;
}

// Pass 1: L_c and P_c of every chunk but the last.
extern "C" __global__ void __launch_bounds__(POLE_THREADS)
    pole_residue_chunk_maps(
        const index_t *__restrict__ offsets, const real_t *__restrict__ profile,
        const real_t *__restrict__ poles, const real_t *__restrict__ residues,
        const real_t *__restrict__ signs, const bool is_cr,
        const bool carried_cr, const real_t *__restrict__ carried_charge,
        const real_t carried_lag_dt, const real_t factor, const real_t bin_dt,
        const int n_bins, const int n_poles, const int chunk_length,
        const int n_chunks, real_t *__restrict__ chunk_maps) {
  __shared__ PoleTileShared sh;
  const int first_pole = blockIdx.x * POLE_BLOCK;
  const int chunk = blockIdx.y;
  if (chunk >= n_chunks - 1)
    return;
  pole_tile_load(sh, poles, residues, signs, nullptr, is_cr, carried_cr,
                 carried_charge[0], carried_lag_dt, 0, bin_dt, first_pole,
                 n_poles, false);
  const int start = chunk * chunk_length;
  const int stop = start + chunk_length; // < n_bins: not the last chunk
  int first = start;
  while (first < stop) {
    pole_tile_fetch(sh, offsets, profile, factor, first, stop);
    pole_tile_own(sh);
    __syncthreads();
    const int next = first + sh.tile_length;
    pole_tile_step(sh, offsets[first], offsets[next], first == 0);
    __syncthreads();
    first = next;
  }
  const int j = threadIdx.x;
  if (j < POLE_BLOCK && first_pole + j < n_poles) {
    real_t p_re, p_im;
    const real_t span = (real_t)(offsets[stop] - offsets[start]);
    pole_cexp(sh.pole_re[j] * span, sh.pole_im[j] * span, p_re, p_im);
    real_t *map = chunk_maps + ((size_t)chunk * n_poles + first_pole + j) * 4;
    map[0] = p_re;
    map[1] = p_im;
    map[2] = sh.state_re[j];
    map[3] = sh.state_im[j];
  }
}

// Pass 2: the read-outs of every chunk; the last hands the state over.
extern "C" __global__ void __launch_bounds__(POLE_THREADS)
    wake_from_pole_residue(
        const index_t *__restrict__ offsets, const real_t *__restrict__ profile,
        const real_t *__restrict__ poles, const real_t *__restrict__ residues,
        const real_t *__restrict__ signs, const bool is_cr,
        const bool carried_cr, const real_t *__restrict__ carried_charge,
        const real_t carried_lag_dt, const real_t factor, const real_t bin_dt,
        const int n_bins, const int n_poles, const int chunk_length,
        const int n_chunks, const real_t *__restrict__ chunk_maps,
        const real_t *__restrict__ wake, const real_t state_lag_dt,
        real_t *__restrict__ states, real_t *__restrict__ voltage) {
  __shared__ PoleTileShared sh;
  const int first_pole = blockIdx.x * POLE_BLOCK;
  const int chunk = blockIdx.y;
  pole_tile_load(sh, poles, residues, signs, states, is_cr, carried_cr,
                 carried_charge[0], carried_lag_dt, state_lag_dt, bin_dt,
                 first_pole, n_poles, true);
  const int j = threadIdx.x;
  if (j < POLE_BLOCK && first_pole + j < n_poles) {
    // fold the maps of the earlier chunks into the state at the chunk start
    real_t y_re = sh.state_re[j], y_im = sh.state_im[j];
    for (int earlier = 0; earlier < chunk; ++earlier) {
      const real_t *map =
          chunk_maps + ((size_t)earlier * n_poles + first_pole + j) * 4;
      const real_t n_re = map[0] * y_re - map[1] * y_im + map[2];
      y_im = map[0] * y_im + map[1] * y_re + map[3];
      y_re = n_re;
    }
    sh.state_re[j] = y_re;
    sh.state_im[j] = y_im;
  }
  __syncthreads();
  pole_tile_read_state(sh);
  __syncthreads();
  const bool add_wake = blockIdx.x == 0;
  const int start = chunk * chunk_length;
  const int stop = min(start + chunk_length, n_bins);
  int first = start;
  while (first < stop) {
    pole_tile_fetch(sh, offsets, profile, factor, first, stop);
    pole_tile_own(sh);
    pole_tile_read(sh, wake, add_wake, first == 0, first, voltage);
    __syncthreads();
    const int next = first + sh.tile_length;
    if (next == n_bins) { // hand over: one bin before the last bin
      if (j < POLE_BLOCK && first_pole + j < n_poles) {
        const int length = sh.tile_length;
        const real_t p_re = sh.power_re[length][j],
                     p_im = sh.power_im[length][j];
        real_t h_re =
            p_re * sh.state_re[j] - p_im * sh.state_im[j] + sh.own_re[j];
        real_t h_im =
            p_re * sh.state_im[j] + p_im * sh.state_re[j] + sh.own_im[j];
        if (first == 0) { // a single tile: the carried charge too
          real_t c_re, c_im;
          const real_t to = (real_t)(offsets[n_bins - 1] - offsets[0]);
          pole_cexp(sh.pole_re[j] * to, sh.pole_im[j] * to, c_re, c_im);
          h_re += c_re * sh.carried_re[j] - c_im * sh.carried_im[j];
          h_im += c_re * sh.carried_im[j] + c_im * sh.carried_re[j];
        }
        states[2 * (first_pole + j)] = h_re;
        states[2 * (first_pole + j) + 1] = h_im;
      }
    } else if (next < stop) {
      pole_tile_step(sh, offsets[first], offsets[next], first == 0);
      pole_tile_read_state(sh); // same thread as the step: no barrier
    }
    __syncthreads();
    first = next;
  }
}
