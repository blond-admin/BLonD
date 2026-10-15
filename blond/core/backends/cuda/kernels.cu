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

// Far field of a pole-residue (vector fitting) wake. Mirrors cpp/poles.cpp;
// see `Specials.wake_from_pole_residue` in blond/core/backends/backend.py
// for the contract. Complex arrays (poles, residues, states) are stored as
// interleaved real/imag: [re0, im0, re1, im1, ...]. Positions are in whole
// bins from the first bin: a bin is read out two bins behind its own
// position, and the state is handed over one bin before the last bin.
//
// One thread per pole rarely fills a GPU, so the read-outs are also split
// into chunks run in parallel. The recursion is linear: running a chunk
// maps a state S to P * S + L, with P = exp(pole * span) over the chunk's
// span of whole bins and L its own contribution from a zero state.
// `pole_residue_chunk_maps` computes (P, L) for every chunk but the last, and
// `wake_from_pole_residue` folds the maps of the earlier chunks into the
// true state at the start of its chunk before it reads out. With one chunk
// it is the plain recursion.
//
// The 32 lanes of a warp are 32 poles of one chunk, so they read out the
// same bins in lockstep: control flow depends on the bins only, never on
// the pole. Each read-out is summed over the warp first and added by one
// lane, one atomicAdd per warp and bin instead of one per pole and bin,
// which on the same address serialize. Lanes past the last pole therefore
// stay in the warp and read out zero.

// The pole kernels are launched with at most POLE_THREADS_PER_BLOCK
// threads. Their launch bounds override the global -maxrregcount, which
// would spill the pole constants to local memory and reload them on every
// read-out.
#define POLE_THREADS_PER_BLOCK 128

// One pole's constants. The read-out flip is folded into the residue.
struct PoleConstants {
  real_t pole_re;
  real_t pole_im;
  real_t flipped_res_re;
  real_t flipped_res_im;
  real_t decay_re;
  real_t decay_im;
  real_t injection;
  real_t carried_injection;
};

__device__ static inline PoleConstants load_pole_constants(
    const real_t *__restrict__ poles, const real_t *__restrict__ residues,
    const real_t *__restrict__ counterrotating_pole_signs, const int pole_i,
    const bool is_counterrotating_beam, const bool carried_is_counterrotating,
    const real_t carried_charge, const real_t factor, const real_t bin_dt) {
  PoleConstants pole;
  pole.pole_re = poles[2 * pole_i];
  pole.pole_im = poles[2 * pole_i + 1];
  // The flip is applied to both the injection and the read-out, so a
  // beam's own wake never flips; a charge carried from the other beam
  // keeps that beam's flip.
  const bool flipped = counterrotating_pole_signs[pole_i] == real_t(-1);
  const real_t flip = (is_counterrotating_beam && flipped) ? -1 : 1;
  const real_t carried_flip = (carried_is_counterrotating && flipped) ? -1 : 1;
  pole.flipped_res_re = flip * residues[2 * pole_i];
  pole.flipped_res_im = flip * residues[2 * pole_i + 1];
  // A complex pole stands in for its unstored conjugate partner.
  const real_t pair = (pole.pole_im == real_t(0)) ? real_t(1) : real_t(2);
  pole.injection = flip * pair * factor;
  pole.carried_injection = carried_flip * pair * carried_charge;
  const real_t decay_abs = exp(pole.pole_re * bin_dt);
  pole.decay_re = decay_abs * cos(pole.pole_im * bin_dt);
  pole.decay_im = decay_abs * sin(pole.pole_im * bin_dt);
  return pole;
}

// (out_re, out_im) = exp(pole * span)
__device__ static inline void pole_exp(const PoleConstants &pole,
                                       const real_t span, real_t &out_re,
                                       real_t &out_im) {
  const real_t e_abs = exp(pole.pole_re * span);
  out_re = e_abs * cos(pole.pole_im * span);
  out_im = e_abs * sin(pole.pole_im * span);
}

// state *= (e_re + i e_im)
__device__ static inline void multiply_state(const real_t e_re,
                                             const real_t e_im,
                                             real_t &state_re,
                                             real_t &state_im) {
  const real_t new_re = state_re * e_re - state_im * e_im;
  state_im = state_re * e_im + state_im * e_re;
  state_re = new_re;
}

// Decay the state over `span`, in [s].
__device__ static inline void decay_pole_state(const PoleConstants &pole,
                                               const real_t span,
                                               real_t &state_re,
                                               real_t &state_im) {
  real_t e_re = 0;
  real_t e_im = 0;
  pole_exp(pole, span, e_re, e_im);
  multiply_state(e_re, e_im, state_re, state_im);
}

// Advance the state by `n_steps` whole bins.
__device__ static inline void
advance_pole_state(const PoleConstants &pole, const index_t n_steps,
                   const real_t bin_dt, real_t &state_re, real_t &state_im) {
  if (n_steps == 1) {
    multiply_state(pole.decay_re, pole.decay_im, state_re, state_im);
  } else if (n_steps > 1) {
    decay_pole_state(pole, real_t(n_steps) * bin_dt, state_re, state_im);
  }
}

// Add the carried charge, emitted `lag` before the current position.
__device__ static inline void inject_carried(const PoleConstants &pole,
                                             const real_t lag, real_t &state_re,
                                             real_t &state_im) {
  real_t e_re = 0;
  real_t e_im = 0;
  pole_exp(pole, lag, e_re, e_im);
  state_re += pole.carried_injection * e_re;
  state_im += pole.carried_injection * e_im;
}

// Where the state stands before read-out `first_read`: its position, and
// the number of bins entered by then (every bin but the last at most).
__device__ static inline void
chunk_start(const index_t *__restrict__ bin_offsets, const int n_bins,
            const int first_read, index_t &position, int &next_bin) {
  if (first_read == 0) {
    position = -2;
    next_bin = 0;
    return;
  }
  position = bin_offsets[first_read - 1] - 2;
  // the first bin past `position`, by bisection; bin_offsets ascends
  int low = 0;
  int high = n_bins - 1;
  while (low < high) {
    const int middle = (low + high) / 2;
    if (bin_offsets[middle] <= position) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  next_bin = low;
}

// Sum of `value` over the 32 lanes of the warp, valid in lane 0. Every
// lane of the warp must call it.
__device__ static inline real_t warp_sum(real_t value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// Read-outs [first_read, end_read) of one pole, from `position` and
// `next_bin` on; adds them, summed over the warp, to `voltage` if READ_OUT.
template <bool READ_OUT>
__device__ static inline void run_pole_reads(
    const PoleConstants &pole, const index_t *__restrict__ bin_offsets,
    const real_t *__restrict__ profile, const int n_bins, const int first_read,
    const int end_read, const real_t carried_lag_dt, const real_t bin_dt,
    index_t &position, int &next_bin, real_t &state_re, real_t &state_im,
    real_t *__restrict__ voltage) {
  for (int read_i = first_read; read_i < end_read; ++read_i) {
    const index_t read_position = bin_offsets[read_i] - 2;
    // every bin but the last enters the state when it is due
    for (; next_bin < n_bins - 1 && bin_offsets[next_bin] <= read_position;
         ++next_bin) {
      advance_pole_state(pole, bin_offsets[next_bin] - position, bin_dt,
                         state_re, state_im);
      position = bin_offsets[next_bin];
      state_re += pole.injection * profile[next_bin];
    }
    advance_pole_state(pole, read_position - position, bin_dt, state_re,
                       state_im);
    position = read_position;
    if (read_i == 1) {
      // The carried charge enters after the first read-out.
      inject_carried(pole, real_t(position) * bin_dt + carried_lag_dt, state_re,
                     state_im);
    }
    if (READ_OUT) {
      const real_t warp_voltage = warp_sum(pole.flipped_res_re * state_re -
                                           pole.flipped_res_im * state_im);
      if ((threadIdx.x & 31) == 0) {
        atomicAdd(&voltage[read_i], warp_voltage);
      }
    }
  }
}

// Map (P, L) of chunk blockIdx.y for every chunk but the last; one thread
// per pole and chunk. chunk_maps holds [n_chunks - 1][n_poles][4] values:
// P_re, P_im, L_re, L_im.
extern "C" __global__ void __launch_bounds__(POLE_THREADS_PER_BLOCK)
    pole_residue_chunk_maps(
        const index_t *__restrict__ bin_offsets,
        const real_t *__restrict__ profile,
        const real_t *__restrict__ carried_charge,
        const bool carried_is_counterrotating, const real_t carried_lag_dt,
        const real_t *__restrict__ poles, const real_t *__restrict__ residues,
        const bool is_counterrotating_beam,
        const real_t *__restrict__ counterrotating_pole_signs,
        const real_t factor, const real_t bin_dt, const int n_bins,
        const int n_poles, const int chunk_length, const int n_chunks,
        real_t *__restrict__ chunk_maps) {
  const int pole_i = blockIdx.x * blockDim.x + threadIdx.x;
  const int chunk_i = blockIdx.y;
  if (pole_i >= n_poles || chunk_i >= n_chunks - 1)
    return;
  const PoleConstants pole =
      load_pole_constants(poles, residues, counterrotating_pole_signs, pole_i,
                          is_counterrotating_beam, carried_is_counterrotating,
                          carried_charge[0], factor, bin_dt);
  const int first_read = chunk_i * chunk_length;
  index_t position = 0;
  int next_bin = 0;
  chunk_start(bin_offsets, n_bins, first_read, position, next_bin);
  const index_t start_position = position;
  real_t state_re = 0;
  real_t state_im = 0;
  run_pole_reads<false>(pole, bin_offsets, profile, n_bins, first_read,
                        first_read + chunk_length, carried_lag_dt, bin_dt,
                        position, next_bin, state_re, state_im, nullptr);
  real_t *map = chunk_maps + ((size_t)chunk_i * n_poles + pole_i) * 4;
  pole_exp(pole, real_t(position - start_position) * bin_dt, map[0], map[1]);
  map[2] = state_re;
  map[3] = state_im;
}

// The read-outs of chunk blockIdx.y, added to `voltage`; one thread per
// pole and chunk. The last chunk hands its state over in `states`. Its
// arguments up to `chunk_maps` are those of `pole_residue_chunk_maps`.
extern "C" __global__ void __launch_bounds__(POLE_THREADS_PER_BLOCK)
    wake_from_pole_residue(
        const index_t *__restrict__ bin_offsets,
        const real_t *__restrict__ profile,
        const real_t *__restrict__ carried_charge,
        const bool carried_is_counterrotating, const real_t carried_lag_dt,
        const real_t *__restrict__ poles, const real_t *__restrict__ residues,
        const bool is_counterrotating_beam,
        const real_t *__restrict__ counterrotating_pole_signs,
        const real_t factor, const real_t bin_dt, const int n_bins,
        const int n_poles, const int chunk_length, const int n_chunks,
        const real_t *__restrict__ chunk_maps, const real_t state_lag_dt,
        real_t *__restrict__ states, real_t *__restrict__ voltage) {
  const int pole_i = blockIdx.x * blockDim.x + threadIdx.x;
  const int chunk_i = blockIdx.y;
  if (chunk_i >= n_chunks)
    return;
  // A lane past the last pole stays for the warp sum: it runs the last
  // pole with a zero residue and keeps its state to itself.
  const bool has_pole = pole_i < n_poles;
  const int run_i = has_pole ? pole_i : n_poles - 1;
  PoleConstants pole =
      load_pole_constants(poles, residues, counterrotating_pole_signs, run_i,
                          is_counterrotating_beam, carried_is_counterrotating,
                          carried_charge[0], factor, bin_dt);
  if (!has_pole) {
    pole.flipped_res_re = 0;
    pole.flipped_res_im = 0;
  }

  // The incoming state at the first read-out, then folded over the earlier
  // chunks: S_{c+1} = P_c S_c + L_c.
  real_t state_re = states[2 * run_i];
  real_t state_im = states[2 * run_i + 1];
  decay_pole_state(pole, state_lag_dt, state_re, state_im);
  for (int earlier_i = 0; earlier_i < chunk_i; ++earlier_i) {
    const real_t *map = chunk_maps + ((size_t)earlier_i * n_poles + run_i) * 4;
    multiply_state(map[0], map[1], state_re, state_im);
    state_re += map[2];
    state_im += map[3];
  }

  const int first_read = chunk_i * chunk_length;
  const int end_read = min(first_read + chunk_length, n_bins);
  index_t position = 0;
  int next_bin = 0;
  chunk_start(bin_offsets, n_bins, first_read, position, next_bin);
  run_pole_reads<true>(pole, bin_offsets, profile, n_bins, first_read, end_read,
                       carried_lag_dt, bin_dt, position, next_bin, state_re,
                       state_im, voltage);
  if (chunk_i != n_chunks - 1)
    return;
  // Hand over one bin before the last bin, every other bin in.
  for (; next_bin < n_bins - 1; ++next_bin) {
    advance_pole_state(pole, bin_offsets[next_bin] - position, bin_dt, state_re,
                       state_im);
    position = bin_offsets[next_bin];
    state_re += pole.injection * profile[next_bin];
  }
  const index_t handover_position = bin_offsets[n_bins - 1] - 1;
  advance_pole_state(pole, handover_position - position, bin_dt, state_re,
                     state_im);
  if (n_bins == 1) {
    inject_carried(pole, real_t(handover_position) * bin_dt + carried_lag_dt,
                   state_re, state_im);
  }
  if (has_pole) {
    states[2 * pole_i] = state_re;
    states[2 * pole_i + 1] = state_im;
  }
}
