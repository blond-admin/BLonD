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
// loop. Note that the guards protecting the conversion of a bin index
// to `int` are written as `index < lo || index >= hi`: a NaN index
// compares false against both bounds, passes the guard and reaches the
// conversion, which is undefined behaviour. The caller must not produce
// non-finite coordinates. See `Specials` in blond/core/backends/backend.py.

#include <cstring>
#include <type_traits>

#ifdef USEFLOAT
using real_t = float;
#else
using real_t = double;
#endif

// Integer type of macro-particle counts and particle loop counters.
// Must match `INDEX_DTYPE` in blond/core/backends/backend.py.
using index_t = long long;

// Needs `real_t` and `index_t` above.
#include "kernel_call_records.h"

// Start and stride of a grid-stride loop over the macro-particles. They
// are computed in 32 bits, which is exact: callables.py launches
// 2 * n_SM blocks of at most 1024 threads, far below 2^31 threads. Only
// the particle index itself needs `index_t`.
namespace {
__device__ __forceinline__ index_t particle_loop_start() {
  return static_cast<int>(threadIdx.x + blockDim.x * blockIdx.x);
}

__device__ __forceinline__ index_t particle_loop_stride() {
  const unsigned int stride = blockDim.x * gridDim.x;
  return stride;
}
} // namespace

// Per-particle kernel bodies, one overload of `apply_to_particle` per
// kernel call record (kernel_call_records.h). The eager kernels below and
// the deferred (fused) kernel call the same overload, so each formula
// exists once on the GPU.
//
// A record's loop-invariant factors (the FP64 divisions of the drifts)
// are split off into `prepare`, which returns them as a small `*Factors`
// struct; `apply_to_particle` takes the record and its factors. The
// eager kernels call `prepare` in their particle loop and nvcc hoists it
// (their SASS is the same as with the factors inside the overloads). The
// fused tile loop of the deferred executor defeats that hoisting: there
// every tile repeated a slow FP64 division per drift record. So the
// executor prepares every record of a batch once per block, before its
// tile loop. Records without such factors take `NoFactors`.
namespace {
struct NoFactors {};

// Every record type without its own `prepare` overload below.
template <class Args>
__device__ __forceinline__ NoFactors prepare(const Args &) {
  return {};
}

__device__ __forceinline__ void
apply_to_particle(const KickSingleHarmonicArgs &args, NoFactors /*factors*/,
                  const real_t &dt, real_t &dE) {
  dE += args.charge * args.voltage * sin(args.omega_rf * dt + args.phi_rf) +
        args.acceleration_kick;
}

// The multi-harmonic kick, templated on whatever holds the per-harmonic
// `voltage`, `omega_rf` and `phi_rf` arrays: the record's trailing
// columns (`RfHarmonics`), or the eager kernel's `RFParamsBatch` read in
// place from the parameter space.
template <class RFParams>
__device__ __forceinline__ void
kick_multi_harmonic_particle(const RFParams &rf_params, const int n_rf,
                             const real_t charge, const real_t acc_kick,
                             const real_t &dt, real_t &dE) {
  // Starting from acc_kick rather than zero saves an FP64 add per
  // particle, measurable on GPUs with low FP64 throughput.
  real_t dE_sum = acc_kick;
  for (int j = 0; j < n_rf; j++) {
    dE_sum += charge * rf_params.voltage[j] *
              sin(rf_params.omega_rf[j] * dt + rf_params.phi_rf[j]);
  }
  dE += dE_sum;
}

__device__ __forceinline__ void
apply_to_particle(const KickMultiHarmonicArgs &args, NoFactors /*factors*/,
                  const real_t &dt, real_t &dE) {
  kick_multi_harmonic_particle(harmonics_of(args), args.n_rf, args.charge,
                               args.acceleration_kick, dt, dE);
}

struct DriftSimpleFactors {
  real_t coeff; // T eta_0 / (beta^2 E)
};

__device__ __forceinline__ DriftSimpleFactors
prepare(const DriftSimpleArgs &args) {
  return {args.T * args.eta_0 / (args.beta * args.beta * args.energy)};
}

__device__ __forceinline__ void
apply_to_particle(const DriftSimpleArgs & /*args*/,
                  const DriftSimpleFactors &factors, real_t &dt,
                  const real_t &dE) {
  dt += factors.coeff * dE;
}

// 1 / beta^2 and 1 / E of the relativistic delta, shared by the drifts
// below that compute it.
struct RelativisticDeltaFactors {
  real_t inv_beta_sq;
  real_t inv_energy;
};

__device__ __forceinline__ RelativisticDeltaFactors
relativistic_delta_factors(const real_t beta, const real_t energy) {
  return {1.0 / (beta * beta), 1.0 / energy};
}

__device__ __forceinline__ RelativisticDeltaFactors
prepare(const DriftLikeLineSegmentArgs &args) {
  return relativistic_delta_factors(args.beta, args.energy);
}

// Drift with the linear slip factor but the exact relativistic delta;
// reproduces the longitudinal drift of an xsuite LineSegmentMap.
__device__ __forceinline__ void
apply_to_particle(const DriftLikeLineSegmentArgs &args,
                  const RelativisticDeltaFactors &factors, real_t &dt,
                  const real_t &dE) {
  const real_t inv_beta_sq = factors.inv_beta_sq;
  const real_t inv_energy = factors.inv_energy;
  const real_t delta =
      sqrt(1.0 + inv_beta_sq * (dE * dE * inv_energy * inv_energy +
                                2.0 * dE * inv_energy)) -
      1.0;
  dt += args.T * args.eta_0 * delta;
}

// The polynomial in delta, `1 + alpha_0 delta + sum_k higher_alpha[k]
// delta^(k+2)`; shared by the record overload and the eager kernel.
__device__ __forceinline__ void
drift_exact_particle(const real_t T, const real_t alpha_zero,
                     const real_t *higher_alpha, const int n_alpha,
                     const RelativisticDeltaFactors &factors, real_t &dt,
                     const real_t dE) {
  const real_t inv_beta_sq = factors.inv_beta_sq;
  const real_t inv_energy = factors.inv_energy;
  const real_t inv_energy_sq = inv_energy * inv_energy;

  const real_t delta = sqrt(1.0 + inv_beta_sq * (dE * dE * inv_energy_sq +
                                                 2.0 * dE * inv_energy)) -
                       1.0;

  real_t poly = 1.0 + alpha_zero * delta;

  real_t delta_power = delta * delta; // starts at δ²
  for (int k = 0; k < n_alpha; ++k) {
    // NOLINTNEXTLINE(*-pointer-arithmetic)
    poly += higher_alpha[k] * delta_power;
    delta_power *= delta; // next power
  }

  dt += T * (poly * (1.0 + dE * inv_energy) / (1.0 + delta) - 1.0);
}

__device__ __forceinline__ RelativisticDeltaFactors
prepare(const DriftExactArgs &args) {
  return relativistic_delta_factors(args.beta, args.energy);
}

__device__ __forceinline__ void
apply_to_particle(const DriftExactArgs &args,
                  const RelativisticDeltaFactors &factors, real_t &dt,
                  const real_t &dE) {
  drift_exact_particle(args.T, args.alpha_0, &args.higher_alpha[0],
                       args.n_alpha, factors, dt, dE);
}

// Reads the table of `build_voltage_kick_table`.
__device__ __forceinline__ void
apply_to_particle(const KickInterpolatedArgs &args, NoFactors /*factors*/,
                  const real_t &dt, real_t &dE) {
  const real_t *table = args.voltage_kick_table;
  const int n_bins = static_cast<int>((args.voltage_kick_table_length - 2) / 2);
  // Range-check before the conversion to `int` (see `hybrid_histogram`).
  // NOLINTBEGIN(*-pointer-arithmetic)
  const real_t fbin_real = floor((dt - table[0]) * table[1]);
  if (fbin_real >= static_cast<real_t>(0) &&
      fbin_real < static_cast<real_t>(n_bins)) {
    const int pair = 2 + 2 * static_cast<int>(fbin_real);
    dE += dt * table[pair] + table[pair + 1];
  } else {
    // Out of range only the interpolated voltage is undefined.
    dE += args.acceleration_kick;
  }
  // NOLINTEND(*-pointer-arithmetic)
}
} // namespace

extern "C" __global__ void drift_simple(real_t *__restrict__ beam_dt,
                                        const real_t *__restrict__ beam_dE,
                                        const real_t T, const real_t eta_zero,
                                        const real_t beta, const real_t energy,
                                        const index_t n_macroparticles) {
  const DriftSimpleArgs args = {T, eta_zero, beta, energy};
  for (index_t i = particle_loop_start(); i < n_macroparticles;
       i += particle_loop_stride()) {
    apply_to_particle(args, prepare(args), beam_dt[i], beam_dE[i]);
  }
}

extern "C" __global__ void
drift_like_line_segment(real_t *__restrict__ beam_dt,
                        const real_t *__restrict__ beam_dE, const real_t T,
                        const real_t eta_zero, const real_t beta,
                        const real_t energy, const index_t n_macroparticles) {
  const DriftLikeLineSegmentArgs args = {T, eta_zero, beta, energy};
  for (index_t i = particle_loop_start(); i < n_macroparticles;
       i += particle_loop_stride()) {
    apply_to_particle(args, prepare(args), beam_dt[i], beam_dE[i]);
  }
}

extern "C" __global__ void
kick_single_harmonic(const real_t *__restrict__ beam_dt,
                     real_t *__restrict__ beam_dE, const real_t charge,
                     const real_t voltage, const real_t omega_RF,
                     const real_t phi_RF, const index_t n_macroparticles,
                     const real_t acc_kick) {
  const KickSingleHarmonicArgs args = {voltage, omega_RF, phi_RF, charge,
                                       acc_kick};
  for (index_t i = particle_loop_start(); i < n_macroparticles;
       i += particle_loop_stride()) {
    apply_to_particle(args, prepare(args), beam_dt[i], beam_dE[i]);
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
constexpr int MAX_RF_HARMONICS_PER_LAUNCH = 32;
// Plain C arrays and external linkage: the layout is fixed by the NumPy
// structured dtype the host fills, and the type is part of the signature
// of an `extern "C"` kernel.
// NOLINTBEGIN(*-avoid-c-arrays,misc-use-internal-linkage)
struct RFParamsBatch {
  real_t voltage[MAX_RF_HARMONICS_PER_LAUNCH];
  real_t omega_rf[MAX_RF_HARMONICS_PER_LAUNCH];
  real_t phi_rf[MAX_RF_HARMONICS_PER_LAUNCH];
};
// NOLINTEND(*-avoid-c-arrays,misc-use-internal-linkage)

extern "C" __global__ void
kick_multi_harmonic(const real_t *__restrict__ beam_dt,
                    real_t *__restrict__ beam_dE,
                    const RFParamsBatch rf_params_batch,
                    const int n_rf_in_batch, const real_t charge,
                    const index_t n_macroparticles, const real_t acc_kick) {
  // The batch is read in place from the parameter space. Copying it into
  // a KickMultiHarmonicArgs first -- per thread, or per block in shared
  // memory behind a barrier -- makes the harmonic loop spill registers
  // under -maxrregcount 32.
  for (index_t i = particle_loop_start(); i < n_macroparticles;
       i += particle_loop_stride()) {
    kick_multi_harmonic_particle(rf_params_batch, n_rf_in_batch, charge,
                                 acc_kick, beam_dt[i], beam_dE[i]);
  }
}

extern "C" __global__ void beam_phase(const real_t *__restrict__ hist_x,
                                      const real_t *__restrict__ hist_y,
                                      real_t *result, real_t alpha,
                                      real_t omega_rf, real_t phi_rf,
                                      int n_bins) {
  // No `bin_size`: the trapezoidal step cancels in the sin/cos ratio the
  // caller takes.
  extern __shared__ real_t shared[];

  real_t *sin_partial = &shared[0];
  real_t *cos_partial = &shared[blockDim.x];

  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

  real_t sin_val = 0.0;
  real_t cos_val = 0.0;

  if (i < n_bins) {
    const real_t x = hist_x[i];
    const real_t prof = hist_y[i];
    const real_t phase = omega_rf * x + phi_rf;
    const real_t base = exp(alpha * x) * prof;

    const real_t coeff = ((i == 0) || (i == n_bins - 1)) ? 1.0 : 2.0;

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
  while (reduction_width < blockDim.x) {
    reduction_width <<= 1;
  }
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

extern "C" __global__ void
hybrid_histogram(const real_t *__restrict__ input, real_t *__restrict__ output,
                 const real_t cut_left, const real_t cut_right,
                 const unsigned int n_slices, const index_t n_macroparticles,
                 const int capacity) {
  extern __shared__ int block_hist[];
  const int block_thread = static_cast<int>(threadIdx.x);
  //reset shared memory
  for (int i = block_thread; i < capacity;
       i = static_cast<int>(i + blockDim.x)) {
    block_hist[i] = 0;
  }
  __syncthreads();
  real_t const inv_bin_width = n_slices / (cut_right - cut_left);

  const int low_tbin = static_cast<int>(n_slices / 2) - (capacity / 2);
  const int high_tbin = low_tbin + capacity;

  for (index_t i = particle_loop_start(); i < n_macroparticles;
       i += particle_loop_stride()) {
    // Range-check in floating point *before* the conversion:
    // converting an out-of-range value to `int` is undefined
    // behaviour.
    real_t target_bin_real = floor((input[i] - cut_left) * inv_bin_width);
    // Scaling is not exact: a value at or just below cut_right can land
    // on n_slices. Fold it back into the last bin, as np.histogram
    // does, instead of dropping the particle.
    if (target_bin_real >= real_t(n_slices) && input[i] <= cut_right) {
      target_bin_real = real_t(n_slices - 1);
    }
    if (target_bin_real < real_t(0) || target_bin_real >= real_t(n_slices)) {
      continue;
    }
    const int target_bin = (int)target_bin_real;
    if (target_bin >= low_tbin && target_bin < high_tbin) {
      atomicAdd(&(block_hist[target_bin - low_tbin]), 1);
    } else {
      atomicAdd(&(output[target_bin]), 1);
    }
  }
  __syncthreads();
  for (int i = block_thread; i < capacity;
       i = static_cast<int>(i + blockDim.x)) {
    atomicAdd(&output[low_tbin + i], (real_t)block_hist[i]);
  }
}

extern "C" __global__ void
sm_histogram(const real_t *__restrict__ input, real_t *__restrict__ output,
             const real_t cut_left, const real_t cut_right,
             const unsigned int n_slices, const index_t n_macroparticles) {
  // Named apart from `hybrid_histogram`'s: both alias the launch's dynamic
  // shared memory.
  extern __shared__ int slice_hist[];
  for (unsigned int i = threadIdx.x; i < n_slices; i += blockDim.x) {
    slice_hist[i] = 0;
  }
  __syncthreads();
  real_t const inv_bin_width = n_slices / (cut_right - cut_left);
  for (index_t i = particle_loop_start(); i < n_macroparticles;
       i += particle_loop_stride()) {
    // See `hybrid_histogram`: range-check before converting to `int`,
    // and fold a value that scales onto n_slices back into the last
    // bin instead of dropping it.
    real_t target_bin_real = floor((input[i] - cut_left) * inv_bin_width);
    if (target_bin_real >= real_t(n_slices) && input[i] <= cut_right) {
      target_bin_real = real_t(n_slices - 1);
    }
    if (target_bin_real < real_t(0) || target_bin_real >= real_t(n_slices)) {
      continue;
    }
    const int target_bin = (int)target_bin_real;

    atomicAdd(&(slice_hist[target_bin]), 1);
  }
  __syncthreads();
  for (unsigned int i = threadIdx.x; i < n_slices; i += blockDim.x) {
    atomicAdd(&output[i], (real_t)slice_hist[i]);
  }
}

namespace {
// (slope, offset) of the linear voltage in bin `i`, with `charge` and
// `acc_kick` folded in.
__device__ __forceinline__ void
voltage_kick_pair(const int i, const real_t *__restrict__ voltage_array,
                  const real_t *__restrict__ bin_centers, const real_t charge,
                  const real_t inv_bin_width, const real_t acc_kick,
                  real_t &slope, real_t &offset) {
  // NOLINTBEGIN(*-pointer-arithmetic)
  slope = charge * (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
  offset = (charge * voltage_array[i] - bin_centers[i] * slope) + acc_kick;
  // NOLINTEND(*-pointer-arithmetic)
}
} // namespace

extern "C" __global__ void lik_only_gm_copy(
    real_t *__restrict__ /*beam_dt*/, real_t *__restrict__ /*beam_dE*/,
    const real_t *__restrict__ voltage_array,
    const real_t *__restrict__ bin_centers, const real_t charge,
    const int n_slices, const index_t /*n_macroparticles*/,
    const real_t acc_kick, real_t *__restrict__ glob_vkick_factor) {
  // The unnamed parameters keep the signature of `lik_only_gm_comp`.
  // The loop steps `i` by an unsigned addition converted back to `int`.
  // A signed `int` stride lets nvcc hoist the division below above the
  // early exit of threads with no bin to fill: 1.1x slower here, 1.7x in
  // `lik_sparse_gm_copy` (T400).
  const int tid = static_cast<int>(threadIdx.x + blockDim.x * blockIdx.x);
  const unsigned int stride = gridDim.x * blockDim.x;
  real_t const inv_bin_width =
      (n_slices - 1) / (bin_centers[n_slices - 1] - bin_centers[0]);

  for (int i = tid; i < n_slices - 1; i = static_cast<int>(i + stride)) {
    const int factor_i = 2 * i;
    voltage_kick_pair(i, voltage_array, bin_centers, charge, inv_bin_width,
                      acc_kick, glob_vkick_factor[factor_i],
                      glob_vkick_factor[factor_i + 1]);
  }
}

// Table read by the deferred interpolated kick: `2 * n_slices` entries,
// [bin_centers[0], inverse bin width, (slope, offset) per bin], as
// `linear_interp_kick_table` builds it in C++.
extern "C" __global__ void
build_voltage_kick_table(const real_t *__restrict__ voltage_array,
                         const real_t *__restrict__ bin_centers,
                         const real_t charge, const int n_slices,
                         const real_t acc_kick, real_t *__restrict__ table) {
  // See `lik_only_gm_copy` for the unsigned stride.
  const int tid = static_cast<int>(threadIdx.x + blockDim.x * blockIdx.x);
  const unsigned int stride = gridDim.x * blockDim.x;
  const real_t inv_bin_width =
      (n_slices - 1) / (bin_centers[n_slices - 1] - bin_centers[0]);
  if (tid == 0) {
    table[0] = bin_centers[0];
    table[1] = inv_bin_width;
  }
  for (int i = tid; i < n_slices - 1; i = static_cast<int>(i + stride)) {
    voltage_kick_pair(i, voltage_array, bin_centers, charge, inv_bin_width,
                      acc_kick, table[2 + 2 * i], table[3 + 2 * i]);
  }
}

extern "C" __global__ void lik_only_gm_comp(
    const real_t *__restrict__ beam_dt, real_t *__restrict__ beam_dE,
    const real_t *__restrict__ /*voltage_array*/,
    const real_t *__restrict__ bin_centers, const real_t /*charge*/,
    const int n_slices, const index_t n_macroparticles, const real_t acc_kick,
    const real_t *__restrict__ glob_vkick_factor) {
  // The unnamed parameters keep the signature of `lik_only_gm_copy`.
  real_t const inv_bin_width =
      (n_slices - 1) / (bin_centers[n_slices - 1] - bin_centers[0]);
  const real_t bin0 = bin_centers[0];
  for (index_t i = particle_loop_start(); i < n_macroparticles;
       i += particle_loop_stride()) {
    // Range-check before the conversion to `int` (see `hybrid_histogram`).
    const real_t fbin_real = floor((beam_dt[i] - bin0) * inv_bin_width);
    if (fbin_real >= real_t(0) && fbin_real < real_t(n_slices - 1)) {
      const int factor_i = 2 * (int)fbin_real;
      beam_dE[i] += beam_dt[i] * glob_vkick_factor[factor_i] +
                    glob_vkick_factor[factor_i + 1];
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
  // Unsigned stride: see `lik_only_gm_copy`.
  const int tid = static_cast<int>(threadIdx.x + blockDim.x * blockIdx.x);
  const unsigned int stride = gridDim.x * blockDim.x;
  const real_t inv_bin_width = real_t(bins_per_profile) / cut_width;

  for (int i = tid; i < n_slices_total - 1; i = static_cast<int>(i + stride)) {
    // (slope, offset) of the linear voltage in bin `i`.
    const int factor_i = 2 * i;
    glob_vkick_factor[factor_i] =
        charge * (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
    glob_vkick_factor[factor_i + 1] =
        (charge * voltage_array[i] -
         bin_centers[i] * glob_vkick_factor[factor_i]) +
        acc_kick;
  }
}

extern "C" __global__ void lik_sparse_gm_comp(
    const real_t *__restrict__ beam_dt, real_t *__restrict__ beam_dE,
    const index_t n_macroparticles, const real_t first_left_cut,
    const real_t left_cut_distance, const real_t cut_width,
    const int bins_per_profile, const int n_buckets,
    const bool *__restrict__ filling_pattern,
    const int *__restrict__ bucket_index_to_memory_index, const real_t acc_kick,
    const real_t *__restrict__ glob_vkick_factor) {
  const real_t inv_hist_dist = real_t(1) / left_cut_distance;
  const real_t inv_bin_width = real_t(bins_per_profile) / cut_width;
  const real_t bin_width = cut_width / real_t(bins_per_profile);

  for (index_t i = particle_loop_start(); i < n_macroparticles;
       i += particle_loop_stride()) {
    const real_t dt = beam_dt[i];
    // Range-check before the conversion to `int` (see `hybrid_histogram`).
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

    const int factor_i =
        2 * (bucket_index_to_memory_index[bucket_i] + local_bin);
    beam_dE[i] +=
        dt * glob_vkick_factor[factor_i] + glob_vkick_factor[factor_i + 1];
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
  for (index_t i = particle_loop_start(); i < n_macroparticles;
       i += particle_loop_stride()) {
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

namespace {
// N(0, 1) draw at the backend's real_t precision.
__device__ __forceinline__ real_t
curand_standard_normal(curandStatePhilox4_32_10_t *state) {
#ifdef USEFLOAT
  return curand_normal(state);
#else
  return curand_normal_double(state);
#endif
}
} // namespace

extern "C" __global__ void apply_sr_without_quantum_excitation(
    real_t *__restrict__ beam_dE, const real_t damping_factor,
    const real_t energy_lost, const index_t n_macroparticles) {
  for (index_t i = particle_loop_start(); i < n_macroparticles;
       i += particle_loop_stride()) {
    beam_dE[i] = damping_factor * beam_dE[i] - energy_lost;
  }
}

extern "C" __global__ void apply_sr_with_quantum_excitation(
    real_t *__restrict__ beam_dE, const real_t damping_factor,
    const real_t energy_lost, const real_t noise_scale,
    const unsigned long long base_seed, const index_t n_macroparticles) {

  // One cuRAND state per thread. `base_seed` is unique per launch and
  // `tid` selects the cuRAND subsequence, so the streams are independent
  // across threads and across launches.
  const index_t tid = particle_loop_start();
  curandStatePhilox4_32_10_t state;
  curand_init(base_seed, tid, 0, &state);

  for (index_t i = tid; i < n_macroparticles; i += particle_loop_stride()) {
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
  // The coefficients are read in place from global memory. Copying them
  // into a per-thread DriftExactArgs puts the dynamically indexed array
  // in local memory, which costs more than the global reads it saves.
  const int n_used = (higher_alpha == nullptr) ? 0 : n_alpha;
  for (index_t i = particle_loop_start(); i < n_macroparticles;
       i += particle_loop_stride()) {
    drift_exact_particle(T, alpha_zero, higher_alpha, n_used,
                         relativistic_delta_factors(beta, energy), beam_dt[i],
                         beam_dE[i]);
  }
}

// A batch of kernel call records, passed by value in the kernel's
// parameter space like `RFParamsBatch`: no host-to-device copy per
// flush. 8-byte slots keep every record 8-byte aligned. Must match
// `_KERNEL_CALL_BATCH_DTYPE` in callables.py, which splits larger
// batches over several launches.
// NOLINTBEGIN(*-avoid-c-arrays,misc-use-internal-linkage)
struct KernelCallBatch {
  unsigned long long slots[KERNEL_CALL_BATCH_CAPACITY_BYTES / 8];
};
// NOLINTEND(*-avoid-c-arrays,misc-use-internal-linkage)

// The batch plus the other parameters of `execute_kernel_call_batch`
// (`n_bytes`, `store_flags`, `beam_dt`, `beam_dE`, `n_macroparticles`)
// must fit the 4096-byte kernel parameter limit of CUDA < 12.1 and
// pre-Volta GPUs: kernels.cu is one translation unit, so overflowing it
// would break every kernel on those targets.
static_assert(sizeof(KernelCallBatch) + sizeof(unsigned int) * 2 +
                      sizeof(real_t *) * 2 + sizeof(index_t) <=
                  4096,
              "execute_kernel_call_batch parameters exceed 4096 bytes");

// Bits of `store_flags`: the coordinates any record of the batch writes.
// Must match `STORE_DT` and `STORE_DE` in callables.py.
constexpr unsigned int STORE_DT = 1U;
constexpr unsigned int STORE_DE = 2U;

// Compiled Args sizes, compared with the numpy dtypes when loading.
extern "C" __device__ const unsigned int kernel_call_args_sizes[KERNEL_COUNT] =
    KERNEL_CALL_ARGS_SIZES_INITIALIZER;

// Particles each thread carries through the whole batch at once. Every
// record is applied to all of them in one `visit_kernel_call`. A power
// of two: the beam's tail is covered by its halvings.
constexpr int PARTICLES_PER_THREAD = 8;
static_assert(PARTICLES_PER_THREAD > 0 &&
                  (PARTICLES_PER_THREAD & (PARTICLES_PER_THREAD - 1)) == 0,
              "PARTICLES_PER_THREAD must be a power of two");
// Blocks of the executor launched per SM (`default_blocks` in
// callables.py). `__launch_bounds__` keeps its registers low enough for
// them to be resident at once, so the whole grid runs in one wave.
constexpr int EXECUTOR_BLOCK_SIZE = 256;
constexpr int EXECUTOR_BLOCKS_PER_SM = 2;

namespace {
// Size of the smallest record, which bounds the records per launch.
constexpr std::size_t smallest_record_size() {
  std::size_t smallest = KERNEL_CALL_ARGS_SIZES[0];
  for (const std::uint32_t size : KERNEL_CALL_ARGS_SIZES) {
    smallest = size < smallest ? size : smallest;
  }
  return sizeof(KernelCallHeader) + smallest;
}
constexpr int MAX_RECORDS_PER_LAUNCH =
    static_cast<int>(KERNEL_CALL_BATCH_CAPACITY_BYTES / smallest_record_size());

// Storage of any record's `prepare` result, one per record of a launch.
// Written and read with memcpy as the type `prepare` returns.
union RecordFactors {
  DriftSimpleFactors drift_simple;
  RelativisticDeltaFactors relativistic_delta;
};

template <class Factors>
__device__ __forceinline__ void store_factors(const Factors &factors,
                                              RecordFactors &slot) {
  static_assert(sizeof(Factors) <= sizeof(RecordFactors),
                "add the Factors type to RecordFactors");
  if (!std::is_empty<Factors>::value) {
    memcpy(&slot, &factors, sizeof(Factors));
  }
}

template <class Factors>
__device__ __forceinline__ Factors load_factors(const RecordFactors &slot) {
  Factors factors{};
  if (!std::is_empty<Factors>::value) {
    memcpy(&factors, &slot, sizeof(Factors));
  }
  return factors;
}

// Visitor: the record's loop-invariant factors into its slot.
struct PrepareRecord {
  RecordFactors *slot;
  template <class Args> __device__ void operator()(const Args &args) const {
    store_factors(prepare(args), *slot);
  }
};

// Visitor: the record, with the factors in its slot, on a tile.
// NOLINTBEGIN(*-avoid-c-arrays,*-constant-array-index)
template <int TILE> struct ApplyToParticleTile {
  real_t (*dt)[TILE];
  real_t (*dE)[TILE];
  const RecordFactors *slot;
  template <class Args> __device__ void operator()(const Args &args) const {
    using Factors = decltype(prepare(args));
    const Factors factors = load_factors<Factors>(*slot);
#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      apply_to_particle(args, factors, (*dt)[k], (*dE)[k]);
    }
  }
};

// Every record of the batch on the tile of TILE particles starting at
// `tile_start`, `stride` apart. Past the end of the beam the tile
// computes on zeros, never stored.
template <int TILE>
__device__ __forceinline__ void
apply_batch_to_tile(const KernelCallHeader *first, const KernelCallHeader *last,
                    const RecordFactors *factors,
                    const unsigned int store_flags,
                    real_t *__restrict__ beam_dt, real_t *__restrict__ beam_dE,
                    const index_t tile_start, const index_t stride,
                    const index_t n_macroparticles) {
  real_t dt[TILE];
  real_t dE[TILE];
#pragma unroll
  for (int k = 0; k < TILE; ++k) {
    const index_t i = tile_start + k * stride;
    dt[k] = i < n_macroparticles ? beam_dt[i] : real_t(0);
    dE[k] = i < n_macroparticles ? beam_dE[i] : real_t(0);
  }
  int index = 0;
  for (const KernelCallHeader *record = first; record != last;
       record = next_record(record), ++index) {
    visit_kernel_call(record,
                      ApplyToParticleTile<TILE>{&dt, &dE, &factors[index]});
  }
  const bool store_dt = (store_flags & STORE_DT) != 0U;
  const bool store_dE = (store_flags & STORE_DE) != 0U;
#pragma unroll
  for (int k = 0; k < TILE; ++k) {
    const index_t i = tile_start + k * stride;
    if (i < n_macroparticles) {
      if (store_dt) {
        beam_dt[i] = dt[k];
      }
      if (store_dE) {
        beam_dE[i] = dE[k];
      }
    }
  }
}

// The tail of the beam from `sweep_start` on, which needs `tail_length`
// (< 2 * TILE) particles per thread: a tile of TILE particles if bit TILE
// of `tail_length` is set, then the rest with the halvings of TILE.
template <int TILE>
__device__ __forceinline__ void
apply_batch_to_tail(const KernelCallHeader *first, const KernelCallHeader *last,
                    const RecordFactors *factors,
                    const unsigned int store_flags,
                    real_t *__restrict__ beam_dt, real_t *__restrict__ beam_dE,
                    index_t sweep_start, const index_t tail_length,
                    const index_t stride, const index_t n_macroparticles) {
  if ((tail_length & TILE) != 0) {
    apply_batch_to_tile<TILE>(first, last, factors, store_flags, beam_dt,
                              beam_dE, sweep_start + particle_loop_start(),
                              stride, n_macroparticles);
    sweep_start += TILE * stride;
  }
  apply_batch_to_tail<TILE / 2>(first, last, factors, store_flags, beam_dt,
                                beam_dE, sweep_start, tail_length, stride,
                                n_macroparticles);
}

// No particles left after the tile of one.
template <>
__device__ __forceinline__ void apply_batch_to_tail<0>(
    const KernelCallHeader * /*first*/, const KernelCallHeader * /*last*/,
    const RecordFactors * /*factors*/, const unsigned int /*store_flags*/,
    real_t *__restrict__ /*beam_dt*/, real_t *__restrict__ /*beam_dE*/,
    index_t /*sweep_start*/, const index_t /*tail_length*/,
    const index_t /*stride*/, const index_t /*n_macroparticles*/) {}
// NOLINTEND(*-avoid-c-arrays,*-constant-array-index)
} // namespace

// Every record of the batch on every particle, dt/dE kept in registers
// between records. The batch is staged once per block in shared memory:
// records are addressed through a runtime pointer, and addressing the
// parameter space that way makes nvcc copy the whole batch to local
// memory in every thread (a 4 KiB stack frame). Then the threads of the
// block prepare the records' factors (`prepare`) into a shared table,
// one record each, so the tile loop reads them instead of repeating the
// drifts' FP64 divisions once per tile. All threads of a warp read the
// same record and slot (a shared-memory broadcast), so the switch in
// visit_kernel_call does not diverge. A thread's tile is strided by the
// grid size, which keeps the loads and stores coalesced. Only the
// coordinates in `store_flags` are stored: a kick-only batch leaves dt
// untouched, a drift-only batch dE, so storing them would be pure
// memory traffic. The flags are uniform per launch, so no divergence.
// The grid sweeps the beam with full tiles while every thread needs all
// PARTICLES_PER_THREAD of them. The tail after that needs fewer
// particles per thread, the same number for the whole grid: it is
// covered by the halvings of the tile that add up to that number, so
// each thread computes ceil(n_macroparticles / n_threads) particles
// instead of that rounded up to a multiple of PARTICLES_PER_THREAD.
// Which halvings run depends on n_macroparticles and the grid only, so
// no divergence either.
extern "C" __global__ void __launch_bounds__(EXECUTOR_BLOCK_SIZE,
                                             EXECUTOR_BLOCKS_PER_SM)
    execute_kernel_call_batch(const KernelCallBatch batch,
                              const unsigned int n_bytes,
                              const unsigned int store_flags,
                              real_t *__restrict__ beam_dt,
                              real_t *__restrict__ beam_dE,
                              const index_t n_macroparticles) {
  __shared__ KernelCallBatch staged;
  // NOLINTNEXTLINE(*-avoid-c-arrays)
  __shared__ RecordFactors factors[MAX_RECORDS_PER_LAUNCH];
  const auto n_slots = static_cast<int>(n_bytes / sizeof(staged.slots[0]));
  for (int j = static_cast<int>(threadIdx.x); j < n_slots;
       j = static_cast<int>(j + blockDim.x)) {
    staged.slots[j] = batch.slots[j];
  }
  __syncthreads();
  // NOLINTBEGIN(*-reinterpret-cast,*-pointer-arithmetic)
  const auto *bytes = reinterpret_cast<const char *>(staged.slots);
  const auto *first = reinterpret_cast<const KernelCallHeader *>(bytes);
  const auto *last =
      reinterpret_cast<const KernelCallHeader *>(bytes + n_bytes);
  // NOLINTEND(*-reinterpret-cast,*-pointer-arithmetic)
  // NOLINTBEGIN(*-avoid-c-arrays,*-constant-array-index)
  {
    int index = 0;
    for (const KernelCallHeader *record = first; record != last;
         record = next_record(record), ++index) {
      if (index % static_cast<int>(blockDim.x) ==
          static_cast<int>(threadIdx.x)) {
        visit_kernel_call(record, PrepareRecord{&factors[index]});
      }
    }
  }
  __syncthreads();
  const index_t stride = particle_loop_stride();
  const index_t sweep_length = stride * PARTICLES_PER_THREAD;
  index_t sweep_start = 0;
  // While some thread needs all PARTICLES_PER_THREAD particles of a tile.
  for (; sweep_start + sweep_length - stride < n_macroparticles;
       sweep_start += sweep_length) {
    apply_batch_to_tile<PARTICLES_PER_THREAD>(
        first, last, factors, store_flags, beam_dt, beam_dE,
        sweep_start + particle_loop_start(), stride, n_macroparticles);
  }
  // Particles per thread still to do, < PARTICLES_PER_THREAD.
  const index_t tail_length =
      (n_macroparticles - sweep_start + stride - 1) / stride;
  apply_batch_to_tail<PARTICLES_PER_THREAD / 2>(
      first, last, factors, store_flags, beam_dt, beam_dE, sweep_start,
      tail_length, stride, n_macroparticles);
  // NOLINTEND(*-avoid-c-arrays,*-constant-array-index)
}

extern "C" __global__ void
histogram_sparse(const real_t *__restrict__ input, real_t *__restrict__ output,
                 const real_t first_left_cut, const real_t left_cut_distance,
                 const real_t cut_width, const int bins_per_profile,
                 const int n_buckets, const index_t n_macroparticles,
                 const bool *__restrict__ filling_pattern,
                 const int *__restrict__ bucket_index_to_memory_index) {

  const real_t cut_left0 = first_left_cut;
  const real_t inv_hist_dist = real_t(1) / left_cut_distance;
  const real_t inv_bin_width = real_t(bins_per_profile) / cut_width;

  // Loop through input particles and update the histograms in global
  // memory.
  for (index_t i = particle_loop_start(); i < n_macroparticles;
       i += particle_loop_stride()) {
    const real_t dt = input[i];

    // Range-check before the conversion to `int` (see `hybrid_histogram`).
    const real_t bucket_real = (dt - cut_left0) * inv_hist_dist;
    if (bucket_real < real_t(0) || bucket_real >= real_t(n_buckets)) {
      continue;
    }
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
    if (dt < cut_left || dt >= cut_right) {
      continue;
    }

    // Calculate the bin index
    const int bin = (int)((dt - cut_left) * inv_bin_width);
    if ((unsigned)bin < (unsigned)bins_per_profile) {
      atomicAdd(&output[bucket_index_to_memory_index[bucket_i] + bin], 1);
    }
  }
}

// Apply pole-residue (vector fitting) model to a beam profile to generate
// induced voltage. Mirrors the CPU/OpenMP implementation in cpp/poles.cpp but
// is parallelized one thread per pole. The per-pole state evolution is
// sequential across bins; different poles are fully independent and contend
// only on the output `voltage` buffer via atomicAdd.
//
// Complex arrays (poles, residues, states) are stored as interleaved real/imag:
//   [re0, im0, re1, im1, ...]
// The last complex element of `states` stores t_start in its real part.
extern "C" __global__ void wake_from_pole_residue(
    const real_t *__restrict__ profile, const real_t *__restrict__ profile_dts,
    const real_t *__restrict__ poles, const real_t *__restrict__ residues,
    const bool is_counterrotating_beam,
    const real_t *__restrict__ cr_pole_signs,
    const int *__restrict__ update_on_bin, const real_t factor,
    real_t *__restrict__ states, real_t *__restrict__ voltage, const int n_bins,
    const int n_poles, const int n_updates, const int /*n_profile_dts*/) {
  const int pole_i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (pole_i >= n_poles) {
    return;
  }

  const real_t two_factor = real_t(2) * factor;
  const int t_start_n = 2 * n_poles;
  const real_t t_start = states[t_start_n];

  // `cr_pole_flip` is intentionally applied to BOTH the state injection
  // and the output amplitude: for the counter-rotating beam's own wake
  // the two factors cancel (flip * flip == 1); only contributions of
  // the other beam, accumulated in the shared `states`, see a net
  // sign flip.
  real_t cr_pole_flip = real_t(1);
  if (is_counterrotating_beam && cr_pole_signs[pole_i] == real_t(-1)) {
    cr_pole_flip = real_t(-1);
  }

  const int pole_n = 2 * pole_i;
  const real_t pole_re = poles[pole_n];
  const real_t pole_im = poles[pole_n + 1];
  const real_t res_re = residues[pole_n];
  const real_t res_im = residues[pole_n + 1];

  real_t state_re = states[pole_n];
  real_t state_im = states[pole_n + 1];

  // A real pole has no implicit complex conjugate (vector-fitting
  // convention): only a pole with pole_im != 0 stands in for an
  // unstored conjugate partner and needs the doubled injection.
  const real_t injection_factor = (pole_im == real_t(0)) ? factor : two_factor;

  int i_update = 0;
  int update_on_bin_i = (n_updates > 0) ? update_on_bin[0] : -1;

  real_t decay_re = real_t(0);
  real_t decay_im = real_t(0);

  for (int bin_i = 0; bin_i < n_bins; ++bin_i) {
    if (bin_i == update_on_bin_i) {
      const real_t t_jump = (bin_i == 0)
                                ? (profile_dts[0] - t_start)
                                : (profile_dts[bin_i] - profile_dts[bin_i - 1]);

      // state *= exp(pole * t_jump)
      {
        const real_t jump_abs = exp(pole_re * t_jump);
        const real_t jump_re = jump_abs * cos(pole_im * t_jump);
        const real_t jump_im = jump_abs * sin(pole_im * t_jump);
        const real_t new_state_re = state_re * jump_re - state_im * jump_im;
        const real_t new_state_imag = state_re * jump_im + state_im * jump_re;
        state_re = new_state_re;
        state_im = new_state_imag;
      }

      // decay = exp(pole * dt)
      const real_t dt = profile_dts[bin_i + 1] - profile_dts[bin_i];
      {
        const real_t decay_abs = exp(pole_re * dt);
        const real_t cos_tmp = cos(pole_im * dt);
        const real_t sin_tmp = sin(pole_im * dt);
        decay_re = decay_abs * cos_tmp;
        decay_im = decay_abs * sin_tmp;
      }

      ++i_update;
      if (i_update < n_updates) {
        update_on_bin_i = update_on_bin[i_update];
      }
    } else {
      // state *= decay
      const real_t new_state_re = state_re * decay_re - state_im * decay_im;
      const real_t new_state_imag = state_re * decay_im + state_im * decay_re;
      state_re = new_state_re;
      state_im = new_state_imag;
    }

    const real_t half_step =
        cr_pole_flip * (real_t(0.5) * profile[bin_i]) * injection_factor;

    // First half of the trapezoidal rule.
    state_re += half_step;

    // amp = Re(residue * state)
    const real_t amp = res_re * state_re - res_im * state_im;
    atomicAdd(&voltage[bin_i], cr_pole_flip * amp);

    // Second half of the trapezoidal rule.
    state_re += half_step;
  }

  // Persist state for the next call. `t_start` for the next call is
  // written by the caller after the launch: writing it here would race
  // with pole threads that have not yet read it.
  states[pole_n] = state_re;
  states[pole_n + 1] = state_im;
}
