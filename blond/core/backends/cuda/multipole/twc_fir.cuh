// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Included by kernels.cu, which defines real_t and index_t.
#pragma once

// ---------------------------------------------------------------------------
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
