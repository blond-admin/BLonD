// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Included by kernels.cu, which defines real_t and index_t.
#pragma once

#include "pole_residue.cuh" // pole_cexp

// ---------------------------------------------------------------------------
// Far field of a pole-residue wake as tiled GEMMs, and the near field of
// the same model. Not part of the `Specials` parity contract; see
// `CudaNoParitySpecials.wake_from_pole_residue_gemm` in multipole/pole_gemm.py for
// the scheme. The bins are cut into tiles of K bins; per tile t the first
// GEMM gives Z_t, the tile's own sum over its bins for every pole, the scan
// carries the state Y_t from tile to tile:
//   Y_{t+1} = A Y_t + G Z_t + inj h_last LB   (+ the carried charge
//                                              after tile 0)
// and the second GEMM reads out [Y_t, h_t] . [E; T]. A, G and LB are
// tabulated per step type `type_ids[t]` (one per distinct gap between
// tiles; -1: the last tile, no step after it). The scan is two-level:
// `pole_gemm_scan_local` scans each group of tiles from zero and keeps the
// group's propagator, `pole_gemm_scan_groups` stitches the groups in order,
// and `pole_gemm_scan_fix` adds each group's true start state.
//
// A slab of tiles lives in one row-major workspace, a row per tile:
// [Y_re (poles) | Y_im (poles) | h (halo, then the K charges)], so both
// GEMMs read it in place; a row of Z is [Z_re | Z_im]. Tile indices into
// the plan's tables are global, `t_first` is the slab's first tile. Arrays
// over the poles are planar, except the interleaved complex `states` and
// `pole_dt`.

__device__ static inline void pole_gemm_cmul(const real_t ar, const real_t ai,
                                             const real_t br, const real_t bi,
                                             real_t &cr, real_t &ci) {
  cr = ar * br - ai * bi;
  ci = ar * bi + ai * br;
}

// The slab's charges h into the workspace and, on the first slab
// (`n_poles` > 0), per pole: the carried charge at -1, the incoming state
// at -2 and the carried charge's injection after tile 0.
extern "C" __global__ void pole_gemm_prep(
    const real_t *__restrict__ profile, const real_t factor,
    const long long *__restrict__ gather, const int n_bins, const int t_first,
    const int n_slab_tiles, const int tile_length, real_t *__restrict__ work,
    const int work_ld, const int h_col, const int n_poles,
    const real_t *__restrict__ states, const real_t *__restrict__ pole_dt,
    const real_t *__restrict__ pair, const real_t *__restrict__ signs,
    const real_t *__restrict__ carried_charge,
    const int carried_is_counterrotating, const real_t carried_bins,
    const real_t state_bins, const real_t tile_1_bins,
    real_t *__restrict__ state_re, real_t *__restrict__ state_im,
    real_t *__restrict__ carry_re, real_t *__restrict__ carry_im,
    real_t *__restrict__ carried_re, real_t *__restrict__ carried_im) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int row_length = tile_length + 1;
  const long long n_charges = (long long)n_slab_tiles * row_length;
  if (i < n_charges) {
    const long long row = i / row_length;
    const int column = (int)(i - row * row_length);
    const long long bin =
        gather[((long long)t_first + row) * row_length + column];
    work[row * work_ld + h_col + column] =
        bin < n_bins ? factor * profile[bin] : (real_t)0;
    return;
  }
  const long long p = i - n_charges;
  if (p >= n_poles)
    return;
  const real_t dr = pole_dt[2 * p], di = pole_dt[2 * p + 1];
  const real_t flip =
      (signs[p] == -1 && carried_is_counterrotating) ? (real_t)-1 : (real_t)1;
  const real_t charge = flip * pair[p] * carried_charge[0];
  real_t er, ei;
  pole_cexp(dr * carried_bins, di * carried_bins, er, ei);
  const real_t cr = charge * er, ci = charge * ei;
  carried_re[p] = cr;
  carried_im[p] = ci;
  pole_cexp(dr * state_bins, di * state_bins, er, ei);
  pole_gemm_cmul(states[2 * p], states[2 * p + 1], er, ei, state_re[p],
                 state_im[p]);
  pole_cexp(dr * tile_1_bins, di * tile_1_bins, er, ei);
  pole_gemm_cmul(cr, ci, er, ei, carry_re[p], carry_im[p]);
}

extern "C" __global__ void pole_gemm_scan_local(
    const real_t *__restrict__ z, const int z_ld,
    const int *__restrict__ type_ids, const long long *__restrict__ last_bin,
    const real_t *__restrict__ profile, const real_t factor, const int n_bins,
    const real_t *__restrict__ a_re, const real_t *__restrict__ a_im,
    const real_t *__restrict__ g_re, const real_t *__restrict__ g_im,
    const real_t *__restrict__ lb_re, const real_t *__restrict__ lb_im,
    const real_t *__restrict__ injection, const real_t *__restrict__ carry_re,
    const real_t *__restrict__ carry_im, const int t_first, const int n_tiles,
    const int n_poles, const int group_length, real_t *__restrict__ y,
    const int y_ld, real_t *__restrict__ end_re, real_t *__restrict__ end_im,
    real_t *__restrict__ prop_re, real_t *__restrict__ prop_im) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int group = blockIdx.y;
  if (p >= n_poles)
    return;
  const int t0 = group * group_length;
  const int t1 = min(t0 + group_length, n_tiles);
  real_t yr = 0, yi = 0, pr = 1, pi = 0;
  const real_t inj = injection[p];
  for (int t = t0; t < t1; ++t) {
    const size_t row = (size_t)t * y_ld;
    y[row + p] = yr;
    y[row + n_poles + p] = yi;
    const int tile = t_first + t;
    const int type = type_ids[tile];
    if (type < 0)
      continue; // the last tile: no step after it
    const size_t k = (size_t)type * n_poles + p;
    const real_t ar = a_re[k], ai = a_im[k];
    const size_t z_row = (size_t)t * z_ld;
    const long long last = last_bin[tile];
    const real_t last_charge = last < n_bins ? factor * profile[last] : 0;
    real_t nr, ni, br, bi;
    pole_gemm_cmul(ar, ai, yr, yi, nr, ni);
    pole_gemm_cmul(g_re[k], g_im[k], z[z_row + p], z[z_row + n_poles + p], br,
                   bi);
    nr += br + inj * last_charge * lb_re[k];
    ni += bi + inj * last_charge * lb_im[k];
    if (tile == 0) {
      nr += carry_re[p];
      ni += carry_im[p];
    }
    yr = nr;
    yi = ni;
    real_t qr, qi;
    pole_gemm_cmul(ar, ai, pr, pi, qr, qi);
    pr = qr;
    pi = qi;
  }
  const size_t j = (size_t)group * n_poles + p;
  end_re[j] = yr;
  end_im[j] = yi;
  prop_re[j] = pr;
  prop_im[j] = pi;
}

extern "C" __global__ void pole_gemm_scan_groups(
    const real_t *__restrict__ end_re, const real_t *__restrict__ end_im,
    const real_t *__restrict__ prop_re, const real_t *__restrict__ prop_im,
    const int n_groups, const int n_poles, real_t *__restrict__ start_re,
    real_t *__restrict__ start_im, real_t *__restrict__ state_re,
    real_t *__restrict__ state_im) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n_poles)
    return;
  real_t sr = state_re[p], si = state_im[p];
  for (int group = 0; group < n_groups; ++group) {
    const size_t j = (size_t)group * n_poles + p;
    start_re[j] = sr;
    start_im[j] = si;
    real_t nr, ni;
    pole_gemm_cmul(prop_re[j], prop_im[j], sr, si, nr, ni);
    sr = nr + end_re[j];
    si = ni + end_im[j];
  }
  state_re[p] = sr; // the state at the next slab's first tile
  state_im[p] = si;
}

extern "C" __global__ void pole_gemm_scan_fix(
    const int *__restrict__ type_ids, const real_t *__restrict__ a_re,
    const real_t *__restrict__ a_im, const real_t *__restrict__ start_re,
    const real_t *__restrict__ start_im, const int t_first, const int n_tiles,
    const int n_poles, const int group_length, real_t *__restrict__ y,
    const int y_ld) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int group = blockIdx.y;
  if (p >= n_poles)
    return;
  const int t0 = group * group_length;
  const int t1 = min(t0 + group_length, n_tiles);
  const size_t j = (size_t)group * n_poles + p;
  real_t cr = start_re[j], ci = start_im[j];
  for (int t = t0; t < t1; ++t) {
    const size_t row = (size_t)t * y_ld;
    y[row + p] += cr;
    y[row + n_poles + p] += ci;
    const int type = type_ids[t_first + t];
    if (type < 0)
      break;
    const size_t k = (size_t)type * n_poles + p;
    real_t nr, ni;
    pole_gemm_cmul(a_re[k], a_im[k], cr, ci, nr, ni);
    cr = nr;
    ci = ni;
  }
}

// The slab's read-outs into `voltage`, from `bin_first` on. On the first
// slab, bins 1 to `n_carried_bins` of tile 0 also take the carried
// charge's read-outs, Re(carried . E) over the poles with E the rows
// [E_re; -E_im] of the read-out GEMM's [E_re; -E_im; T]: one block per such
// bin sums over the poles. The other blocks copy the remaining bins.
// Launched with GEMM_THREADS threads per block.
#define GEMM_THREADS 128
extern "C" __global__ void
pole_gemm_readout(const real_t *__restrict__ reads, const long long bin_first,
                  const long long n_slab_bins, const int n_carried_bins,
                  const int tile_length, const int n_poles,
                  const real_t *__restrict__ carried_re,
                  const real_t *__restrict__ carried_im,
                  const real_t *__restrict__ e, real_t *__restrict__ voltage) {
  if ((int)blockIdx.x < n_carried_bins) {
    __shared__ real_t partial[GEMM_THREADS];
    const int column = blockIdx.x; // read-out of bin column + 1
    real_t sum = 0;
    for (int p = threadIdx.x; p < n_poles; p += blockDim.x)
      sum += carried_re[p] * e[(size_t)p * tile_length + column] +
             carried_im[p] * e[((size_t)n_poles + p) * tile_length + column];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half /= 2) {
      if (threadIdx.x < half)
        partial[threadIdx.x] += partial[threadIdx.x + half];
      __syncthreads();
    }
    if (threadIdx.x == 0)
      voltage[bin_first + column + 1] = reads[column + 1] + partial[0];
    return;
  }
  const long long i =
      (long long)(blockIdx.x - n_carried_bins) * blockDim.x + threadIdx.x;
  if (i >= n_slab_bins || (i >= 1 && i <= n_carried_bins))
    return;
  voltage[bin_first + i] = reads[i];
}

// Hand-over: the state one bin before the last bin, from the last tile's
// state and its charges but the last, h . M with M [length][poles]; a
// single tile also takes the carried charge, advanced by `last_offset`.
extern "C" __global__ void
pole_gemm_handover(const real_t *__restrict__ work, const int work_ld,
                   const int last_row, const int h_col, const int length,
                   const int n_poles, const real_t *__restrict__ m_re,
                   const real_t *__restrict__ m_im,
                   const real_t *__restrict__ advance_re,
                   const real_t *__restrict__ advance_im, const int single_tile,
                   const real_t *__restrict__ carried_re,
                   const real_t *__restrict__ carried_im,
                   const real_t *__restrict__ carried_advance_re,
                   const real_t *__restrict__ carried_advance_im,
                   real_t *__restrict__ states) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n_poles)
    return;
  const size_t row = (size_t)last_row * work_ld;
  real_t hr, hi;
  pole_gemm_cmul(advance_re[p], advance_im[p], work[row + p],
                 work[row + n_poles + p], hr, hi);
#pragma unroll 8
  for (int j = 0; j < length; ++j) {
    const real_t charge = work[row + h_col + j];
    hr += charge * m_re[(size_t)j * n_poles + p];
    hi += charge * m_im[(size_t)j * n_poles + p];
  }
  if (single_tile) {
    real_t cr, ci;
    pole_gemm_cmul(carried_re[p], carried_im[p], carried_advance_re[p],
                   carried_advance_im[p], cr, ci);
    hr += cr;
    hi += ci;
  }
  states[2 * p] = hr;
  states[2 * p + 1] = hi;
}

#define POLE_NEAR_THREADS 256

// Near field of a pole-residue model: `voltage` += factor * (the taps at
// lags of +1, 0 and -1 bin applied to the bin's neighbours, where
// adjacent). The first bin also takes the carried charge's wake: the
// constant `carried_wake` or, with `n_carried_poles` > 0, summed over those
// poles as weight Re(residue exp(pole shift_dt)). Then the last bin's
// charge is carried to the next call. Launched with POLE_NEAR_THREADS
// threads per block.
extern "C" __global__ void pole_near_field(
    const real_t *__restrict__ profile, const real_t factor,
    const real_t *__restrict__ taps, const bool *__restrict__ adjacent,
    const int n_bins, real_t *__restrict__ carried_charge,
    const int n_carried_poles, const real_t *__restrict__ carried_poles,
    const real_t *__restrict__ carried_residues,
    const real_t *__restrict__ carried_weights, const real_t shift_dt,
    const real_t carried_wake, real_t *__restrict__ voltage) {
  __shared__ real_t partial[POLE_NEAR_THREADS];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x == 0) { // the carried charge's wake at the first bin
    real_t sum = 0;
    for (int p = threadIdx.x; p < n_carried_poles; p += blockDim.x) {
      real_t er, ei;
      pole_cexp(carried_poles[2 * p] * shift_dt,
                carried_poles[2 * p + 1] * shift_dt, er, ei);
      sum += carried_weights[p] *
             (carried_residues[2 * p] * er - carried_residues[2 * p + 1] * ei);
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half /= 2) {
      if (threadIdx.x < half)
        partial[threadIdx.x] += partial[threadIdx.x + half];
      __syncthreads();
    }
  }
  if (i >= n_bins)
    return;
  real_t value = taps[1] * profile[i];
  if (i > 0 && adjacent[i - 1])
    value += taps[0] * profile[i - 1];
  if (i + 1 < n_bins && adjacent[i])
    value += taps[2] * profile[i + 1];
  value *= factor;
  if (i == 0) { // the only thread touching the carried charge
    const real_t wake = n_carried_poles > 0 ? partial[0] : carried_wake;
    value += carried_charge[0] * wake;
    carried_charge[0] = factor * profile[n_bins - 1];
  }
  voltage[i] += value;
}
