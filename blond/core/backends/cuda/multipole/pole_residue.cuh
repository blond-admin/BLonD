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
