// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// References: Juan F. Esteban Mueller, Alexandre Lasheen, D. Quartullo, K. Iliakis

// Optimised C++ routine that calculates the kick of a voltage array on
// particles

#include <cmath>
#include <math.h>
#include <stdlib.h>

#include "blond_common.h"

extern "C" void linear_interp_kick(real_t *__restrict__ beam_dt,
                                   real_t *__restrict__ beam_dE,
                                   const real_t *__restrict__ voltage_array,
                                   const real_t *__restrict__ bin_centers,
                                   const real_t charge, const int n_slices,
                                   const int n_macroparticles,
                                   const real_t acc_kick) {

  const int STEP = 64;
  const real_t inv_bin_width =
      (n_slices - 1) / (bin_centers[n_slices - 1] - bin_centers[0]);

  real_t *voltageKick = (real_t *)malloc((n_slices - 1) * sizeof(real_t));
  real_t *factor = (real_t *)malloc((n_slices - 1) * sizeof(real_t));

#pragma omp parallel
  {
    // Keep the bin index in double until it is range-checked: converting
    // an out-of-range double to an integer type is undefined behaviour
    // (a huge positive index can wrap back into the valid bin range).
    double fbin[STEP];

#pragma omp for
    for (int i = 0; i < n_slices - 1; i++) {
      voltageKick[i] =
          charge * (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
      factor[i] =
          (charge * voltage_array[i] - bin_centers[i] * voltageKick[i]) +
          acc_kick;
    }

#pragma omp for
    for (int i = 0; i < n_macroparticles; i += STEP) {

      const int loop_count =
          n_macroparticles - i > STEP ? STEP : n_macroparticles - i;

      for (int j = 0; j < loop_count; j++) {
        fbin[j] = std::floor((beam_dt[i + j] - bin_centers[0]) *
                             inv_bin_width);
      }

      for (int j = 0; j < loop_count; j++) {
        if (fbin[j] >= 0.0 && fbin[j] < (double)(n_slices - 1)) {
          const int bin = (int)fbin[j];
          beam_dE[i + j] +=
              beam_dt[i + j] * voltageKick[bin] + factor[bin];
        }
      }
    }
  }
  free(voltageKick);
  free(factor);
}

// Sparse variant of linear_interp_kick: bin_centers/voltage are a
// concatenation of one dense island per active RF bucket (see
// EquidistantMultiProfile / histogram_sparse.cpp), with gaps between
// islands whenever the filling pattern skips a bucket. inv_bin_width is
// derived from bins_per_profile/cut_width (constant per-bucket, since all
// buckets share the same size) instead of from the array's global
// endpoints, which would be wrong across a gap. Each particle is first
// resolved to its bucket (mirroring histogram_sparse.cpp), then
// interpolated within that bucket's own bins using the same
// voltageKick/factor formula as the dense kernel.
extern "C" void linear_interp_kick_sparse(
    real_t *__restrict__ beam_dt, real_t *__restrict__ beam_dE,
    const real_t *__restrict__ voltage_array,
    const real_t *__restrict__ bin_centers, const real_t charge,
    const int n_slices_total, const int n_macroparticles,
    const real_t acc_kick, const real_t first_left_cut,
    const real_t left_cut_distance, const real_t cut_width,
    const int bins_per_profile, const int n_buckets,
    const bool *__restrict__ filling_pattern,
    const int *__restrict__ bucket_index_to_memory_index) {

  const real_t inv_bin_width = real_t(bins_per_profile) / cut_width;
  const real_t bin_width = cut_width / real_t(bins_per_profile);
  const real_t inv_hist_dist = real_t(1) / left_cut_distance;

  real_t *voltageKick =
      (real_t *)malloc((n_slices_total - 1) * sizeof(real_t));
  real_t *factor = (real_t *)malloc((n_slices_total - 1) * sizeof(real_t));

#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < n_slices_total - 1; i++) {
      voltageKick[i] =
          charge * (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
      factor[i] =
          (charge * voltage_array[i] - bin_centers[i] * voltageKick[i]) +
          acc_kick;
    }

#pragma omp for
    for (int i = 0; i < n_macroparticles; i++) {
      const real_t dt = beam_dt[i];
      const int bucket_i =
          (int)std::floor((dt - first_left_cut) * inv_hist_dist);
      if (bucket_i < 0 || bucket_i >= n_buckets)
        continue;
      if (!filling_pattern[bucket_i])
        continue;

      const real_t cut_left = first_left_cut + bucket_i * left_cut_distance;
      const real_t bucket_bin_center0 = cut_left + bin_width / real_t(2);
      const int local_bin =
          (int)std::floor((dt - bucket_bin_center0) * inv_bin_width);
      if (local_bin < 0 || local_bin >= bins_per_profile - 1)
        continue;

      const int bin = bucket_index_to_memory_index[bucket_i] + local_bin;
      beam_dE[i] += dt * voltageKick[bin] + factor[bin];
    }
  }
  free(voltageKick);
  free(factor);
}

// Optimised C++ routine that interpolates the induced voltage
// assuming constant slice width and a shift of the time array by a constant.
// Only right extrapolation is assumed; it gives zero values.
// This routine contributes to the computation of multi-turn wake with
// acceleration
extern "C" void linear_interp_time_translation(real_t *__restrict__ xp,
                                               real_t *__restrict__ yp,
                                               real_t *__restrict__ x,
                                               real_t *__restrict__ y,
                                               const int len_xp) {

  const real_t inv_bin_width = (len_xp - 1) / (xp[len_xp - 1] - xp[0]);

  const int ffbin0 = (int)((x[0] - xp[0]) * inv_bin_width);
  const int diff = len_xp - ffbin0;

#pragma omp parallel for
  for (int i = 0; i < diff - 1; i++) {
    int ffbin;
    ffbin = ffbin0 + i;
    y[i] = yp[ffbin] +
           (x[i] - xp[ffbin]) * (yp[ffbin + 1] - yp[ffbin]) * inv_bin_width;
  }
}
