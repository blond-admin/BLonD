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

#include <array>
#include <cmath>

#include "blond_common.h"
#include "particle_kernels.h"
#include "scratch_buffer.h"

namespace {

// Writes the header of the kick table (see particle_kernels.h) and
// returns the inverse bin width the pairs need.
real_t write_kick_table_header(const real_t *bin_centers, const int n_slices,
                               real_t *table) {
  const real_t inv_bin_width =
      (n_slices - 1) / (bin_centers[n_slices - 1] - bin_centers[0]);
  table[0] = bin_centers[0];
  table[1] = inv_bin_width;
  return inv_bin_width;
}

// (slope, offset) of the linear voltage in each bin. An orphaned
// `omp for`: inside a parallel region the bins are shared out over the
// threads and the implicit barrier publishes the whole table; outside
// one it runs serially.
void write_kick_table_pairs(const real_t *voltage, const real_t *bin_centers,
                            const real_t charge, const int n_slices,
                            const real_t acc_kick, const real_t inv_bin_width,
                            real_t *table) {
  real_t *const pairs = table + 2;
#pragma omp for
  for (int i = 0; i < n_slices - 1; i++) {
    const real_t slope = charge * (voltage[i + 1] - voltage[i]) * inv_bin_width;
    const index_t pair = 2 * static_cast<index_t>(i);
    pairs[pair] = slope;
    pairs[pair + 1] = (charge * voltage[i] - bin_centers[i] * slope) + acc_kick;
  }
}

} // namespace

extern "C" void linear_interp_kick_table(const real_t *voltage,
                                         const real_t *bin_centers,
                                         const real_t charge,
                                         const int n_slices,
                                         const real_t acc_kick, real_t *table) {
  const real_t inv_bin_width =
      write_kick_table_header(bin_centers, n_slices, table);
#pragma omp parallel
  write_kick_table_pairs(voltage, bin_centers, charge, n_slices, acc_kick,
                         inv_bin_width, table);
}

void apply_to_chunk(const KickInterpolatedArgs &args,
                    const real_t *__restrict__ beam_dt,
                    real_t *__restrict__ beam_dE, const index_t begin,
                    const index_t end) {
  constexpr int STEP = 64;
  const real_t *const table = args.voltage_kick_table;
  const real_t bin0 = table[0];
  const real_t inv_bin_width = table[1];
  const real_t *__restrict__ pairs = table + 2;
  const int n_bins = static_cast<int>((args.voltage_kick_table_length - 2) / 2);
  const real_t acc_kick = args.acceleration_kick;

  // Keep the bin index in double until it is range-checked: converting
  // an out-of-range double to an integer type is undefined behaviour
  // (a huge positive index can wrap back into the valid bin range).
  // Left uninitialised: every element is written before it is read, and
  // value-initialising it costs a memset per call.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  std::array<double, STEP> fbin;

  for (index_t i = begin; i < end; i += STEP) {

    const index_t loop_count = end - i > STEP ? STEP : (end - i);

    for (index_t j = 0; j < loop_count; j++) {
      fbin[j] = std::floor((beam_dt[i + j] - bin0) * inv_bin_width);
    }

    for (index_t j = 0; j < loop_count; j++) {
      if (fbin[j] >= 0.0 && fbin[j] < static_cast<double>(n_bins)) {
        const index_t pair = 2 * static_cast<index_t>(fbin[j]);
        beam_dE[i + j] += beam_dt[i + j] * pairs[pair] + pairs[pair + 1];
      } else {
        // Out of range only the interpolated voltage is undefined.
        // acc_kick carries the reference energy change, which applies
        // to the whole beam (the pairs already fold it in above).
        beam_dE[i + j] += acc_kick;
      }
    }
  }
}

extern "C" void linear_interp_kick(const real_t *__restrict__ beam_dt,
                                   real_t *__restrict__ beam_dE,
                                   const real_t *__restrict__ voltage_array,
                                   const real_t *__restrict__ bin_centers,
                                   const real_t charge, const int n_slices,
                                   const index_t n_macroparticles,
                                   const real_t acc_kick) {
  static thread_local std::vector<real_t> table_buffer;
  real_t *const table =
      reuse_scratch(table_buffer, 2 * static_cast<std::size_t>(n_slices));
  KickInterpolatedArgs args{};
  args.voltage_kick_table = table;
  args.voltage_kick_table_length = 2 * static_cast<index_t>(n_slices);
  args.acceleration_kick = acc_kick;
  const real_t inv_bin_width =
      write_kick_table_header(bin_centers, n_slices, table);
  // One parallel region for table and kick: the table's `omp for`
  // ends in a barrier, after which every thread reads the whole table.
#pragma omp parallel
  {
    write_kick_table_pairs(voltage_array, bin_centers, charge, n_slices,
                           acc_kick, inv_bin_width, table);
    index_t begin = 0;
    index_t end = 0;
    this_thread_range(n_macroparticles, begin, end);
    apply_to_chunk(args, beam_dt, beam_dE, begin, end);
  }
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
    const real_t *__restrict__ beam_dt, real_t *__restrict__ beam_dE,
    const real_t *__restrict__ voltage_array,
    const real_t *__restrict__ bin_centers, const real_t charge,
    const int n_slices_total, const index_t n_macroparticles,
    const real_t acc_kick, const real_t first_left_cut,
    const real_t left_cut_distance, const real_t cut_width,
    const int bins_per_profile, const int n_buckets,
    const bool *__restrict__ filling_pattern,
    const int *__restrict__ bucket_index_to_memory_index) {

  // Fetched first: the thread_local lookup is a call that would otherwise
  // force the constants below onto the stack in the single-core build.
  static thread_local std::vector<real_t> voltageKick_buffer;
  static thread_local std::vector<real_t> factor_buffer;
  real_t *const voltageKick =
      reuse_scratch(voltageKick_buffer, n_slices_total - 1);
  real_t *const factor = reuse_scratch(factor_buffer, n_slices_total - 1);

  const real_t inv_bin_width = real_t(bins_per_profile) / cut_width;
  const real_t bin_width = cut_width / real_t(bins_per_profile);
  const real_t inv_hist_dist = real_t(1) / left_cut_distance;

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
    for (index_t i = 0; i < n_macroparticles; i++) {
      const real_t dt = beam_dt[i];
      // Range-check in floating point *before* the conversion:
      // converting an out-of-range value to `int` is undefined
      // behaviour. The dense loop above already does this.
      const real_t bucket_real =
          std::floor((dt - first_left_cut) * inv_hist_dist);
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
          std::floor((dt - bucket_bin_center0) * inv_bin_width);
      if (local_bin_real < real_t(0) ||
          local_bin_real >= real_t(bins_per_profile - 1)) {
        beam_dE[i] += acc_kick;
        continue;
      }
      const int local_bin = (int)local_bin_real;

      const int bin = bucket_index_to_memory_index[bucket_i] + local_bin;
      beam_dE[i] += dt * voltageKick[bin] + factor[bin];
    }
  }
}

// Optimised C++ routine that interpolates the induced voltage
// assuming constant slice width and a shift of the time array by a constant.
// Only right extrapolation is assumed; it gives zero values.
// This routine contributes to the computation of multi-turn wake with
// acceleration
extern "C" void linear_interp_time_translation(const real_t *__restrict__ xp,
                                               const real_t *__restrict__ yp,
                                               const real_t *__restrict__ x,
                                               real_t *__restrict__ y,
                                               const int len_xp) {

  const real_t inv_bin_width = (len_xp - 1) / (xp[len_xp - 1] - xp[0]);

  const int ffbin0 = (int)((x[0] - xp[0]) * inv_bin_width);
  const int diff = len_xp - ffbin0;

#pragma omp parallel for
  for (int i = 0; i < diff - 1; i++) {
    const int ffbin = ffbin0 + i;
    y[i] = yp[ffbin] +
           (x[i] - xp[ffbin]) * (yp[ffbin + 1] - yp[ffbin]) * inv_bin_width;
  }
}
