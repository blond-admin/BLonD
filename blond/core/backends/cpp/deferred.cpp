// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Executor of a deferred batch: every queued kernel call record applied
// to one cache-sized chunk of the beam before moving to the next, so the
// particles stream through memory once per batch instead of once per
// kernel. The records and the only switch over their kernel ids are
// generated (kernel_call_records.h); this file has no kernel-specific
// code.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "blond_common.h"
#include "kernel_call_records.h"
#include "openmp.h"
#include "particle_kernels.h"
#include "scratch_buffer.h"

namespace {
// Visitor: forwards each record to the `apply_to_chunk` overload of its
// Args type (particle_kernels.h).
struct ApplyToChunk {
  real_t *beam_dt;
  real_t *beam_dE;
  index_t begin;
  index_t end;

  // The calling thread's histogram counts, if the batch bins dt.
  // Public like the members above: built as an aggregate.
  // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
  index_t *histogram_counts;

  template <class Args> void operator()(const Args &args) const {
    apply_to_chunk(args, beam_dt, beam_dE, begin, end);
  }
  void operator()(const HistogramArgs &args) const {
    bin_chunk(args, beam_dt, histogram_counts, begin, end);
  }
};

// The batch's histogram record, or nullptr; queuing one runs the batch,
// so a batch holds at most one, as its last record.
const HistogramArgs *find_histogram(const KernelCallHeader *first,
                                    const KernelCallHeader *last) {
  const HistogramArgs *histogram = nullptr;
  for (const KernelCallHeader *record = first; record != last;
       record = next_record(record)) {
    if (record->kernel_id == KernelId::Histogram) {
      histogram = &record_args<HistogramArgs>(record);
    }
  }
  return histogram;
}
} // namespace

extern "C" void execute_kernel_call_batch(const std::uint8_t *batch,
                                          const std::size_t n_bytes,
                                          real_t *beam_dt, real_t *beam_dE,
                                          const index_t n_macroparticles,
                                          const index_t chunk_size) {
  // NOLINTBEGIN(*-reinterpret-cast,*-pointer-arithmetic)
  const auto *first = reinterpret_cast<const KernelCallHeader *>(batch);
  const auto *last =
      reinterpret_cast<const KernelCallHeader *>(batch + n_bytes);
  // NOLINTEND(*-reinterpret-cast,*-pointer-arithmetic)
  const HistogramArgs *const histogram = find_histogram(first, last);
  const index_t n_bins =
      histogram != nullptr ? histogram->array_write_length : 0;
  // One row of counts per thread, summed into hist_y after the chunks;
  // index_t, so a bin can count more than 2^31 - 1 particles.
  static thread_local std::vector<index_t> counts_buffer;
  index_t *const counts = reuse_scratch(
      counts_buffer, static_cast<std::size_t>(omp_get_max_threads()) * n_bins);
#pragma omp parallel
  {
    index_t *const thread_counts =
        counts + static_cast<std::size_t>(omp_get_thread_num()) * n_bins;
    std::memset(thread_counts, 0, n_bins * sizeof(index_t));
    index_t thread_begin = 0;
    index_t thread_end = 0;
    this_thread_range(n_macroparticles, thread_begin, thread_end);
    for (index_t chunk_begin = thread_begin; chunk_begin < thread_end;
         chunk_begin += chunk_size) {
      const ApplyToChunk apply = {
          beam_dt, beam_dE, chunk_begin,
          std::min(chunk_begin + chunk_size, thread_end), thread_counts};
      for (const KernelCallHeader *record = first; record != last;
           record = next_record(record)) {
        visit_kernel_call(record, apply);
      }
    }
    if (histogram != nullptr) {
      const int n_threads = omp_get_num_threads();
#pragma omp barrier
#pragma omp for
      for (index_t bin = 0; bin < n_bins; bin++) {
        index_t count = 0;
        for (int thread = 0; thread < n_threads; thread++) {
          count += counts[static_cast<std::size_t>(thread) * n_bins + bin];
        }
        // exact while a bin holds fewer than 2^53 particles
        histogram->array_write[bin] = static_cast<real_t>(count);
      }
    }
  }
}

// `thread_range` as compiled, so the tests can check the split.
extern "C" void blond_thread_range(const index_t n, const int thread_id,
                                   const int n_threads, index_t *begin,
                                   index_t *end) {
  thread_range(n, thread_id, n_threads, *begin, *end);
}

// Size of each Args struct as compiled, compared against the numpy
// dtypes once when the library is loaded (callables.py).
extern "C" std::uint32_t kernel_call_args_size(const int kernel_id) {
  return (kernel_id >= 0 && kernel_id < KERNEL_COUNT)
             ? KERNEL_CALL_ARGS_SIZES[kernel_id]
             : 0;
}
