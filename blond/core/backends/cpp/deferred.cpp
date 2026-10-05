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
// Visitor: forwards each record to its `apply_to_chunk_counting`
// overload (particle_kernels.h), with the calling thread's counters of
// that record, then moves `counters` past them.
struct ApplyToChunk {
  real_t *beam_dt;
  real_t *beam_dE;
  index_t begin;
  index_t end;
  // Public like the members above: built as an aggregate.
  // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
  index_t **counters;

  template <class Args> void operator()(const Args &args) const {
    apply_to_chunk_counting(args, beam_dt, beam_dE, begin, end, *counters);
    *counters += counters_per_thread(args);
  }
};

// Visitor: adds up the counters per thread of the records.
struct CountersPerThread {
  index_t *total;
  template <class Args> void operator()(const Args &args) const {
    *total += counters_per_thread(args);
  }
};

// Visitor: lets each record merge its counters of every thread
// (`merge_counters`), then moves `counters` past them. Called by every
// thread of the parallel region, after all chunks.
struct MergeCounters {
  const index_t **counters;
  std::size_t row_length;
  int n_threads;
  template <class Args> void operator()(const Args &args) const {
    merge_counters(args, *counters, row_length, n_threads);
    *counters += counters_per_thread(args);
  }
};
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
  // Records that count across particles (e.g. binning them) get zeroed
  // counters, one row per thread holding every such record's counters in
  // batch order; they merge them after the last chunk.
  index_t row_length = 0;
  for (const KernelCallHeader *record = first; record != last;
       record = next_record(record)) {
    visit_kernel_call(record, CountersPerThread{&row_length});
  }
  static thread_local std::vector<index_t> counters_buffer;
  index_t *const counters = reuse_scratch(
      counters_buffer,
      static_cast<std::size_t>(omp_get_max_threads()) * row_length);
#pragma omp parallel
  {
    // NOLINTNEXTLINE(*-pointer-arithmetic)
    index_t *const thread_counters =
        counters + static_cast<std::size_t>(omp_get_thread_num()) * row_length;
    std::memset(thread_counters, 0, row_length * sizeof(index_t));
    index_t thread_begin = 0;
    index_t thread_end = 0;
    this_thread_range(n_macroparticles, thread_begin, thread_end);
    for (index_t chunk_begin = thread_begin; chunk_begin < thread_end;
         chunk_begin += chunk_size) {
      index_t *record_counters = thread_counters;
      const ApplyToChunk apply = {
          beam_dt, beam_dE, chunk_begin,
          std::min(chunk_begin + chunk_size, thread_end), &record_counters};
      for (const KernelCallHeader *record = first; record != last;
           record = next_record(record)) {
        visit_kernel_call(record, apply);
      }
    }
    if (row_length > 0) {
#pragma omp barrier
      const index_t *record_counters = counters;
      const MergeCounters merge = {&record_counters,
                                   static_cast<std::size_t>(row_length),
                                   omp_get_num_threads()};
      for (const KernelCallHeader *record = first; record != last;
           record = next_record(record)) {
        visit_kernel_call(record, merge);
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
