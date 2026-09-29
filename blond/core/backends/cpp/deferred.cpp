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

#include "blond_common.h"
#include "kernel_call_records.h"
#include "openmp.h"
#include "particle_kernels.h"

namespace {
// Visitor: forwards each record to the `apply_to_chunk` overload of its
// Args type (particle_kernels.h).
struct ApplyToChunk {
  real_t *beam_dt;
  real_t *beam_dE;
  index_t begin;
  index_t end;

  template <class Args> void operator()(const Args &args) const {
    apply_to_chunk(args, beam_dt, beam_dE, begin, end);
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
#pragma omp parallel
  {
    index_t thread_begin = 0;
    index_t thread_end = 0;
    this_thread_range(n_macroparticles, thread_begin, thread_end);
    for (index_t chunk_begin = thread_begin; chunk_begin < thread_end;
         chunk_begin += chunk_size) {
      const ApplyToChunk apply = {
          beam_dt, beam_dE, chunk_begin,
          std::min(chunk_begin + chunk_size, thread_end)};
      for (const KernelCallHeader *record = first; record != last;
           record = next_record(record)) {
        visit_kernel_call(record, apply);
      }
    }
  }
}

// Size of each Args struct as compiled, compared against the numpy
// dtypes once when the library is loaded (callables.py).
extern "C" std::uint32_t kernel_call_args_size(const int kernel_id) {
  return (kernel_id >= 0 && kernel_id < KERNEL_COUNT)
             ? KERNEL_CALL_ARGS_SIZES[kernel_id]
             : 0;
}
