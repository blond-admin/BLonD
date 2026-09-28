// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Executor of a queue of particle ops (see particle_ops.h), chunk by chunk.
//
// The eager kernels stream the whole beam from memory once per kernel. Here
// every thread walks its share of the beam in chunks small enough to stay
// in its cache and runs *all* queued ops on one chunk before moving to the
// next, so a queue of k memory-bound ops costs about one pass over DRAM
// instead of k. Results equal the eager kernels': the ops are the same
// code, particles are independent, and the histogram counts integers.
//
// The Python side (deferred.py) packs the queue into flat arrays: per op
// its id, its scalars as doubles and its arrays as pointers, each in the
// order its `scalars()`/`arrays()` name them.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "blond_common.h"
#include "openmp.h"
#include "particle_ops.h"

// Every op the queue can hold; the position in this list is the op id.
// The name is the `Specials` method the op implements, which is how
// deferred.py maps its methods onto the ids. An X-macro, because it is
// the one list every per-op table below is generated from.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define BLOND_PARTICLE_OPS(X)                                                  \
  X(KickSingleHarmonic, "kick_single_harmonic")                                \
  X(KickMultiHarmonic, "kick_multi_harmonic")                                  \
  X(DriftSimple, "drift_simple")                                               \
  X(DriftLikeLineSegment, "drift_like_line_segment")                           \
  X(LinearInterpKick, "kick_interpolated")                                     \
  X(Histogram, "histogram")

namespace {

// One queued op with its arguments and any state it needs across chunks.
class QueuedOp {
public:
  // Subclasses bring their own args and state.
  QueuedOp() = default;
  // Not copyable: an op owns per-flush buffers (tables, histogram rows).
  QueuedOp(const QueuedOp &) = delete;
  QueuedOp &operator=(const QueuedOp &) = delete;
  // Not movable: a queued op stays in place, held by pointer.
  QueuedOp(QueuedOp &&) = delete;
  QueuedOp &operator=(QueuedOp &&) = delete;
  // Virtual, so deleting through a QueuedOp* destroys the derived op.
  virtual ~QueuedOp() = default;
  // Once, before the first chunk, outside the parallel region.
  // Optional setup, e.g. build the interpolation table or zero the
  // per-thread histogram rows; no-op by default.
  virtual void prepare(int /*n_threads*/) {}
  // Per chunk, on the thread `thread_id`.
  // Required: run the op on particles [begin, end); `thread_id` selects
  // the thread-private scratch, if any.
  virtual void apply(real_t *beam_dt, real_t *beam_dE, index_t begin,
                     index_t end, int thread_id) = 0;
  // Once, after the last chunk, outside the parallel region.
  // Optional teardown, e.g. sum the per-thread histogram rows into the
  // output; no-op by default.
  virtual void finalize(int /*n_threads*/) {}
};

// Default: an op that only touches the particles of the chunk.
template <class Op> class Queued : public QueuedOp {
public:
  explicit Queued(const typename Op::Args &args) : args_(args) {}
  void apply(real_t *beam_dt, real_t *beam_dE, const index_t begin,
             const index_t end, int /*thread_id*/) override {
    Op::apply(args_, beam_dt, beam_dE, begin, end);
  }

private:
  typename Op::Args args_;
};

// The interpolation tables are filled once per flush, before the chunks.
template <> class Queued<LinearInterpKick> : public QueuedOp {
public:
  explicit Queued(const LinearInterpKick::Args &args) : args_(args) {}
  void prepare(int /*n_threads*/) override {
    voltage_kick_.resize(args_.n_slices);
    factor_.resize(args_.n_slices);
    args_.voltage_kick = voltage_kick_.data();
    args_.factor = factor_.data();
    const real_t inv_bin_width = LinearInterpKick::inv_bin_width(args_);
    LinearInterpKick::fill_table(args_, inv_bin_width, 0, args_.n_slices - 1);
    LinearInterpKick::fill_trash_entry(args_);
  }
  void apply(real_t *beam_dt, real_t *beam_dE, const index_t begin,
             const index_t end, int /*thread_id*/) override {
    LinearInterpKick::apply(args_, beam_dt, beam_dE, begin, end);
  }

private:
  LinearInterpKick::Args args_;
  std::vector<real_t> voltage_kick_;
  std::vector<real_t> factor_;
};

// Every thread counts its chunks into its own row; the rows are summed
// after the last chunk.
template <> class Queued<Histogram> : public QueuedOp {
public:
  explicit Queued(const Histogram::Args &args) : args_(args) {}
  void prepare(const int n_threads) override {
    counts_.assign((size_t)n_threads * row(), 0);
  }
  void apply(real_t *beam_dt, real_t *beam_dE, const index_t begin,
             const index_t end, const int thread_id) override {
    Histogram::count(args_, counts_.data() + (size_t)thread_id * row(),
                     args_.reads_dE ? beam_dE : beam_dt, begin, end);
  }
  void finalize(const int n_threads) override {
    Histogram::reduce(args_, counts_.data(), n_threads, 0, args_.n_slices);
  }

private:
  size_t row() const { return Histogram::row_size(args_.n_slices); }
  Histogram::Args args_;
  std::vector<index_t> counts_;
};

enum class OpId : std::uint8_t {
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define BLOND_OP_ID(op, name) op,
  BLOND_PARTICLE_OPS(BLOND_OP_ID)
#undef BLOND_OP_ID
      N_OPS
};

// `visitor.template visit<Op>(name)` for the op with id `op_id`, or
// `visitor.unknown()`. The one place the op list becomes code; every
// per-op table is a visitor.
template <class Visitor>
typename Visitor::result_type visit_op(const int op_id,
                                       const Visitor &visitor) {
  switch (static_cast<OpId>(op_id)) {
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define BLOND_OP_CASE(op, name)                                                \
  case OpId::op:                                                               \
    return visitor.template visit<op>(name);
    BLOND_PARTICLE_OPS(BLOND_OP_CASE)
#undef BLOND_OP_CASE
  default:
    return visitor.unknown();
  }
}

class MakeQueuedOp {
public:
  using result_type = std::unique_ptr<QueuedOp>;
  MakeQueuedOp(const double *scalars, void *const *pointers)
      : scalars_(scalars), pointers_(pointers) {}
  template <class Op> result_type visit(const char * /*name*/) const {
    return result_type(new Queued<Op>(Op::unpack(scalars_, pointers_)));
  }
  static result_type unknown() { return nullptr; }

private:
  const double *scalars_;
  void *const *pointers_;
};

struct OpName {
  using result_type = const char *;
  template <class Op> result_type visit(const char *name) const { return name; }
  static result_type unknown() { return ""; }
};

struct OpScalars {
  using result_type = const char *;
  template <class Op> static result_type visit(const char * /*name*/) {
    return Op::scalars();
  }
  static result_type unknown() { return ""; }
};

struct OpArrays {
  using result_type = const char *;
  template <class Op> static result_type visit(const char * /*name*/) {
    return Op::arrays();
  }
  static result_type unknown() { return ""; }
};

} // namespace

// Describe the op `op_id` to Python, which packs its parameters by name.
extern "C" int deferred_n_ops() { return static_cast<int>(OpId::N_OPS); }

extern "C" const char *deferred_op_name(const int op_id) {
  return visit_op(op_id, OpName());
}

// Names of the scalars `unpack` of op `op_id` reads, in its order.
extern "C" const char *deferred_op_scalars(const int op_id) {
  return visit_op(op_id, OpScalars());
}

// Names of the arrays `unpack` of op `op_id` reads, in its order.
extern "C" const char *deferred_op_arrays(const int op_id) {
  return visit_op(op_id, OpArrays());
}

namespace {

// Number of space-separated names in `names`.
int count_names(const char *names) {
  int count = 0;
  bool in_name = false;
  for (; *names != '\0'; names++) {
    const bool is_space = *names == ' ';
    count += (!is_space && !in_name) ? 1 : 0;
    in_name = !is_space;
  }
  return count;
}

} // namespace

// Run the `n_ops` queued ops on beam_dt/beam_dE, `chunk_size` particles at
// a time. Returns 0, or -1 for an unknown op id (nothing is run then).
extern "C" int deferred_execute(real_t *__restrict__ beam_dt,
                                real_t *__restrict__ beam_dE,
                                const index_t n_macroparticles, const int n_ops,
                                const int *op_ids, const double *scalars,
                                void *const *pointers,
                                const index_t chunk_size) {
  std::vector<std::unique_ptr<QueuedOp>> ops;
  ops.reserve(n_ops);
  for (int k = 0; k < n_ops; k++) {
    const MakeQueuedOp make(scalars, pointers);
    ops.push_back(visit_op(op_ids[k], make));
    if (!ops.back()) {
      return -1;
    }
    scalars += count_names(deferred_op_scalars(op_ids[k]));
    pointers += count_names(deferred_op_arrays(op_ids[k]));
  }

  // At least one chunk per thread: waking a thread for less work costs
  // more than it saves, which dominates small beams.
  const index_t n_chunks = (n_macroparticles + chunk_size - 1) / chunk_size;
  const int max_threads = omp_get_max_threads();
  const int n_threads = static_cast<int>(
      std::max<index_t>(1, std::min<index_t>(n_chunks, max_threads)));
  for (auto &op : ops) {
    op->prepare(n_threads);
  }

#pragma omp parallel num_threads(n_threads)
  {
    const int thread_id = omp_get_thread_num();
    index_t begin = 0;
    index_t end = 0;
    this_thread_range(n_macroparticles, begin, end);
    for (index_t chunk_begin = begin; chunk_begin < end;
         chunk_begin += chunk_size) {
      const index_t chunk_end = std::min(end, chunk_begin + chunk_size);
      for (auto &op : ops) {
        op->apply(beam_dt, beam_dE, chunk_begin, chunk_end, thread_id);
      }
    }
  }

  for (auto &op : ops) {
    op->finalize(n_threads);
  }
  return 0;
}
