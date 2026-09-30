# Deferred kernel execution for the cpp and cuda backends — design

Status: draft for review · Branch: `blonder_feature/deffered-backends` · 2026-09-29

## 1. Purpose

Each turn, BLonD calls several per-particle kernels in a row: RF kicks,
drifts and the interpolated induced-voltage kick. Every call streams the
full `dt`/`dE` arrays through memory. The idea here is to defer those calls
and run them fused, so the particles are read once per batch instead of once
per kernel.

The prototype `blonder_coding_experiments/cpp-deferred-queue` (`cc12aa382`)
showed this works on the CPU (§9).

This design keeps that gain and fixes the prototype's three shortcomings:

1. **GPU support:** add a fused interpreter kernel for `cuda`.
2. **Readability and extensibility.**
   - Replace the X-macro, the visitors and the virtual wrappers with a single
     generated dispatch, `visit_kernel_call`, plus one plain overload per
     kernel.
   - Replace the positional packing, which had three unchecked sources of
     truth, with one generated layout.
   - Remove every special case from the executors: no roles, no
     setup/finish steps, and no reduction plumbing.
3. **xsuite synergy.** Reuse the patterns of xtrack's fused `track_line`,
   without depending on xobjects:
   - typed POD records tagged with a type id;
   - one dispatch over that type id;
   - a chunk-outer loop on the CPU and one thread per particle on the GPU.

The work is **implemented fresh on the current branch**. The prototype serves
only as a reference: its side-branch merges and its "Faster" commit are not
taken over.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| kernel call | One deferred invocation of a `Specials` method |
| kernel call record | A `KernelCallHeader` followed by that kernel's `…Args` struct, in 8-byte slots |
| batch | The records one flush executes, fused into one pass over the particles |
| kernel call queue | The per-thread Python object that collects records until the next flush |

These words appear as-is in file, class and function names.

## 3. Scope

**Deferrable kernels:** only per-particle updates that read and write
nothing but `dt`/`dE` and their own record.

| Kernel | Condition |
|---|---|
| `kick_single_harmonic` | |
| `kick_multi_harmonic` | |
| `drift_simple` | |
| `drift_like_line_segment` | |
| `drift_exact` | Up to 8 higher-order `alpha` coefficients. More than that falls back to flush-then-call |
| `kick_interpolated` | Dense path only. The voltage-kick table is built when the call is queued (§5.2) |

**Reductions are not fused. `histogram` runs eagerly, after a flush.** The
spike in §9 shows this costs nothing measurable once a turn computes one
profile. It keeps the executors free of accumulators, of per-batch setup and
reduction steps, and of output parameters. A fused `histogram_sparse` would
have been the same effort a second time.

**Everything else runs eagerly after a flush.** `make_deferred_specials`
wraps every `Specials` method that has no class in `KERNEL_CALL_ARGS` as
*flush-then-call*. The wrapper first runs the queued batch, then calls the
eager method unchanged. Results are identical to eager, but nothing fuses
across that call.

One uniform rule applies instead of a decision per method, because:
- a method added to the ABC later is safe by default;
- flushing an empty queue costs one Python call;
- a test checks that every public `Specials` method is either deferred or
  wrapped.

The non-deferred methods:

| `Specials` method | Why not deferred | Caller |
|---|---|---|
| `histogram` | A reduction; not fused, see above | `generals/distributed/distributed_array.py` |
| `histogram_sparse` | A reduction into many per-bunch profiles | `generals/distributed/distributed_array.py` |
| `loss_box` | Per particle, but it also writes the `flags` array, which the batch does not carry. Follow-up candidate | `physics/losses.py` |
| `apply_synchrotron_radiation_and_quantum_excitation_energy_kick` | Quantum excitation draws random numbers. Fusing changes the RNG stream order, so deferred would no longer reproduce eager | `physics/synchrotron_radiation/base.py` |
| `move_flagged_elements_to_end` | Reorders particles and returns a count; not a per-particle update | `core/beam/base.py` |
| `music_track` | A sequential algorithm over sorted particles | `physics/impedances/music_algorithm.py` |
| `beam_phase` | Works on the profile and returns a scalar to the host | `physics/feedbacks/beam_feedback.py` |
| `wake_from_pole_residue` | Works on profile/wake arrays, not particles | `physics/impedances/solvers.py` |
| `sum_1d_array`, `dot_product_1d_array` | Return a scalar. The MPI beam statistics pass `array_local` straight in, so the flush is what makes them correct | `core/backends/mpi_distributed/callables.py` |
| `get_max_threads` | A query only. It flushes like the rest for uniformity, which is harmless | thread setup |

Two paths of deferrable kernels are also flush-then-call:
- the sparse path of `kick_interpolated` (`first_left_cut is not None`);
- `drift_exact` with more than 8 higher-order coefficients.

**Specials names.** The new modes are `set_specials("cpp_deferred")` and
`set_specials("cuda_deferred")`. The eager `cpp`/`cuda` backends keep working
as today.

**Out of scope:**
- fused reductions;
- generating the existing `RFParamsBatch` dtype;
- an xtrack-element adapter.

## 4. Record layout — one definition

### 4.1 `blond/core/backends/deferred/kernel_call_records.py`

**Each deferrable kernel is a frozen dataclass**, a subclass of
`KernelCallArgs`. The dataclass *is* the layout definition, in the spirit of
xobjects' `_xofields` or a ctypes `Structure`:

```python
@dataclass(frozen=True, eq=False)
class DriftSimpleArgs(KernelCallArgs):
    T: Real
    eta_0: Real
    beta: Real
    energy: Real
```

**Everything is a Python symbol, not a string**, so the IDE autocompletes it
and a typo fails at import or at construction instead of during a
simulation:
- **One name on both sides.** The class name is the name of the generated C
  struct. The `Specials` method is derived from it
  (`DriftSimpleArgs` → `drift_simple`). `__init_subclass__` raises
  `TypeError` if `Specials` has no such method.
- **Field kinds are typed markers.** A field's kind is an `Annotated` alias:
  `Real`, `Int32`, `Index`, `InputArray`, `RfParameters`, `HigherAlphas`.
  Each alias carries a `RecordField` object, and that object knows its C
  declaration, its numpy dtype and how to pack a value. The generator and the
  queue ask the marker, so neither needs a chain of `if kind == ...`.
- **Records are built through one classmethod.** It is
  `from_specials_call(arguments, eager_specials) -> list[Self] | None`, where
  `None` means "run eagerly". The default fills every field from the kwarg
  of the same name. A kernel overrides it when its arguments need
  transforming, and the override returns typed instances such as
  `KickMultiHarmonicArgs(n_rf=…, voltage=…)`.
- **Registry.** `KERNEL_CALL_ARGS` is a tuple of the classes; its order is the
  order of `KernelId`.

The design needs Python ≥3.11 for `typing.Self`. BLonD drops 3.10 before
this lands.

The beam's `dt`/`dE` are **not** fields. They are bound once per queue and
passed once per batch (§5.1).

**Field markers:**

| Alias (marker) | C member(s) | numpy dtype | When queued |
|---|---|---|---|
| `Real` (`RealField`) | `real_t name` | `float64` | Captured by value |
| `Int32` (`Int32Field`) | `std::int32_t name` | `int32` | Captured by value; overflow raises |
| `Index` (`IndexField`) | `index_t name` | `INDEX_DTYPE` | Captured by value; `index_abi.cpp` guards the width |
| `InputArray` (`InputArrayField`) | `const real_t *name` + `index_t name_length` | `uintp` + `INDEX_DTYPE` | Host address on cpp, device address on cuda. Kept alive until the flush. `from_specials_call` must hand over an array that nobody writes before the flush, such as a fresh table |
| `RfParameters`, `HigherAlphas` (`InlineRealArrayField(32)`, `InlineRealArrayField(8)`) | `real_t name[max_length]` | `(float64, (max_length,))` | Copied from a **host** array. Unused slots are zeroed. The number of used slots is a separate `Int32` field (`n_rf`, `n_alpha`) |

**Rules for every field:**
- **Order and padding.** Fields keep their declared order. The generator
  aligns each member to its element size and pads the struct to 8 bytes with
  explicit `padding_<k>` members. C and numpy get the same offsets, so no
  compiler padding is assumed.
- **Precision.** Reals are 64-bit, as in the only supported backends. The
  header asserts this with `static_assert`, and the load-time ABI check
  (§4.4) catches a stale library.
- **Arrays** must already have the float dtype and be C-contiguous. This is
  checked by an `assert` in `pack`, following the wrapper convention.
- **Inline arrays reject GPU arrays**, because reading one would force a sync
  on every call. `RFParamsBatch` makes the same assumption today.
- **More harmonics than 32.** `KickMultiHarmonicArgs.from_specials_call`
  returns several records. `acceleration_kick` is applied only in the last
  one, as `CudaSpecials.kick_multi_harmonic` does today. At least one record
  is always returned, so `n_rf == 0` still applies the kick.

**The kernels:**

| Class | Fields | `from_specials_call` |
|---|---|---|
| `KickSingleHarmonicArgs` | `voltage`, `omega_rf`, `phi_rf`, `charge`, `acceleration_kick`: `Real` | default |
| `KickMultiHarmonicArgs` | `n_rf`: `Int32`<br>`voltage`, `omega_rf`, `phi_rf`: `RfParameters`<br>`charge`, `acceleration_kick`: `Real` | splits into records of 32 harmonics |
| `DriftSimpleArgs` | `T`, `eta_0`, `beta`, `energy`: `Real` | default |
| `DriftLikeLineSegmentArgs` | `T`, `eta_0`, `beta`, `energy`: `Real` | default |
| `DriftExactArgs` | `T`, `alpha_0`, `beta`, `energy`: `Real`<br>`n_alpha`: `Int32`<br>`higher_alpha`: `HigherAlphas` | inlines the coefficients; more than 8 → `None` (eager) |
| `KickInterpolatedArgs` | `voltage_kick_table`: `InputArray`<br>`acceleration_kick`: `Real` | builds the table; sparse profiles → `None` (eager) |

**`drift_exact` coefficients are inlined; its caller and CUDA wrapper change.**
- **The polynomial can't be split.** Unlike RF harmonics, the higher-order
  terms can't be spread over several records. More than 8 coefficients
  therefore falls back to flush-then-call. Realistic lattices use ≤ 3.
- **Today the caller copies to the GPU every turn.** `physics/drifts.py:684`
  builds `higher_alpha` with `backend.array(...)`, which is a small
  host-to-device copy per turn on the GPU. `CudaSpecials.drift_exact` also
  asserts a device array, which contradicts the ABC's `NumpyArray`
  annotation.
- **Change: the caller passes a host array.** `drifts.py` passes
  `np.asarray(..., dtype=backend.float)`. The deferred path copies it into the
  record, with no transfer. The eager `CudaSpecials.drift_exact` accepts the
  host array and does the copy itself. That is the same cost as today, just
  moved into the wrapper.
- **The CPU keeps its tuning.** The `apply_to_chunk` overload for
  `DriftExactArgs` dispatches on `n_alpha` to the existing
  `drift_exact_unrolled<N>`, so the unrolled specialisations for 0–4
  coefficients are kept.

**`kick_interpolated` needs no per-batch setup.** Its `from_specials_call`
builds the voltage-kick table when the call is queued. The table touches no
particles, so building it before the flush is exact.

**Table layout.** Both backends use the same layout:
`[first_bin_center, inverse_bin_width, (slope, offset) × n_bins]`, with
`charge` and `acceleration_kick` folded into the pairs. The GPU path can
therefore read the bin geometry without a sync to the host.

**Table builders.** Each backend's eager specials builds the table through
`_build_voltage_kick_table`, which calls:

| Backend | Builder |
|---|---|
| cpp | `linear_interp_kick_table`, factored out of `linear_interp_kick.cpp`. The eager kick uses it too |
| cuda | A new `build_voltage_kick_table` kernel. It shares its pair loop with `lik_only_gm_copy` through a `__device__` helper |

The table is a fresh array owned by the queue, so it needs no snapshot.

**The same module also:**
- builds the numpy structured dtypes;
- writes the header when run as
  `python -m blond.core.backends.deferred.kernel_call_records`.

### 4.2 `blond/core/backends/deferred/kernel_call_records.h` (generated, checked in)

This header is generated, so it must not be edited by hand. It is plain
C++11, which both g++ (`-std=c++11`) and nvcc accept. It contains:

```cpp
enum class KernelId : uint32_t { KickSingleHarmonic, ..., KickInterpolated };
struct KernelCallHeader { KernelId kernel_id; uint32_t record_size_bytes; };
struct DriftSimpleArgs { real_t T; real_t eta_0; real_t beta; real_t energy; };
...
static_assert(sizeof(DriftSimpleArgs) == 32, "regenerate kernel_call_records.h");

// The only switch over KernelId; generated, so it cannot miss a kernel.
template <class Visitor>
BLOND_HOST_DEVICE void visit_kernel_call(const KernelCallHeader *record,
                                         const Visitor &visitor) {
  switch (record->kernel_id) {
    case KernelId::DriftSimple:
      visitor(record_args<DriftSimpleArgs>(record)); break;
    ...
  }
}
```

`BLOND_HOST_DEVICE` expands to `__host__ __device__` under nvcc and to
nothing under g++. Both compile scripts add `-I blond/core/backends/deferred`.

### 4.3 Batch layout

A batch is a contiguous byte buffer of records packed back to back, each
padded to 8 bytes. `next_record(record)` advances by `record_size_bytes`, so
no offset table is needed.

### 4.4 Guards

1. **Header up to date.** A unit test regenerates the header and compares it
   with the checked-in file.
2. **Record sizes checked at load time.**
   - The cpp library exports `kernel_call_record_size(KernelId)`, following
     `cpp/index_abi.cpp`.
   - When the library is loaded, Python compares each size with the dtype's
     `itemsize` and raises on any mismatch. Without this check, a mismatch
     would silently corrupt particle data.
3. **Missing kernel implementation fails the build.** The executor's visitor
   calls one overload per `Args` type, so a kernel without an implementation
   in a backend does not compile.

### 4.5 Adding a deferrable kernel

This checklist goes into the module docstring:
1. Add a `<Kernel>Args(KernelCallArgs)` dataclass to `KERNEL_CALL_ARGS`, and override `from_specials_call` only if
   the kernel's arguments need transforming.
2. Regenerate the header.
3. Add one overload per backend:
   - `apply_to_chunk(const XArgs&, …)` in `cpp/particle_kernels.h`;
   - `apply_to_particle(const XArgs&, …)` in `cuda/kernels.cu`.

The executors are never touched.

## 5. Python kernel call queue (shared by both backends)

`blond/core/backends/deferred/kernel_call_queue.py`

### 5.1 `KernelCallQueue(threading.local)`

- **One beam per queue.** The queue is bound to one beam's `dt`/`dE`. A call
  on a different beam flushes first.
- **Record storage.** Records go into a preallocated, growable `uint8`
  buffer, written through structured-dtype views.
- **Array lifetime.** `keep_alive` holds every referenced array until the
  flush. Values are captured as described in §4.1.

**Why `threading.local`.** There is one `backend.specials` per process, and
every simulation in that process calls it.
- Running separate simulations in Python threads of one process is
  supported: `ctypes` releases the GIL, and
  `tests/unittests/core/backends/cpp/test_thread_safety.py` covers this case.
- With a single shared queue, two threads' records would interleave into one
  batch and be applied to the wrong beam. `threading.local` gives each
  Python thread its own queue.
- OpenMP worker threads never touch the queue.
- On CuPy, each queue launches on its thread's current stream, as eager CuPy
  calls do.
- **Limitation:** records can't be queued in one thread and flushed in
  another. Nothing in BLonD does that.

### 5.1a MPI

Under `mpirun`, every rank is a separate process with its own `backend.specials`
and its own queue, which holds calls on that rank's `array_local`.
`threading.local` plays no role across ranks.

**`flush()` is purely rank-local.** It never communicates, so ranks cannot
deadlock on it.

**Every collective must see up-to-date local data:**
- **Histogram.** `DistributedArray.histogram` calls `Specials.histogram`, then
  `Allreduce`s the result (`distributed_array.py:354`). Because `histogram`
  is flush-then-call, all queued kicks and drifts have been applied first.
- **Statistics through `Specials`** (`sum_1d_array`, `dot_product_1d_array`
  in `mpi_distributed/callables.py`) are flush-then-call, so they are
  correct.
- **Statistics that bypass `Specials`** are
  `DistributedArray.min/max/mean/std/sum/mpi_gather/mpi_scatter/copy_as_*`,
  which run numpy directly on `array_local`, and the Beam statistics
  `dt_min`/`dE_max`/… built on them. The Beam stores its coordinates as a
  `FlushingDistributedArray` (`blond/core/beam/flushing_distributed_array.py`),
  whose `array_local` flushes on every read, so all of these see flushed
  data without a flush of their own. The base `DistributedArray`, used
  beyond the Beam, stays unaware of deferral.
- **GPU with MPI.** CUDA-aware MPI on CuPy arrays already needs the stream
  synchronised before a collective. Deferral changes nothing here, because
  its launches go on the same stream.

### 5.2 `make_deferred_specials(eager_specials_class, execute_batch)`

The factory returns a subclass of the eager specials class:
- **Generated queuing methods.** One per `KERNEL_CALL_ARGS` class, built
  through its `from_specials_call`. No method is
  written by hand for any kernel.
- **Every other method** becomes flush-then-call (§3).
- **`flush()` on the `Specials` ABC.** It is added there as a concrete no-op,
  so callers can flush unconditionally on every backend.

### 5.3 `execute_batch`, one per backend

Signature: `execute_batch(batch_bytes, n_bytes, dt, dE, n_macroparticles)`.

**cpp:** one ctypes call to `execute_kernel_call_batch`.

**cuda:** one launch of the `execute_kernel_call_batch` kernel.
- The batch is passed **by value** as a fixed-capacity `KernelCallBatch`
  struct, the same way `RFParamsBatch` is passed today. There is no
  host-to-device copy per flush.
- Capacity is `KERNEL_CALL_BATCH_CAPACITY_BYTES` = 4064 B (4096 - 32), so the
  whole kernel-parameter list still fits the classic 4096 B kernel-parameter
  limit alongside the batch's other fields; every supported GPU accepts it.
- A larger batch is split into several launches. The result is still
  correct, only less fused.
- A multi-harmonic record is about 800 B (3 × `RfParameters` of 32 reals).
- The fused interpreter kernel tiles 8 particles per thread, to amortise the
  per-thread cost of decoding the queued record stream over more work.

### 5.4 Flush points

Two choke points replace per-method flushes:

1. **Beam coordinate storage.** `BeamBaseClass._dt`, `_dE`, `_flags` and
   `_ids` are `FlushingCoordinates` descriptors. Whatever is assigned
   (`setup_beam`, `add_beam`/`add_particles` via
   `distributed_array.concatenate`, `copy_coordinates_from`, a plain
   `beam._dE = DistributedArray(...)` in a script) is stored as a
   `FlushingDistributedArray`; a raw array is rejected. The descriptor
   flushes *before* replacing, because queued calls reference the old
   arrays. On the stored array:
   - reading `array_local` flushes, so every inherited method
     (`min`/`max`/`mean`/`std`/`sum`/`histogram`/`histogram_sparse`/
     `mpi_gather`/`mpi_scatter`/`copy_as_*`) and every Beam method or
     script that reads the data sees queued calls applied, with no flush
     of its own;
   - assigning `array_local` flushes before replacing it;
   - `copy`/`deepcopy`/pickling flush (`__getstate__`), and the copy is
     again flushing;
   - `local_size`/`global_size` do **not** flush: queued kernels never
     change the particle count, and `RFStation._track` asks for
     `common_array_size` between the queued kick and drift every turn.
2. **Deferred specials** (`make_deferred_specials`): every non-deferrable
   method is flush-then-call, and a deferrable call on a different beam
   flushes first.

Also: a change of backend or specials flushes, and `Simulation.mainloop`
flushes once after the execution model returns (normal end and the
`until_section_index` early return). The latter is not needed for
readouts in the same thread, which flush through (1); it is needed
because the queue is `threading.local`: a simulation run in a worker
thread would otherwise leave its last turn in that thread's queue, which
is dropped when the thread ends, so a later read from the main thread
would silently see stale coordinates. The execution models themselves
contain no flush; observables and callbacks read through the Beam.

**Kernel-argument accessors, which do not flush.** If passing the beam to
a deferred kernel flushed, every deferred call would flush just before
being queued, and nothing would fuse. `BeamBaseClass` therefore has two
properties, `kernel_call_dt` and `kernel_call_dE`, which return
`FlushingDistributedArray.array_local_without_flush`. They are the only
users of that raw accessor; a test greps `blond/` and fails on any other
use.

**Accessor rules:**
- `backend.specials.<kernel>(...)` call sites use `kernel_call_dt` /
  `kernel_call_dE`.
- Any other code may read the coordinates any way it likes, including
  `beam._dt.array_local`: every path goes through the flushing storage.
  The earlier grep guard against `._dt`/`._dE` outside `core/beam/` is
  replaced by the guard on `array_local_without_flush`.

**Remaining hole.** A raw NumPy/CuPy array obtained earlier (from
`read_partial_*`, `kernel_call_*` or `array_local`) is a plain array: if
more kernel calls are queued afterwards, reading that same object later
does not flush and may be stale. Re-read through the Beam. The
`kernel_call_*` and `read_partial_*` docstrings say so.

## 6. CPU executor (`blond/core/backends/cpp/deferred.cpp`)

### 6.1 Kernel bodies (`blond/core/backends/cpp/particle_kernels.h`, new)

The header has one overload per kernel:

```cpp
BLOND_PREFER_VECTOR_WIDTH_512 BLOND_NOINLINE void
apply_to_chunk(const DriftSimpleArgs &args, real_t *dt, const real_t *dE,
               index_t begin, index_t end);
```

It loops over the particles in `[begin, end)`.

**Why `noinline` is load-bearing.** GCC falls back to 256-bit code when a
compute-bound body is inlined into an OpenMP region that lacks the
vector-width attribute.

**One formula per kernel on the CPU.** The eager `extern "C"` kernels in
`kick.cpp`, `drift.cpp`, `drift_exact.cpp` and `linear_interp_kick.cpp`
become thin wrappers.
Each builds its `Args` and calls the same overload across the whole beam,
split over OpenMP threads. The eager `linear_interp_kick` calls the new
`linear_interp_kick_table` first.

### 6.2 Executor

```cpp
extern "C" void execute_kernel_call_batch(const uint8_t *batch,
                                          size_t n_bytes, real_t *dt,
                                          real_t *dE, index_t n);
```

The whole executor is one OpenMP region:
- Each thread walks its particle range in chunks.
- For each chunk, it runs every record through `visit_kernel_call` with a
  small C++11 functor, `ApplyToChunk { dt, dE, begin, end }`, whose templated
  `operator()` calls `apply_to_chunk(args, …)`.
- The executor contains no kernel-specific code and needs no scratch.

**Chunk size.** The default is 4096 particles, overridable with
`BLOND_DEFERRED_CHUNK_SIZE`, as in the prototype.

## 7. CUDA executor (`blond/core/backends/cuda/kernels.cu`, same cubin)

### 7.1 Per-particle device functions

Each deferrable kernel has one overload:

```cpp
__device__ void apply_to_particle(const DriftSimpleArgs &args,
                                  real_t &dt, real_t &dE);
```

This is the loop body of the eager kernel, moved out of it. The eager kernels
call the same function, so each formula exists once within CUDA. Only these
six kernels change.

### 7.2 Fused kernel

`execute_kernel_call_batch(const KernelCallBatch batch, n_bytes, dt, dE, n)`
is a grid-stride loop. For each particle it:
1. loads `dt` and `dE` into registers;
2. runs every record through `visit_kernel_call` with an `ApplyToParticle`
   functor;
3. writes `dt` and `dE` back once.

The `switch` never diverges, because all threads of a warp see the same
kernel id.

### 7.3 Registers and transfers

- **Registers.** The cubin is built with `-maxrregcount 32`. The fused kernel
  sets its own limit via `__launch_bounds__` / `__maxnreg__`. Spills are
  checked with `-Xptxas -v`.
- **Transfers.** Nothing on the execute path synchronises with the host or
  calls `.get()`.

## 8. Testing

TDD applies throughout, with the RED stage shown for each test.

| # | Test | Location |
|---|---|---|
| 1 | Header is current | `tests/unittests/core/backends/deferred/test_kernel_call_records.py` |
| 2 | Record sizes match the dtypes at library load | same file |
| 3 | Deferred matches eager | `tests/unittests/core/backends/deferred/test_deferred_specials.py` |
| 4 | Flush semantics | same file |
| 5 | Coordinate storage flushes on every read, statistic, copy and replacement, and every assignment path keeps the flushing type; sizes and `kernel_call_dt`/`kernel_call_dE` do not flush; `array_local_without_flush` is used nowhere else | `tests/unittests/core/beam/test_kernel_call_accessors.py`, `tests/unittests/core/beam/test_flushing_distributed_array.py` |
| 6 | Main loop: deferred vs eager | `tests/unittests/core/simulation/execution_models/test_single_beam.py` |

**Test 3** runs 3 turns of every kernel on both deferred backends and
compares coordinates with eager within `rtol`. It covers:
- n ∈ {1, 7, 1000, 100003};
- a range of chunk sizes;
- a forced split of a CUDA batch larger than the capacity;
- more than 32 harmonics;
- `drift_exact` with 0, 1, 4 and 5 coefficients (unrolled and generic), and
  with 9 (fallback).

`cuda_deferred` runs behind the `cupy` marker.

**Test 4** covers:
- a non-deferred call flushes;
- a call on another beam flushes;
- scalars are captured at enqueue;
- arrays are kept alive;
- queues are per thread;
- switching specials flushes;
- every public `Specials` method is either deferred or wrapped.

**Test 6** requires deferred to match eager at `rtol=1e-12`, checks that a
callback sees the flushed beam although the main loop does not flush, and
that a simulation run in a worker thread leaves nothing queued for the
main thread to miss.

All test classes inherit from `BLonDTestCase`.

## 9. Measurements and verification

**Spike: should the histogram be fused?** Both variants ran on the prototype,
which fuses the histogram, and its throwaway variant, which runs the
histogram flush-then-eager. Setup:
- i5-11500, 12 threads;
- EX_23 PSB with two wakefields, 1e7 macroparticles, 10 000 bins;
- 30 turns, 3 interleaved runs.

All variants produced identical checksums.

| Setup | eager | deferred, histogram fused | deferred, histogram eager |
|---|---|---|---|
| EX_23 as-is: 3 histograms/turn | 48–50 ms | 29–31 ms | 35–40 ms |
| `track_profile=False` on both wakefields: 1 histogram/turn | 43–48 ms | 18.0–19.3 ms | 18.5–20.8 ms |

With one profile per turn, fusing the histogram is within noise. The
remaining gain comes from fusing the kicks and drifts, about 2.4×.

**Side finding (not part of this MR).** Every `WakeField` defaults to
`track_profile = True` (`impedances/base.py:433`). In EX_23 the
`StaticProfile` element and both wakefields therefore recompute the same
histogram, 3 times per turn. That costs 10 % in eager. In deferred it costs
far more (29 → 18 ms), because every extra histogram flushes the batch. This
needs its own issue: is the re-tracking intended when the profile is already
an element of the ring?

**Verification of the MR:**
- **Tests:** `BLOND_FORCE_TEST_ALL_BACKENDS=True python -m pytest tests/unittests/ -v`.
  On a CPU-only machine, CUDA failures are expected noise.
- **Pre-commit and docs:** `pre-commit run --all-files` and
  `cd docs && bash create_docs.sh`, which builds with `-W`.
- **Benchmark:** the same EX_23 setup, deferred vs eager.
  - CPU: match the spike's deferred numbers. Also record single-core cycles.
  - T400: record both variants.

**MR benchmark table.** Produced by
`dev_tools/performance_blond3/backends/deferred_psb.py`: EX_23 PSB,
1e7 macroparticles, 10 000 bins, 30 timed turns after 3 warm-up turns,
`--one-histogram` (`track_profile=False` on both wakefields, one
histogram/turn). 3 interleaved runs per mode; checksums of `dE` were
identical within each device across eager/deferred.

CPU (i5-11500, 12 threads):

| mode | run 1 | run 2 | run 3 |
|---|---|---|---|
| cpp | 41.4 ms | 45.2 ms | 38.0 ms |
| cpp_deferred | 27.0 ms | 20.7 ms | 17.5 ms |

T400 (4 GB):

| mode | run 1 | run 2 | run 3 |
|---|---|---|---|
| cuda | 33.7 ms | 33.1 ms | 33.8 ms |
| cuda_deferred | 34.9 ms | 31.7 ms | 31.4 ms |

Flushes/turn were 1.33 in every configuration. That counter only saw
explicit `specials.flush()` calls, not the flush-then-call inside
`histogram`; counting executed batches instead (`batches/turn`, added in
Task 13) gives exactly 1.00 per turn for the deferred modes: one fused
kick+drift batch, run by the `StaticProfile` histogram.

After Task 13 (self-flushing Beam storage, main-loop flushes removed),
same setup, 3 interleaved runs, 12-thread desktop: cpp 43.2 / 39.1 /
40.6 ms, cpp_deferred 26.0 / 23.0 / 19.0 ms (before: cpp 39.7 / 62.9 /
44.1, cpp_deferred 27.0 / 26.9 / 22.4); batches/turn 1.00 before and
after, `flush()` calls/turn 1.33 → 1.60 (extra calls on an empty queue),
identical checksums. One T400 run each: cuda 36.2 ms, cuda_deferred
34.2 ms. On the T400, cuda_deferred is within noise
of cuda -- the fused interpreter kernel amortises launch overhead less
than on the CPU, where per-call dispatch cost dominates; this matches the
prediction in the risk table that GPU gains would be smaller.

Single-core cycle counts (`OMP_NUM_THREADS` does not reach the compiled
backend -- importing `blond` overrides it before user code can set it, so
threads were pinned through BLonD's own `cpp_single_core`/`--single-core`
mechanism, i.e. the non-OMP compiled library, not the environment
variable), 1e6 macroparticles, 10 turns, `--one-histogram`,
`perf stat -e cycles`:

| mode | ms/turn | process cycles (includes ~3 s import/compile overhead) |
|---|---|---|
| cpp | 14.25 | 13.14e9 |
| cpp_deferred | 8.47 | 12.42e9 |

`dE` checksums matched exactly between `cpp` and `cpp_deferred` in both
the multi-threaded and single-core runs.

## 10. Risks and open points

| Risk | Consequence | Mitigation |
|---|---|---|
| Direct `_dt`/`_dE` access outside `core/beam/` | Stale reads | Route through the accessors; a grep test fails on any new hit |
| Extra flushes from avoidable non-deferred calls, such as redundant histograms | Batches break up and the gain shrinks (§9) | Handle as separate issues; count flushes per turn in the benchmark |
| Register pressure in the fused CUDA kernel | Spills eat the gain | Per-kernel register limit; measure |
| 4064 B kernel-parameter capacity | Multi-harmonic-heavy batches get split | Correct, only less fused; measure how often it happens |
| Refactoring the eager cpp kernels into `particle_kernels.h` | May regress their AVX-512 code generation | Benchmark the eager kernels before and after |
