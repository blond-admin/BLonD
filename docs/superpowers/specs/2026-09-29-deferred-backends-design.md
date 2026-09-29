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
wraps every `Specials` method that is not in `KERNEL_CALL_RECORDS` as
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

`KERNEL_CALL_RECORDS` declares three things for each deferrable kernel:
- **name:** the `Specials` method name;
- **fields:** named after the `Specials` arguments where the value comes
  straight from a kwarg;
- **optional `prepare_on_enqueue`:** a Python function that turns the kwargs
  into field values. `kick_interpolated` is the only kernel that needs it.

The beam's `dt`/`dE` are **not** fields. They are bound once per queue and
passed once per batch (§5.1).

There are five field kinds:

| Kind | C type in the record | numpy dtype | When queued |
|---|---|---|---|
| `real` | `real_t` | `backend.float` | Captured by value |
| `int32` | `int32_t` | `np.int32` | Captured by value; a range check raises on overflow |
| `index` | `index_t` | `INDEX_DTYPE` (`backend.py`) | Captured by value. `index_abi.cpp` already guards the width |
| `input_array` | `const real_t *` + `index_t <name>_length` | `np.uintp` + `INDEX_DTYPE` | Pointer is a host address on cpp and a device address (`.data.ptr`) on cuda. The array is kept alive until the flush. It is snapshotted into a same-device copy unless the queue created it itself (`prepare_on_enqueue`) |
| `inline_real_array(max_length)` | `real_t <name>[max_length]` + an `int32_t` count | `(backend.float, max_length)` + `np.int32` | Copied from a **host** array into the record. Unused slots are zero. Several fields can share one count field |

**Rules for every field:**
- **Order and padding.** Fields keep their declared order. The generator
  pads each record to a multiple of 8 bytes, and C and numpy get the same
  explicit `offsets`. No compiler padding is assumed.
- **Precision.** `real_t` in the record must match `backend.float`. The
  load-time ABI check (§4.4) catches a library compiled at the other
  precision.
- **Arrays** must already have the backend float dtype and be C-contiguous.
  This is checked by an `assert` in the queue, following the backend-wrapper
  convention. Nothing is coerced.
- **`inline_real_array` rejects GPU arrays**, because reading one would force
  a sync on every call. `RFParamsBatch` makes the same assumption today.
- **More values than `max_length`.** The call is queued as several
  consecutive records. `acceleration_kick` is applied only in the last one,
  as `CudaSpecials.kick_multi_harmonic` does today. At least one record is
  always queued, so `n_rf == 0` still applies the kick.

**Fields per kernel:**

| Kernel | Fields |
|---|---|
| `kick_single_harmonic` | `voltage`, `omega_rf`, `phi_rf`, `charge`, `acceleration_kick`: all `real` |
| `kick_multi_harmonic` | `voltage`, `omega_rf`, `phi_rf`: `inline_real_array(32)` sharing the count `n_rf`<br>`charge`, `acceleration_kick`: `real` |
| `drift_simple` | `T`, `eta_0`, `beta`, `energy`: all `real` |
| `drift_like_line_segment` | `T`, `eta_0`, `beta`, `energy`: all `real` |
| `drift_exact` | `T`, `alpha_0`, `beta`, `energy`: `real`<br>`higher_alpha`: `inline_real_array(8)` with count `n_alpha` |
| `kick_interpolated` (dense) | `voltage_kick_table`: `input_array`, built by the queue<br>`first_bin_center`, `inverse_bin_width`: `real` |

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

**`kick_interpolated` needs no per-batch setup.** Its `prepare_on_enqueue`
builds the voltage-kick table (slope and offset per bin, with `charge` and
`acceleration_kick` folded in) when the call is queued. The table is built by
the backend's own table builder, which the eager path uses as well:

| Backend | Table builder |
|---|---|
| cuda | The existing `lik_only_gm_copy` kernel |
| cpp | A new `linear_interp_kick_table` entry point, factored out of `linear_interp_kick.cpp` |

The table touches no particles, so building it before the flush is correct.
The table's layout differs between backends (CUDA interleaves slope and
offset), but to the record it is just an opaque pointer. This also removes
the snapshot copy of `voltage`: the table is a fresh array owned by the
queue.

The same module:
- builds the numpy structured dtypes at run time, with `backend.float` as the
  float width;
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
1. Add an entry to `KERNEL_CALL_RECORDS`, and a `prepare_on_enqueue` only if
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
- **Statistics that bypass `Specials`** are:
  - `DistributedArray.min/max/mean/std/gather/scatter`, which run numpy
    directly on `array_local`;
  - the Beam statistics `dt_min`/`dE_max`/… in `core/beam/beams.py`, which
    read `self._dt` directly.

  Every public Beam method or property that reads particle data therefore
  flushes first, through one helper, `_flush_kernel_calls()`.
  `DistributedArray` stays unaware of deferral; its Beam callers do the
  flushing.
- **GPU with MPI.** CUDA-aware MPI on CuPy arrays already needs the stream
  synchronised before a collective. Deferral changes nothing here, because
  its launches go on the same stream.

### 5.2 `make_deferred_specials(eager_specials_class, execute_batch)`

The factory returns a subclass of the eager specials class:
- **Generated queuing methods.** One per `KERNEL_CALL_RECORDS` entry, mapping
  kwargs to fields by name, or through `prepare_on_enqueue`. No method is
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
- Capacity is 4 KB, which every supported GPU accepts.
- A larger batch is split into several launches. The result is still
  correct, only less fused.
- A multi-harmonic record is about 800 B (`inline_real_array(32)` × 3).

### 5.4 Flush points

**Existing points (as in the prototype):**
- a non-deferrable call;
- a call on a different beam;
- a change of backend or specials;
- `flush_before_readout` in the execution models, before any active
  observable or callback;
- the early-return path and the end of the main loop.

**New point: Beam data accessors.** `dt`, `dE`, `read_partial_*`,
`write_partial_*`, and every public Beam method that reads particle data
(§5.1a).

**Kernel-argument accessors, which do not flush.** The `Specials` call sites
in `rf_station.py`, `drifts.py`, `impedances/base.py` and `barrier_bucket.py`
currently fetch their arrays through `read_partial_dt()` /
`write_partial_dE()`. If those flushed, every deferred call would flush just
before being queued, and nothing would fuse. `BeamBaseClass` therefore gets
two new properties:

```python
@property
def kernel_call_dt(self) -> NumpyArray | CupyArray:
    """Local `dt` to pass to a `Specials` kernel call; does not flush."""
```

`kernel_call_dE` is the same for `dE`. They are properties because they have
no side effect; the flushing `read_/write_partial_*` stay methods.

**Accessor rules:**
- `backend.specials.<kernel>(...)` call sites use `kernel_call_dt` /
  `kernel_call_dE`.
- Any other Python code that touches particle data uses the flushing
  accessors.
- Direct `_dt` / `_dE` access is allowed only inside `blond/core/beam/`, and
  only after `_flush_kernel_calls()` or in the two kernel-argument
  properties.

**Direct `_dt` / `_dE` users outside `blond/core/beam/`** (excluding
`legacy/` and `experimental/`) are:
- `simulation.py`
- `single_beam.py`
- `observables*.py`
- `profiles*.py`
- `synchrotron_radiation/base.py`
- `beam_preparation/helpers.py`
- `muon_collider/beam_preparation.py`
- two examples

The implementation routes each one through the right accessor. A test then
greps `blond/` for `._dt` / `._dE` outside `core/beam/` and fails on any
hit.

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
| 5 | Data accessors flush; `kernel_call_dt`/`kernel_call_dE` do not; no direct `_dt`/`_dE` outside `core/beam/` | `tests/unittests/core/beam/` |
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

**Test 6** requires deferred to match eager at `rtol=1e-12`, and checks that a
callback sees the flushed beam.

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

## 10. Risks and open points

| Risk | Consequence | Mitigation |
|---|---|---|
| Direct `_dt`/`_dE` access outside `core/beam/` | Stale reads | Route through the accessors; a grep test fails on any new hit |
| Extra flushes from avoidable non-deferred calls, such as redundant histograms | Batches break up and the gain shrinks (§9) | Handle as separate issues; count flushes per turn in the benchmark |
| Register pressure in the fused CUDA kernel | Spills eat the gain | Per-kernel register limit; measure |
| 4 KB kernel-parameter capacity | Multi-harmonic-heavy batches get split | Correct, only less fused; measure how often it happens |
| Refactoring the eager cpp kernels into `particle_kernels.h` | May regress their AVX-512 code generation | Benchmark the eager kernels before and after |
