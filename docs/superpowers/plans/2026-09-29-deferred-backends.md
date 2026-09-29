# Deferred cpp and cuda kernel execution: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal.** Add `cpp_deferred` and `cuda_deferred` specials. They queue per-particle kernel calls and run each batch as one fused pass over the particles.

**Architecture.**
- A Python definition (`kernel_call_records.py`) is the single source of truth for the record layout. It generates the C++/CUDA header, including the only `switch` over kernel ids, and the numpy dtypes.
- A shared Python queue (`kernel_call_queue.py`) packs records and flushes them through one backend-specific `execute_batch`.
- Each backend has one executor loop and one overload per kernel, written against the generated structs:
  - cpp: `apply_to_chunk`;
  - cuda: `apply_to_particle`.
- Every other `Specials` method is flush-then-call.

**Tech stack.** Python ≥3.10, numpy, ctypes, CuPy `RawModule` (precompiled cubin), C++11 with OpenMP, CUDA via nvcc.

**Spec.** `docs/superpowers/specs/2026-09-29-deferred-backends-design.md`, committed in `a50b13dbb`. Read it before starting a task.

## Global Constraints

**Branch and commits**
- Work on `blonder_feature/deffered-backends`. Implement on top of the current HEAD. Do not merge or cherry-pick from `blonder_coding_experiments/cpp-deferred-queue`.
- Run `pre-commit run --files <changed>` before **every** commit and commit only when it is all green. If hooks auto-fix files, re-stage them and rerun.
- Write commit messages in the past tense ("Added …"). The body explains *why*. End with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- TDD with a visible RED: run each new test and show it failing before implementing.

**Tests**
- Test classes inherit `BLonDTestCase` (`from blond.testing.backend_testing import BLonDTestCase`) and use `self.assert*`, never a bare `assert`.
- Tests that change `backend` are marked `@pytest.mark.backend_mutation`. They restore `Numpy64Bit` and `"python"` specials in `tearDown`.
- CUDA tests are gated with `@skip_if_no_cupy` and marked `@pytest.mark.cupy`. This machine has a T400, so run them.
- Compare results with `rtol`/`atol`, never bit-exact.

**Code**
- Line length is 79. Python docstrings are NumPy style.
- Every new file under `blond/` starts with the copyright header from `dev_tools/copyright_notice.txt`: `#` comments in `.py`, `//` comments in `.h`, `.cpp` and `.cu`.
- Use `assert` for dtype/contiguity validation in backend wrappers and the queue. Never turn it into `raise`.
- Use `backend.<fn>`, `copy_to_cpu` and `is_cupy_array` (`blond.generals.cupy_.no_cupy_import`). Never import `cupy` at module top level in code that must load CPU-only.
- Do not put guards or allocations inside per-particle loops. Do not transfer between host and device in the execute path.
- The C++ standard stays C++11 (`-std=c++11`): no generic lambdas and no `if constexpr`.
- SI units: seconds, eV, radians.

**Naming**
- Specials names: `"cpp_deferred"`, `"cuda_deferred"`.
- Vocabulary, used as-is in names: *kernel call*, *kernel call record*, *batch*, *kernel call queue*.
- Beam accessors: `kernel_call_dt` / `kernel_call_dE` (properties, no flush) and `_flush_kernel_calls()` (helper).
- Record field names are the `Specials` argument names (`T`, `eta_0`, `alpha_0`, `acceleration_kick`, …).

**Capacities and defaults**
- Inline capacities: `kick_multi_harmonic` 32 harmonics per record; `drift_exact` 8 higher-order coefficients (more falls back to eager).
- CUDA batch capacity is 4096 bytes, passed by value.
- Default chunk size is 4096 particles, overridable with `BLOND_DEFERRED_CHUNK_SIZE`.

**Environment**
- The venv is `.venv` at the repo root: `/home/slauber/PycharmProjects/deleteme/BLonD_uv/.venv/bin/python`.

## Review Focus

These are the input classes most likely to bite a user that the per-task tests would not naturally hit. Each has a pinned test in the task named in brackets.

1. **The caller mutates or reuses an input array after queuing.** An induced-voltage buffer overwritten in place before the flush must not change the kick already queued. [Task 5, `test_voltage_mutated_after_queue`]
2. **A kernel called on a view or slice of the beam, or on another array.** It must flush first and still match eager. It must never apply the queued batch to the wrong arrays. [Task 5, `test_kernel_on_a_slice_of_the_beam`]
3. **`execute_batch` raises mid-flush.** The queue must be empty afterwards, so a later flush never re-applies the same kicks. [Task 5, `test_failed_flush_clears_queue`]
4. **A beam with zero macroparticles.** Queuing and flushing must be a no-op, with no crash. [Task 4 `test_zero_macroparticles` and Task 10 `test_zero_macroparticles`]
5. **The beam is copied or its arrays replaced while calls are pending.** `deepcopy` in muon-collider preparation and `setup_beam` again must see up-to-date coordinates. [Task 6, `test_copy_sees_pending_kicks`]

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `blond/core/backends/deferred/__init__.py` | new | Package marker with a docstring |
| `blond/core/backends/deferred/kernel_call_records.py` | new | Record definitions, `prepare_on_enqueue` hooks, dtypes, header generator, header digest |
| `blond/core/backends/deferred/kernel_call_records.h` | new, generated | Structs, `KernelId`, `visit_kernel_call`, `next_record`, sizes |
| `blond/core/backends/deferred/kernel_call_queue.py` | new | `KernelCallQueue`, `make_deferred_specials`, `deferred_chunk_size` |
| `blond/core/backends/cpp/particle_kernels.h` | new | `apply_to_chunk` overloads, `thread_range`, `run_on_all_particles`, `linear_interp_kick_table` declaration |
| `blond/core/backends/cpp/deferred.cpp` | new | `execute_kernel_call_batch`, `kernel_call_args_size` |
| `blond/core/backends/cpp/kick.cpp`, `drift.cpp`, `drift_exact.cpp`, `linear_interp_kick.cpp` | modify | Eager kernels become thin wrappers around `particle_kernels.h` |
| `blond/core/backends/cpp/compile.py`, `compiled_dir_handler.py` | modify | Add `deferred.cpp`, `-I`, header digest |
| `blond/core/backends/cpp/callables.py` | modify | `deferred=` flag, `execute_kernel_call_batch`, `_build_voltage_kick_table`, ABI check |
| `blond/core/backends/cuda/kernels.cu` | modify | `apply_to_particle` overloads, `build_voltage_kick_table`, fused kernel, size table |
| `blond/core/backends/cuda/compile.py`, `compiled_dir_handler.py` | modify | `-I`, header digest |
| `blond/core/backends/cuda/callables.py` | modify | `_build_voltage_kick_table`, `CudaDeferredSpecials`, host `higher_alpha` |
| `blond/core/backends/backend.py` | modify | `Specials.flush`, flush on change, `cpp_deferred` / `cuda_deferred` |
| `blond/core/beam/base.py`, `beams.py` | modify | `kernel_call_dt/dE`, `_flush_kernel_calls`, flushing readers |
| Kernel call sites | modify | Use `kernel_call_dt/dE`: `physics/rf_station.py`, `drifts.py`, `impedances/base.py`, `barrier_bucket.py`, `experimental/physics/kick_pooling.py` |
| Direct `_dt`/`_dE` users | modify | Route through the accessors: `physics/profiles.py`, `profiles_sparse.py`, `handle_results/observables*.py`, `core/simulation/simulation.py`, `execution_models/single_beam.py`, `beam_preparation/helpers.py`, `specifics/muon_collider/beam_preparation.py`, `examples/scripts/EX_20…`, `EX_28…` |
| `blond/core/simulation/execution_models/base.py`, `single_beam.py`, `conterrotating_beams.py` | modify | Flush points |
| `dev_tools/run_clang_tidy.py` | modify | `-I` for the deferred header |
| `dev_tools/performance_blond3/backends/deferred_psb.py` | new | Benchmark |
| `tests/unittests/core/backends/deferred/{__init__,test_kernel_call_records,test_cpp_executor,test_deferred_specials}.py` | new | Tests |
| `tests/unittests/core/beam/test_kernel_call_accessors.py` | new | Tests |
| `tests/unittests/core/simulation/execution_models/test_deferred_mainloop.py` | new | Tests |

---

### Task 1: `Specials.flush()` and flushing on backend/specials changes

**Files:**
- Modify: `blond/core/backends/backend.py`. The `Specials` class starts at line 77. Edit `change_backend` (around line 755), `NumpyBackend.set_specials` (around line 1115) and `CupyBackend.set_specials` (around line 1268).
- Test: `tests/unittests/core/backends/deferred/__init__.py` (empty) and `tests/unittests/core/backends/deferred/test_specials_flush.py`

**Interfaces:**
- Produces:
  - `Specials.flush() -> None`, a concrete static method that does nothing;
  - `BackendBaseClass.change_backend` and every `set_specials` call `self.specials.flush()` before replacing `self.specials`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unittests/core/backends/deferred/test_specials_flush.py
from unittest import mock

import pytest

from blond.core.backends.backend import Numpy64Bit, Specials, backend
from blond.testing.backend_testing import BLonDTestCase


class TestSpecialsFlush(BLonDTestCase):
    def tearDown(self) -> None:
        backend.change_backend(Numpy64Bit)
        backend.set_specials("python")

    def test_every_specials_has_flush(self) -> None:
        backend.set_specials("python")
        self.assertIsNone(Specials.flush())
        self.assertIsNone(backend.specials.flush())

    @pytest.mark.backend_mutation
    def test_set_specials_flushes_previous(self) -> None:
        backend.set_specials("python")
        with mock.patch.object(
            type(backend.specials), "flush", create=True
        ) as flush:
            backend.set_specials("cpp")
        flush.assert_called_once_with()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/unittests/core/backends/deferred/test_specials_flush.py -v`

Expected: `test_every_specials_has_flush` FAILS with `AttributeError: type object 'Specials' has no attribute 'flush'`. `test_set_specials_flushes_previous` FAILS because `flush` was not called.

- [ ] **Step 3: Implement**

In `class Specials(ABC)`, directly after the class docstring, add:

```python
    @staticmethod
    def flush() -> None:
        """
        Run all kernel calls that are still queued.

        Eager specials queue nothing, so this is a no-op; deferred
        specials (``cpp_deferred``, ``cuda_deferred``) override it.
        Deliberately not abstract: callers flush unconditionally,
        whatever specials are active.
        """
```

In `change_backend`, immediately before `_new_backend = new_backend()`, add:

```python
        # Queued kernel calls belong to the old specials; run them first.
        self.specials.flush()
```

In both `set_specials` implementations, add this as the first statement of the body, after the docstring:

```python
        # Queued kernel calls belong to the old specials; run them first.
        if getattr(self, "specials", None) is not None:
            self.specials.flush()
```

The `getattr` guard is needed because `CupyBackend.__init__` sets `self.specials` itself, and `set_specials` can run before any specials exist.

- [ ] **Step 3b: Check whether `mock.patch.object` with `create=True` works on the python specials**

Run: `.venv/bin/python -m pytest tests/unittests/core/backends/deferred/test_specials_flush.py -v`

Expected: PASS. If the patch target is an *instance* (`backend.specials` is `PythonSpecials()`), `type(...)` still resolves to the class, so the test works as written.

- [ ] **Step 4: Run the backend suite for regressions**

Run: `.venv/bin/python -m pytest tests/unittests/core/backends/test_backend.py -q -x`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
pre-commit run --files blond/core/backends/backend.py tests/unittests/core/backends/deferred/__init__.py tests/unittests/core/backends/deferred/test_specials_flush.py
git add blond/core/backends/backend.py tests/unittests/core/backends/deferred/
git commit -m "Added Specials.flush and flush the old specials on every change" -m "Deferred specials will hold queued kernel calls; replacing the specials or the backend must run them first so no kick is lost. Eager specials get a no-op so callers never need to know which kind is active." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Record definitions, dtypes, generated header, and build wiring

**Files:**
- Create: `blond/core/backends/deferred/__init__.py`
- Create: `blond/core/backends/deferred/kernel_call_records.py`
- Create (generated): `blond/core/backends/deferred/kernel_call_records.h`
- Modify: `blond/core/backends/cpp/compiled_dir_handler.py` (the `hash_build_target(...)` call near line 124: add to `extra`)
- Modify: `blond/core/backends/cuda/compiled_dir_handler.py` (the `hash_build_target(...)` call near line 73: add `extra`)
- Modify: `blond/core/backends/cpp/compile.py` (compile command: add `-I<deferred dir>`)
- Modify: `blond/core/backends/cuda/compile.py` (the nvcc command near line 156: add `-I<deferred dir>`)
- Modify: `dev_tools/run_clang_tidy.py` (`COMPILER_FLAGS` at line 39)
- Test: `tests/unittests/core/backends/deferred/test_kernel_call_records.py`

**Interfaces:**
- Produces, in `kernel_call_records.py`:
  - `RecordField(name: str, kind: str, max_length: int = 0, count_field: str = "")`
  - `KernelCallRecord(name: str, fields: tuple[RecordField, ...], prepare_on_enqueue: Callable | None = None)`, with properties `.kernel_id_name -> str` (CamelCase) and `.kernel_id -> int`
  - `KERNEL_CALL_RECORDS: tuple[KernelCallRecord, ...]`
  - `RECORDS_BY_NAME: dict[str, KernelCallRecord]`
  - `HEADER_DTYPE: np.dtype`, with fields `kernel_id` (u4) and `record_size_bytes` (u4)
  - `args_dtype(record) -> np.dtype`, `record_dtype(record) -> np.dtype`
  - `generate_header() -> str`, `HEADER_PATH: str`, `header_digest() -> str`
  - `KERNEL_CALL_BATCH_CAPACITY_BYTES = 4096`
  - `MAX_RF_HARMONICS_PER_RECORD = 32`, `MAX_HIGHER_ALPHA = 8`
- Produces, in the header:
  - `enum class KernelId : std::uint32_t`
  - `struct KernelCallHeader`
  - `<Kernel>Args` structs
  - `KERNEL_COUNT`, `KERNEL_CALL_BATCH_CAPACITY_BYTES`, `KERNEL_CALL_ARGS_SIZES[]`, `KERNEL_CALL_ARGS_SIZES_INITIALIZER`
  - `record_args<Args>(record)`, `next_record(record)`, `visit_kernel_call(record, visitor)`
  - `BLOND_HOST_DEVICE`
- The `prepare_on_enqueue` hooks are defined in this task **as data only**. Tasks 5 and 10 exercise their behaviour. Every hook has the signature `hook(arguments: dict[str, Any], eager_specials) -> list[dict[str, Any]] | None`, where `None` means "run eagerly".

- [ ] **Step 1: Write the failing tests**

```python
# tests/unittests/core/backends/deferred/test_kernel_call_records.py
import inspect

import numpy as np

from blond.core.backends.backend import Specials
from blond.core.backends.deferred import kernel_call_records as records
from blond.testing.backend_testing import BLonDTestCase


class TestKernelCallRecords(BLonDTestCase):
    def test_header_is_current(self) -> None:
        with open(records.HEADER_PATH) as file:
            on_disk = file.read()
        self.assertEqual(
            on_disk,
            records.generate_header(),
            "kernel_call_records.h is stale; run `python -m "
            "blond.core.backends.deferred.kernel_call_records`",
        )

    def test_record_names_are_specials_methods(self) -> None:
        for record in records.KERNEL_CALL_RECORDS:
            self.assertTrue(hasattr(Specials, record.name), record.name)

    def test_direct_fields_are_specials_arguments(self) -> None:
        # Fields without a prepare hook are filled from the kwargs by name.
        for record in records.KERNEL_CALL_RECORDS:
            if record.prepare_on_enqueue is not None:
                continue
            parameters = inspect.signature(
                getattr(Specials, record.name)
            ).parameters
            for field in record.fields:
                self.assertIn(field.name, parameters, record.name)

    def test_dtypes_are_8_byte_padded(self) -> None:
        for record in records.KERNEL_CALL_RECORDS:
            self.assertEqual(records.record_dtype(record).itemsize % 8, 0)
            self.assertEqual(
                records.record_dtype(record).itemsize,
                records.HEADER_DTYPE.itemsize
                + records.args_dtype(record).itemsize,
            )

    def test_kick_multi_harmonic_layout(self) -> None:
        dtype = records.args_dtype(records.RECORDS_BY_NAME["kick_multi_harmonic"])
        self.assertEqual(dtype.fields["n_rf"][1], 0)
        self.assertEqual(dtype.fields["voltage"][1], 8)  # 4 bytes of padding
        self.assertEqual(dtype.fields["voltage"][0].shape, (32,))

    def test_kernel_ids_are_positions(self) -> None:
        for position, record in enumerate(records.KERNEL_CALL_RECORDS):
            self.assertEqual(record.kernel_id, position)

    def test_digest_changes_with_header(self) -> None:
        self.assertEqual(len(records.header_digest()), 64)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unittests/core/backends/deferred/test_kernel_call_records.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'blond.core.backends.deferred'`.

- [ ] **Step 3: Implement `kernel_call_records.py`**

```python
# <copyright header, 7 lines, from dev_tools/copyright_notice.txt>

"""
Layout of deferred kernel call records -- the single source of truth.

A deferred specials (``cpp_deferred``, ``cuda_deferred``) turns every call
of a deferrable kernel into a *kernel call record*: a `HEADER_DTYPE`
header followed by the kernel's ``<Kernel>Args`` struct. The records of
one flush form a *batch*, which the backend's executor applies to the
particles in a single fused pass.

This module defines those records. From it, ``kernel_call_records.h``
(C++/CUDA) is generated and the matching numpy dtypes are built, so the
layout is written down exactly once.

Adding a deferrable kernel
--------------------------
1. Add a `KernelCallRecord` to `KERNEL_CALL_RECORDS`, with a
   ``prepare_on_enqueue`` hook only if its arguments need transforming.
2. Regenerate the header:
   ``python -m blond.core.backends.deferred.kernel_call_records``.
3. Add one overload per backend: ``apply_to_chunk(const XArgs&, ...)`` in
   ``cpp/particle_kernels.h`` and ``apply_to_particle(const XArgs&, ...)``
   in ``cuda/kernels.cu``. A missing overload does not compile.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

from blond.core.backends.backend import INDEX_DTYPE
from blond.generals.cupy_.no_cupy_import import is_cupy_array

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

HEADER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kernel_call_records.h"
)
KERNEL_CALL_BATCH_CAPACITY_BYTES = 4096
MAX_RF_HARMONICS_PER_RECORD = 32
MAX_HIGHER_ALPHA = 8

# Record fields are laid out for 64-bit reals and indices; the generated
# header static_asserts the same on the C++/CUDA side.
_REAL = np.dtype(np.float64)
_INDEX = np.dtype(INDEX_DTYPE)
assert _INDEX.itemsize == 8, "records assume a 64-bit index_t"

_C_TYPES = {"real": "real_t", "int32": "std::int32_t", "index": "index_t"}
_DTYPES = {"real": _REAL, "int32": np.dtype(np.int32), "index": _INDEX}


@dataclass(frozen=True)
class RecordField:
    """
    One field of a kernel's ``Args`` struct.

    Parameters
    ----------
    name
        Field name; the `Specials` argument name when filled from kwargs.
    kind
        ``"real"``, ``"int32"``, ``"index"``, ``"input_array"`` (a
        ``const real_t *`` plus ``index_t <name>_length``) or
        ``"inline_real_array"`` (``real_t <name>[max_length]``).
    max_length
        Capacity of an ``inline_real_array``.
    count_field
        Name of the ``int32`` field holding how many values of an
        ``inline_real_array`` are used.
    """

    name: str
    kind: str
    max_length: int = 0
    count_field: str = ""


@dataclass(frozen=True)
class KernelCallRecord:
    """
    Definition of one deferrable kernel.

    Parameters
    ----------
    name
        The `Specials` method name.
    fields
        The ``Args`` struct fields, in layout order.
    prepare_on_enqueue
        Optional ``hook(arguments, eager_specials)`` returning the field
        values of one or more records, or None to run the call eagerly.
        Without a hook, each field is taken from the kwarg of its name.
    """

    name: str
    fields: tuple[RecordField, ...]
    prepare_on_enqueue: Callable | None = None

    @property
    def kernel_id_name(self) -> str:
        """CamelCase name used for ``KernelId`` and the ``Args`` struct."""
        return "".join(part.title() for part in self.name.split("_"))

    @property
    def kernel_id(self) -> int:
        """Position in `KERNEL_CALL_RECORDS`, the value of ``KernelId``."""
        return KERNEL_CALL_RECORDS.index(self)


def _real(name: str) -> RecordField:
    return RecordField(name, "real")


def _assert_host(array: Any, name: str) -> None:
    # A device array would force a device-to-host sync on every call.
    assert not is_cupy_array(array), f"`{name}` must be a host array"


def _split_harmonics(
    arguments: dict[str, Any], eager_specials: Any
) -> list[dict[str, Any]]:
    """
    Inline the RF parameters, 32 harmonics per record.

    ``acceleration_kick`` goes into the last record only, and one record
    is always queued, so ``n_rf == 0`` still applies it -- as
    ``CudaSpecials.kick_multi_harmonic`` splits its launches.
    """
    n_rf = int(arguments["n_rf"])
    for name in ("voltage", "omega_rf", "phi_rf"):
        _assert_host(arguments[name], name)
    values = []
    for first in range(0, max(n_rf, 1), MAX_RF_HARMONICS_PER_RECORD):
        last = min(first + MAX_RF_HARMONICS_PER_RECORD, n_rf)
        values.append(
            {
                "n_rf": last - first,
                "voltage": arguments["voltage"][first:last],
                "omega_rf": arguments["omega_rf"][first:last],
                "phi_rf": arguments["phi_rf"][first:last],
                "charge": arguments["charge"],
                "acceleration_kick": (
                    arguments["acceleration_kick"] if last == n_rf else 0.0
                ),
            }
        )
    return values


def _inline_higher_alpha(
    arguments: dict[str, Any], eager_specials: Any
) -> list[dict[str, Any]] | None:
    """Inline the higher-order alphas; more than 8 runs eagerly."""
    higher_alpha = arguments["higher_alpha"]
    _assert_host(higher_alpha, "higher_alpha")
    if len(higher_alpha) > MAX_HIGHER_ALPHA:
        return None
    return [
        {
            "T": arguments["T"],
            "alpha_0": arguments["alpha_0"],
            "beta": arguments["beta"],
            "energy": arguments["energy"],
            "n_alpha": len(higher_alpha),
            "higher_alpha": higher_alpha,
        }
    ]


def _build_voltage_kick_table(
    arguments: dict[str, Any], eager_specials: Any
) -> list[dict[str, Any]] | None:
    """
    Build the table of the dense interpolated kick when queuing.

    The table touches no particles, so building it before the flush is
    exact. Its layout, shared by both backends, is
    ``[first_bin_center, inverse_bin_width, (slope, offset) * n_bins]``
    with ``charge`` and ``acceleration_kick`` folded into the pairs. It is
    a fresh array owned by the queue, so it needs no snapshot.
    """
    if arguments["first_left_cut"] is not None:
        return None  # sparse profiles run eagerly
    table = eager_specials._build_voltage_kick_table(
        voltage=arguments["voltage"],
        bin_centers=arguments["bin_centers"],
        charge=arguments["charge"],
        acceleration_kick=arguments["acceleration_kick"],
    )
    return [
        {
            "voltage_kick_table": table,
            "acceleration_kick": arguments["acceleration_kick"],
        }
    ]


_DRIFT_FIELDS = (_real("T"), _real("eta_0"), _real("beta"), _real("energy"))

KERNEL_CALL_RECORDS: tuple[KernelCallRecord, ...] = (
    KernelCallRecord(
        "kick_single_harmonic",
        (
            _real("voltage"),
            _real("omega_rf"),
            _real("phi_rf"),
            _real("charge"),
            _real("acceleration_kick"),
        ),
    ),
    KernelCallRecord(
        "kick_multi_harmonic",
        (
            RecordField("n_rf", "int32"),
            RecordField(
                "voltage", "inline_real_array", MAX_RF_HARMONICS_PER_RECORD,
                "n_rf",
            ),
            RecordField(
                "omega_rf", "inline_real_array", MAX_RF_HARMONICS_PER_RECORD,
                "n_rf",
            ),
            RecordField(
                "phi_rf", "inline_real_array", MAX_RF_HARMONICS_PER_RECORD,
                "n_rf",
            ),
            _real("charge"),
            _real("acceleration_kick"),
        ),
        prepare_on_enqueue=_split_harmonics,
    ),
    KernelCallRecord("drift_simple", _DRIFT_FIELDS),
    KernelCallRecord("drift_like_line_segment", _DRIFT_FIELDS),
    KernelCallRecord(
        "drift_exact",
        (
            _real("T"),
            _real("alpha_0"),
            _real("beta"),
            _real("energy"),
            RecordField("n_alpha", "int32"),
            RecordField(
                "higher_alpha", "inline_real_array", MAX_HIGHER_ALPHA,
                "n_alpha",
            ),
        ),
        prepare_on_enqueue=_inline_higher_alpha,
    ),
    KernelCallRecord(
        "kick_interpolated",
        (
            RecordField("voltage_kick_table", "input_array"),
            _real("acceleration_kick"),
        ),
        prepare_on_enqueue=_build_voltage_kick_table,
    ),
)
RECORDS_BY_NAME = {record.name: record for record in KERNEL_CALL_RECORDS}

HEADER_DTYPE = np.dtype(
    {
        "names": ["kernel_id", "record_size_bytes"],
        "formats": [np.uint32, np.uint32],
        "offsets": [0, 4],
        "itemsize": 8,
    }
)


def _layout(record: KernelCallRecord) -> tuple[list[tuple], int]:
    """
    Return ``[(c_declaration, name, dtype, offset)], size`` of the Args.

    Every field is aligned to its own size and the struct is padded to 8
    bytes with explicit ``std::int32_t padding_<k>`` members, so C and
    numpy never depend on compiler padding.
    """
    entries: list[tuple] = []
    offset = 0
    n_padding = 0

    def pad_to_8() -> None:
        nonlocal offset, n_padding
        if offset % 8:
            entries.append(
                (f"std::int32_t padding_{n_padding};", None, None, offset)
            )
            n_padding += 1
            offset += 4

    for field in record.fields:
        if field.kind in _C_TYPES:
            dtype = _DTYPES[field.kind]
            if dtype.itemsize == 8:
                pad_to_8()
            entries.append(
                (f"{_C_TYPES[field.kind]} {field.name};", field.name, dtype,
                 offset)
            )
            offset += dtype.itemsize
        elif field.kind == "input_array":
            pad_to_8()
            entries.append(
                (f"const real_t *{field.name};", field.name,
                 np.dtype(np.uintp), offset)
            )
            entries.append(
                (f"index_t {field.name}_length;", f"{field.name}_length",
                 _INDEX, offset + 8)
            )
            offset += 16
        elif field.kind == "inline_real_array":
            pad_to_8()
            dtype = np.dtype((_REAL, (field.max_length,)))
            entries.append(
                (f"real_t {field.name}[{field.max_length}];", field.name,
                 dtype, offset)
            )
            offset += dtype.itemsize
        else:  # pragma: no cover - a definition error
            raise ValueError(f"unknown field kind {field.kind!r}")
    pad_to_8()
    return entries, offset


@cache
def args_dtype(record: KernelCallRecord) -> np.dtype:
    """
    Numpy dtype of the kernel's ``Args`` struct.

    Parameters
    ----------
    record
        The kernel definition.

    Returns
    -------
    np.dtype
        Structured dtype with the same offsets as the C struct.
    """
    entries, size = _layout(record)
    named = [entry for entry in entries if entry[1] is not None]
    return np.dtype(
        {
            "names": [entry[1] for entry in named],
            "formats": [entry[2] for entry in named],
            "offsets": [entry[3] for entry in named],
            "itemsize": size,
        }
    )


@cache
def record_dtype(record: KernelCallRecord) -> np.dtype:
    """
    Numpy dtype of a whole record: header followed by the ``Args``.

    Parameters
    ----------
    record
        The kernel definition.

    Returns
    -------
    np.dtype
        Structured dtype with fields ``kernel_id``, ``record_size_bytes``
        and ``args``.
    """
    args = args_dtype(record)
    return np.dtype(
        {
            "names": ["kernel_id", "record_size_bytes", "args"],
            "formats": [np.uint32, np.uint32, args],
            "offsets": [0, 4, HEADER_DTYPE.itemsize],
            "itemsize": HEADER_DTYPE.itemsize + args.itemsize,
        }
    )


def _copyright_lines() -> list[str]:
    notice = os.path.join(
        os.path.dirname(HEADER_PATH), "..", "..", "..", "..",
        "dev_tools", "copyright_notice.txt",
    )
    with open(notice) as file:
        return [re.sub(r"^#", "//", line.rstrip("\n")) for line in file]


def generate_header() -> str:
    """
    Return the text of ``kernel_call_records.h``.

    Returns
    -------
    str
        The header, byte for byte as it must be on disk.
    """
    lines = [
        *_copyright_lines(),
        "",
        "// GENERATED by `python -m "
        "blond.core.backends.deferred.kernel_call_records`",
        "// from kernel_call_records.py next to this file. Do not edit.",
        "//",
        "// Include after `real_t` and `index_t` are defined:",
        "// blond_common.h on the C++ side, kernels.cu on the CUDA side.",
        "",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "#ifdef __CUDACC__",
        "#define BLOND_HOST_DEVICE __host__ __device__",
        "#else",
        "#define BLOND_HOST_DEVICE",
        "#endif",
        "",
        'static_assert(sizeof(real_t) == 8, "records assume 64-bit real_t");',
        'static_assert(sizeof(index_t) == 8, "records assume 64-bit index_t");',
        "",
        "enum class KernelId : std::uint32_t {",
        *(
            f"  {record.kernel_id_name} = {record.kernel_id},"
            for record in KERNEL_CALL_RECORDS
        ),
        "};",
        f"constexpr int KERNEL_COUNT = {len(KERNEL_CALL_RECORDS)};",
        "constexpr std::size_t KERNEL_CALL_BATCH_CAPACITY_BYTES = "
        f"{KERNEL_CALL_BATCH_CAPACITY_BYTES};",
        "",
        "struct KernelCallHeader {",
        "  KernelId kernel_id;",
        "  std::uint32_t record_size_bytes; // header + Args, multiple of 8",
        "};",
        "",
        "// Plain C arrays: the layout is fixed by the numpy dtypes.",
        "// NOLINTBEGIN(*-avoid-c-arrays)",
    ]
    for record in KERNEL_CALL_RECORDS:
        entries, size = _layout(record)
        lines.append(f"struct {record.kernel_id_name}Args {{")
        lines.extend(f"  {entry[0]}" for entry in entries)
        lines.append("};")
        lines.append(
            f"static_assert(sizeof({record.kernel_id_name}Args) == {size},"
            ' "regenerate kernel_call_records.h");'
        )
        for entry in entries:
            if entry[1] is not None and not entry[1].endswith("_length"):
                lines.append(
                    f"static_assert(offsetof({record.kernel_id_name}Args, "
                    f"{entry[1]}) == {entry[3]},"
                    ' "regenerate kernel_call_records.h");'
                )
        lines.append("")
    lines += [
        "// A macro, so the CUDA side can initialise a __device__ array",
        "// from the same list (kernels.cu).",
        "#define KERNEL_CALL_ARGS_SIZES_INITIALIZER \\",
        "  { \\",
        *(
            f"    sizeof({record.kernel_id_name}Args), \\"
            for record in KERNEL_CALL_RECORDS
        ),
        "  }",
        "constexpr std::uint32_t KERNEL_CALL_ARGS_SIZES[KERNEL_COUNT] ="
        " KERNEL_CALL_ARGS_SIZES_INITIALIZER;",
        "// NOLINTEND(*-avoid-c-arrays)",
        "",
        "// The records are packed back to back in a byte buffer, hence the",
        "// casts from the header to its Args and to the next header.",
        "// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast,"
        "cppcoreguidelines-pro-bounds-pointer-arithmetic)",
        "template <class Args>",
        "BLOND_HOST_DEVICE inline const Args &",
        "record_args(const KernelCallHeader *record) {",
        "  return *reinterpret_cast<const Args *>(",
        "      reinterpret_cast<const char *>(record) + "
        "sizeof(KernelCallHeader));",
        "}",
        "",
        "BLOND_HOST_DEVICE inline const KernelCallHeader *",
        "next_record(const KernelCallHeader *record) {",
        "  return reinterpret_cast<const KernelCallHeader *>(",
        "      reinterpret_cast<const char *>(record) + "
        "record->record_size_bytes);",
        "}",
        "// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast,"
        "cppcoreguidelines-pro-bounds-pointer-arithmetic)",
        "",
        "// The only switch over KernelId. `visitor(args)` resolves to the",
        "// backend's overload for that Args type; a missing overload does",
        "// not compile.",
        "template <class Visitor>",
        "BLOND_HOST_DEVICE inline void",
        "visit_kernel_call(const KernelCallHeader *record, "
        "const Visitor &visitor) {",
        "  switch (record->kernel_id) {",
    ]
    for record in KERNEL_CALL_RECORDS:
        lines += [
            f"  case KernelId::{record.kernel_id_name}:",
            f"    visitor(record_args<{record.kernel_id_name}Args>(record));",
            "    break;",
        ]
    lines += ["  }", "}", ""]
    return "\n".join(lines)


def header_digest() -> str:
    """
    Return the SHA-256 of the header on disk.

    Folded into the compiled-library cache keys of the cpp and cuda
    backends, whose own source hashes do not cover this folder.

    Returns
    -------
    str
        Hex digest of ``kernel_call_records.h``.
    """
    with open(HEADER_PATH, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    with open(HEADER_PATH, "w") as file:
        file.write(generate_header())
```

`blond/core/backends/deferred/__init__.py` contains the copyright header plus:

```python
"""Deferred (queued, fused) execution of per-particle kernels."""
```

- [ ] **Step 4: Generate the header and run the tests**

Run: `.venv/bin/python -m blond.core.backends.deferred.kernel_call_records && .venv/bin/python -m pytest tests/unittests/core/backends/deferred/test_kernel_call_records.py -v`

Expected: PASS.

Then open `kernel_call_records.h` and check that `KickMultiHarmonicArgs` begins with `std::int32_t n_rf; std::int32_t padding_0; real_t voltage[32];`.

- [ ] **Step 5: Check that both compilers accept the header**

Run:

```bash
printf '#include <cstdint>\nusing real_t=double;\nusing index_t=std::int64_t;\n#include "kernel_call_records.h"\nint main(){return KERNEL_COUNT;}\n' > /tmp/slauber/claude-174044/-home-slauber-PycharmProjects-deleteme-BLonD-uv/95585d69-62af-4541-a176-0ad67c77c134/scratchpad/h.cpp
g++ -std=c++11 -Wall -Werror -I blond/core/backends/deferred /tmp/slauber/claude-174044/-home-slauber-PycharmProjects-deleteme-BLonD-uv/95585d69-62af-4541-a176-0ad67c77c134/scratchpad/h.cpp -o /dev/null && echo OK-gcc
cp /tmp/slauber/claude-174044/-home-slauber-PycharmProjects-deleteme-BLonD-uv/95585d69-62af-4541-a176-0ad67c77c134/scratchpad/h.cpp /tmp/slauber/claude-174044/-home-slauber-PycharmProjects-deleteme-BLonD-uv/95585d69-62af-4541-a176-0ad67c77c134/scratchpad/h.cu
nvcc -std=c++11 -I blond/core/backends/deferred -c /tmp/slauber/claude-174044/-home-slauber-PycharmProjects-deleteme-BLonD-uv/95585d69-62af-4541-a176-0ad67c77c134/scratchpad/h.cu -o /dev/null && echo OK-nvcc
```

Expected: `OK-gcc` and `OK-nvcc`.

- [ ] **Step 6: Wire the header into both builds and into clang-tidy**

In `cpp/compiled_dir_handler.py`, inside the `hash_build_target` call's `extra=(...)`, add:

```python
            f"kernel_call_records={_kernel_call_records_digest()}",
```

Define the helper at module level:

```python
def _kernel_call_records_digest() -> str:
    # The generated header lives in ../deferred, outside the hashed folder.
    from blond.core.backends.deferred.kernel_call_records import (
        header_digest,
    )

    return header_digest()
```

In `cuda/compiled_dir_handler.py`, add the same helper and pass `extra=(f"kernel_call_records={_kernel_call_records_digest()}",)` to its `hash_build_target` call.

In `cpp/compile.py`, find where the flag list passed to `g++` is built (near the `-std=c++11` flag, line 197). Append:

```python
        "-I" + os.path.join(os.path.dirname(_basepath), "deferred"),
```

In `cuda/compile.py`, in the nvcc command near line 156 (`+ ["-o", libname_double, "-I" + cupyloc]`), add a second element: `"-I" + os.path.join(os.path.dirname(folder), "deferred")`.

In `dev_tools/run_clang_tidy.py`, append `f"-I{ROOT / 'blond' / 'core' / 'backends' / 'deferred'}"` to `COMPILER_FLAGS`.

- [ ] **Step 7: Run the compiled-dir tests**

Run: `.venv/bin/python -m pytest tests/unittests/core/backends/cpp/test_compiled_dir_handler.py tests/unittests/core/backends/deferred -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
pre-commit run --files blond/core/backends/deferred/* blond/core/backends/cpp/compiled_dir_handler.py blond/core/backends/cuda/compiled_dir_handler.py blond/core/backends/cpp/compile.py blond/core/backends/cuda/compile.py dev_tools/run_clang_tidy.py tests/unittests/core/backends/deferred/test_kernel_call_records.py
git add blond/core/backends/deferred dev_tools/run_clang_tidy.py blond/core/backends/cpp/compiled_dir_handler.py blond/core/backends/cuda/compiled_dir_handler.py blond/core/backends/cpp/compile.py blond/core/backends/cuda/compile.py tests/unittests/core/backends/deferred/test_kernel_call_records.py
git commit -m "Added the kernel call record definitions and their generated header" -m "One Python definition now fixes the layout that Python packs and C++/CUDA read, replacing positional packing with three unchecked sources of truth. The header digest joins both library cache keys because their source hashes do not cover the deferred folder." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: CPU kernel bodies in `particle_kernels.h`, with the eager kernels as wrappers

**Files:**
- Create: `blond/core/backends/cpp/particle_kernels.h`
- Modify:
  - `blond/core/backends/cpp/kick.cpp`: `kick_single_harmonic`, `kick_multi_harmonic`; leave `rf_volt_comp` untouched
  - `blond/core/backends/cpp/drift.cpp`
  - `blond/core/backends/cpp/drift_exact.cpp`
  - `blond/core/backends/cpp/linear_interp_kick.cpp`: the dense `linear_interp_kick` only; leave `linear_interp_kick_sparse` untouched
- Modify: `blond/core/backends/cpp/callables.py`. Inside `CppSpecials`, add `_build_voltage_kick_table`, and set `_LIBBLOND.linear_interp_kick_table`.
- Test: `tests/unittests/core/backends/cpp/test_voltage_kick_table.py`

**Interfaces:**
- Consumes: the `<Kernel>Args` structs from Task 2.
- Produces:
  - `void apply_to_chunk(const XArgs&, real_t *beam_dt, real_t *beam_dE, index_t begin, index_t end)` for all six `Args`
  - `thread_range(n, thread_id, n_threads, begin&, end&)` and `this_thread_range(n, begin&, end&)`
  - `run_on_all_particles(const Args&, dt, dE, n)`
  - `extern "C" void linear_interp_kick_table(const real_t *voltage, const real_t *bin_centers, real_t charge, int n_slices, real_t acc_kick, real_t *table)`. The table has `2 * n_slices` entries: `[bin_centers[0], inv_bin_width, (slope, offset) × (n_slices-1)]`.
  - `CppSpecials._build_voltage_kick_table(voltage, bin_centers, charge, acceleration_kick) -> NDArray` (numpy, length `2 * len(bin_centers)`)

- [ ] **Step 1: Write the failing test**

```python
# tests/unittests/core/backends/cpp/test_voltage_kick_table.py
import numpy as np
import pytest

from blond.core.backends.backend import Numpy64Bit, backend
from blond.testing.backend_testing import BLonDTestCase


@pytest.mark.backend_mutation
class TestVoltageKickTable(BLonDTestCase):
    def setUp(self) -> None:
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")

    def tearDown(self) -> None:
        backend.set_specials("python")

    def test_table_layout(self) -> None:
        bin_centers = np.linspace(-1e-9, 1e-9, 5)
        voltage = np.array([1.0, 3.0, 2.0, -1.0, 0.5])
        charge, acceleration_kick = 2.0, 0.25
        table = backend.specials._build_voltage_kick_table(
            voltage=voltage,
            bin_centers=bin_centers,
            charge=charge,
            acceleration_kick=acceleration_kick,
        )
        inverse_bin_width = 4 / (bin_centers[-1] - bin_centers[0])
        slope = charge * np.diff(voltage) * inverse_bin_width
        offset = charge * voltage[:-1] - bin_centers[:-1] * slope
        offset += acceleration_kick
        expected = np.concatenate(
            [[bin_centers[0], inverse_bin_width],
             np.column_stack([slope, offset]).ravel()]
        )
        np.testing.assert_allclose(table, expected, rtol=1e-14)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/unittests/core/backends/cpp/test_voltage_kick_table.py -v`

Expected: FAIL with `AttributeError: ... '_build_voltage_kick_table'`.

- [ ] **Step 3: Write `particle_kernels.h`**

Copy each loop body **verbatim** from its current eager kernel, so the arithmetic, and with it bitwise eager parity, is unchanged.

```cpp
// <copyright header>

// Per-particle kernel bodies, one overload of `apply_to_chunk` per
// kernel call record (kernel_call_records.h). Both drivers use them:
//  - the eager extern "C" kernels, via `run_on_all_particles`, which
//    gives every OpenMP thread its share of the whole beam;
//  - the deferred executor (deferred.cpp), which applies all queued
//    records to one cache-sized chunk before moving to the next.
// A formula therefore exists once on the CPU.
//
// Compute-bound overloads carry BLOND_PREFER_VECTOR_WIDTH_512 together
// with `noinline`: inlined into an OpenMP region without the attribute,
// GCC silently falls back to 256-bit code.

#pragma once

#include <cmath>

#include "blond_common.h"
#include "kernel_call_records.h"
#include "openmp.h"

#define BLOND_NOINLINE __attribute__((noinline))

// Split [0, n) evenly over the threads, in whole `granule`s, so two
// threads never share a cache line of particles.
inline void thread_range(const index_t n, const int thread_id,
                         const int n_threads, index_t &begin, index_t &end,
                         const index_t granule = 128) {
  const index_t n_granules = (n + granule - 1) / granule;
  const index_t per_thread = n_granules / n_threads;
  const index_t rest = n_granules % n_threads;
  const index_t first =
      thread_id * per_thread + (thread_id < rest ? thread_id : rest);
  const index_t count = per_thread + (thread_id < rest ? 1 : 0);
  begin = first * granule < n ? first * granule : n;
  end = (first + count) * granule < n ? (first + count) * granule : n;
}

// The calling thread's share of [0, n); call inside a parallel region.
inline void this_thread_range(const index_t n, index_t &begin,
                              index_t &end) {
  thread_range(n, omp_get_thread_num(), omp_get_num_threads(), begin, end);
}

BLOND_PREFER_VECTOR_WIDTH_512 BLOND_NOINLINE inline void
apply_to_chunk(const KickSingleHarmonicArgs &args,
               real_t *__restrict__ beam_dt, real_t *__restrict__ beam_dE,
               const index_t begin, const index_t end) {
  for (index_t i = begin; i < end; i++) {
    beam_dE[i] += args.charge * args.voltage *
                      FAST_SIN(args.omega_rf * beam_dt[i] + args.phi_rf) +
                  args.acceleration_kick;
  }
}

BLOND_PREFER_VECTOR_WIDTH_512 BLOND_NOINLINE inline void
apply_to_chunk(const KickMultiHarmonicArgs &args,
               real_t *__restrict__ beam_dt, real_t *__restrict__ beam_dE,
               const index_t begin, const index_t end) {
  // Body: the five n_rf branches of kick_multi_harmonic in kick.cpp
  // (unrolled 1..4, loop for more), verbatim, with `#pragma omp parallel
  // for` removed, the loops running over [begin, end), and
  // voltage/omega_RF/phi_RF/charge/acc_kick/n_rf read from `args`
  // (args.voltage, args.omega_rf, args.phi_rf, args.charge,
  // args.acceleration_kick, args.n_rf). Keep the NOLINTNEXTLINE comment.
  const real_t *__restrict__ voltage = args.voltage;
  const real_t *__restrict__ omega_RF = args.omega_rf;
  const real_t *__restrict__ phi_RF = args.phi_rf;
  const real_t charge = args.charge;
  const real_t acc_kick = args.acceleration_kick;
  const int n_rf = args.n_rf;
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (n_rf == 1) {
    for (index_t i = begin; i < end; i++) {
      const real_t dE_sum =
          voltage[0] * FAST_SIN(omega_RF[0] * beam_dt[i] + phi_RF[0]);
      beam_dE[i] += charge * dE_sum + acc_kick;
    }
  } else if (n_rf == 2) {
    for (index_t i = begin; i < end; i++) {
      const real_t dE_sum =
          voltage[0] * FAST_SIN(omega_RF[0] * beam_dt[i] + phi_RF[0]) +
          voltage[1] * FAST_SIN(omega_RF[1] * beam_dt[i] + phi_RF[1]);
      beam_dE[i] += charge * dE_sum + acc_kick;
    }
  } else if (n_rf == 3) {
    for (index_t i = begin; i < end; i++) {
      const real_t dE_sum =
          voltage[0] * FAST_SIN(omega_RF[0] * beam_dt[i] + phi_RF[0]) +
          voltage[1] * FAST_SIN(omega_RF[1] * beam_dt[i] + phi_RF[1]) +
          voltage[2] * FAST_SIN(omega_RF[2] * beam_dt[i] + phi_RF[2]);
      beam_dE[i] += charge * dE_sum + acc_kick;
    }
  } else if (n_rf == 4) {
    for (index_t i = begin; i < end; i++) {
      const real_t dE_sum =
          voltage[0] * FAST_SIN(omega_RF[0] * beam_dt[i] + phi_RF[0]) +
          voltage[1] * FAST_SIN(omega_RF[1] * beam_dt[i] + phi_RF[1]) +
          voltage[2] * FAST_SIN(omega_RF[2] * beam_dt[i] + phi_RF[2]) +
          voltage[3] * FAST_SIN(omega_RF[3] * beam_dt[i] + phi_RF[3]);
      beam_dE[i] += charge * dE_sum + acc_kick;
    }
  } else {
    for (index_t i = begin; i < end; i++) {
      real_t dE_sum = 0.0;
      for (int j = 0; j < n_rf; j++) {
        dE_sum += voltage[j] * FAST_SIN(omega_RF[j] * beam_dt[i] + phi_RF[j]);
      }
      beam_dE[i] += charge * dE_sum + acc_kick;
    }
  }
}

inline void apply_to_chunk(const DriftSimpleArgs &args,
                           real_t *__restrict__ beam_dt,
                           real_t *__restrict__ beam_dE, const index_t begin,
                           const index_t end) {
  const real_t coeff =
      args.T * args.eta_0 / (args.beta * args.beta * args.energy);
  for (index_t i = begin; i < end; i++) {
    beam_dt[i] += coeff * beam_dE[i];
  }
}

inline void apply_to_chunk(const DriftLikeLineSegmentArgs &args,
                           real_t *__restrict__ beam_dt,
                           real_t *__restrict__ beam_dE, const index_t begin,
                           const index_t end) {
  const real_t inv_beta_sq = 1.0 / (args.beta * args.beta);
  const real_t inv_energy = 1.0 / args.energy;
  const real_t inv_energy_sq = inv_energy * inv_energy;
  for (index_t i = begin; i < end; i++) {
    const real_t dE = beam_dE[i];
    const real_t delta =
        std::sqrt(1.0 + inv_beta_sq *
                            (dE * dE * inv_energy_sq + 2.0 * dE * inv_energy)) -
        1.0;
    beam_dt[i] += args.T * args.eta_0 * delta;
  }
}

// drift_exact: declared here, defined in drift_exact.cpp, which keeps its
// unrolled drift_exact_unrolled<N> templates and dispatches on n_alpha.
void apply_to_chunk(const DriftExactArgs &args, real_t *beam_dt,
                    real_t *beam_dE, index_t begin, index_t end);

// kick_interpolated: declared here, defined in linear_interp_kick.cpp.
void apply_to_chunk(const KickInterpolatedArgs &args, real_t *beam_dt,
                    real_t *beam_dE, index_t begin, index_t end);

extern "C" void linear_interp_kick_table(const real_t *voltage,
                                         const real_t *bin_centers,
                                         real_t charge, int n_slices,
                                         real_t acc_kick, real_t *table);

// Eager driver: `args` on all particles, each thread on its own share.
template <class Args>
inline void run_on_all_particles(const Args &args, real_t *beam_dt,
                                 real_t *beam_dE,
                                 const index_t n_macroparticles) {
#pragma omp parallel
  {
    index_t begin = 0;
    index_t end = 0;
    this_thread_range(n_macroparticles, begin, end);
    apply_to_chunk(args, beam_dt, beam_dE, begin, end);
  }
}
```

**Important:** the per-kernel attributes must match what the eager kernels carry today.
- `kick_single_harmonic` and `kick_multi_harmonic` carry `BLOND_PREFER_VECTOR_WIDTH_512` in `kick.cpp`.
- `drift_simple` and `drift_like_line_segment` carry none, so their overloads above are plain `inline`.
- Check `drift_exact.cpp` and `linear_interp_kick.cpp` for attributes and copy them onto their definitions.

- [ ] **Step 4: Turn the eager kernels into wrappers**

In `kick.cpp`, replace the bodies (keep the signatures and `extern "C"` exactly). `beam_dt` is `const` in the signature and `run_on_all_particles` takes `real_t *`, so use `const_cast` with a comment. The kick only reads `dt`.

```cpp
#include "particle_kernels.h"

extern "C" void kick_single_harmonic(const real_t *__restrict__ beam_dt,
                     real_t *__restrict__ beam_dE, const real_t charge,
                     const real_t voltage, const real_t omega_RF,
                     const real_t phi_RF, const index_t n_macroparticles,
                     const real_t acc_kick) {
  const KickSingleHarmonicArgs args = {voltage, omega_RF, phi_RF, charge,
                                       acc_kick};
  // The kick only reads dt; apply_to_chunk has one signature for all.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  run_on_all_particles(args, const_cast<real_t *>(beam_dt), beam_dE,
                       n_macroparticles);
}
```

Follow the same pattern for the others:
- `kick_multi_harmonic`: it takes pointer arrays, so copy them into the inline arrays in chunks of 32. Build one `KickMultiHarmonicArgs` per chunk of ≤32 harmonics and pass `acc_kick` only to the last. The multi-harmonic kernel has no `n_rf > 32` limit today, so this chunking is required for correctness.
- `drift_simple` / `drift_like_line_segment`: `DriftSimpleArgs{T, eta_zero, beta, energy}`, with a `const_cast` on `beam_dE`.

In `drift_exact.cpp`:
- Keep the anonymous-namespace templates.
- Add `void apply_to_chunk(const DriftExactArgs &args, ...)`. It builds the same `switch (args.n_alpha)` over `drift_exact_unrolled<0..4>` / `drift_exact_generic`, changed so they take `[begin, end)` instead of `n_macroparticles` and have **no** `#pragma omp parallel for`.
- The eager `drift_exact` checks `n_alpha <= MAX_HIGHER_ALPHA (8)`.
  - If it holds, it builds `DriftExactArgs` (copying `higher_alpha`) and calls `run_on_all_particles`.
  - Otherwise it runs the generic path directly under `#pragma omp parallel` with `this_thread_range`. That is the eager fallback for more than 8 coefficients.

In `linear_interp_kick.cpp`:
- Add `linear_interp_kick_table`. Its loop is the existing table loop, writing `table[2 + 2*i]` (slope) and `table[3 + 2*i]` (offset), with `table[0] = bin_centers[0]` and `table[1] = inv_bin_width`.
- Add `apply_to_chunk(const KickInterpolatedArgs &args, ...)`. It reads `bin0 = table[0]`, `inv_bin_width = table[1]` and `n_bins = (args.voltage_kick_table_length - 2) / 2`. It keeps the existing `STEP = 64` staged `fbin` loop and the out-of-range `+= acc_kick` branch, reading the pairs from `table + 2`.
- The eager `linear_interp_kick` gets a `reuse_scratch` table of `2 * n_slices` entries, fills it with `linear_interp_kick_table`, and calls `run_on_all_particles`.

- [ ] **Step 5: Add `_build_voltage_kick_table` to `CppSpecials`**

In `reload_cpp_backend`, next to the other `restype` lines, add:

```python
    _LIBBLOND.linear_interp_kick_table.restype = None
```

Inside `class CppSpecials`, add:

```python
        @staticmethod
        def _build_voltage_kick_table(
            voltage: NumpyArray,
            bin_centers: NumpyArray,
            charge: float,
            acceleration_kick: float,
        ) -> NumpyArray:
            """Table read by the deferred interpolated kick."""
            assert _is_valid((voltage, floattype), (bin_centers, floattype))
            n_slices = len(bin_centers)
            assert n_slices >= 2  # noqa: PLR2004
            table = np.empty(2 * n_slices, dtype=floattype)
            _LIBBLOND.linear_interp_kick_table(
                _get_pointer(voltage),
                _get_pointer(bin_centers),
                c_real(floattype(charge), floattype),
                ct.c_int(n_slices),
                c_real(floattype(acceleration_kick), floattype),
                ct.c_void_p(table.ctypes.data),
            )
            return table
```

Use `ct.c_void_p(table.ctypes.data)`, not `_get_pointer`. The table is fresh every call, and caching its `id` would pollute the pointer cache.

- [ ] **Step 6: Rebuild and run the tests**

Run: `.venv/bin/python -m blond.core.backends.cpp.compile && .venv/bin/python -m pytest tests/unittests/core/backends/cpp tests/unittests/core/backends/test_backend.py -q`

Expected: PASS. `test_backend.py` compares every kernel against the python reference, so it guards the refactor.

- [ ] **Step 7: Check vectorisation and eager speed**

Run: `.venv/bin/python -m pytest tests/unittests/core/backends/cpp/test_vector_width.py -v`

Expected: PASS (it checks the 512-bit code).

Then run `.venv/bin/python dev_tools/performance_blond3/backends/kick.py` and `.venv/bin/python dev_tools/performance_blond3/backends/drift.py`, both before (`git stash`) and after. The per-kernel times must stay within noise. Record both sets of numbers in the commit body.

- [ ] **Step 8: Commit**

```bash
pre-commit run --files blond/core/backends/cpp/*.h blond/core/backends/cpp/*.cpp blond/core/backends/cpp/callables.py tests/unittests/core/backends/cpp/test_voltage_kick_table.py
git add blond/core/backends/cpp tests/unittests/core/backends/cpp/test_voltage_kick_table.py
git commit -m "Moved the cpp particle kernel bodies into particle_kernels.h" -m "The deferred executor and the eager kernels must share one formula per kernel. The interpolated-kick table is now its own entry point so it can be built when a deferred call is queued. <paste before/after kernel timings>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The cpp executor and the record-size check at load time

**Files:**
- Create: `blond/core/backends/cpp/deferred.cpp`
- Modify: `blond/core/backends/cpp/compile.py`. Add `"deferred.cpp",` to `cpp_files` (line 67).
- Modify: `blond/core/backends/cpp/callables.py`. Add `check_kernel_call_record_abi(library)`, next to `check_index_abi`, and call it in `reload_cpp_backend` after `check_index_abi(_LIBBLOND)`.
- Test: `tests/unittests/core/backends/deferred/test_cpp_executor.py`

**Interfaces:**
- Consumes: `record_dtype`, `args_dtype`, `KERNEL_CALL_RECORDS` (Task 2); `apply_to_chunk` overloads (Task 3).
- Produces:
  - `extern "C" void execute_kernel_call_batch(const std::uint8_t *batch, std::size_t n_bytes, real_t *beam_dt, real_t *beam_dE, index_t n_macroparticles, index_t chunk_size)`
  - `extern "C" std::uint32_t kernel_call_args_size(int kernel_id)`, which returns 0 for an unknown id
  - `check_kernel_call_record_abi(library) -> None`, which raises `AssertionError` on a mismatch

- [ ] **Step 1: Write the failing tests**

The tests drive the executor directly through ctypes, with a hand-packed batch.

```python
# tests/unittests/core/backends/deferred/test_cpp_executor.py
import ctypes as ct

import numpy as np
import pytest

from blond.core.backends.backend import Numpy64Bit, backend
from blond.core.backends.cpp.callables import c_index_t
from blond.core.backends.deferred import kernel_call_records as records
from blond.testing.backend_testing import BLonDTestCase


def _pack(*records_and_values) -> np.ndarray:
    parts = []
    for record, values in records_and_values:
        item = np.zeros((), dtype=records.record_dtype(record))
        item["kernel_id"] = record.kernel_id
        item["record_size_bytes"] = item.dtype.itemsize
        for name, value in values.items():
            item["args"][name] = value
        parts.append(item.tobytes())
    return np.frombuffer(b"".join(parts), dtype=np.uint8).copy()


@pytest.mark.backend_mutation
class TestCppExecutor(BLonDTestCase):
    def setUp(self) -> None:
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")
        self.eager = backend.specials
        self.library = self.eager._library  # set in Step 3

    def tearDown(self) -> None:
        backend.set_specials("python")

    def _execute(self, batch, dt, dE, chunk_size=4096) -> None:
        self.library.execute_kernel_call_batch(
            ct.c_void_p(batch.ctypes.data),
            ct.c_size_t(batch.size),
            ct.c_void_p(dt.ctypes.data),
            ct.c_void_p(dE.ctypes.data),
            c_index_t(len(dt)),
            c_index_t(chunk_size),
        )

    def test_kick_then_drift_matches_eager(self) -> None:
        rng = np.random.default_rng(1)
        for n in (1, 7, 1000, 100003):
            for chunk_size in (1, 64, 4096, 10**9):
                dt = rng.uniform(-1e-9, 1e-9, n)
                dE = rng.uniform(-1e6, 1e6, n)
                dt_eager, dE_eager = backend.copy(dt), backend.copy(dE)
                kick = dict(voltage=8e3, omega_rf=2e7, phi_rf=0.1,
                            charge=1.0, acceleration_kick=12.0)
                drift = dict(T=1e-6, eta_0=0.01, beta=0.9, energy=2e9)
                self.eager.kick_single_harmonic(
                    dt=dt_eager, dE=dE_eager, **kick)
                self.eager.drift_simple(dt=dt_eager, dE=dE_eager, **drift)
                batch = _pack(
                    (records.RECORDS_BY_NAME["kick_single_harmonic"], kick),
                    (records.RECORDS_BY_NAME["drift_simple"], drift),
                )
                self._execute(batch, dt, dE, chunk_size)
                np.testing.assert_allclose(dE, dE_eager, rtol=1e-12)
                np.testing.assert_allclose(dt, dt_eager, rtol=1e-12)

    def test_zero_macroparticles(self) -> None:
        dt, dE = np.empty(0), np.empty(0)
        batch = _pack((records.RECORDS_BY_NAME["drift_simple"],
                       dict(T=1.0, eta_0=1.0, beta=1.0, energy=1.0)))
        self._execute(batch, dt, dE)  # must not crash

    def test_args_sizes_match_dtypes(self) -> None:
        for record in records.KERNEL_CALL_RECORDS:
            self.assertEqual(
                self.library.kernel_call_args_size(record.kernel_id),
                records.args_dtype(record).itemsize,
            )
        self.assertEqual(self.library.kernel_call_args_size(999), 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unittests/core/backends/deferred/test_cpp_executor.py -v`

Expected: FAIL with `AttributeError: ... '_library'`. After Step 3 adds `_library`, the failure is `undefined symbol: execute_kernel_call_batch`.

- [ ] **Step 3: Implement `deferred.cpp`, the ABI check and `CppSpecials._library`**

```cpp
// <copyright header>

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
  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
  const auto *first = reinterpret_cast<const KernelCallHeader *>(batch);
  const auto *last = reinterpret_cast<const KernelCallHeader *>(batch + n_bytes);
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
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
```

In `callables.py`, add `check_kernel_call_record_abi` below `check_index_abi`:

```python
def check_kernel_call_record_abi(library: CDLL) -> None:
    """
    Assert every compiled ``Args`` struct matches its numpy dtype.

    A mismatch would make the deferred executor read wrong parameters;
    comparing once at load time turns that into a loud failure.

    Parameters
    ----------
    library
        The freshly loaded ``libblond``.
    """
    from blond.core.backends.deferred.kernel_call_records import (
        KERNEL_CALL_RECORDS,
        args_dtype,
    )

    library.kernel_call_args_size.restype = ct.c_uint32
    for record in KERNEL_CALL_RECORDS:
        compiled = int(library.kernel_call_args_size(record.kernel_id))
        expected = args_dtype(record).itemsize
        assert compiled == expected, (
            f"{record.kernel_id_name}Args is {compiled} bytes in libblond "
            f"but {expected} in kernel_call_records.py; rebuild the C++ "
            "backend with `blond-compile-cpp`."
        )
```

In `reload_cpp_backend`:
- after `check_index_abi(_LIBBLOND)`, call `check_kernel_call_record_abi(_LIBBLOND)`;
- set `_LIBBLOND.execute_kernel_call_batch.restype = None`;
- inside `class CppSpecials`, add a class attribute `_library = _LIBBLOND` with the comment `# the loaded libblond, for the deferred executor and its tests`.

- [ ] **Step 4: Rebuild and run the tests**

Run: `.venv/bin/python -m blond.core.backends.cpp.compile && .venv/bin/python -m pytest tests/unittests/core/backends/deferred/test_cpp_executor.py tests/unittests/core/backends/cpp -q`

Expected: PASS.

- [ ] **Step 5: Run clang-tidy on the new files**

Run: `.venv/bin/python dev_tools/run_clang_tidy.py`

Expected: no new findings in `deferred.cpp`, `particle_kernels.h` or `kernel_call_records.h`. Fix any finding in the source, or in the generator for the header, and regenerate.

- [ ] **Step 6: Commit**

```bash
pre-commit run --files blond/core/backends/cpp/deferred.cpp blond/core/backends/cpp/compile.py blond/core/backends/cpp/callables.py tests/unittests/core/backends/deferred/test_cpp_executor.py
git add blond/core/backends/cpp/deferred.cpp blond/core/backends/cpp/compile.py blond/core/backends/cpp/callables.py tests/unittests/core/backends/deferred/test_cpp_executor.py
git commit -m "Added the cpp executor for deferred kernel call batches" -m "It walks the batch per cache-sized chunk through the generated visit_kernel_call, so it holds no kernel-specific code. The Args sizes are checked against the numpy dtypes when libblond is loaded." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The kernel call queue, `make_deferred_specials` and `cpp_deferred`

**Files:**
- Create: `blond/core/backends/deferred/kernel_call_queue.py`
- Modify: `blond/core/backends/cpp/callables.py`
  - `reload_cpp_backend(floattype, parallel=True, deferred=False)`;
  - when `deferred`, return `make_deferred_specials(CppSpecials, execute_batch)`.
- Modify: `blond/core/backends/backend.py`
  - `NumpyBackend.set_specials` accepts `"cpp_deferred"`;
  - its `Literal[...]` and docstring list the new mode.
- Test: `tests/unittests/core/backends/deferred/test_deferred_specials.py`

**Interfaces:**
- Consumes:
  - `KERNEL_CALL_RECORDS`, `RECORDS_BY_NAME`, `record_dtype` (Task 2);
  - `CppSpecials._library`, `CppSpecials._build_voltage_kick_table` (Tasks 3–4);
  - `Specials.flush` (Task 1).
- Produces:
  - `KernelCallQueue` (a `threading.local` subclass) with `.bind(dt, dE)`, `.holds(dt, dE) -> bool`, `.append(record, values: dict)`, `.clear()`, and the attributes `.buffer`, `.n_bytes`, `.record_sizes`, `.dt`, `.dE`, `.keep_alive`.
  - `make_deferred_specials(eager_specials: type, execute_batch: Callable[[np.ndarray, list[int], Any, Any], None]) -> type`. The returned class has `flush()` and the class attribute `kernel_call_queue`.
  - `deferred_chunk_size() -> int`, which reads `BLOND_DEFERRED_CHUNK_SIZE` (default 4096) and raises `ValueError` if the value is < 1.
  - `address_of(array) -> int`, which returns the host or device address.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unittests/core/backends/deferred/test_deferred_specials.py
import inspect
import threading

import numpy as np
import pytest

from blond.core.backends.backend import Numpy64Bit, Specials, backend
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.testing.backend_testing import BLonDTestCase

RNG = np.random.default_rng(3)


def _beam(n):
    """Beam coordinates on the active backend's device."""
    return (
        backend.array(RNG.uniform(-1e-8, 1e-8, n), dtype=backend.float),
        backend.array(RNG.uniform(-1e6, 1e6, n), dtype=backend.float),
    )


def _close(actual, expected, **tolerances):
    np.testing.assert_allclose(
        copy_to_cpu(actual), copy_to_cpu(expected), **tolerances
    )


def _equal(a, b) -> bool:
    return np.array_equal(copy_to_cpu(a), copy_to_cpu(b))


KICK = dict(voltage=8e3, omega_rf=2e7, phi_rf=0.1, charge=1.0,
            acceleration_kick=12.0)
DRIFT = dict(T=1e-6, eta_0=0.01, beta=0.9, energy=2e9)


def _turn(specials, dt, dE, n_rf=3, n_alpha=2, n_bins=64):
    specials.kick_single_harmonic(dt=dt, dE=dE, **KICK)
    specials.kick_multi_harmonic(
        dt=dt, dE=dE,
        voltage=np.linspace(1e3, 2e3, n_rf),
        omega_rf=np.linspace(1e7, 3e7, n_rf),
        phi_rf=np.linspace(0, 1, n_rf),
        charge=1.0, n_rf=n_rf, acceleration_kick=5.0,
    )
    specials.drift_simple(dt=dt, dE=dE, **DRIFT)
    specials.drift_like_line_segment(dt=dt, dE=dE, **DRIFT)
    specials.drift_exact(
        dt=dt, dE=dE, T=1e-6, alpha_0=0.01,
        higher_alpha=np.linspace(1e-3, 2e-3, n_alpha),
        beta=0.9, energy=2e9,
    )
    # Host arrays for the inlined RF/alpha values (the ABC's NumpyArray),
    # device arrays for the profile-sized interpolated kick inputs.
    specials.kick_interpolated(
        dt=dt, dE=dE,
        voltage=backend.array(np.sin(np.linspace(0, 3, n_bins)) * 1e3),
        bin_centers=backend.array(np.linspace(-1e-8, 1e-8, n_bins)),
        charge=1.0, acceleration_kick=3.0,
    )


@pytest.mark.backend_mutation
class TestCppDeferredSpecials(BLonDTestCase):
    mode = "cpp_deferred"
    eager_mode = "cpp"

    def setUp(self) -> None:
        backend.change_backend(Numpy64Bit)
        backend.set_specials(self.eager_mode)
        self.eager = backend.specials
        backend.set_specials(self.mode)
        self.deferred = backend.specials

    def tearDown(self) -> None:
        self.deferred.flush()
        backend.set_specials("python")

    def _assert_matches_eager(self, **turn_kwargs) -> None:
        for n in (1, 7, 1000, 100003):
            dt, dE = _beam(n)
            dt_eager, dE_eager = backend.copy(dt), backend.copy(dE)
            for _ in range(3):
                _turn(self.eager, dt_eager, dE_eager, **turn_kwargs)
                _turn(self.deferred, dt, dE, **turn_kwargs)
            self.deferred.flush()
            _close(dt, dt_eager, rtol=1e-11, atol=0)
            _close(dE, dE_eager, rtol=1e-11, atol=1e-6)

    def test_matches_eager(self) -> None:
        self._assert_matches_eager()

    def test_more_than_32_harmonics(self) -> None:
        self._assert_matches_eager(n_rf=40)

    def test_drift_exact_coefficient_counts(self) -> None:
        for n_alpha in (0, 1, 4, 5, 9):  # 9 falls back to eager
            self._assert_matches_eager(n_alpha=n_alpha)

    def test_chunk_sizes(self) -> None:
        for chunk_size in ("1", "64", "1000000000"):
            with pytest.MonkeyPatch.context() as patch:
                patch.setenv("BLOND_DEFERRED_CHUNK_SIZE", chunk_size)
                self._assert_matches_eager()

    def test_queued_until_flush(self) -> None:
        dt, dE = _beam(10)
        before = backend.copy(dE)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **KICK)
        self.assertTrue(_equal(dE, before))
        self.deferred.flush()
        self.assertFalse(_equal(dE, before))

    def test_unqueued_kernel_flushes_first(self) -> None:
        dt, dE = _beam(10)
        before = backend.copy(dE)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **KICK)
        total = self.deferred.sum_1d_array(dE)  # not deferrable: flushes
        self.assertEqual(self.deferred.kernel_call_queue.n_bytes, 0)
        self.assertFalse(_equal(dE, before))
        self.assertAlmostEqual(
            total, float(np.sum(copy_to_cpu(dE))), delta=1e-3
        )

    def test_kernel_on_a_slice_of_the_beam(self) -> None:  # Review Focus 2
        dt, dE = _beam(100)
        dt_eager, dE_eager = backend.copy(dt), backend.copy(dE)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **KICK)
        self.deferred.drift_simple(dt=dt[:50], dE=dE[:50], **DRIFT)
        self.deferred.flush()
        self.eager.kick_single_harmonic(dt=dt_eager, dE=dE_eager, **KICK)
        self.eager.drift_simple(dt=dt_eager[:50], dE=dE_eager[:50], **DRIFT)
        _close(dt, dt_eager, rtol=1e-12)
        _close(dE, dE_eager, rtol=1e-12)

    def test_voltage_mutated_after_queue(self) -> None:  # Review Focus 1
        dt, dE = _beam(1000)
        dt_eager, dE_eager = backend.copy(dt), backend.copy(dE)
        voltage = backend.array(np.sin(np.linspace(0, 3, 64)) * 1e3)
        bins = backend.array(np.linspace(-1e-8, 1e-8, 64))
        self.eager.kick_interpolated(dt=dt_eager, dE=dE_eager,
            voltage=backend.copy(voltage), bin_centers=bins, charge=1.0,
            acceleration_kick=0.0)
        self.deferred.kick_interpolated(dt=dt, dE=dE, voltage=voltage,
            bin_centers=bins, charge=1.0, acceleration_kick=0.0)
        voltage[:] = 0.0  # caller reuses its buffer before the flush
        self.deferred.flush()
        _close(dE, dE_eager, rtol=1e-12)

    def test_failed_flush_clears_queue(self) -> None:  # Review Focus 3
        dt, dE = _beam(10)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **KICK)
        queue = self.deferred.kernel_call_queue
        original = self.deferred._execute_batch
        def failing(*args):
            raise RuntimeError("boom")
        type(self.deferred)._execute_batch = staticmethod(failing)
        try:
            with self.assertRaises(RuntimeError):
                self.deferred.flush()
        finally:
            type(self.deferred)._execute_batch = staticmethod(original)
        self.assertEqual(queue.n_bytes, 0)
        self.assertEqual(queue.keep_alive, [])

    def test_scalars_captured_at_enqueue(self) -> None:
        dt, dE = _beam(10)
        dt_e, dE_e = backend.copy(dt), backend.copy(dE)
        kick = dict(KICK)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **kick)
        kick["voltage"] = 0.0
        self.deferred.flush()
        self.eager.kick_single_harmonic(dt=dt_e, dE=dE_e, **KICK)
        _close(dE, dE_e, rtol=1e-12)

    def test_positional_arguments(self) -> None:
        dt, dE = _beam(10)
        dt_e, dE_e = backend.copy(dt), backend.copy(dE)
        self.deferred.drift_simple(dt, dE, 1e-6, 0.01, 0.9, 2e9)
        self.deferred.flush()
        self.eager.drift_simple(dt_e, dE_e, 1e-6, 0.01, 0.9, 2e9)
        _close(dt, dt_e, rtol=1e-12)

    def test_queues_are_per_thread(self) -> None:
        results = {}
        def worker(key):
            dt, dE = _beam(10000)
            dt_e, dE_e = backend.copy(dt), backend.copy(dE)
            for _ in range(20):
                _turn(self.deferred, dt, dE)
                _turn(self.eager, dt_e, dE_e)
            self.deferred.flush()
            results[key] = (
                np.allclose(copy_to_cpu(dt), copy_to_cpu(dt_e), rtol=1e-11),
                np.allclose(copy_to_cpu(dE), copy_to_cpu(dE_e), rtol=1e-11,
                            atol=1e-6),
            )
        threads = [threading.Thread(target=worker, args=(k,))
                   for k in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(results, {k: (True, True) for k in range(4)})

    def test_every_specials_method_is_deferred_or_wrapped(self) -> None:
        for name, value in vars(Specials).items():
            if name.startswith("_") or not isinstance(value, staticmethod):
                continue
            self.assertIn(name, vars(type(self.deferred)), name)

    def test_switching_specials_flushes(self) -> None:
        dt, dE = _beam(10)
        before = backend.copy(dE)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **KICK)
        backend.set_specials(self.eager_mode)
        self.assertFalse(_equal(dE, before))

    def test_invalid_chunk_size(self) -> None:
        from blond.core.backends.deferred.kernel_call_queue import (
            deferred_chunk_size,
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("BLOND_DEFERRED_CHUNK_SIZE", "0")
            with self.assertRaises(ValueError):
                deferred_chunk_size()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unittests/core/backends/deferred/test_deferred_specials.py -v`

Expected: FAIL with `UnknownBackendMode: Unknown specials mode 'cpp_deferred'`.

- [ ] **Step 3: Implement `kernel_call_queue.py`**

```python
# <copyright header>

"""
Queue of deferred kernel calls, shared by ``cpp_deferred``/``cuda_deferred``.

`make_deferred_specials` derives deferred specials from eager ones: each
deferrable kernel (`KERNEL_CALL_RECORDS`) packs a kernel call record into
a per-thread `KernelCallQueue` instead of running; every other method
flushes the queue and then runs eagerly. A flush hands the batch to the
backend's ``execute_batch``, which applies it in one fused pass.
"""

from __future__ import annotations

import functools
import inspect
import os
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from blond.core.backends.backend import Specials, backend
from blond.core.backends.deferred.kernel_call_records import (
    KERNEL_CALL_RECORDS,
    RECORDS_BY_NAME,
    KernelCallRecord,
    record_dtype,
)
from blond.generals.cupy_.no_cupy_import import is_cupy_array

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

DEFAULT_CHUNK_SIZE = 4096
_INITIAL_CAPACITY_BYTES = 4096


def deferred_chunk_size() -> int:
    """
    Return the particles per chunk of the cpp executor.

    Returns
    -------
    int
        ``BLOND_DEFERRED_CHUNK_SIZE`` if set, else `DEFAULT_CHUNK_SIZE`.

    Raises
    ------
    ValueError
        If the environment variable is not a positive integer.
    """
    chunk_size = int(
        os.environ.get("BLOND_DEFERRED_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)
    )
    if chunk_size < 1:
        raise ValueError(
            f"BLOND_DEFERRED_CHUNK_SIZE must be >= 1, got {chunk_size}"
        )
    return chunk_size


def address_of(array: Any) -> int:
    """
    Return the data address of a host or device array.

    Parameters
    ----------
    array
        NumPy or CuPy array.

    Returns
    -------
    int
        Host address for NumPy, device address for CuPy.
    """
    return array.data.ptr if is_cupy_array(array) else array.ctypes.data


class KernelCallQueue(threading.local):
    """
    Kernel call records of one Python thread, waiting for the next flush.

    Per thread because one specials object serves every simulation in the
    process, and simulations may run in separate Python threads (ctypes
    releases the GIL). The queue is bound to one beam's ``dt``/``dE``.
    """

    def __init__(self) -> None:
        self.buffer = np.zeros(_INITIAL_CAPACITY_BYTES, dtype=np.uint8)
        self.n_bytes = 0
        self.record_sizes: list[int] = []
        self.keep_alive: list[Any] = []
        self.dt: Any = None
        self.dE: Any = None

    def holds(self, dt: Any, dE: Any) -> bool:
        """
        Return whether the queued records belong to this beam.

        Parameters
        ----------
        dt, dE
            Beam coordinate arrays of the next call.

        Returns
        -------
        bool
            True for the very same array objects.
        """
        return self.dt is dt and self.dE is dE

    def bind(self, dt: Any, dE: Any) -> None:
        """
        Bind the empty queue to a beam.

        Parameters
        ----------
        dt, dE
            Beam coordinate arrays all queued records will act on.
        """
        assert self.n_bytes == 0, "bind only an empty queue"
        assert dt.dtype == backend.float and dE.dtype == backend.float
        assert dt.flags.c_contiguous and dE.flags.c_contiguous
        self.dt, self.dE = dt, dE

    def append(self, record: KernelCallRecord, values: dict[str, Any]) -> None:
        """
        Pack one kernel call record at the end of the batch.

        Parameters
        ----------
        record
            The kernel definition.
        values
            Field name -> value, as the record's fields declare.
        """
        dtype = record_dtype(record)
        size = dtype.itemsize
        self._reserve(size)
        item = self.buffer[self.n_bytes : self.n_bytes + size].view(dtype)[0]
        item["kernel_id"] = record.kernel_id
        item["record_size_bytes"] = size
        args = item["args"]
        for field in record.fields:
            value = values[field.name]
            if field.kind == "input_array":
                assert value.dtype == backend.float
                assert value.flags.c_contiguous
                args[field.name] = address_of(value)
                args[f"{field.name}_length"] = value.size
                self.keep_alive.append(value)
            elif field.kind == "inline_real_array":
                n_values = len(value)
                args[field.name][:n_values] = value
                args[field.name][n_values:] = 0.0
            else:
                args[field.name] = value
        self.n_bytes += size
        self.record_sizes.append(size)

    def clear(self) -> None:
        """Drop all records and array references; keep the buffer."""
        self.n_bytes = 0
        self.record_sizes = []
        self.keep_alive = []
        self.dt = self.dE = None

    def _reserve(self, size: int) -> None:
        needed = self.n_bytes + size
        if needed > self.buffer.size:
            grown = np.zeros(max(2 * self.buffer.size, needed), np.uint8)
            grown[: self.n_bytes] = self.buffer[: self.n_bytes]
            self.buffer = grown


def _values_from_kwargs(
    record: KernelCallRecord, arguments: dict[str, Any]
) -> list[dict[str, Any]]:
    return [{field.name: arguments[field.name] for field in record.fields}]


def make_deferred_specials(
    eager_specials: type,
    execute_batch: Callable[[np.ndarray, list[int], Any, Any], None],
) -> type:
    """
    Derive deferred specials from eager ones.

    Parameters
    ----------
    eager_specials
        The eager specials class, e.g. ``CppSpecials``.
    execute_batch
        ``execute_batch(batch, record_sizes, dt, dE)`` applying the batch
        bytes, in one fused pass where possible, to the beam.

    Returns
    -------
    type
        A subclass of ``eager_specials`` named ``Deferred<name>``.
    """
    queue = KernelCallQueue()
    namespace: dict[str, Any] = {"kernel_call_queue": queue}

    def flush() -> None:
        """Run all queued kernel calls of this thread."""
        if queue.n_bytes == 0:
            return
        batch = queue.buffer[: queue.n_bytes]
        try:
            deferred_class._execute_batch(
                batch, queue.record_sizes, queue.dt, queue.dE
            )
        finally:
            queue.clear()  # never re-apply a batch, even after an error

    namespace["flush"] = staticmethod(flush)
    namespace["_execute_batch"] = staticmethod(execute_batch)

    for name, value in vars(Specials).items():
        if name.startswith("_") or name == "flush":
            continue
        if not isinstance(value, staticmethod):
            continue
        eager_method = getattr(eager_specials, name)
        if name in RECORDS_BY_NAME:
            method = _queuing_method(
                RECORDS_BY_NAME[name], eager_method, eager_specials, queue,
                flush,
            )
        else:
            method = _flushing_method(eager_method, flush)
        namespace[name] = staticmethod(method)

    deferred_class = type(
        f"Deferred{eager_specials.__name__}", (eager_specials,), namespace
    )
    return deferred_class


def _flushing_method(eager_method: Callable, flush: Callable) -> Callable:
    @functools.wraps(eager_method)
    def flush_then_call(*args: Any, **kwargs: Any) -> Any:
        flush()
        return eager_method(*args, **kwargs)

    return flush_then_call


def _queuing_method(
    record: KernelCallRecord,
    eager_method: Callable,
    eager_specials: type,
    queue: KernelCallQueue,
    flush: Callable,
) -> Callable:
    signature = inspect.signature(eager_method)
    prepare = record.prepare_on_enqueue

    @functools.wraps(eager_method)
    def queue_kernel_call(*args: Any, **kwargs: Any) -> None:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        if prepare is None:
            values = _values_from_kwargs(record, arguments)
        else:
            values = prepare(arguments, eager_specials)
        if values is None:  # this call must run eagerly
            flush()
            eager_method(*args, **kwargs)
            return
        dt, dE = arguments["dt"], arguments["dE"]
        if not queue.holds(dt, dE):
            flush()
            queue.bind(dt, dE)
        for record_values in values:
            queue.append(record, record_values)

    return queue_kernel_call
```

`deferred_class` is referenced inside `flush` before the class exists. That works because `flush` runs only after `type(...)` has returned; `_execute_batch` is looked up at call time, which is also what lets the Review Focus 3 test patch it. Note this in a one-line comment above `flush`.

`KERNEL_CALL_RECORDS` is imported only for documentation: `vars(Specials)` drives the loop. Drop the unused import if ruff complains.

- [ ] **Step 4: Wire up `cpp_deferred`**

In `reload_cpp_backend`, add the parameter `deferred: bool = False` and document it: "If True, return deferred specials (``cpp_deferred``)." Before `return CppSpecials`, add:

```python
    if deferred:
        from blond.core.backends.deferred.kernel_call_queue import (
            deferred_chunk_size,
            make_deferred_specials,
        )

        def execute_batch(batch, record_sizes, dt, dE) -> None:
            _LIBBLOND.execute_kernel_call_batch(
                ct.c_void_p(batch.ctypes.data),
                ct.c_size_t(batch.size),
                _get_pointer(dt),
                _get_pointer(dE),
                _get_beam_len(dt),
                c_index_t(deferred_chunk_size()),
            )

        return make_deferred_specials(CppSpecials, execute_batch)
```

In `NumpyBackend.set_specials`, add `"cpp_deferred"` to the `Literal`, and add this branch:

```python
        elif mode == "cpp_deferred":
            from blond.core.backends.cpp.callables import reload_cpp_backend

            self.specials = reload_cpp_backend(
                self.float, parallel=True, deferred=True
            )
            self.specials_mode = mode
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/unittests/core/backends/deferred -v`

Expected: PASS.

If `test_matches_eager` fails only on `kick_multi_harmonic` with `n_rf=40`, the cause is the summation order across split records, compared with the eager single-pass sum. Because Task 3 made the eager cpp kernel chunk by 32 the same way, the two must agree to `rtol=1e-11`. Any other mismatch is a bug.

- [ ] **Step 6: Commit**

```bash
pre-commit run --files blond/core/backends/deferred/kernel_call_queue.py blond/core/backends/cpp/callables.py blond/core/backends/backend.py tests/unittests/core/backends/deferred/test_deferred_specials.py
git add blond/core/backends/deferred/kernel_call_queue.py blond/core/backends/cpp/callables.py blond/core/backends/backend.py tests/unittests/core/backends/deferred/test_deferred_specials.py
git commit -m "Added cpp_deferred specials backed by a shared kernel call queue" -m "Deferrable kernels now pack generated records instead of running, and every other Specials method flushes first, so results equal eager while kicks and drifts fuse into one pass over the particles. The queue is per thread because one specials object serves all simulations of a process." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Beam accessors, migrating the call sites, and the grep guard

**Files:**
- Modify: `blond/core/beam/base.py`
  - add `kernel_call_dt`, `kernel_call_dE` and `_flush_kernel_calls`;
  - flush in `dt`, `dE`, `read_partial_*` and `write_partial_*`, and in `sort_by_dt` and any other public method that reads `_dt`/`_dE`.
- Modify: `blond/core/beam/beams.py`
  - `dt_min`, `dt_max`, `dE_min`, `dE_max`, `rms_emittance`, `plot_hist2d`, `plot_scatter` and `plot_hist` call `self._flush_kernel_calls()` first;
  - `common_array_size` does not flush, because it reads no coordinates.
- Modify the kernel call sites to `dt=beam.kernel_call_dt, dE=beam.kernel_call_dE`:
  - `physics/rf_station.py:939`, `:1384`, `:1961`
  - `physics/drifts.py:345`, `:693`, `:829`
  - `physics/impedances/base.py:556`
  - `physics/barrier_bucket.py:242`
  - `experimental/physics/kick_pooling.py:214`
- Modify the direct `_dt`/`_dE` users to use the flushing accessors:

  | File | Change |
  |---|---|
  | `physics/profiles.py:411` | `beam.dt.histogram(` |
  | `physics/profiles_sparse.py:467`, `:475` | `beam.read_partial_dt()`, `beam.dt.histogram_sparse(` |
  | `handle_results/observables_as_elements.py:318-321` | `beam.dt.std()`, … |
  | `handle_results/observables.py:471`, `:771`, `:810`, `:981-983` | `self._beam.dt…` |
  | `core/simulation/simulation.py:1073-1086` | `beam.dt…` / `beam.dE…` |
  | `core/simulation/execution_models/single_beam.py:109` | `beam.common_array_size` |
  | `beam_preparation/helpers.py:76-77` | `beam.write_partial_dt()` / `beam.write_partial_dE()` |
  | `specifics/muon_collider/beam_preparation.py:106-107` | `deepcopy(other_beam.dt)` / `deepcopy(other_beam.dE)` |
  | `examples/scripts/EX_20_Acceleration_sparse_profiles.py:105-118` | `beam1.dt.global_size` |
  | `examples/scripts/EX_28_Multiturn_sparse_sps.py:255` | `_bunch.dE.min()` / `.max()` |

- Test: `tests/unittests/core/beam/test_kernel_call_accessors.py`

**Interfaces:**
- Consumes: `backend.specials.flush()` (Task 1), `cpp_deferred` (Task 5).
- Produces:
  - `BeamBaseClass.kernel_call_dt` / `kernel_call_dE`, properties that return `self._dt.array_local` / `self._dE.array_local`;
  - `BeamBaseClass._flush_kernel_calls() -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unittests/core/beam/test_kernel_call_accessors.py
import pathlib
import re
from copy import deepcopy

import numpy as np
import pytest

from blond import Beam, proton
from blond.core.backends.backend import Numpy64Bit, backend
from blond.testing.backend_testing import BLonDTestCase

KICK = dict(voltage=8e3, omega_rf=2e7, phi_rf=0.1, charge=1.0,
            acceleration_kick=12.0)
ROOT = pathlib.Path(__file__).resolve().parents[4] / "blond"


@pytest.mark.backend_mutation
class TestKernelCallAccessors(BLonDTestCase):
    def setUp(self) -> None:
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp_deferred")
        self.beam = Beam(intensity=1e11, particle_type=proton)
        self.beam.setup_beam(
            dt=np.linspace(-1e-9, 1e-9, 100),
            dE=np.linspace(-1e6, 1e6, 100),
        )

    def tearDown(self) -> None:
        backend.specials.flush()
        backend.set_specials("python")

    def _queue_kick(self) -> np.ndarray:
        before = self.beam.kernel_call_dE.copy()
        backend.specials.kick_single_harmonic(
            dt=self.beam.kernel_call_dt, dE=self.beam.kernel_call_dE, **KICK
        )
        return before

    def test_kernel_call_accessors_do_not_flush(self) -> None:
        before = self._queue_kick()
        np.testing.assert_array_equal(self.beam.kernel_call_dE, before)
        self.assertGreater(backend.specials.kernel_call_queue.n_bytes, 0)

    def test_data_accessors_flush(self) -> None:
        for read in (
            lambda: self.beam.read_partial_dE(),
            lambda: self.beam.write_partial_dE(),
            lambda: self.beam.dE.array_local,
            lambda: self.beam.dE_max,
        ):
            before = self._queue_kick()
            read()
            self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)
            self.assertFalse(
                np.array_equal(self.beam.kernel_call_dE, before)
            )

    def test_copy_sees_pending_kicks(self) -> None:  # Review Focus 5
        self._queue_kick()
        copied = deepcopy(self.beam.dE)  # flushing accessor, as muon code
        np.testing.assert_array_equal(
            copied.array_local, self.beam.kernel_call_dE
        )


class TestNoDirectCoordinateAccess(BLonDTestCase):
    def test_no_direct_dt_dE_outside_core_beam(self) -> None:
        pattern = re.compile(r"\._d(t|E)\b")
        allowed = ("legacy", "experimental", "core/beam", "core/backends")
        hits = []
        for path in ROOT.rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith(allowed):
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line) and "`" not in line:
                    hits.append(f"{relative}:{number}: {line.strip()}")
        self.assertEqual(hits, [], "use the Beam accessors:\n" +
                         "\n".join(hits))
```

The backtick exclusion skips docstring mentions such as ``Mutates ``beam._dE`` in place`` (`synchrotron_radiation/base.py:104`). Update that docstring anyway, to say `beam.kernel_call_dE`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unittests/core/beam/test_kernel_call_accessors.py -v`

Expected:
- FAIL with `AttributeError: 'Beam' object has no attribute 'kernel_call_dE'`;
- the grep test FAILS, listing about 20 hits.

- [ ] **Step 3: Implement the accessors in `base.py`**

Add them next to `read_partial_dt`:

```python
    @property
    def kernel_call_dt(self) -> NumpyArray | CupyArray:
        """
        Local dt-array to pass to a `Specials` kernel call, in [s].

        Unlike `read_partial_dt`, this does not run queued kernel calls
        first: passing it to a deferred kernel must not flush the queue
        the call is about to join. Use it only as a kernel argument.

        Returns
        -------
        dt
            Dt-array on the current node, in [s].
        """
        return self._dt.array_local

    @property
    def kernel_call_dE(self) -> NumpyArray | CupyArray:
        """
        Local dE-array to pass to a `Specials` kernel call, in [eV].

        Unlike `read_partial_dE`, this does not run queued kernel calls
        first: passing it to a deferred kernel must not flush the queue
        the call is about to join. Use it only as a kernel argument.

        Returns
        -------
        dE
            DE-array on the current node, in [eV].
        """
        return self._dE.array_local

    @staticmethod
    def _flush_kernel_calls() -> None:
        """Run queued kernel calls before Python reads the coordinates."""
        from blond.core.backends.backend import backend

        backend.specials.flush()
```

Add `self._flush_kernel_calls()` as the first statement of these bodies:
- the `dE`, `dt` and `flags` properties;
- `read_partial_ids`, `read_partial_dt`, `write_partial_dt`, `read_partial_dE`, `write_partial_dE`, `read_partial_flags` and `write_partial_flags`;
- `sort_by_dt`;
- in `beams.py`: `dt_min`, `dt_max`, `dE_min`, `dE_max`, `rms_emittance`, `plot_hist2d`, `plot_scatter` and `plot_hist`.

In each `read_partial_*` / `write_partial_*` docstring `Notes`, add the line: "Runs queued (deferred) kernel calls first; kernel arguments use `kernel_call_dt` / `kernel_call_dE`."

- [ ] **Step 4: Migrate the call sites and the direct users**

Apply the table under **Files** above.

In `rf_station.py:939`, for example:

```python
            backend.specials.kick_interpolated(
                dt=beam.kernel_call_dt,
                dE=beam.kernel_call_dE,
```

- [ ] **Step 5: Run the tests and the physics suite**

Run: `.venv/bin/python -m pytest tests/unittests/core/beam tests/unittests/physics tests/unittests/handle_results -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
pre-commit run --files $(git diff --name-only) tests/unittests/core/beam/test_kernel_call_accessors.py
git add -u blond tests/unittests/core/beam/test_kernel_call_accessors.py
git commit -m "Added kernel_call_dt/dE and made the Beam data accessors flush" -m "Kernel call sites used read/write_partial_*, which must now flush so Python never reads stale coordinates; giving kernel arguments their own non-flushing accessors keeps deferred calls queued. Direct _dt/_dE access outside core/beam is gone and a test keeps it that way." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Flush points in the main loops

**Files:**
- Modify: `blond/core/simulation/execution_models/base.py`. Add `flush_before_readout(observe, callbacks, turn_i) -> None`.
- Modify: `blond/core/simulation/execution_models/single_beam.py`. The loop is at lines 112–140.
- Modify: `blond/core/simulation/execution_models/conterrotating_beams.py`. Apply the same three points.
- Test: `tests/unittests/core/simulation/execution_models/test_deferred_mainloop.py`. Create `tests/unittests/core/simulation/execution_models/__init__.py` if it is missing.

**Interfaces:**
- Consumes: `cpp_deferred` (Task 5); the accessors (Task 6).
- Produces: `flush_before_readout(observe: Sequence, callbacks: Sequence, turn_i: int) -> None`.

- [ ] **Step 1: Write the failing test**

Build the EX_23 simulation with a smaller setup: 1e4 macroparticles, `n_bins=1000`, 5 turns. Run it once with `cpp` and once with `cpp_deferred`, each with a callback that records `beam.read_partial_dE().copy()` at every turn.

```python
# tests/unittests/core/simulation/execution_models/test_deferred_mainloop.py
import numpy as np
import pytest

from blond.core.backends.backend import Numpy64Bit, backend
from blond.testing.backend_testing import BLonDTestCase


def _run(mode):
    backend.change_backend(Numpy64Bit)
    backend.set_specials(mode)
    from blond.examples.scripts import EX_23_Main_long_ps_booster as ex

    seen = []

    def record(simulation, beam):
        seen.append(beam.read_partial_dE().copy())

    record.each_turn_i = 1
    sim, beam = ex.build(n_macroparticles=10_000, n_bins=1000)  # Step 3
    sim.run_simulation(beams=(beam,), n_turns=5, callbacks=[record])
    return seen, beam.read_partial_dt().copy()


@pytest.mark.backend_mutation
class TestDeferredMainloop(BLonDTestCase):
    def tearDown(self) -> None:
        backend.specials.flush()
        backend.set_specials("python")

    def test_deferred_matches_eager_and_callbacks_see_flushed_beam(self):
        eager_turns, eager_dt = _run("cpp")
        deferred_turns, deferred_dt = _run("cpp_deferred")
        self.assertEqual(len(eager_turns), len(deferred_turns))
        for eager, deferred in zip(eager_turns, deferred_turns):
            np.testing.assert_allclose(deferred, eager, rtol=1e-12)
        np.testing.assert_allclose(deferred_dt, eager_dt, rtol=1e-12)
        self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unittests/core/simulation/execution_models/test_deferred_mainloop.py -v`

Expected: FAIL with `AttributeError: module ... has no attribute 'build'`.

- [ ] **Step 3: Factor out `build()` in EX_23**

Split `main()` in `blond/examples/scripts/EX_23_Main_long_ps_booster.py`:
- add `build(n_macroparticles=1001, n_bins=10_000) -> tuple[Simulation, Beam]`, which contains everything up to `prepare_beam`;
- keep `main()`, which calls `build()` and then `run_simulation(n_turns=2)`.

The example's behaviour does not change.

Run the test again. Expected: it may already PASS, because the callback reads through a flushing accessor. It must also pass *after* Step 4. The final assertion (`n_bytes == 0`) is the one that needs Step 4: the end-of-loop flush.

- [ ] **Step 4: Add the flush points**

In `execution_models/base.py`:

```python
def flush_before_readout(
    observe: Sequence, callbacks: Sequence, turn_i: int
) -> None:
    """
    Run queued kernel calls if anything reads the beam this turn.

    Without an active observable or callback the queue may span the turn
    boundary, fusing the end of one turn with the start of the next.

    Parameters
    ----------
    observe
        Observables of the main loop.
    callbacks
        Callbacks of the main loop, each with ``each_turn_i``.
    turn_i
        Current turn.
    """
    from blond.core.backends.backend import backend

    if any(o.is_active_this_turn(turn_i=turn_i) for o in observe) or any(
        turn_i % callback.each_turn_i == 0 for callback in callbacks
    ):
        backend.specials.flush()
```

In `single_beam.py`, make three changes:
1. Replace `return` inside the element loop with `backend.specials.flush()` followed by `return`.
2. Call `flush_before_readout(observe, callbacks, simulation.turn_counter.value)` after the element loop and before `for observable in observe:`.
3. Call `backend.specials.flush()` after the `for turn_i` loop and before `simulation.turn_counter.value += 1`.

Apply the same three changes in `conterrotating_beams.py`.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/unittests/core/simulation -q && .venv/bin/python blond/examples/scripts/EX_23_Main_long_ps_booster.py`

Expected: PASS, and the example exits with 0.

- [ ] **Step 6: Commit**

```bash
pre-commit run --files $(git diff --name-only) tests/unittests/core/simulation/execution_models/test_deferred_mainloop.py
git add -u blond tests/unittests/core/simulation/execution_models
git commit -m "Flushed queued kernel calls at the main loop's readout points" -m "Observables and callbacks must see the tracked beam, and the loop must leave nothing queued when it returns; when nothing reads the beam the queue may span turns." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: `drift_exact` takes host coefficients

**Files:**
- Modify: `blond/physics/drifts.py:684-689`. Build `higher_alpha` with numpy.
- Modify: `blond/core/backends/cuda/callables.py`, `drift_exact` (around line 355). Accept a host array and copy it to the device.
- Test: `tests/unittests/core/backends/test_backend.py`. The existing `drift_exact` tests pass `higher_alpha` built with `backend.array`; add a host-array case.

**Interfaces:**
- Produces: `Specials.drift_exact(higher_alpha: NumpyArray)` is a host array on **every** backend, matching the ABC annotation.

- [ ] **Step 1: Write the failing test**

Add to `TestSpecials` in `test_backend.py` (the class that has `special_modes`):

```python
    @pytest.mark.backend_mutation
    def test_drift_exact_host_coefficients(self) -> None:
        reference = None
        for special in self.special_modes:
            self._setUp(dtype=np.float64, special_mode=special)
            higher_alpha = np.array([1e-3, 2e-3])  # host, on every backend
            backend.specials.drift_exact(
                dt=self.dt, dE=self.dE, T=self.t_rev, alpha_0=self.alpha_0,
                higher_alpha=higher_alpha, beta=self.beta,
                energy=self.energy,
            )
            result = copy_to_cpu(self.dt)
            if reference is None:
                reference = result
            else:
                np.testing.assert_allclose(result, reference, rtol=1e-12,
                                           err_msg=special)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/unittests/core/backends/test_backend.py -k drift_exact_host -v`

Expected: FAIL for `cuda` with `AssertionError`, raised by `assert higher_alpha.device != "cpu"` or by the dtype/attribute check.

- [ ] **Step 3: Implement**

In `CudaSpecials.drift_exact`:
- replace the device assertions on `higher_alpha` with `assert not is_cupy_array(higher_alpha) or higher_alpha.dtype == FLOAT`;
- then add `higher_alpha = cp.asarray(higher_alpha, dtype=FLOAT)` with the comment `# host coefficients, as the ABC declares; one tiny copy per call, as the caller used to do`.

In `drifts.py`, replace `backend.array(...)` with `np.asarray(..., dtype=backend.float)`.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/unittests/core/backends/test_backend.py -k drift_exact -v && .venv/bin/python -m pytest tests/unittests/physics -q -k drift`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
pre-commit run --files blond/physics/drifts.py blond/core/backends/cuda/callables.py tests/unittests/core/backends/test_backend.py
git add blond/physics/drifts.py blond/core/backends/cuda/callables.py tests/unittests/core/backends/test_backend.py
git commit -m "Passed drift_exact coefficients as a host array on every backend" -m "The ABC declares a NumpyArray but the CUDA wrapper required a device array; host coefficients let the deferred queue inline them without a transfer, and eager CUDA does the same one copy the caller did before." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: CUDA per-particle device functions

**Files:**
- Modify: `blond/core/backends/cuda/kernels.cu`. Include the header; add `apply_to_particle` overloads and `build_voltage_kick_table`; make the six eager kernels call them.
- Modify: `blond/core/backends/cuda/callables.py`. Add `CudaSpecials._build_voltage_kick_table`.
- Test: `tests/unittests/core/backends/deferred/test_cuda_voltage_kick_table.py`

**Interfaces:**
- Consumes: `kernel_call_records.h` (Task 2).
- Produces:
  - `__device__ void apply_to_particle(const XArgs&, real_t &dt, real_t &dE)` for all six `Args`;
  - `extern "C" __global__ void build_voltage_kick_table(const real_t *voltage, const real_t *bin_centers, real_t charge, int n_slices, real_t acc_kick, real_t *table)`, in the same layout as Task 3;
  - `CudaSpecials._build_voltage_kick_table(voltage, bin_centers, charge, acceleration_kick) -> CupyArray`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unittests/core/backends/deferred/test_cuda_voltage_kick_table.py
import numpy as np
import pytest

from blond.core.backends.backend import Numpy64Bit, backend
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.testing.backend_testing import BLonDTestCase, skip_if_no_cupy


@pytest.mark.cupy
@pytest.mark.backend_mutation
class TestCudaVoltageKickTable(BLonDTestCase):
    def tearDown(self) -> None:
        backend.change_backend(Numpy64Bit)
        backend.set_specials("python")

    @skip_if_no_cupy
    def test_matches_cpp_table(self) -> None:
        bin_centers = np.linspace(-1e-9, 1e-9, 50)
        voltage = np.sin(np.linspace(0, 3, 50)) * 1e3
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")
        expected = backend.specials._build_voltage_kick_table(
            voltage=voltage, bin_centers=bin_centers, charge=2.0,
            acceleration_kick=0.5)
        from blond.core.backends.backend import Cupy64Bit
        backend.change_backend(Cupy64Bit)
        table = backend.specials._build_voltage_kick_table(
            voltage=backend.array(voltage),
            bin_centers=backend.array(bin_centers),
            charge=2.0, acceleration_kick=0.5)
        np.testing.assert_allclose(copy_to_cpu(table), expected,
                                   rtol=1e-13)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/unittests/core/backends/deferred/test_cuda_voltage_kick_table.py -v`

Expected: FAIL with `AttributeError: ... '_build_voltage_kick_table'`.

- [ ] **Step 3: Implement it in `kernels.cu`**

Immediately after the `index_t` typedef, add `#include "kernel_call_records.h"`. The header needs `real_t` and `index_t` first.

Add the overloads before the eager kernels. Each body is the loop body of its eager kernel, **verbatim**, working on the scalars `dt` / `dE`. For example:

```cpp
__device__ __forceinline__ void
apply_to_particle(const DriftSimpleArgs &args, real_t &dt, real_t &dE) {
  dt += args.T * args.eta_0 / (args.beta * args.beta * args.energy) * dE;
}
```

The eager `drift_simple` hoists `coeff` out of the loop. To keep the eager arithmetic identical, write it as `const real_t coeff = T * eta_zero / (beta * beta * energy); dt += coeff * dE;` inside the device function too. nvcc hoists the loop-invariant product when it inlines, so the result is identical.

Then:
- **Kicks:** `kick_single_harmonic` becomes `dE += charge * voltage * sin(omega_rf * dt + phi_rf) + acc_kick`. `kick_multi_harmonic` keeps the existing CUDA form: start from `acc_kick`, then add `charge * v[j] * sin(...)` for `j < args.n_rf`.
- **`drift_like_line_segment`:** a verbatim copy.
- **`drift_exact`:** the verbatim loop over `args.n_alpha`, using `args.higher_alpha`.
- **`KickInterpolatedArgs`:**

```cpp
__device__ __forceinline__ void
apply_to_particle(const KickInterpolatedArgs &args, real_t &dt, real_t &dE) {
  const real_t *table = args.voltage_kick_table;
  const int n_bins = static_cast<int>((args.voltage_kick_table_length - 2) / 2);
  // Range-check before the conversion to `int` (see `hybrid_histogram`).
  const real_t fbin_real = floor((dt - table[0]) * table[1]);
  if (fbin_real >= real_t(0) && fbin_real < real_t(n_bins)) {
    const int pair = 2 + 2 * static_cast<int>(fbin_real);
    dE += dt * table[pair] + table[pair + 1];
  } else {
    dE += args.acceleration_kick;
  }
}
```

The eager kernels call them. For `drift_simple`:

```cpp
extern "C" __global__ void drift_simple(real_t *__restrict__ beam_dt,
                                        const real_t *__restrict__ beam_dE,
                                        const real_t T, const real_t eta_zero,
                                        const real_t beta, const real_t energy,
                                        const index_t n_macroparticles) {
  const DriftSimpleArgs args = {T, eta_zero, beta, energy};
  for (index_t i = particle_loop_start(); i < n_macroparticles;
       i += particle_loop_stride()) {
    real_t dE = beam_dE[i];
    apply_to_particle(args, beam_dt[i], dE);
  }
}
```

Do the same for `drift_like_line_segment`, `drift_exact`, `kick_single_harmonic` and `kick_multi_harmonic`:
- `drift_exact` copies `higher_alpha[0..n_alpha)` into `args.higher_alpha` when `n_alpha <= 8`, and otherwise keeps its existing loop.
- `kick_multi_harmonic` copies `rf_params_batch` into the `Args` inline arrays.

Leave the eager dense `lik_only_gm_comp` unchanged. It reads the old pair layout.

`build_voltage_kick_table` reuses the loop body of `lik_only_gm_copy`. Factor that body into a `__device__` helper, `voltage_kick_pair(i, voltage, bin_centers, charge, inv_bin_width, acc_kick, slope&, offset&)`, and call it from both kernels:

```cpp
extern "C" __global__ void build_voltage_kick_table(
    const real_t *__restrict__ voltage_array,
    const real_t *__restrict__ bin_centers, const real_t charge,
    const int n_slices, const real_t acc_kick, real_t *__restrict__ table) {
  const int tid = static_cast<int>(threadIdx.x + blockDim.x * blockIdx.x);
  const unsigned int stride = gridDim.x * blockDim.x;
  const real_t inv_bin_width =
      (n_slices - 1) / (bin_centers[n_slices - 1] - bin_centers[0]);
  if (tid == 0) {
    table[0] = bin_centers[0];
    table[1] = inv_bin_width;
  }
  for (int i = tid; i < n_slices - 1; i = static_cast<int>(i + stride)) {
    voltage_kick_pair(i, voltage_array, bin_centers, charge, inv_bin_width,
                      acc_kick, table[2 + 2 * i], table[3 + 2 * i]);
  }
}
```

In `cuda/callables.py`:
- load it with `_build_voltage_kick_table_kernel = gpu_module.get_function("build_voltage_kick_table")`;
- add the static method `_build_voltage_kick_table` to `CudaSpecials`. It allocates `cp.empty(2 * n, FLOAT)`, launches the kernel with `grid_size`/`block_size`, and returns the table.

- [ ] **Step 4: Rebuild and run the tests**

Run: `.venv/bin/python -m blond.core.backends.cuda.compile && BLOND_FORCE_TEST_ALL_BACKENDS=True .venv/bin/python -m pytest tests/unittests/core/backends/test_backend.py tests/unittests/core/backends/deferred/test_cuda_voltage_kick_table.py -q`

Expected: PASS. `test_backend.py` guards the eager CUDA refactor against the python reference.

- [ ] **Step 5: Check that eager CUDA speed has not changed**

Run `.venv/bin/python dev_tools/performance_blond3/backends/kick.py`, and likewise `drift.py`, with the CUDA mode both before (`git stash`) and after. Expect the times to stay within noise, and paste both into the commit body.

- [ ] **Step 6: Commit**

```bash
pre-commit run --files blond/core/backends/cuda/kernels.cu blond/core/backends/cuda/callables.py tests/unittests/core/backends/deferred/test_cuda_voltage_kick_table.py
git add blond/core/backends/cuda tests/unittests/core/backends/deferred/test_cuda_voltage_kick_table.py
git commit -m "Moved the CUDA particle kernel bodies into apply_to_particle overloads" -m "The fused deferred kernel and the eager kernels must share one formula per kernel within CUDA. <paste before/after timings>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: The fused CUDA kernel and `cuda_deferred`

**Files:**
- Modify: `blond/core/backends/cuda/kernels.cu`
  - add `KernelCallBatch` and `execute_kernel_call_batch`;
  - add `kernel_call_args_sizes`.
- Modify: `blond/core/backends/cuda/compile.py`
  - add `-Werror=switch` only if nvcc accepts it; otherwise rely on the generated switch;
  - keep `-maxrregcount 32` global, and give the fused kernel `__launch_bounds__`.
- Modify: `blond/core/backends/cuda/callables.py`
  - add `_execute_kernel_call_batch`, the ABI check, and `CudaDeferredSpecials`.
- Modify: `blond/core/backends/backend.py`
  - `CupyBackend.set_specials` accepts `"cuda_deferred"`;
  - `_backend_class_for_mode` maps `"cuda_deferred"` to `Cupy64Bit`.
- Test: `tests/unittests/core/backends/deferred/test_deferred_specials.py`. Add a CUDA subclass.

**Interfaces:**
- Consumes:
  - the Task 9 overloads and `_build_voltage_kick_table`;
  - `make_deferred_specials`, `KERNEL_CALL_BATCH_CAPACITY_BYTES` (Tasks 2 and 5).
- Produces:
  - `CudaDeferredSpecials`;
  - `set_specials("cuda_deferred")`;
  - `_split_batch(record_sizes, capacity) -> list[tuple[int, int]]`, the byte ranges for each launch.

- [ ] **Step 1: Write the failing tests**

Append to `test_deferred_specials.py`:

```python
from blond.testing.backend_testing import cupy_available


@pytest.mark.cupy
@pytest.mark.backend_mutation
class TestCudaDeferredSpecials(TestCppDeferredSpecials):
    """Every cpp_deferred test, rerun on the GPU."""

    mode = "cuda_deferred"
    eager_mode = "cuda"

    def setUp(self) -> None:
        if not cupy_available:
            self.skipTest("CuPy is not available")
        from blond.core.backends.backend import Cupy64Bit

        backend.change_backend(Cupy64Bit)
        backend.set_specials(self.eager_mode)
        self.eager = backend.specials
        backend.set_specials(self.mode)
        self.deferred = backend.specials

    def tearDown(self) -> None:
        self.deferred.flush()
        backend.change_backend(Numpy64Bit)
        backend.set_specials("python")

    def test_chunk_sizes(self) -> None:
        self.skipTest("the chunk size exists on the cpp executor only")
```

The Task 5 helpers (`_beam`, `_close`, `_equal`, host RF/alpha arrays, backend arrays for the interpolated kick) are already device-agnostic, so the whole cpp suite reruns unchanged. Check that `cupy_available` is exported by `blond/testing/backend_testing.py` (it is used by `skip_if_no_cupy`); import it from wherever that module takes it.

Also add these CUDA-only tests to the subclass:

```python
    def test_batch_larger_than_capacity_is_split(self) -> None:
        # 40 harmonics x 2 = 4 multi-harmonic records (~800 B) + more
        dt, dE = _beam(1000)
        dt_e, dE_e = backend.copy(dt), backend.copy(dE)
        for _ in range(3):
            _turn(self.deferred, dt, dE, n_rf=40)
            _turn(self.eager, dt_e, dE_e, n_rf=40)
        self.assertGreater(
            self.deferred.kernel_call_queue.n_bytes, 4096
        )
        self.deferred.flush()
        np.testing.assert_allclose(copy_to_cpu(dE), copy_to_cpu(dE_e),
                                   rtol=1e-11, atol=1e-6)

    def test_split_batch_ranges(self) -> None:
        from blond.core.backends.cuda.callables import _split_batch
        self.assertEqual(
            _split_batch([1000, 1000, 1000, 2000, 100], 4096),
            [(0, 3000), (3000, 5100)],
        )

    def test_zero_macroparticles(self) -> None:  # Review Focus 4
        dt, dE = backend.zeros(0), backend.zeros(0)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **KICK)
        self.deferred.flush()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `BLOND_FORCE_TEST_ALL_BACKENDS=True .venv/bin/python -m pytest tests/unittests/core/backends/deferred/test_deferred_specials.py -v -k Cuda`

Expected: FAIL with `UnknownBackendMode: Unknown specials mode 'cuda_deferred'`.

- [ ] **Step 3: Implement the fused kernel**

```cpp
// A batch of kernel call records, passed by value in the kernel's
// parameter space like `RFParamsBatch`: no host-to-device copy per
// flush. 8-byte slots keep every record 8-byte aligned.
// NOLINTBEGIN(*-avoid-c-arrays,misc-use-internal-linkage)
struct KernelCallBatch {
  unsigned long long slots[KERNEL_CALL_BATCH_CAPACITY_BYTES / 8];
};
// NOLINTEND(*-avoid-c-arrays,misc-use-internal-linkage)

// Compiled Args sizes, compared with the numpy dtypes when loading.
extern "C" __device__ const unsigned int
    kernel_call_args_sizes[KERNEL_COUNT] = KERNEL_CALL_ARGS_SIZES_INITIALIZER;

namespace {
struct ApplyToParticle {
  real_t *dt;
  real_t *dE;
  template <class Args> __device__ void operator()(const Args &args) const {
    apply_to_particle(args, *dt, *dE);
  }
};
} // namespace

// Every record of the batch on every particle, dt/dE kept in registers
// between records. All threads of a warp read the same record, so the
// switch in visit_kernel_call does not diverge.
extern "C" __global__ void __launch_bounds__(256)
    execute_kernel_call_batch(const KernelCallBatch batch,
                              const unsigned int n_bytes,
                              real_t *__restrict__ beam_dt,
                              real_t *__restrict__ beam_dE,
                              const index_t n_macroparticles) {
  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
  const auto *bytes = reinterpret_cast<const char *>(batch.slots);
  const auto *first = reinterpret_cast<const KernelCallHeader *>(bytes);
  const auto *last = reinterpret_cast<const KernelCallHeader *>(bytes + n_bytes);
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
  for (index_t i = particle_loop_start(); i < n_macroparticles;
       i += particle_loop_stride()) {
    real_t dt = beam_dt[i];
    real_t dE = beam_dE[i];
    const ApplyToParticle apply = {&dt, &dE};
    for (const KernelCallHeader *record = first; record != last;
         record = next_record(record)) {
      visit_kernel_call(record, apply);
    }
    beam_dt[i] = dt;
    beam_dE[i] = dE;
  }
}
```

`__launch_bounds__(256)`: the launch below must use ≤256 threads per block for this kernel. Check the spills with:

`nvcc --cubin -O3 --use_fast_math -arch sm_75 -Xptxas -v -I blond/core/backends/deferred -I <cupy include> blond/core/backends/cuda/kernels.cu -o /dev/null 2>&1 | grep -A2 execute_kernel_call_batch`

Expected: `0 bytes spill stores`. If it reports spills under the global `-maxrregcount 32`, then:
1. drop `-maxrregcount` from `cuda/compile.py`;
2. add `__launch_bounds__(1024, 1)` to the eager kernels that relied on it, so their limit stays;
3. re-measure the eager speed as in Task 9, Step 5.

- [ ] **Step 4: Implement the Python side**

In `cuda/callables.py`:

```python
from blond.core.backends.deferred.kernel_call_records import (
    KERNEL_CALL_BATCH_CAPACITY_BYTES,
    KERNEL_CALL_RECORDS,
    args_dtype,
)
from blond.core.backends.deferred.kernel_call_queue import (
    make_deferred_specials,
)

_execute_kernel_call_batch_kernel = gpu_module.get_function(
    "execute_kernel_call_batch"
)
_KERNEL_CALL_BATCH_DTYPE = np.dtype(
    [("slots", np.uint64, (KERNEL_CALL_BATCH_CAPACITY_BYTES // 8,))]
)
_deferred_block_size = (min(threads, 256), 1, 1)  # __launch_bounds__(256)


def _check_kernel_call_record_abi() -> None:
    sizes = cp.ndarray(
        (len(KERNEL_CALL_RECORDS),), dtype=np.uint32,
        memptr=gpu_module.get_global("kernel_call_args_sizes"),
    ).get()
    for record in KERNEL_CALL_RECORDS:
        assert sizes[record.kernel_id] == args_dtype(record).itemsize, (
            f"{record.kernel_id_name}Args differs between the cubin and "
            "kernel_call_records.py; rebuild with `blond-compile-cuda`."
        )


_check_kernel_call_record_abi()


def _split_batch(
    record_sizes: list[int], capacity: int
) -> list[tuple[int, int]]:
    """Byte ranges of consecutive records, each fitting `capacity`."""
    ranges, start, end = [], 0, 0
    for size in record_sizes:
        if end + size - start > capacity:
            ranges.append((start, end))
            start = end
        end += size
    ranges.append((start, end))
    return ranges


def _execute_batch(batch, record_sizes, dt, dE) -> None:
    for start, end in _split_batch(
        record_sizes, KERNEL_CALL_BATCH_CAPACITY_BYTES
    ):
        parameters = np.zeros((), dtype=_KERNEL_CALL_BATCH_DTYPE)
        parameters["slots"].view(np.uint8)[: end - start] = batch[start:end]
        _execute_kernel_call_batch_kernel(
            args=(
                parameters,
                np.uint32(end - start),
                dt,
                dE,
                INDEX_DTYPE(dt.size),
            ),
            grid=grid_size,
            block=_deferred_block_size,
        )
```

After the `CudaSpecials` class, add:

```python
CudaDeferredSpecials = make_deferred_specials(CudaSpecials, _execute_batch)
```

No single record may exceed the capacity. The largest is multi-harmonic at about 800 B, so this holds. Add `assert size <= capacity` in `_split_batch`.

In `backend.py`:
- `_backend_class_for_mode`: `return Cupy64Bit if mode.lower() in ("cuda", "cuda_deferred") else Numpy64Bit`.
- `CupyBackend.set_specials`: `Literal["cuda", "cuda_deferred"]`, plus this branch:

```python
        elif mode == "cuda_deferred":
            from blond.core.backends.cuda.callables import (
                CudaDeferredSpecials,
            )

            self.specials = CudaDeferredSpecials()
```

Also set `self.specials_mode = mode` in both branches if `specials_mode` is tracked there. Check how `CupyBackend` tracks it and follow that.

- [ ] **Step 5: Rebuild and run the tests**

Run: `.venv/bin/python -m blond.core.backends.cuda.compile && BLOND_FORCE_TEST_ALL_BACKENDS=True .venv/bin/python -m pytest tests/unittests/core/backends -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
pre-commit run --files blond/core/backends/cuda/kernels.cu blond/core/backends/cuda/compile.py blond/core/backends/cuda/callables.py blond/core/backends/backend.py blond/core/backends/deferred/* tests/unittests/core/backends/deferred/test_deferred_specials.py
git add -u blond tests
git commit -m "Added cuda_deferred specials with a fused interpreter kernel" -m "Queuing alone gains nothing on the GPU; one launch that applies every queued record to a particle held in registers removes the per-kernel dt/dE round trips. The batch travels by value like RFParamsBatch and is split over launches only beyond 4 KB." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Benchmark, docs and full verification

**Files:**
- Create: `dev_tools/performance_blond3/backends/deferred_psb.py`
- Modify: the docstrings of both `set_specials` methods (list the new modes). Also update any backend-modes doc page: grep `docs/` for `cpp_single_core` to find where the modes are listed and add the two new ones.
- Modify: `.agents/skills/blond-dev/SKILL.md`, *Backend conventions*. Add one bullet about deferred specials and `kernel_call_dt/dE`. Do **not** edit the root `CLAUDE.md`; the `sync` pre-commit hook regenerates it.

- [ ] **Step 1: Write the benchmark**

Base it on EX_23 `build()` from Task 7.
- Arguments: `mode` (`cpp` | `cpp_deferred` | `cuda` | `cuda_deferred`), `n_macroparticles` (default 1e7) and `n_turns` (default 30).
- Warm up with 3 turns first.
- Print ms/turn, flushes per turn (count them by wrapping `flush`), and a checksum of `dE`.
- Add the option `--one-histogram`, which sets `track_profile=False` on both wakefields. This reproduces the spec §9 rows.

- [ ] **Step 2: Run the benchmark**

Run each configuration 3 times, interleaved, on the CPU (12 threads) and on the T400:

```bash
for r in 1 2 3; do for m in cpp cpp_deferred cuda cuda_deferred; do
  .venv/bin/python dev_tools/performance_blond3/backends/deferred_psb.py $m --one-histogram; done; done
```

Expected on the CPU: `cpp_deferred` at about 18–19 ms/turn, against 43–48 ms for `cpp` (spec §9). The checksums must be identical across modes on the same device. Record the GPU numbers, whatever they are. Also record a single-core cycle count for `cpp` vs `cpp_deferred` with `OMP_NUM_THREADS=1 perf stat -e cycles`. Mind the memory note: `import blond` overrides `OMP_NUM_THREADS`, so pin threads through BLonD's own mechanism.

- [ ] **Step 3: Run the full verification**

```bash
BLOND_FORCE_TEST_ALL_BACKENDS=True .venv/bin/python -m pytest tests/unittests/ -q
pre-commit run --all-files
cd docs && bash create_docs.sh
.venv/bin/python dev_tools/run_clang_tidy.py
```

Expected: all green. Report any failure verbatim; don't paper over it.

- [ ] **Step 4: Commit**

```bash
pre-commit run --files dev_tools/performance_blond3/backends/deferred_psb.py blond/core/backends/backend.py .agents/skills/blond-dev/SKILL.md
git add dev_tools/performance_blond3/backends/deferred_psb.py -u
git commit -m "Added the deferred PSB benchmark and documented the deferred specials" -m "<paste the benchmark table: CPU and T400, eager vs deferred, 1 and 3 histograms per turn>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Self-Review Notes

**Spec coverage.** Each spec section maps to a task:

| Spec section | Task(s) |
|---|---|
| §3 Scope and flush-then-call | 5 |
| §4.1–4.3 Records | 2 |
| §4.4 Guards | 2 (header check), 4 (cpp size check), 10 (cuda size check) |
| §4.5 Checklist | 2 (module docstring) |
| §5.1 Per-thread queue | 5 |
| §5.1a MPI | 6 (Beam readers flush) and 5 (`sum_1d_array` / `dot_product_1d_array` are flush-then-call) |
| §5.2 Queue | 5 |
| §5.3 Execute per backend | 5, 10 |
| §5.4 Flush points | 1, 6, 7 |
| §6 CPU executor | 3, 4 |
| §7 CUDA executor | 9, 10 |
| §8 Tests | spread over the tasks |
| §9 Benchmark | 11 |
| §10 Risks | 3, 9 (codegen benchmarks), 10 (spills), 6 (grep guard) |

**Deviations from the spec, deliberate.**
- **No `input_array` snapshot.** The only `input_array`, `voltage_kick_table`, is always a fresh array built by its hook. That is exactly the spec's "unless the queue created it itself" case, so no snapshot code exists.
- **Beam binding compares arrays by identity (`is`),** not by address. A view is therefore a different beam (it flushes), which is correct.
- **The cpp eager `kick_multi_harmonic` now also chunks by 32.** It has to build inline `Args`. Summation order for `n_rf > 32` changes, within `rtol=1e-12`; mention this in the Task 3 commit.
