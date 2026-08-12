# Sparse-Aware `kick_interpolated` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `kick_interpolated` produce correct physics when handed a
sparse, multi-island `EquidistantMultiProfile.hist_x`, and make it raise a
clear error instead of silently computing wrong physics for any other
non-uniform `bin_centers`.

**Architecture:** `kick_interpolated` keeps its existing signature and gains
six new **optional** keyword arguments carrying the same island metadata
`histogram_sparse` already consumes (`first_left_cut`, `left_cut_distance`,
`cut_width`, `bins_per_profile`, `filling_pattern`,
`bucket_index_to_memory_index`). Omitted → today's dense path, now guarded by
a one-time (not per-particle) uniform-spacing check that raises `ValueError`
instead of silently misbehaving. Provided → a new per-particle bucket
resolution step (identical structure to `histogram_sparse`) replaces the
naive global `floor()` index, while reusing the *same* segment-interpolation
math (`voltageKick`/`factor`) as the dense path — the two paths differ only
in how a particle's array index is resolved, not in the interpolation
formula itself. Implemented identically (backend parity) in
`python`/`numba`/`cpp`/`cuda`.

**Tech Stack:** Python, Numba (`njit`/`prange`), C++ (OpenMP, ctypes), CUDA
(RawModule kernels, CuPy).

## Global Constraints

- Python ≥3.10, line length 79 (ruff formats; don't crush variable names to fit).
- Branch: `blonder_bugfix/sparse-interp` (already checked out). Never commit to `blonder` directly.
- Strict TDD: write the failing test, run it, show RED, then implement.
- `pre-commit run --all-files` (or `--files <changed>`) must pass before every `git commit`.
- Backend parity is mandatory: any kernel signature change must land in `python`, `numba`, `cpp`, and `cuda` — not just one.
- Run backend-touching tests with `BLOND_FORCE_TEST_ALL_BACKENDS=True`.
- `assert` is for hot-loop dtype/contiguity checks only (stripped under `-O`); the new uniform-spacing guard is a correctness check and must be `raise ValueError`, not `assert`.
- NumPy-style docstrings (numpydoc-validation enforced) on all touched public methods.
- Commit messages: past tense, body explains why.

---

### Task 1: Uniform-spacing guard on the dense path (all 4 backends)

**Files:**
- Modify: `blond/core/backends/python/callables.py:379-421` (`PythonSpecials.kick_interpolated`)
- Modify: `blond/core/backends/numba/callables.py:148-155,536-562` (`sig_kick_interpolated`, `NumbaSpecials.kick_interpolated`)
- Modify: `blond/core/backends/cpp/callables.py:247-278` (`CppSpecials.kick_interpolated`)
- Modify: `blond/core/backends/cuda/callables.py:335-381` (`CudaSpecials.kick_interpolated`)
- Test: `tests/unittests/core/backends/test_backend.py`

**Interfaces:**
- Produces: `kick_interpolated(dt, dE, voltage, bin_centers, charge, acceleration_kick)` now raises `ValueError` when `bin_centers` is not uniformly spaced. Behavior for uniformly-spaced `bin_centers` is unchanged (verified by existing `test_kick_interpolated`, `test_kick_interpolated_edges`, `test_kick_interpolated_far_outside_window`, which must keep passing).

- [ ] **Step 1: Write the failing test**

Add to `tests/unittests/core/backends/test_backend.py`, in `TestSpecials`, right
after `test_kick_interpolated_far_outside_window`:

```python
    @pytest.mark.backend_mutation
    def test_kick_interpolated_rejects_non_uniform_bin_centers(self) -> None:
        """Non-uniform bin_centers (e.g. a sparse multi-island hist_x from
        EquidistantMultiProfile) must raise, not silently compute the wrong
        physics by assuming a global uniform grid."""
        dtype = np.float64
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            dt = backend.linspace(-5, 5, 20, dtype=backend.float)
            dE = backend.zeros_like(dt, dtype=backend.float)
            # islands: uniform within [0, 4) and [10, 14), gap in between
            bin_centers_np = np.concatenate(
                [
                    np.linspace(0, 4, 10, endpoint=False),
                    np.linspace(10, 14, 10, endpoint=False),
                ]
            )
            bin_centers = backend.array(bin_centers_np, dtype=backend.float)
            voltage = bin_centers**2
            charge = backend.float(10)
            acceleration_kick = backend.float(0.5)
            with self.assertRaises(ValueError):
                backend.specials.kick_interpolated(
                    dt=dt,
                    dE=dE,
                    voltage=voltage,
                    bin_centers=bin_centers,
                    charge=charge,
                    acceleration_kick=acceleration_kick,
                )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `BLOND_FORCE_TEST_ALL_BACKENDS=True python -m pytest tests/unittests/core/backends/test_backend.py::TestSpecials::test_kick_interpolated_rejects_non_uniform_bin_centers -v`
Expected: FAIL — no `ValueError` is raised today (this is the silent bug).

- [ ] **Step 3: Implement the guard in the Python backend**

In `blond/core/backends/python/callables.py`, replace the body of
`kick_interpolated` (keep the existing signature and docstring for now —
Task 2 extends both):

```python
        n_slices = len(bin_centers)
        if n_slices >= 2:
            diffs = np.diff(bin_centers)
            if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=0.0):
                raise ValueError(
                    "bin_centers is not uniformly spaced (looks like a "
                    "sparse/multi-island EquidistantMultiProfile.hist_x). "
                    "Either pass this profile's sparse metadata "
                    "(first_left_cut, left_cut_distance, cut_width, "
                    "bins_per_profile, filling_pattern, "
                    "bucket_index_to_memory_index), e.g. via "
                    "`profile.sparse_kick_metadata`, or use "
                    "EquidistantMultiProfile.profiles[i].hist_x for a "
                    "single bucket."
                )
        inv_bin_width = (n_slices - 1) / (bin_centers[-1] - bin_centers[0])

        fbin = np.floor((dt - bin_centers[0]) * inv_bin_width).astype(np.int32)

        helper1 = charge * (voltage[1:] - voltage[:-1]) * inv_bin_width
        helper2 = (
            charge * voltage[:-1] - bin_centers[:-1] * helper1
        ) + acceleration_kick

        for i in range(len(dt)):
            if (fbin[i] >= 0) and (fbin[i] < n_slices - 1):
                dE[i] += dt[i] * helper1[fbin[i]] + helper2[fbin[i]]
```

- [ ] **Step 4: Implement the guard in the Numba backend**

In `blond/core/backends/numba/callables.py`, `NumbaSpecials.kick_interpolated`
is currently the `@njit`-decorated function itself. Rename its body to a
private helper `_kick_interpolated_dense_nb` and make the public
`kick_interpolated` a thin, non-jitted dispatcher that runs the guard first
(numba nopython mode cannot format a dynamic exception message, so the guard
must live in plain Python, outside the jitted kernel):

```python
@njit(
    sig_kick_interpolated,
    parallel=True,
    fastmath=True,
    cache=True,
)
def _kick_interpolated_dense_nb(  # NOQA PLR0915
    dt: NumpyArray,
    dE: NumpyArray,
    voltage: NumpyArray,
    bin_centers: NumpyArray,
    charge: float,
    acceleration_kick: float,
) -> None:
    dx = (bin_centers[-1] - bin_centers[0]) / (len(bin_centers) - 1)
    inv_dx = 1 / dx
    x_min = bin_centers[0]
    x_max = bin_centers[-1]
    for i in prange(len(dE)):
        x = dt[i]

        if x < x_min or x >= x_max:
            continue
        else:
            idx = int((x - x_min) * inv_dx)
            x0 = x_min + idx * dx
            y0 = voltage[idx]
            y1 = voltage[idx + 1]

            v = y0 + (y1 - y0) * inv_dx * (x - x0)
            dE[i] += charge * v + acceleration_kick


@staticmethod
@enforce_precision(FLOAT)
def kick_interpolated(  # NOQA: D102
    dt: NumpyArray,
    dE: NumpyArray,
    voltage: NumpyArray,
    bin_centers: NumpyArray,
    charge: float,
    acceleration_kick: float,
) -> None:
    n_slices = len(bin_centers)
    if n_slices >= 2:
        diffs = np.diff(bin_centers)
        if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=0.0):
            raise ValueError(
                "bin_centers is not uniformly spaced (looks like a "
                "sparse/multi-island EquidistantMultiProfile.hist_x). "
                "Either pass this profile's sparse metadata "
                "(first_left_cut, left_cut_distance, cut_width, "
                "bins_per_profile, filling_pattern, "
                "bucket_index_to_memory_index), e.g. via "
                "`profile.sparse_kick_metadata`, or use "
                "EquidistantMultiProfile.profiles[i].hist_x for a "
                "single bucket."
            )
    _kick_interpolated_dense_nb(
        dt, dE, voltage, bin_centers, charge, acceleration_kick
    )
```

Move `_kick_interpolated_dense_nb` and `kick_interpolated` out of the
`NumbaSpecials` class body (module-level function + a `staticmethod` that
calls it), following the exact pattern already used for
`move_flagged_elements_to_end` / `_move_flagged_elements_to_end_nb` in this
same file.

- [ ] **Step 5: Implement the guard in the C++ backend**

In `blond/core/backends/cpp/callables.py`, at the top of
`CppSpecials.kick_interpolated`, right after the existing `assert` block and
before the `_LIBBLOND.linear_interp_kick(...)` call:

```python
            n_slices = len(bin_centers)
            if n_slices >= 2:
                diffs = np.diff(bin_centers)
                if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=0.0):
                    raise ValueError(
                        "bin_centers is not uniformly spaced (looks like a "
                        "sparse/multi-island "
                        "EquidistantMultiProfile.hist_x). Either pass this "
                        "profile's sparse metadata (first_left_cut, "
                        "left_cut_distance, cut_width, bins_per_profile, "
                        "filling_pattern, bucket_index_to_memory_index), "
                        "e.g. via `profile.sparse_kick_metadata`, or use "
                        "EquidistantMultiProfile.profiles[i].hist_x for a "
                        "single bucket."
                    )
```

(No C++ source change needed for this step — the check runs in the Python
wrapper before crossing into compiled code.)

- [ ] **Step 6: Implement the guard in the CUDA backend**

In `blond/core/backends/cuda/callables.py`, at the top of
`CudaSpecials.kick_interpolated`, after the existing `assert` block and
before building `glob_vkick_factor`:

```python
        n_slices = bin_centers.size
        if n_slices >= 2:
            diffs = cp.diff(bin_centers)
            if not cp.allclose(diffs, diffs[0], rtol=1e-6, atol=0.0):
                raise ValueError(
                    "bin_centers is not uniformly spaced (looks like a "
                    "sparse/multi-island EquidistantMultiProfile.hist_x). "
                    "Either pass this profile's sparse metadata "
                    "(first_left_cut, left_cut_distance, cut_width, "
                    "bins_per_profile, filling_pattern, "
                    "bucket_index_to_memory_index), e.g. via "
                    "`profile.sparse_kick_metadata`, or use "
                    "EquidistantMultiProfile.profiles[i].hist_x for a "
                    "single bucket."
                )
```

- [ ] **Step 7: Run the new test and the existing dense-path tests**

Run: `BLOND_FORCE_TEST_ALL_BACKENDS=True python -m pytest tests/unittests/core/backends/test_backend.py -k kick_interpolated -v`
Expected: all PASS, including the new
`test_kick_interpolated_rejects_non_uniform_bin_centers` across every backend
in `self.special_modes` (python, cpp, cpp_single_core, numba, and cuda if
`cupy_available`).

- [ ] **Step 8: Run pre-commit and commit**

```bash
pre-commit run --files blond/core/backends/python/callables.py blond/core/backends/numba/callables.py blond/core/backends/cpp/callables.py blond/core/backends/cuda/callables.py tests/unittests/core/backends/test_backend.py
git add blond/core/backends/python/callables.py blond/core/backends/numba/callables.py blond/core/backends/cpp/callables.py blond/core/backends/cuda/callables.py tests/unittests/core/backends/test_backend.py
git commit -m "Guarded kick_interpolated against silently wrong physics on non-uniform bin_centers"
```

---

### Task 2: Sparse-aware kick path — Python and Numba backends

**Files:**
- Modify: `blond/core/backends/python/callables.py` (`PythonSpecials.kick_interpolated`)
- Modify: `blond/core/backends/numba/callables.py` (new `sig_kick_interpolated_sparse`, `_kick_interpolated_sparse_nb`, `NumbaSpecials.kick_interpolated`)
- Test: `tests/unittests/core/backends/test_backend.py`

**Interfaces:**
- Consumes: guard logic from Task 1 (only runs when the new args are omitted).
- Produces: `kick_interpolated(..., first_left_cut=None, left_cut_distance=None, cut_width=None, bins_per_profile=None, filling_pattern=None, bucket_index_to_memory_index=None)`. When `first_left_cut is not None`, all six new args must be given together; the dense path/guard is skipped entirely.

- [ ] **Step 1: Write the failing test**

Add to `tests/unittests/core/backends/test_backend.py`, in `TestSpecials`:

```python
    @pytest.mark.backend_mutation
    def test_kick_interpolated_sparse(self) -> None:
        """A particle sitting exactly on the first bin of the *second*
        island must be kicked using that island's own voltage segment, not
        misindexed into a neighboring island by a naive global floor()."""
        dtype = np.float64
        bins_per_profile = 4
        # bucket 0 and 3 filled, buckets 1 and 2 empty (a real gap)
        filling_pattern_np = np.array([True, False, False, True])
        bucket_index_to_memory_index_np = np.array(
            [0, 0, 0, bins_per_profile], dtype=np.int32
        )
        first_left_cut = 0.0
        left_cut_distance = 1.0
        cut_width = 1.0  # == profile_width, one bucket
        bin_width = cut_width / bins_per_profile

        # memory layout: [bucket0 bins..., bucket3 bins...]
        bin_centers_np = np.concatenate(
            [
                first_left_cut
                + b * left_cut_distance
                + bin_width * (np.arange(bins_per_profile) + 0.5)
                for b in (0, 3)
            ]
        )
        voltage_np = np.array(
            [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
        )

        for i, special in enumerate(self.special_modes):
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue

            # particle exactly on bin_centers_np[4] == first bin of the
            # second island; a global-uniform-grid bug maps this to
            # memory index 5 instead of 4.
            dt = backend.array(
                np.array([bin_centers_np[4]]), dtype=backend.float
            )
            dE = backend.zeros_like(dt, dtype=backend.float)
            voltage = backend.array(voltage_np, dtype=backend.float)
            filling_pattern = backend.array(filling_pattern_np, dtype=bool)
            bucket_index_to_memory_index = backend.array(
                bucket_index_to_memory_index_np, dtype=np.int32
            )
            bin_centers = backend.array(bin_centers_np, dtype=backend.float)
            charge = backend.float(1.0)
            acceleration_kick = backend.float(0.0)

            backend.specials.kick_interpolated(
                dt=dt,
                dE=dE,
                voltage=voltage,
                bin_centers=bin_centers,
                charge=charge,
                acceleration_kick=acceleration_kick,
                first_left_cut=first_left_cut,
                left_cut_distance=left_cut_distance,
                cut_width=cut_width,
                bins_per_profile=bins_per_profile,
                filling_pattern=filling_pattern,
                bucket_index_to_memory_index=bucket_index_to_memory_index,
            )
            result = dE
            if special == "cuda":
                result = result.get()

            # Ground truth: same particle kicked against ONLY the second
            # island's own 4-bin dense profile (island-local, no gap).
            dt_local = backend.array(
                np.array(
                    [bin_centers_np[4] - 3 * left_cut_distance]
                ),
                dtype=backend.float,
            )
            dE_local = backend.zeros_like(dt_local, dtype=backend.float)
            voltage_local = backend.array(
                voltage_np[bins_per_profile:], dtype=backend.float
            )
            bin_centers_local = backend.array(
                bin_centers_np[bins_per_profile:] - 3 * left_cut_distance,
                dtype=backend.float,
            )
            backend.specials.kick_interpolated(
                dt=dt_local,
                dE=dE_local,
                voltage=voltage_local,
                bin_centers=bin_centers_local,
                charge=charge,
                acceleration_kick=acceleration_kick,
            )
            expected = dE_local
            if special == "cuda":
                expected = expected.get()

            np.testing.assert_allclose(
                result,
                expected,
                rtol=self.rtol,
                err_msg=f"Failed sparse test `{special}` with {dtype}",
            )
            if i == 0:
                result_python = result
            else:
                np.testing.assert_allclose(
                    result,
                    result_python,
                    rtol=self.rtol,
                    err_msg=f"Cross-backend mismatch `{special}` {dtype}",
                )

    @pytest.mark.backend_mutation
    def test_kick_interpolated_sparse_skips_unfilled_bucket(self) -> None:
        """A particle whose dt falls into an unfilled bucket's time window
        must receive no kick (mirrors histogram_sparse's `continue`)."""
        dtype = np.float64
        bins_per_profile = 4
        filling_pattern_np = np.array([True, False, False, True])
        bucket_index_to_memory_index_np = np.array(
            [0, 0, 0, bins_per_profile], dtype=np.int32
        )
        for special in self.special_modes:
            try:
                self._setUp(dtype=dtype, special_mode=special)
            except (FileNotFoundError, OSError):
                print(f"Could not perform `{special}` test for {dtype}")
                continue
            dt = backend.array(np.array([1.5]), dtype=backend.float)  # bucket 1, unfilled
            dE = backend.zeros_like(dt, dtype=backend.float)
            voltage = backend.array(
                np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]),
                dtype=backend.float,
            )
            filling_pattern = backend.array(filling_pattern_np, dtype=bool)
            bucket_index_to_memory_index = backend.array(
                bucket_index_to_memory_index_np, dtype=np.int32
            )
            bin_centers = backend.array(
                np.arange(8) * 0.25, dtype=backend.float
            )
            backend.specials.kick_interpolated(
                dt=dt,
                dE=dE,
                voltage=voltage,
                bin_centers=bin_centers,
                charge=backend.float(1.0),
                acceleration_kick=backend.float(0.0),
                first_left_cut=0.0,
                left_cut_distance=1.0,
                cut_width=1.0,
                bins_per_profile=bins_per_profile,
                filling_pattern=filling_pattern,
                bucket_index_to_memory_index=bucket_index_to_memory_index,
            )
            result = dE
            if special == "cuda":
                result = result.get()
            np.testing.assert_allclose(
                result, np.zeros(1), err_msg=f"Failed `{special}` {dtype}"
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `BLOND_FORCE_TEST_ALL_BACKENDS=True python -m pytest tests/unittests/core/backends/test_backend.py -k "kick_interpolated_sparse" -v`
Expected: FAIL with `TypeError: kick_interpolated() got an unexpected keyword argument 'first_left_cut'` for every backend.

- [ ] **Step 3: Implement the sparse path in the Python backend**

In `blond/core/backends/python/callables.py`, replace `kick_interpolated`
entirely with:

```python
    @staticmethod
    def kick_interpolated(
        dt: NumpyArray,
        dE: NumpyArray,
        voltage: NumpyArray,
        bin_centers: NumpyArray,
        charge: float,
        acceleration_kick: float,
        first_left_cut: float | None = None,
        left_cut_distance: float | None = None,
        cut_width: float | None = None,
        bins_per_profile: int | None = None,
        filling_pattern: NumpyArray | None = None,
        bucket_index_to_memory_index: NumpyArray | None = None,
    ) -> None:
        """
        Interpolated kick method.

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s].
        dE
            Macro-particle energy coordinates, in [eV].
        voltage
            Array of voltages along `bin_centers`, in [V].
        bin_centers
            Positions of `voltage`, in [s].
        charge
            Particle charge, as number of elementary charges `e` [].
        acceleration_kick
            Energy, in [eV], which is added to all particles.
            This is intended to subtract the target energy from the RF
            energy gain in one common call.
        first_left_cut
            Left edge of the first bucket's histogram. Pass this together
            with the other sparse-metadata arguments below (e.g. via
            `EquidistantMultiProfile.sparse_kick_metadata`) when
            `bin_centers` is a gapped, multi-island array such as
            `EquidistantMultiProfile.hist_x`. When omitted, `bin_centers`
            must be uniformly spaced.
        left_cut_distance
            Distance between the left edge of each bucket's histogram.
        cut_width
            Distance between left and right edge of one bucket's
            histogram.
        bins_per_profile
            Number of bins per bucket.
        filling_pattern
            Filling pattern as a boolean array where `True` means filled
            bucket.
        bucket_index_to_memory_index
            Maps bucket index to memory index, see
            `_gen_array_bucket_index_to_memory_index`.
        """
        sparse = first_left_cut is not None
        n_slices = len(bin_centers)

        if sparse:
            inv_bin_width = bins_per_profile / cut_width
        else:
            if n_slices >= 2:
                diffs = np.diff(bin_centers)
                if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=0.0):
                    raise ValueError(
                        "bin_centers is not uniformly spaced (looks like "
                        "a sparse/multi-island "
                        "EquidistantMultiProfile.hist_x). Either pass "
                        "this profile's sparse metadata (first_left_cut, "
                        "left_cut_distance, cut_width, bins_per_profile, "
                        "filling_pattern, bucket_index_to_memory_index), "
                        "e.g. via `profile.sparse_kick_metadata`, or use "
                        "EquidistantMultiProfile.profiles[i].hist_x for "
                        "a single bucket."
                    )
            inv_bin_width = (n_slices - 1) / (
                bin_centers[-1] - bin_centers[0]
            )

        helper1 = charge * (voltage[1:] - voltage[:-1]) * inv_bin_width
        helper2 = (
            charge * voltage[:-1] - bin_centers[:-1] * helper1
        ) + acceleration_kick

        if not sparse:
            fbin = np.floor(
                (dt - bin_centers[0]) * inv_bin_width
            ).astype(np.int32)
            for i in range(len(dt)):
                if (fbin[i] >= 0) and (fbin[i] < n_slices - 1):
                    dE[i] += dt[i] * helper1[fbin[i]] + helper2[fbin[i]]
            return

        n_buckets = len(filling_pattern)
        inv_hist_dist = 1.0 / left_cut_distance
        bin_width = cut_width / bins_per_profile
        for i in range(len(dt)):
            bucket_i = int(
                np.floor((dt[i] - first_left_cut) * inv_hist_dist)
            )
            if bucket_i < 0 or bucket_i >= n_buckets:
                continue
            if not filling_pattern[bucket_i]:
                continue
            cut_left = first_left_cut + bucket_i * left_cut_distance
            bucket_bin_center0 = cut_left + bin_width / 2.0
            local_bin = int(
                np.floor((dt[i] - bucket_bin_center0) * inv_bin_width)
            )
            if local_bin < 0 or local_bin >= bins_per_profile - 1:
                continue
            fbin = bucket_index_to_memory_index[bucket_i] + local_bin
            dE[i] += dt[i] * helper1[fbin] + helper2[fbin]
```

- [ ] **Step 4: Implement the sparse path in the Numba backend**

In `blond/core/backends/numba/callables.py`, add a new signature next to
`sig_kick_interpolated`:

```python
sig_kick_interpolated_sparse = void(
    sig_dt,
    sig_dE,
    sig_voltage,
    sig_bin_centers,
    sig_charge,
    sig_acceleration_kick,
    nb_f,  # first_left_cut
    nb_f,  # left_cut_distance
    nb_f,  # cut_width
    numba.int32,  # bins_per_profile
    numba.bool[:],  # filling_pattern
    numba.int32[:],  # bucket_index_to_memory_index
)
```

Add the private jitted kernel next to `_kick_interpolated_dense_nb` (from
Task 1):

```python
@njit(
    sig_kick_interpolated_sparse,
    parallel=True,
    fastmath=True,
    cache=True,
)
def _kick_interpolated_sparse_nb(  # NOQA PLR0915
    dt: NumpyArray,
    dE: NumpyArray,
    voltage: NumpyArray,
    bin_centers: NumpyArray,
    charge: float,
    acceleration_kick: float,
    first_left_cut: float,
    left_cut_distance: float,
    cut_width: float,
    bins_per_profile: int,
    filling_pattern: NumpyArray,
    bucket_index_to_memory_index: NumpyArray,
) -> None:
    n_buckets = len(filling_pattern)
    inv_hist_dist = 1.0 / left_cut_distance
    inv_bin_width = bins_per_profile / cut_width
    bin_width = cut_width / bins_per_profile
    for i in prange(len(dE)):
        x = dt[i]
        bucket_i = int(np.floor((x - first_left_cut) * inv_hist_dist))
        if bucket_i < 0 or bucket_i >= n_buckets:
            continue
        if not filling_pattern[bucket_i]:
            continue
        cut_left = first_left_cut + bucket_i * left_cut_distance
        bucket_bin_center0 = cut_left + bin_width / 2.0
        local_bin = int(
            np.floor((x - bucket_bin_center0) * inv_bin_width)
        )
        if local_bin < 0 or local_bin >= bins_per_profile - 1:
            continue
        idx = bucket_index_to_memory_index[bucket_i] + local_bin
        v = voltage[idx] + (voltage[idx + 1] - voltage[idx]) * inv_bin_width * (
            x - bin_centers[idx]
        )
        dE[i] += charge * v + acceleration_kick
```

Replace the `kick_interpolated` staticmethod (from Task 1) with a dispatcher
that also accepts the six new optional args:

```python
@staticmethod
@enforce_precision(FLOAT)
def kick_interpolated(  # NOQA: D102
    dt: NumpyArray,
    dE: NumpyArray,
    voltage: NumpyArray,
    bin_centers: NumpyArray,
    charge: float,
    acceleration_kick: float,
    first_left_cut: float | None = None,
    left_cut_distance: float | None = None,
    cut_width: float | None = None,
    bins_per_profile: int | None = None,
    filling_pattern: NumpyArray | None = None,
    bucket_index_to_memory_index: NumpyArray | None = None,
) -> None:
    if first_left_cut is None:
        n_slices = len(bin_centers)
        if n_slices >= 2:
            diffs = np.diff(bin_centers)
            if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=0.0):
                raise ValueError(
                    "bin_centers is not uniformly spaced (looks like a "
                    "sparse/multi-island EquidistantMultiProfile.hist_x). "
                    "Either pass this profile's sparse metadata "
                    "(first_left_cut, left_cut_distance, cut_width, "
                    "bins_per_profile, filling_pattern, "
                    "bucket_index_to_memory_index), e.g. via "
                    "`profile.sparse_kick_metadata`, or use "
                    "EquidistantMultiProfile.profiles[i].hist_x for a "
                    "single bucket."
                )
        _kick_interpolated_dense_nb(
            dt, dE, voltage, bin_centers, charge, acceleration_kick
        )
        return
    _kick_interpolated_sparse_nb(
        dt,
        dE,
        voltage,
        bin_centers,
        charge,
        acceleration_kick,
        first_left_cut,
        left_cut_distance,
        cut_width,
        np.int32(bins_per_profile),
        filling_pattern,
        bucket_index_to_memory_index,
    )
```

- [ ] **Step 5: Run tests to verify they pass for python and numba**

Run: `BLOND_FORCE_TEST_ALL_BACKENDS=True python -m pytest tests/unittests/core/backends/test_backend.py -k "kick_interpolated_sparse" -v`
Expected: PASS for `python` and `numba`; `cpp`/`cpp_single_core`/`cuda` still
FAIL with `TypeError` (implemented in Tasks 3 and 4) — that's expected at
this point, not a regression.

- [ ] **Step 6: Run pre-commit and commit**

```bash
pre-commit run --files blond/core/backends/python/callables.py blond/core/backends/numba/callables.py tests/unittests/core/backends/test_backend.py
git add blond/core/backends/python/callables.py blond/core/backends/numba/callables.py tests/unittests/core/backends/test_backend.py
git commit -m "Added sparse-aware kick_interpolated path for python and numba backends"
```

---

### Task 3: Sparse-aware kick path — C++ backend

**Files:**
- Modify: `blond/core/backends/cpp/linear_interp_kick.cpp` (new `linear_interp_kick_sparse`)
- Modify: `blond/core/backends/cpp/callables.py` (`CppSpecials.kick_interpolated`)
- Test: `tests/unittests/core/backends/test_backend.py` (already written in Task 2 — no new test needed, just gets un-skipped for `cpp`/`cpp_single_core`)

**Interfaces:**
- Consumes: `Task 2`'s `test_kick_interpolated_sparse` / `test_kick_interpolated_sparse_skips_unfilled_bucket`.
- Produces: same behavior as the Python/Numba sparse path, for the `cpp`/`cpp_single_core` special modes.

- [ ] **Step 1: Run the existing sparse tests to confirm cpp still fails**

Run: `BLOND_FORCE_TEST_ALL_BACKENDS=True python -m pytest tests/unittests/core/backends/test_backend.py -k "kick_interpolated_sparse" -v`
Expected: FAIL for `cpp`/`cpp_single_core` with `TypeError` (unexpected
keyword argument), PASS for `python`/`numba`.

- [ ] **Step 2: Add the C++ kernel**

Append to `blond/core/backends/cpp/linear_interp_kick.cpp` (after the
existing `linear_interp_kick` function, before
`linear_interp_time_translation`):

```cpp
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
```

`blond/core/backends/cpp/compile.py`'s `cpp_files` list already includes
`"linear_interp_kick.cpp"` — no build-file change needed.

- [ ] **Step 3: Update the Python wrapper**

Replace `CppSpecials.kick_interpolated` in
`blond/core/backends/cpp/callables.py` with:

```python
        @staticmethod
        def kick_interpolated(
            dt: NumpyArray,
            dE: NumpyArray,
            voltage: NumpyArray,
            bin_centers: NumpyArray,
            charge: float,
            acceleration_kick: float,
            first_left_cut: float | None = None,
            left_cut_distance: float | None = None,
            cut_width: float | None = None,
            bins_per_profile: int | None = None,
            filling_pattern: NumpyArray | None = None,
            bucket_index_to_memory_index: NumpyArray | None = None,
        ) -> None:
            assert dt.dtype == floattype
            assert dE.dtype == floattype
            assert voltage.dtype == floattype
            assert bin_centers.dtype == floattype
            assert dt.flags.c_contiguous
            assert dE.flags.c_contiguous
            assert voltage.flags.c_contiguous
            assert bin_centers.flags.c_contiguous

            charge = floattype(charge)
            acceleration_kick = floattype(acceleration_kick)

            if first_left_cut is None:
                n_slices = len(bin_centers)
                if n_slices >= 2:
                    diffs = np.diff(bin_centers)
                    if not np.allclose(
                        diffs, diffs[0], rtol=1e-6, atol=0.0
                    ):
                        raise ValueError(
                            "bin_centers is not uniformly spaced (looks "
                            "like a sparse/multi-island "
                            "EquidistantMultiProfile.hist_x). Either "
                            "pass this profile's sparse metadata "
                            "(first_left_cut, left_cut_distance, "
                            "cut_width, bins_per_profile, "
                            "filling_pattern, "
                            "bucket_index_to_memory_index), e.g. via "
                            "`profile.sparse_kick_metadata`, or use "
                            "EquidistantMultiProfile.profiles[i].hist_x "
                            "for a single bucket."
                        )
                _LIBBLOND.linear_interp_kick(
                    dt.ctypes.data_as(ct.c_void_p),
                    dE.ctypes.data_as(ct.c_void_p),
                    voltage.ctypes.data_as(ct.c_void_p),
                    bin_centers.ctypes.data_as(ct.c_void_p),
                    c_real(charge, floattype),
                    ct.c_int(len(bin_centers)),
                    ct.c_int(len(dt)),
                    c_real(acceleration_kick, floattype),
                )
                return

            assert filling_pattern.dtype == np.bool_
            assert bucket_index_to_memory_index.dtype == np.int32
            assert filling_pattern.flags.c_contiguous
            assert bucket_index_to_memory_index.flags.c_contiguous

            _LIBBLOND.linear_interp_kick_sparse(
                dt.ctypes.data_as(ct.c_void_p),
                dE.ctypes.data_as(ct.c_void_p),
                voltage.ctypes.data_as(ct.c_void_p),
                bin_centers.ctypes.data_as(ct.c_void_p),
                c_real(charge, floattype),
                ct.c_int(len(bin_centers)),
                ct.c_int(len(dt)),
                c_real(acceleration_kick, floattype),
                c_real(floattype(first_left_cut), floattype),
                c_real(floattype(left_cut_distance), floattype),
                c_real(floattype(cut_width), floattype),
                ct.c_int(bins_per_profile),
                ct.c_int(len(filling_pattern)),
                filling_pattern.ctypes.data_as(ct.c_void_p),
                bucket_index_to_memory_index.ctypes.data_as(ct.c_void_p),
            )
```

- [ ] **Step 4: Recompile the C++ backend**

Run: `blond-compile-cpp --parallel`
Expected: exits 0, no compiler errors/warnings about
`linear_interp_kick_sparse`.

- [ ] **Step 5: Run tests to verify cpp now passes**

Run: `BLOND_FORCE_TEST_ALL_BACKENDS=True python -m pytest tests/unittests/core/backends/test_backend.py -k "kick_interpolated_sparse or kick_interpolated_rejects" -v`
Expected: PASS for `python`, `numba`, `cpp`, `cpp_single_core`.

- [ ] **Step 6: Run pre-commit and commit**

```bash
pre-commit run --files blond/core/backends/cpp/linear_interp_kick.cpp blond/core/backends/cpp/callables.py
git add blond/core/backends/cpp/linear_interp_kick.cpp blond/core/backends/cpp/callables.py
git commit -m "Added sparse-aware kick_interpolated path for C++ backend"
```

---

### Task 4: Sparse-aware kick path — CUDA backend

**Files:**
- Modify: `blond/core/backends/cuda/kernels.cu` (new `lik_sparse_gm_copy`, `lik_sparse_gm_comp`)
- Modify: `blond/core/backends/cuda/callables.py` (`CudaSpecials.kick_interpolated`)
- Test: `tests/unittests/core/backends/test_backend.py` (already written in Task 2)

**Interfaces:**
- Consumes: `Task 2`'s sparse tests.
- Produces: same behavior for the `cuda` special mode (only runs if `cupy_available`).

- [ ] **Step 1: Run the existing sparse tests to confirm cuda still fails**

Run: `BLOND_FORCE_TEST_ALL_BACKENDS=True python -m pytest tests/unittests/core/backends/test_backend.py -k "kick_interpolated_sparse" -v`
Expected: FAIL for `cuda` with `TypeError`, PASS for the other 4 modes.

- [ ] **Step 2: Add the CUDA kernels**

Append to `blond/core/backends/cuda/kernels.cu` (after `lik_only_gm_comp`,
before `loss_box`):

```cuda
// Sparse variants of lik_only_gm_copy/lik_only_gm_comp: bin_centers/voltage
// are a concatenation of one dense island per active RF bucket (gaps
// between islands whenever the filling pattern skips a bucket). Unlike the
// dense kernels, inv_bin_width is derived from bins_per_profile/cut_width
// (constant per bucket) instead of the array's global endpoints, and each
// particle is first resolved to its bucket (mirroring histogram_sparse)
// before indexing into glob_vkick_factor.
extern "C"
__global__ void lik_sparse_gm_copy(
    const real_t * __restrict__ voltage_array,
    const real_t * __restrict__ bin_centers,
    const real_t charge,
    const int n_slices_total,
    const real_t acc_kick,
    const real_t cut_width,
    const int bins_per_profile,
    real_t * __restrict__ glob_vkick_factor
)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    const real_t inv_bin_width = real_t(bins_per_profile) / cut_width;

    for (int i = tid; i < n_slices_total - 1; i += gridDim.x * blockDim.x) {
        glob_vkick_factor[2*i] = charge * (voltage_array[i + 1] - voltage_array[i])
                              * inv_bin_width;
        glob_vkick_factor[2*i+1] = (charge * voltage_array[i] - bin_centers[i] * glob_vkick_factor[2*i])
                         + acc_kick;
    }
}


extern "C"
__global__ void lik_sparse_gm_comp(
    real_t * __restrict__ beam_dt,
    real_t * __restrict__ beam_dE,
    const int n_macroparticles,
    const real_t first_left_cut,
    const real_t left_cut_distance,
    const real_t cut_width,
    const int bins_per_profile,
    const int n_buckets,
    const bool * __restrict__ filling_pattern,
    const int * __restrict__ bucket_index_to_memory_index,
    real_t * __restrict__ glob_vkick_factor
)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    const real_t inv_hist_dist = real_t(1) / left_cut_distance;
    const real_t inv_bin_width = real_t(bins_per_profile) / cut_width;
    const real_t bin_width = cut_width / real_t(bins_per_profile);

    for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
        const real_t dt = beam_dt[i];
        const int bucket_i = (int)floor((dt - first_left_cut) * inv_hist_dist);
        if (bucket_i < 0 || bucket_i >= n_buckets)
            continue;
        if (!filling_pattern[bucket_i])
            continue;

        const real_t cut_left = first_left_cut + bucket_i * left_cut_distance;
        const real_t bucket_bin_center0 = cut_left + bin_width / real_t(2);
        const int local_bin = (int)floor((dt - bucket_bin_center0) * inv_bin_width);
        if (local_bin < 0 || local_bin >= bins_per_profile - 1)
            continue;

        const int fbin = bucket_index_to_memory_index[bucket_i] + local_bin;
        beam_dE[i] += dt * glob_vkick_factor[2*fbin] + glob_vkick_factor[2*fbin+1];
    }
}
```

- [ ] **Step 3: Update the Python wrapper**

In `blond/core/backends/cuda/callables.py`, add the two kernel handles next
to `_gm_linear_interp_kick_comp`:

```python
_gm_linear_interp_kick_sparse_help = gpu_module.get_function(
    "lik_sparse_gm_copy"
)
_gm_linear_interp_kick_sparse_comp = gpu_module.get_function(
    "lik_sparse_gm_comp"
)
```

Replace `CudaSpecials.kick_interpolated` with:

```python
    @staticmethod
    def kick_interpolated(  # NOQA: D102
        dt: CupyArray,
        dE: CupyArray,
        voltage: CupyArray,
        bin_centers: CupyArray,
        charge: float,
        acceleration_kick: float,
        first_left_cut: float | None = None,
        left_cut_distance: float | None = None,
        cut_width: float | None = None,
        bins_per_profile: int | None = None,
        filling_pattern: CupyArray | None = None,
        bucket_index_to_memory_index: CupyArray | None = None,
    ) -> None:
        assert dt.device != "cpu", f"Requires Cupy array, but got {type(dt)}."
        assert dE.device != "cpu", f"Requires Cupy array, but got {type(dE)}."
        assert voltage.device != "cpu", (
            f"Requires Cupy array, but got {type(voltage)}."
        )
        assert bin_centers.device != "cpu", (
            f"Requires Cupy array, but got {type(bin_centers)}."
        )

        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT
        assert voltage.dtype == FLOAT
        assert bin_centers.dtype == FLOAT
        assert dt.flags.c_contiguous
        assert dE.flags.c_contiguous
        assert voltage.flags.c_contiguous
        assert bin_centers.flags.c_contiguous

        charge = FLOAT(charge)
        acceleration_kick = FLOAT(acceleration_kick)

        if first_left_cut is None:
            n_slices = bin_centers.size
            if n_slices >= 2:
                diffs = cp.diff(bin_centers)
                if not cp.allclose(diffs, diffs[0], rtol=1e-6, atol=0.0):
                    raise ValueError(
                        "bin_centers is not uniformly spaced (looks like "
                        "a sparse/multi-island "
                        "EquidistantMultiProfile.hist_x). Either pass "
                        "this profile's sparse metadata (first_left_cut, "
                        "left_cut_distance, cut_width, bins_per_profile, "
                        "filling_pattern, bucket_index_to_memory_index), "
                        "e.g. via `profile.sparse_kick_metadata`, or use "
                        "EquidistantMultiProfile.profiles[i].hist_x for "
                        "a single bucket."
                    )
            glob_vkick_factor = cp.empty(2 * (bin_centers.size - 1), FLOAT)
            _gm_linear_interp_kick_help(
                args=(
                    dt,
                    dE,
                    voltage,
                    bin_centers,
                    charge,
                    np.int32(bin_centers.size),
                    np.int32(dt.size),
                    acceleration_kick,
                    glob_vkick_factor,
                ),
                grid=grid_size,
                block=block_size,
            )
            _gm_linear_interp_kick_comp(
                args=(
                    dt,
                    dE,
                    voltage,
                    bin_centers,
                    charge,
                    np.int32(bin_centers.size),
                    np.int32(dt.size),
                    acceleration_kick,
                    glob_vkick_factor,
                ),
                grid=grid_size,
                block=block_size,
            )
            return

        assert filling_pattern.device != "cpu"
        assert bucket_index_to_memory_index.device != "cpu"
        assert filling_pattern.dtype == np.bool_
        assert bucket_index_to_memory_index.dtype == np.int32
        assert filling_pattern.flags.c_contiguous
        assert bucket_index_to_memory_index.flags.c_contiguous

        glob_vkick_factor = cp.empty(2 * (bin_centers.size - 1), FLOAT)
        _gm_linear_interp_kick_sparse_help(
            args=(
                voltage,
                bin_centers,
                charge,
                np.int32(bin_centers.size),
                acceleration_kick,
                FLOAT(cut_width),
                np.int32(bins_per_profile),
                glob_vkick_factor,
            ),
            grid=grid_size,
            block=block_size,
        )
        _gm_linear_interp_kick_sparse_comp(
            args=(
                dt,
                dE,
                np.int32(dt.size),
                FLOAT(first_left_cut),
                FLOAT(left_cut_distance),
                FLOAT(cut_width),
                np.int32(bins_per_profile),
                np.int32(len(filling_pattern)),
                filling_pattern,
                bucket_index_to_memory_index,
                glob_vkick_factor,
            ),
            grid=grid_size,
            block=block_size,
        )
```

**Note:** check the existing `_track_interp`(dense) call between
`_gm_linear_interp_kick_help` and `_gm_linear_interp_kick_comp` in the
current file (lines ~365-390 before this edit) for the exact `args=(...)`
ordering used today, and match it — copy the true current ordering rather
than trusting this snippet blindly, since the visible excerpt in this plan
was truncated mid-call.

- [ ] **Step 4: Recompile the CUDA backend**

Run: `blond-compile-cuda`
Expected: exits 0, `.cubin` rebuilt with `lik_sparse_gm_copy` and
`lik_sparse_gm_comp` present (`cuobjdump --dump-elf-symtab` or attempting
`gpu_module.get_function("lik_sparse_gm_copy")` should not raise).

- [ ] **Step 5: Run tests to verify cuda now passes**

Run: `BLOND_FORCE_TEST_ALL_BACKENDS=True python -m pytest tests/unittests/core/backends/test_backend.py -k "kick_interpolated" -v`
Expected: PASS for all 5 special modes (`python`, `cpp`, `cpp_single_core`,
`numba`, `cuda`).

- [ ] **Step 6: Run pre-commit and commit**

```bash
pre-commit run --files blond/core/backends/cuda/kernels.cu blond/core/backends/cuda/callables.py
git add blond/core/backends/cuda/kernels.cu blond/core/backends/cuda/callables.py
git commit -m "Added sparse-aware kick_interpolated path for CUDA backend"
```

---

### Task 5: Update the `Specials` ABC contract

**Files:**
- Modify: `blond/core/backends/backend.py:195-207`

**Interfaces:**
- Consumes: nothing new.
- Produces: documents, for every backend implementer, that `kick_interpolated` raises `ValueError` on non-uniform `bin_centers` and what the six optional sparse-metadata kwargs mean.

- [ ] **Step 1: Update the abstract method**

```python
    @staticmethod
    @abstractmethod  # pragma: no cover
    def kick_interpolated(  # NOQA: D102
        dt: NumpyArray,
        dE: NumpyArray,
        voltage: NumpyArray,
        bin_centers: NumpyArray,
        charge: float,
        acceleration_kick: float,
        first_left_cut: float | None = None,
        left_cut_distance: float | None = None,
        cut_width: float | None = None,
        bins_per_profile: int | None = None,
        filling_pattern: NumpyArray | None = None,
        bucket_index_to_memory_index: NumpyArray | None = None,
    ) -> None:
        """
        Interpolated kick method.

        With the sparse-metadata arguments omitted, `bin_centers` must be
        uniformly spaced; implementations raise `ValueError` otherwise
        (e.g. when handed a gapped, multi-island array such as
        `EquidistantMultiProfile.hist_x` without its metadata). With the
        sparse-metadata arguments given (all six together, typically via
        `EquidistantMultiProfile.sparse_kick_metadata`), particles are
        resolved to their own bucket before interpolation, matching
        `histogram_sparse`'s bucket-resolution semantics.
        """
        raise NotImplementedError(
            "Abstract method `kick_interpolated` is not implemented."
        )
```

- [ ] **Step 2: Run the full backend test module**

Run: `BLOND_FORCE_TEST_ALL_BACKENDS=True python -m pytest tests/unittests/core/backends/ -v`
Expected: all PASS.

- [ ] **Step 3: Run pre-commit and commit**

```bash
pre-commit run --files blond/core/backends/backend.py
git add blond/core/backends/backend.py
git commit -m "Documented kick_interpolated sparse-metadata contract on the Specials ABC"
```

---

### Task 6: `EquidistantMultiProfile.sparse_kick_metadata` accessor

**Files:**
- Modify: `blond/physics/profiles_sparse.py`
- Test: `tests/unittests/physics/test_profiles_sparse.py`

**Interfaces:**
- Consumes: `EquidistantMultiProfile._first_left_cut`, `_left_cut_distance`, `_bins_per_profile`, `_filling_pattern`, `_bucket_index_to_memory_index`, `profiles[0].cut_left`/`cut_right` (all already set by `configure()`).
- Produces: `EquidistantMultiProfile.sparse_kick_metadata -> dict` with keys `first_left_cut`, `left_cut_distance`, `cut_width`, `bins_per_profile`, `filling_pattern`, `bucket_index_to_memory_index` — exactly the kwarg names `kick_interpolated` expects, so callers can do `backend.specials.kick_interpolated(..., **profile.sparse_kick_metadata)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unittests/physics/test_profiles_sparse.py` (check the existing
imports/class name at the top of that file first and match its style):

```python
    def test_sparse_kick_metadata(self):
        profile = EquidistantMultiProfile.headless(
            t_rev=4.0,
            filling_pattern=np.array([1, 0, 0, 1]),
            bins_per_profile=4,
        )
        meta = profile.sparse_kick_metadata
        self.assertEqual(
            set(meta.keys()),
            {
                "first_left_cut",
                "left_cut_distance",
                "cut_width",
                "bins_per_profile",
                "filling_pattern",
                "bucket_index_to_memory_index",
            },
        )
        self.assertAlmostEqual(meta["first_left_cut"], 0.0)
        self.assertAlmostEqual(meta["left_cut_distance"], 1.0)
        self.assertAlmostEqual(meta["cut_width"], 1.0)
        self.assertEqual(meta["bins_per_profile"], 4)
        np.testing.assert_array_equal(
            meta["filling_pattern"], np.array([True, False, False, True])
        )
        np.testing.assert_array_equal(
            meta["bucket_index_to_memory_index"], np.array([0, 0, 0, 4])
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unittests/physics/test_profiles_sparse.py -k test_sparse_kick_metadata -v`
Expected: FAIL with `AttributeError: 'EquidistantMultiProfile' object has
no attribute 'sparse_kick_metadata'`.

- [ ] **Step 3: Implement the property**

In `blond/physics/profiles_sparse.py`, add after the `n_bins` property
(around line 242):

```python
    @property
    def sparse_kick_metadata(self) -> dict:
        """
        Keyword arguments for a sparse-aware kernel call on this profile.

        Bundles the island metadata that `kick_interpolated` (and
        `histogram_sparse`) need to correctly resolve a particle to its
        own bucket within this profile's concatenated, gapped
        `hist_x`/`hist_y` memory, instead of assuming `hist_x` is a
        single uniform grid.

        Returns
        -------
        sparse_kick_metadata
            Dict with keys ``first_left_cut``, ``left_cut_distance``,
            ``cut_width``, ``bins_per_profile``, ``filling_pattern``,
            ``bucket_index_to_memory_index`` -- matching the keyword
            arguments `Specials.kick_interpolated` expects, so callers
            can do
            ``backend.specials.kick_interpolated(..., **profile.sparse_kick_metadata)``.
        """
        return {
            "first_left_cut": self._first_left_cut,
            "left_cut_distance": self._left_cut_distance,
            "cut_width": (
                self.profiles[0].cut_right - self.profiles[0].cut_left
            ),
            "bins_per_profile": self._bins_per_profile,
            "filling_pattern": self._filling_pattern,
            "bucket_index_to_memory_index": (
                self._bucket_index_to_memory_index
            ),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unittests/physics/test_profiles_sparse.py -k test_sparse_kick_metadata -v`
Expected: PASS.

- [ ] **Step 5: Run pre-commit and commit**

```bash
pre-commit run --files blond/physics/profiles_sparse.py tests/unittests/physics/test_profiles_sparse.py
git add blond/physics/profiles_sparse.py tests/unittests/physics/test_profiles_sparse.py
git commit -m "Added EquidistantMultiProfile.sparse_kick_metadata accessor"
```

---

### Task 7: `PooledInterpolationKick` carries sparse metadata

**Files:**
- Modify: `blond/experimental/physics/kick_pooling.py`
- Test: `tests/unittests/experimental/physics/test_kick_pooling.py` (check this file exists first; if the experimental test tree uses a different path, find it with `grep -rl "PooledInterpolationKick" tests/`)

**Interfaces:**
- Consumes: `Task 6`'s `sparse_kick_metadata` dict shape.
- Produces: `PooledInterpolationKick.register(time_axis, voltage, sparse_metadata=None)`; `track()` forwards `sparse_metadata` (if any) to `kick_interpolated` as `**sparse_metadata`.

- [ ] **Step 1: Locate the existing test file and current `register`/`track`**

Run: `grep -rl "PooledInterpolationKick" tests/`

Read the matched file(s) fully before writing new tests, to match existing
fixture/setup style exactly (buffer size, `time_axis` construction, how
`track()` is invoked and asserted on).

- [ ] **Step 2: Write the failing test**

Add a test to the located file (adapt fixture setup to match what's already
there):

```python
    def test_register_and_track_with_sparse_metadata(self):
        pooled_kick = PooledInterpolationKick(buffer_size=4)
        # minimal 2-bucket sparse layout, bucket 0 and 1 both filled
        time_axis = np.array([0.125, 0.375, 0.625, 0.875])
        voltage = np.array([1.0, 2.0, 3.0, 4.0])
        sparse_metadata = {
            "first_left_cut": 0.0,
            "left_cut_distance": 0.5,
            "cut_width": 0.5,
            "bins_per_profile": 2,
            "filling_pattern": np.array([True, True]),
            "bucket_index_to_memory_index": np.array([0, 2], dtype=np.int32),
        }
        pooled_kick.register(
            time_axis=time_axis,
            voltage=voltage,
            sparse_metadata=sparse_metadata,
        )
        dt = np.array([0.125])
        dE = np.zeros_like(dt)
        beam = _make_minimal_beam(dt=dt, dE=dE)  # reuse this test module's existing beam fixture helper
        pooled_kick._track(beam=beam)
        self.assertNotEqual(dE[0], 0.0)
```

(`_make_minimal_beam` is a placeholder name — replace it with whatever
fixture/setup helper the located test file already uses to build a `beam`
with `.read_partial_dt()`/`.write_partial_dE()`; do not invent a new one if
an equivalent already exists.)

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest <located_test_file> -k test_register_and_track_with_sparse_metadata -v`
Expected: FAIL with `TypeError: register() got an unexpected keyword
argument 'sparse_metadata'`.

- [ ] **Step 4: Implement `register`/`track` changes**

Read `blond/experimental/physics/kick_pooling.py` fully (`register`,
`clear_buffer`, `_track`, and the `_buffer_time_axis` `OrderedDict` around
lines 75, 124-175) before editing, then:

- Add a parallel `self._buffer_sparse_metadata = OrderedDict()` in
  `__init__` and `clear_buffer`.
- `register(self, time_axis, voltage, sparse_metadata: dict | None = None)`:
  store `sparse_metadata` in `self._buffer_sparse_metadata[key]` alongside
  the existing `self._buffer_time_axis[key]`/voltage buffering (same `key =
  id(time_axis)`, same LRU eviction — evict from both dicts together).
- `_track`: when calling `backend.specials.kick_interpolated(...)`, pass
  `**(self._buffer_sparse_metadata[key] or {})` in addition to the existing
  `bin_centers=time`/`voltage=...` arguments.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest <located_test_file> -v`
Expected: all PASS, including the new sparse test and all pre-existing
tests in that file (no regression on the dense/no-metadata path).

- [ ] **Step 6: Run pre-commit and commit**

```bash
pre-commit run --files blond/experimental/physics/kick_pooling.py <located_test_file>
git add blond/experimental/physics/kick_pooling.py <located_test_file>
git commit -m "Threaded sparse kick metadata through PooledInterpolationKick"
```

---

### Task 8: `InducedVoltage._track` routes sparse profiles correctly

**Files:**
- Modify: `blond/physics/impedances/base.py:507-535`
- Test: `tests/unittests/physics/impedances/sparse_profile/test_profile_integration.py` (extend existing file)

**Interfaces:**
- Consumes: `Task 6`'s `EquidistantMultiProfile.sparse_kick_metadata`.
- Produces: `InducedVoltage._track` no longer raises `ValueError` when `self.profile` is an `EquidistantMultiProfile` with a gapped filling pattern; physics matches the per-bucket ground truth.

- [ ] **Step 1: Read the current call site in full**

Read `blond/physics/impedances/base.py:479-535` (the whole `_track` method
and its surrounding class) to confirm the exact current variable names
before editing (`bin_centers`, `voltage`, `self._delayed_kick`).

- [ ] **Step 2: Write the failing test**

Add to `tests/unittests/physics/impedances/sparse_profile/test_profile_integration.py`,
reusing `_exec_full_sim_with_profiles`'s ring/beam/RF setup as a starting
point (read that method fully first — it currently uses a fully-filled
pattern; this new test needs a genuinely gapped one):

```python
    def test_induced_voltage_track_with_gapped_filling_pattern_does_not_raise(
        self,
    ):
        # Build the same simulation as _exec_full_sim_with_profiles, but
        # swap the filling pattern used for the EquidistantMultiProfile to
        # one with real gaps (not just the trailing zero-padding), then
        # run InducedVoltage._track() and confirm it completes without
        # ValueError from kick_interpolated's uniform-spacing guard.
        ...
```

Fill in `...` with a minimal, concrete simulation matching this test file's
existing conventions (same `Ring`/`ConstantMagneticCycle`/`Beam`/
`DriftSimple`/`SingleHarmonicRFStation` construction already used a few
lines above in `_exec_full_sim_with_profiles`), but:
- construct the `EquidistantMultiProfile` with a `filling_pattern` that has
  an internal gap, e.g. `[1, 1, 0, 0, 1, 1, 0, 0, ...]` rather than a
  single contiguous run,
- attach a real `InducedVoltage`/resonator impedance (reuse
  `resonator_data`/`R_shunt`/`f_res`/`Q_factor` already defined at module
  level),
- call `._track(beam=...)` once,
- assert it does not raise, and assert `beam.dE`/`common_array.dE` changed
  from all-zero (i.e. a kick was actually applied, not silently skipped).

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/unittests/physics/impedances/sparse_profile/test_profile_integration.py -k gapped_filling_pattern -v`
Expected: FAIL with `ValueError` raised by Task 1's guard (today's
`InducedVoltage._track` passes `self.profile.hist_x` straight into
`kick_interpolated` with no sparse metadata).

- [ ] **Step 4: Implement the fix**

In `blond/physics/impedances/base.py`, at the top of the file add the
import:

```python
from blond.physics.profiles_sparse import EquidistantMultiProfile
```

In `_track`, replace:

```python
        bin_centers = self.profile.hist_x  # base for induced voltage
        if self._delayed_kick is not None:
            # Relies on PooledInterpolationKick.track()
            # being called later.
            self._delayed_kick.register(
                time_axis=bin_centers,
                voltage=voltage,
            )
        else:
            backend.specials.kick_interpolated(
```

with:

```python
        bin_centers = self.profile.hist_x  # base for induced voltage
        sparse_metadata = (
            self.profile.sparse_kick_metadata
            if isinstance(self.profile, EquidistantMultiProfile)
            else None
        )
        if self._delayed_kick is not None:
            # Relies on PooledInterpolationKick.track()
            # being called later.
            self._delayed_kick.register(
                time_axis=bin_centers,
                voltage=voltage,
                sparse_metadata=sparse_metadata,
            )
        else:
            backend.specials.kick_interpolated(
```

and pass `**(sparse_metadata or {})` as additional kwargs on the existing
`backend.specials.kick_interpolated(...)` call a few lines below (read the
full call's current argument list first, then add the new kwarg-splat
without altering the existing `dt`/`dE`/`voltage`/`bin_centers`/`charge`/
`acceleration_kick` arguments).

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unittests/physics/impedances/sparse_profile/ -v`
Expected: all PASS, including the new test and every pre-existing test in
that directory (no regression on fully-filled patterns).

- [ ] **Step 6: Run pre-commit and commit**

```bash
pre-commit run --files blond/physics/impedances/base.py tests/unittests/physics/impedances/sparse_profile/test_profile_integration.py
git add blond/physics/impedances/base.py tests/unittests/physics/impedances/sparse_profile/test_profile_integration.py
git commit -m "Routed InducedVoltage kick through sparse-aware path for EquidistantMultiProfile"
```

---

### Task 9: Cavity feedback kick routes sparse profiles correctly

**Files:**
- Modify: `blond/physics/cavities.py:911-931,1300-1330,1855-1885` (`_track_interp` and its two call sites)
- Test: `tests/unittests/physics/test_cavities.py` (check exact filename with `grep -rl "class.*Cavity" tests/unittests/physics/`)

**Interfaces:**
- Consumes: `Task 6`'s `EquidistantMultiProfile.sparse_kick_metadata`.
- Produces: `_track_interp` accepts an optional `sparse_metadata` kwarg and forwards it the same way `impedances/base.py` does in Task 8.

- [ ] **Step 1: Read the current call sites in full**

Read `blond/physics/cavities.py:895-935` (`_track_interp`) and both call
sites around lines 1300-1330 and 1855-1885 (`time_axis =
self.cavity_feedback_list[0].profile.hist_x`) before editing.

- [ ] **Step 2: Write the failing test**

Follow the same pattern as Task 8's integration test, but exercising an
`RFCavity` (or whichever concrete class owns `_track_interp` /
`cavity_feedback_list`) configured with a `cavity_feedback_list[0].profile`
that is an `EquidistantMultiProfile` with a genuinely gapped filling
pattern; assert `_track()` completes without `ValueError` and applies a
non-zero kick. Base the simulation setup on whatever fixture the located
test file already uses for cavity-feedback tests — read it fully first.

- [ ] **Step 3: Run test to verify it fails**

Run the located test with `-k` matching the new test name.
Expected: FAIL with the Task 1 `ValueError`.

- [ ] **Step 4: Implement the fix**

In `blond/physics/cavities.py`:

- Add `sparse_metadata: dict | None = None` to `_track_interp`'s signature,
  and pass `**(sparse_metadata or {})` on its
  `backend.specials.kick_interpolated(...)` call (and on
  `self._delayed_kick.register(..., sparse_metadata=sparse_metadata)` in the
  `if self._delayed_kick is not None` branch).
- At both call sites building `time_axis =
  self.cavity_feedback_list[0].profile.hist_x`, also compute:

```python
                sparse_metadata = (
                    self.cavity_feedback_list[0].profile.sparse_kick_metadata
                    if isinstance(
                        self.cavity_feedback_list[0].profile,
                        EquidistantMultiProfile,
                    )
                    else None
                )
```

  and pass `sparse_metadata=sparse_metadata` into the `self._track_interp(
  ...)` call right below.
- Add the import: `from blond.physics.profiles_sparse import
  EquidistantMultiProfile`.

- [ ] **Step 5: Run test to verify it passes**

Run the located test file in full.
Expected: all PASS, no regression on the existing (dense/uniform) cavity
feedback tests.

- [ ] **Step 6: Run pre-commit and commit**

```bash
pre-commit run --files blond/physics/cavities.py <located_test_file>
git add blond/physics/cavities.py <located_test_file>
git commit -m "Routed cavity feedback kick through sparse-aware path for EquidistantMultiProfile"
```

---

### Task 10: End-to-end integration test

**Files:**
- Modify: `tests/unittests/physics/impedances/sparse_profile/test_profile_integration.py`

**Interfaces:**
- Consumes: everything from Tasks 1-8 (the sparse kick path end-to-end
  through `InducedVoltage`).
- Produces: a regression test proving a full multi-turn simulation with a
  gapped filling pattern produces the same induced-voltage kick physics as
  an equivalent single-bunch (fully dense) simulation of the same bucket.

- [ ] **Step 1: Write the test**

Extend `_exec_full_sim_with_profiles`'s pattern (or add a sibling method)
to run `n_turns > 1` with a gapped `filling_pattern`, tracking beam `dE`
each turn. In parallel, run the *same* physical bucket in isolation as a
plain `StaticProfile`-backed `InducedVoltage` simulation (no
`EquidistantMultiProfile` at all — just that one bucket, same resonator,
same beam). Assert the two simulations' final `beam.dE` distributions for
the bunch in that bucket agree to a tight `rtol` (e.g. `1e-8`) after
`n_turns` turns.

This directly matches this file's existing
`_test_both_parameters_equal`/`_test_both_results_equal` two-simulation
comparison style — read `_exec_full_sim_with_profiles` and
`test_compare_both_profiles` fully before writing this, and mirror their
structure rather than inventing a new one.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unittests/physics/impedances/sparse_profile/test_profile_integration.py -v`
Expected: FAIL before this task (if Tasks 1-8 are already merged, this
should actually PASS immediately — in that case skip straight to Step 3
and treat this as a verification-only task, not TDD RED/GREEN. If it does
fail, debug against the ground-truth single-bucket run before touching any
backend code again — a failure here after Tasks 1-8 are done indicates a
mismatch in how `cut_width`/`first_left_cut` map between the sparse and
dense representations, not a new bug to fix ad hoc.)

- [ ] **Step 3: Run full test suite**

Run: `BLOND_FORCE_TEST_ALL_BACKENDS=True python -m pytest tests/unittests/ -v`
Expected: all PASS, coverage not decreased vs. `main`/`blonder`.

- [ ] **Step 4: Run pre-commit and commit**

```bash
pre-commit run --all-files
git add tests/unittests/physics/impedances/sparse_profile/test_profile_integration.py
git commit -m "Added end-to-end regression test for sparse-profile induced-voltage kick"
```

---

## Self-Review Notes

- **Spec coverage:** Goal 1 (correct sparse kick) — Tasks 2-4. Goal 2
  (structural guard) — Task 1. Goal 3 (real call sites) — Tasks 7-9.
  Integration verification — Task 10. All three spec goals have tasks.
- **Type consistency:** `sparse_kick_metadata` dict keys (Task 6) match
  `kick_interpolated`'s new kwarg names exactly (Tasks 1-4) and are reused
  verbatim by Tasks 7-9 via `**sparse_metadata`/`**(sparse_metadata or {})`.
- **Known follow-up, not blocking:** `barrier_bucket.py`'s own
  `backend.linspace(0, t_rev, n_bins)`-built `bins` (line 232) is
  self-generated and already uniform — Task 1's guard should pass it
  unchanged, but it's worth a quick manual check during Task 1 that this
  call site isn't accidentally broken by the new guard.
