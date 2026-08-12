# Fixing silent island-mismatch physics bugs from `EquidistantMultiProfile.hist_x`

Date: 2026-08-12
Branch: `blonder_bugfix/sparse-interp`

## Problem

`EquidistantMultiProfile.hist_x` (`blond/physics/profiles_sparse.py`) concatenates
one `StaticProfile.hist_x` per **active** RF bucket, in filling-pattern order. Bin
spacing is uniform *within* a bucket's island, but the gap between the last bin of
one active island and the first bin of the next is wider whenever the filling
pattern skips buckets — i.e. `hist_x` is only globally uniformly spaced by
coincidence (a fully-filled pattern with no gaps).

`kick_interpolated` (implemented identically, per backend parity, in
`python`/`numba`/`cpp`/`cuda` under `blond/core/backends/*/callables.py`) assumes
`bin_centers` *is* globally uniform: it derives one `inv_bin_width` from
`bin_centers[0]`/`bin_centers[-1]` and floor-indexes into `voltage` with it. Fed a
sparse `hist_x` with gaps, particles land in the wrong bin — often a bin belonging
to a different bucket's island entirely — silently, with no exception and no NaN.

Confirmed by reproduction: `filling_pattern=[1,0,0,1]`, 4 bins/bucket, a particle
sitting exactly on `hist_x[4]` (the first bin of the second island) resolves to
`fbin=5` instead of `4`.

This is reachable in production (non-experimental) code:
`impedances/base.py:525` (`InducedVoltage._track`) and `cavities.py:1308`
(`RFCavity._track_interp` via `cavity_feedback_list[0].profile.hist_x`) both pass
`profile.hist_x` straight into `kick_interpolated`, and `EquidistantMultiProfile`
is duck-typed to stand in for `profile` there — exercised in `EX_20`, `EX_28`, and
`tests/unittests/physics/impedances/sparse_profile/`.

The user's stated concern: *every* future consumer of `hist_x` inherits this same
risk unless something structural prevents it, not just this one call site.

## Goals

1. Make `kick_interpolated` produce **correct** physics when handed a sparse,
   island-structured `hist_x` — as fast as the existing per-bucket
   `histogram_sparse` kernel (no python-level per-particle bucket search).
2. Make it **structurally impossible to silently get this wrong**: any call that
   hands a non-uniform `bin_centers` to the plain (non-sparse) code path raises a
   clear, actionable exception instead of computing garbage.
3. Update all real call sites (`cavities.py`, `impedances/base.py`,
   `blond/experimental/physics/kick_pooling.py`) to route sparse profiles through
   the correct path.

## Non-goals

- Densifying `EquidistantMultiProfile` to a full-ring uniform grid (defeats the
  point of "sparse" for large harmonic numbers with few filled buckets).
- Touching `histogram_sparse` itself — it is already correctly island-aware and is
  the reference pattern this design follows.

## Design

### One function, two internal paths

`kick_interpolated`'s signature gains **optional** sparse-island metadata,
mirroring exactly what `EquidistantMultiProfile` already carries and what
`histogram_sparse` already consumes:

```python
def kick_interpolated(
    dt, dE, voltage, bin_centers, charge, acceleration_kick,
    # New, optional — all-or-nothing:
    first_left_cut: float | None = None,
    left_cut_distance: float | None = None,
    cut_width: float | None = None,
    bins_per_profile: int | None = None,
    filling_pattern: NumpyArray | None = None,
    bucket_index_to_memory_index: NumpyArray | None = None,
) -> None:
```

- **Sparse metadata omitted (`None`):** existing fast global-uniform path, but
  first validated for actual uniform spacing (see Guard below).
- **Sparse metadata provided:** new path. For each particle, resolve its bucket
  index via `floor((dt - first_left_cut) / left_cut_distance)`, check
  `filling_pattern[bucket]` (skip/ignore if unfilled — mirrors `histogram_sparse`'s
  `if not active: continue` semantics for out-of-population particles), map to
  its island's memory offset via `bucket_index_to_memory_index`, then linearly
  interpolate **within that island's own bin width** (`cut_width / bins_per_profile`),
  exactly as the dense path does locally. Implemented per-backend the same way
  `histogram_sparse` already is — vectorized/parallelized (numba `prange`, cpp,
  cuda kernel), no python-level per-particle search, no host↔device transfer in
  the loop.

This keeps one public symbol (`backend.specials.kick_interpolated`) and one
signature across all four backends, per `Specials` ABC parity requirements.

### Guard on the dense path

When sparse metadata is *not* passed, `kick_interpolated` checks `bin_centers` for
uniform spacing (`np.diff` + `allclose`, O(n_slices), once per call — not inside
the per-particle loop, so it doesn't violate the hot-path-no-guards convention)
before computing `inv_bin_width`. If non-uniform, raise `ValueError` naming both
correct routes:

> `bin_centers is not uniformly spaced (looks like a sparse/multi-island
> EquidistantMultiProfile.hist_x). Either pass this profile's sparse metadata
> (first_left_cut, left_cut_distance, cut_width, bins_per_profile,
> filling_pattern, bucket_index_to_memory_index) to kick_interpolated, or use
> EquidistantMultiProfile.profiles[i].hist_x for a single bucket.`

This is a `ValueError`, not `assert` — unlike the dtype/contiguity `assert`s
elsewhere in the backend wrappers (which are intentionally stripped by
`python -O` for hot-loop performance), this check guards against silently wrong
*physics*, must survive `-O`, and costs O(n_slices) once per call, not per
particle.

### Call site changes

- `blond/physics/impedances/base.py::InducedVoltage._track` and
  `blond/physics/cavities.py::RFCavity._track_interp` (and its callers building
  `time_axis`): branch on `isinstance(self.profile, EquidistantMultiProfile)` (or
  `self.cavity_feedback_list[0].profile`, respectively) and pass the profile's
  sparse metadata (`_first_left_cut`, `_left_cut_distance`,
  `cut_right - cut_left` of one bucket, `_bins_per_profile`,
  `_filling_pattern`, `_bucket_index_to_memory_index`) through to
  `kick_interpolated`.
- `blond/experimental/physics/kick_pooling.py::PooledInterpolationKick`: `register()`
  must accept and buffer the same optional metadata alongside `time_axis`/`voltage`
  (keyed together), and `track()` forwards it to `kick_interpolated` on replay.

### Backends touched (parity mandatory)

`blond/core/backends/backend.py` (`Specials` ABC signature),
`python/numba/cpp/cuda` under `blond/core/backends/*/callables.py`.

## Testing (TDD)

1. **RED, documents the bug:** feed a sparse `hist_x` (gapped filling pattern)
   into today's `kick_interpolated` with no metadata — assert it currently
   produces a *wrong* `dE` (pins the bug before the fix). After the guard lands,
   this same call must raise `ValueError` instead.
2. **Sparse-path correctness:** for each active bucket, and specifically for a
   particle sitting exactly on an island boundary bin, compare
   `kick_interpolated` (sparse path) against `kick_interpolated` (dense path) run
   on that single bucket's own `StaticProfile.hist_x`/`voltage` slice — results
   must match to tolerance.
3. **Guard correctness:** dense path raises `ValueError` with the documented
   message for non-uniform `bin_centers`; still passes silently (no regression)
   for genuinely uniform ones (existing `StaticProfile` usage, `barrier_bucket.py`'s
   own `linspace`-built bins).
4. **Backend parity:** run all of the above under
   `BLOND_FORCE_TEST_ALL_BACKENDS=True`.
5. **Integration:** `impedances/base.py`/`cavities.py` end-to-end with an
   `EquidistantMultiProfile`, gapped filling pattern — asserts no `ValueError`
   and correct induced-voltage/cavity-kick physics vs. a fully-filled reference
   case (extending the existing
   `tests/unittests/physics/impedances/sparse_profile/test_profile_integration.py`
   pattern).

## Open risk / to confirm during implementation

- `PooledInterpolationKick.register()` keys buffered arrays by `id(time_axis)`
  (`kick_pooling.py:143`) — need to check this still works cleanly once a second,
  optional metadata bundle is threaded through it.
