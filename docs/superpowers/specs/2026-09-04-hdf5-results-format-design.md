# HDF5 results format for BLonD observables

Date: 2026-09-04
Branch: `355-replace-npy-by-hdf-h5`
Status: design approved, implementation plan pending

## Problem

`DenseArrayRecorder.to_disk` (`blond/handle_results/array_recorders.py:187`)
is the only place in the non-legacy tree that writes simulation results. It
emits two files per recorder — `<filepath>.npy` (the array) and
`<filepath>.json` (`_write_idx`, `overwrite`). Since each observable owns
three to six recorders, a single simulation scatters dozens of loose,
version-opaque files across a folder.

That format has three problems:

1. **Closed.** `.npy` is a NumPy-specific container. Reading BLonD results
   from C++, MATLAB, Julia or a generic archive tool means reimplementing
   the NumPy header format.
2. **Unarchivable.** Nothing in a `.npy` file records which BLonD version
   wrote it, when, or what the array means. The array/metadata split across
   two files makes partial or mismatched result sets easy to produce.
3. **Unversioned.** There is no schema version, so any future change to what
   BLonD records silently breaks every previously archived result.

## Goals

- One self-describing HDF5 file per save, readable from any language.
- A stamped schema version plus a forward migration path, so results written
  today stay loadable after the format evolves.
- Loud, specific failures on any mismatch — never a silent misread.

## Non-goals

- Backwards compatibility with existing `.npy` / `.json` results. This is a
  clean break; old result sets are not readable by this branch.
- The read-only `.npz` loaders in
  `blond/specifics/muon_collider/beam_preparation.py` and the `.npy`/`.npz`
  test fixtures. Those are simulation *inputs* and pinned reference data,
  not result saving.
- Dataset compression. Deferred; see Follow-ups.

## On-disk layout

One file per save. Group per observable, dataset per recorder.

```
run1.h5
  attrs:
    blond_results_format_version : int  = 1
    blond_version                : str  (from blond._version.__version__)
    created                      : str  (ISO-8601)
  /beam1                          attrs: observable_class = "BunchStatistics"
      mean_dt                     attrs: write_idx
      sigma_dt                    attrs: write_idx
  /RFStationPhases                attrs: observable_class = "RFStationPhaseObservation"
      phases
      omegas
      voltages
```

**Dataset names** are the recorder's attribute name with the leading
underscore stripped (`_mean_dt` -> `mean_dt`).
`ObservablesBaseClass.get_recorders` (`observables.py:128`) already returns
those attribute names, so no new bookkeeping is required.

**Group names** resolve in this order:

1. the observable's optional `name=` constructor argument;
2. otherwise `type(observable).__name__`;
3. on collision within one file, suffix `_1`, `_2`, ... and emit a
   `UserWarning` telling the user to pass `name=` for a stable, meaningful
   group name.

Positional indexing is deliberately *not* used as the primary scheme: group
names must stay stable across runs so that a file written today reloads
after the `observe=` tuple is reordered.

## Component design

### `blond/handle_results/hdf5_io.py` (new)

Owns everything that knows about the file format:

- `FORMAT_VERSION: int` — currently `1`.
- `ResultsFormatError(Exception)` — every format-level failure.
- File open/create helpers that stamp the root attributes on write and
  validate them on read.
- The migration registry (see below).

### `DenseArrayRecorder` (`array_recorders.py`)

The recorder no longer knows where it lives; the group owns the location.

Removed: `filepath`, `filepath_array`, `filepath_attributes`, `to_disk`,
`from_disk`, `purge_from_disk`, and the `np.save` / `np.load` /
`json.dump` code paths.

Added:

- `to_group(group, name)` — writes `group[name]` from `self._memory`, with
  the dataset attribute `write_idx`.
- `from_group(group, name)` — classmethod returning a populated recorder.

`overwrite` moves from a per-recorder flag to a file-level concern,
expressed as the h5py file mode (`w` for overwrite, `w-` to refuse).

This is a **breaking change** to the `DenseArrayRecorder` constructor
(`filepath` was its first positional argument) and it removes the
`f"{self.common_filepath}_<name>"` argument from roughly twenty call sites
in `observables.py` and `observables_as_elements.py`. Approved as a clean
break; no deprecation shim.

### `ObservablesBaseClass` (`observables.py`)

- `__init__` gains an optional `name: str | None = None`; `self.name`
  defaults to the class name. `folder` is retained and now designates where
  a *standalone* save writes its own file.
- `to_disk(group=None)`: with a group, write into it. Without one — the
  standalone path used by `blond/examples/scripts/EX_19_Observable_as_element.py:104`
  — open `<folder>last.h5`, create the observable's own group, and delegate
  to the same code path. Writes the group attribute `observable_class`.
- `from_disk(group=None)`: symmetric. Validates `observable_class`, applies
  migrations, then repopulates each expected recorder from its dataset.
- `rename(new_name)` now sets the group name instead of rewriting per-
  recorder path strings. The old string-replacement implementation
  (`observables.py:145-173`) and its `NameError` guard are deleted.

### `Simulation.save_results` / `load_results` (`core/simulation/simulation.py:1502`, `:1607`)

- `save_results(observe, common_name=None, overwrite=True)` opens **one**
  `h5py.File`, resolves group names as specified above, and calls
  `observable.to_disk(group)` per observable.
- `common_name` is reinterpreted as the file stem (`<common_name>.h5`),
  falling back to `<folder of the first observable>last.h5`.
- `load_results` already calls `finalize()` before loading, so the expected
  recorder set is fully known at load time. That is what makes the
  validation below possible: it opens the file read-only, matches each
  observable to its group by name, migrates, and repopulates.

## Migration and validation

The root attribute `blond_results_format_version` is the contract. On read:

| File version | Behaviour |
|---|---|
| `== FORMAT_VERSION` | Load directly. |
| `< FORMAT_VERSION` | Apply the registered chain of `v -> v+1` upgraders in order, logging each step. |
| `> FORMAT_VERSION` | Raise `ResultsFormatError` naming the file's version and stating that a newer BLonD is required. |

Each migration step is a small pure function over the in-memory datasets and
attributes. **The file on disk is never rewritten**, so archived results stay
bit-identical no matter how often they are read. Dataset renames, additions
and removals are expressed as migration steps, which is what keeps a
recorder rename from orphaning old archives.

Structural validation on load:

- A missing expected dataset raises `ResultsFormatError` listing exactly
  which datasets are absent.
- An unexpected extra dataset emits a warning and is ignored, so a file
  written by a slightly newer minor version stays usable.
- A group whose `observable_class` does not match the observable being
  loaded into raises rather than silently loading mismatched data.

## Error handling summary

Every failure mode is explicit and names the offending file, group or
dataset: newer format version, missing dataset, class mismatch, refusing to
overwrite under `w-`. Nothing degrades silently to partial data.

## Testing

Strict TDD, visible RED before each implementation step. All test classes
inherit `unittest.TestCase` and use its assertions.

New `tests/unittests/handle_results/test_hdf5_io.py`:

- array round-trip through `to_group` / `from_group`, including `write_idx`
  and non-default dtypes;
- root attributes stamped on write;
- migration chain applied for an older version, with a synthetic old file;
- newer version raises `ResultsFormatError`;
- missing dataset raises and names the dataset;
- extra dataset warns and is ignored;
- group-name collision auto-suffixes and warns;
- `w-` refuses to clobber an existing file.

Rewritten: the `to_disk` / `from_disk` tests in
`tests/unittests/handle_results/test_observables.py`,
`test_observables_as_elements.py`, and
`tests/unittests/core/simulation/test_simulation.py:324-387`.

Updated: `blond/examples/scripts/EX_19_Observable_as_element.py`.

CI rejects any coverage decrease, so the new module ships with its tests.

## Dependencies

`h5py` is promoted from the `legacy` extra (`pyproject.toml:86`, pinned
`>=3.11.0,<=3.15.1`) to a core runtime dependency. It is a mandatory import
for saving results, so it cannot stay optional.

## Follow-ups (out of scope here)

- **Dataset compression** (gzip). Deferred by explicit decision; worth
  revisiting once the format is in place, since archive size is the main
  motivation and enabling it later is a non-breaking change.
- **Physical units as dataset attributes.** BLonD is strictly SI
  (volts, eV, seconds, radians); recording the unit per dataset would make
  files genuinely self-describing for non-Python readers.
- Migrating the muon-collider `.npz` input loaders, if input formats are
  ever unified with output formats.
