# HDF5 Results Format Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-recorder `.npy` + `.json` result files with a single
self-describing, version-stamped HDF5 file per save.

**Architecture:** A new `blond/handle_results/hdf5_io.py` owns everything that
knows about the file format: the schema version, the root attributes, and a
migration chain applied to an in-memory payload on read. `DenseArrayRecorder`
loses all knowledge of file paths and gains `to_group` / `from_payload`.
`ObservablesBaseClass` writes one HDF5 group per observable, either into a
group handed to it by `Simulation.save_results` (one file for the whole run)
or into its own file when used standalone.

**Tech Stack:** Python >=3.10, h5py (promoted to a core dependency), NumPy,
`unittest` + pytest, pre-commit (ruff, isort, numpydoc-validation).

**Spec:** `docs/superpowers/specs/2026-09-04-hdf5-results-format-design.md`

## Global Constraints

- Line length 79 everywhere (ruff + isort). Break lines rather than shortening
  names.
- Every new file under `blond/` needs the copyright header from
  `dev_tools/copyright_notice.txt`. Bulk-apply with
  `python dev_tools/copy_copyright_to_all_files.py`.
- Public API docstrings are NumPy style, enforced by `numpydoc-validation`
  and by the `-W` Sphinx build. Write Parameters / Returns / Raises sections.
- Every test class inherits `unittest.TestCase` and uses its assertions
  (`assertEqual`, `assertRaises`, `assertWarns`). Never a bare `assert` in
  test code.
- **Run `pre-commit run --all-files` and read the output before every
  `git commit`.** If a hook auto-fixes files, `git add` them and re-run until
  green. Never commit on a red hook.
- Work stays on the branch `355-replace-npy-by-hdf-h5`. Never commit to
  `blonder`.
- Strict TDD: write the test, run it, *see it fail*, then implement.
- `h5py` pin is `>=3.11.0,<=3.15.1` (already in `pyproject.toml:86` under the
  `legacy` extra).
- CI rejects any coverage decrease. Every new module ships with its tests.
- Clean break: no `.npy` reading or writing survives anywhere in
  `blond/handle_results/`. No deprecation shims.

## Naming decision carried into this plan

Dataset names are the recorder attribute name with the leading underscore
stripped. That normalizes one pre-existing inconsistency: the attribute
`_rms_emittance` is currently written to the file suffix `_emittance_stat`
(`blond/handle_results/observables_as_elements.py:296`). Under the new format
it becomes the dataset `rms_emittance`. This is deliberate.

---

### Task 1: Format module — version stamping and migration machinery

**Files:**
- Create: `blond/handle_results/hdf5_io.py`
- Modify: `pyproject.toml` (move `h5py` into `[project] dependencies`)
- Test: `tests/unittests/handle_results/test_hdf5_io.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `FORMAT_VERSION: int` (== 1)
  - `ATTR_FORMAT_VERSION: str` (== `"blond_results_format_version"`)
  - `ATTR_OBSERVABLE_CLASS: str` (== `"observable_class"`)
  - `ATTR_WRITE_IDX: str` (== `"write_idx"`)
  - `ResultsFormatError(Exception)`
  - `GroupPayload = dict[str, tuple[NumpyArray, dict[str, Any]]]`
  - `results_filepath(stem: str | PathLike) -> Path`
  - `create_results_file(stem, overwrite: bool = True) -> h5py.File`
  - `open_results_file(stem) -> h5py.File`
  - `read_format_version(file: h5py.File) -> int`
  - `read_group_payload(group: h5py.Group) -> GroupPayload`
  - `migrate_payload(payload, from_version, to_version=FORMAT_VERSION,
    migrations=MIGRATIONS) -> GroupPayload`
  - `MIGRATIONS: dict[int, Callable[[GroupPayload], GroupPayload]]` (empty)

- [ ] **Step 1: Promote h5py to a core dependency**

In `pyproject.toml`, add to the `[project]` `dependencies` array (keep the
existing pin and the trailing comment style of its neighbours):

```toml
  "h5py>=3.11.0,<=3.15.1",      # HDF5 file format
```

Leave the copy inside the `legacy` extra untouched — legacy pins its own
dependency set on purpose.

- [ ] **Step 2: Write the failing tests**

Create `tests/unittests/handle_results/test_hdf5_io.py`:

```python
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from blond.handle_results.hdf5_io import (
    ATTR_FORMAT_VERSION,
    FORMAT_VERSION,
    ResultsFormatError,
    create_results_file,
    migrate_payload,
    open_results_file,
    read_group_payload,
    results_filepath,
)


class TestResultsFilepath(unittest.TestCase):
    def test_appends_suffix(self) -> None:
        self.assertEqual(results_filepath("run1"), Path("run1.h5"))

    def test_keeps_existing_suffix(self) -> None:
        self.assertEqual(results_filepath("run1.h5"), Path("run1.h5"))


class TestResultsFile(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.stem = str(Path(self._tmp.name) / "run")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_root_attributes_stamped(self) -> None:
        with create_results_file(self.stem) as file:
            pass
        with h5py.File(results_filepath(self.stem), "r") as file:
            self.assertEqual(
                int(file.attrs[ATTR_FORMAT_VERSION]), FORMAT_VERSION
            )
            self.assertIn("blond_version", file.attrs)
            self.assertIn("created", file.attrs)

    def test_no_overwrite_refuses_existing_file(self) -> None:
        with create_results_file(self.stem):
            pass
        with self.assertRaises(FileExistsError):
            with create_results_file(self.stem, overwrite=False):
                pass

    def test_open_rejects_newer_format_version(self) -> None:
        with create_results_file(self.stem) as file:
            file.attrs[ATTR_FORMAT_VERSION] = FORMAT_VERSION + 1
        with self.assertRaises(ResultsFormatError):
            with open_results_file(self.stem):
                pass

    def test_open_rejects_file_without_version(self) -> None:
        with h5py.File(results_filepath(self.stem), "w"):
            pass
        with self.assertRaises(ResultsFormatError):
            with open_results_file(self.stem):
                pass

    def test_open_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            with open_results_file(self.stem + "_absent"):
                pass


class TestReadGroupPayload(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.stem = str(Path(self._tmp.name) / "run")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_reads_arrays_and_attributes(self) -> None:
        with create_results_file(self.stem) as file:
            group = file.create_group("obs")
            dataset = group.create_dataset("mean_dt", data=np.arange(4.0))
            dataset.attrs["write_idx"] = 3
        with open_results_file(self.stem) as file:
            payload = read_group_payload(file["obs"])
        array, attrs = payload["mean_dt"]
        np.testing.assert_array_equal(array, np.arange(4.0))
        self.assertEqual(int(attrs["write_idx"]), 3)


class TestMigratePayload(unittest.TestCase):
    def test_no_migration_when_versions_match(self) -> None:
        payload = {"a": (np.zeros(2), {"write_idx": 1})}
        result = migrate_payload(
            payload, from_version=FORMAT_VERSION, migrations={}
        )
        self.assertEqual(list(result), ["a"])

    def test_applies_chain_in_order(self) -> None:
        calls: list[int] = []

        def step_1(payload):
            calls.append(1)
            return {"b": payload["a"]}

        def step_2(payload):
            calls.append(2)
            return {"c": payload["b"]}

        result = migrate_payload(
            {"a": (np.zeros(2), {})},
            from_version=1,
            to_version=3,
            migrations={1: step_1, 2: step_2},
        )
        self.assertEqual(calls, [1, 2])
        self.assertEqual(list(result), ["c"])

    def test_missing_migration_step_raises(self) -> None:
        with self.assertRaises(ResultsFormatError):
            migrate_payload(
                {"a": (np.zeros(2), {})},
                from_version=1,
                to_version=3,
                migrations={1: lambda payload: payload},
            )
```

- [ ] **Step 3: Run the tests and confirm RED**

Run: `python -m pytest -v tests/unittests/handle_results/test_hdf5_io.py`
Expected: collection error, `ModuleNotFoundError: No module named
'blond.handle_results.hdf5_io'`.

- [ ] **Step 4: Implement `hdf5_io.py`**

Create `blond/handle_results/hdf5_io.py` with the copyright header, then:

```python
"""Read and write BLonD simulation results as HDF5 files."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import h5py

from blond._version import __version__

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from os import PathLike
    from typing import Any

    from numpy.typing import NDArray as NumpyArray

    GroupPayload = dict[str, tuple[NumpyArray, dict[str, Any]]]

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
"""Schema version of the results file written by this BLonD version."""

FILE_SUFFIX = ".h5"

ATTR_FORMAT_VERSION = "blond_results_format_version"
ATTR_BLOND_VERSION = "blond_version"
ATTR_CREATED = "created"
ATTR_OBSERVABLE_CLASS = "observable_class"
ATTR_WRITE_IDX = "write_idx"

MIGRATIONS: dict[int, Callable[[GroupPayload], GroupPayload]] = {}
"""Upgraders keyed by the version they migrate *from*, to that version + 1."""


class ResultsFormatError(Exception):
    """Raised when a results file cannot be interpreted safely."""
```

Then the functions. `results_filepath` appends `FILE_SUFFIX` unless already
present. `create_results_file` opens h5py in mode `"w"` when `overwrite` else
`"w-"` and stamps the three root attributes:

```python
def create_results_file(
    stem: str | PathLike,
    overwrite: bool = True,
) -> h5py.File:
    filepath = results_filepath(stem)
    file = h5py.File(filepath, "w" if overwrite else "w-")
    file.attrs[ATTR_FORMAT_VERSION] = FORMAT_VERSION
    file.attrs[ATTR_BLOND_VERSION] = __version__
    file.attrs[ATTR_CREATED] = datetime.now(timezone.utc).isoformat()
    return file
```

`open_results_file` opens mode `"r"`, raising `FileNotFoundError` if the file
is absent, then validates:

```python
def open_results_file(stem: str | PathLike) -> h5py.File:
    filepath = results_filepath(stem)
    if not filepath.is_file():
        raise FileNotFoundError(f"No results file at {filepath}.")
    file = h5py.File(filepath, "r")
    try:
        file_format_version = read_format_version(file)
    except Exception:
        file.close()
        raise
    if file_format_version > FORMAT_VERSION:
        file.close()
        raise ResultsFormatError(
            f"{filepath} was written with results format version"
            f" {file_format_version}, but this BLonD ({__version__})"
            f" understands at most version {FORMAT_VERSION}."
            f" Upgrade BLonD to read this file."
        )
    return file
```

`read_format_version(file)` returns `int(file.attrs[ATTR_FORMAT_VERSION])`
and raises `ResultsFormatError` naming the file when the attribute is absent
(that is how a foreign or pre-format HDF5 file is rejected).

`read_group_payload` copies every dataset in the group into memory:

```python
def read_group_payload(group: h5py.Group) -> GroupPayload:
    return {
        name: (dataset[()], dict(dataset.attrs))
        for name, dataset in group.items()
    }
```

`migrate_payload` walks the chain, logging each step, and raises
`ResultsFormatError` when a step is missing:

```python
def migrate_payload(
    payload: GroupPayload,
    from_version: int,
    to_version: int = FORMAT_VERSION,
    migrations: dict[int, Callable[[GroupPayload], GroupPayload]] | None = None,
) -> GroupPayload:
    if migrations is None:
        migrations = MIGRATIONS
    for version in range(from_version, to_version):
        if version not in migrations:
            raise ResultsFormatError(
                f"No migration registered from results format version"
                f" {version} to {version + 1}."
            )
        logger.info(f"Migrating results payload {version} -> {version + 1}.")
        payload = migrations[version](payload)
    return payload
```

Note the `to_version` default is evaluated at import time; that is fine
because `FORMAT_VERSION` is a module constant.

- [ ] **Step 5: Run the tests and confirm GREEN**

Run: `python -m pytest -v tests/unittests/handle_results/test_hdf5_io.py`
Expected: all PASS.

- [ ] **Step 6: Apply the copyright header and run pre-commit**

```bash
python dev_tools/copy_copyright_to_all_files.py
pre-commit run --all-files
```

Re-stage and re-run until every hook is green.

- [ ] **Step 7: Commit**

```bash
git add blond/handle_results/hdf5_io.py \
        tests/unittests/handle_results/test_hdf5_io.py pyproject.toml
git commit -m "Added HDF5 results format module

Introduced the schema version, root attributes and payload migration
chain that the new results format is built on, and promoted h5py from
the legacy extra to a core dependency."
```

---

### Task 2: `DenseArrayRecorder` writes to a group instead of a file

**Files:**
- Modify: `blond/handle_results/array_recorders.py` (whole file)
- Test: `tests/unittests/handle_results/test_array_recorders.py` (create)
- Test: `tests/unittests/handle_results/test_observables.py:94-118` (delete
  the obsolete `TestDenseArrayRecorder` class; its replacement lives in the
  new test file)

**Interfaces:**
- Consumes: `ATTR_WRITE_IDX`, `create_results_file` from Task 1.
- Produces:
  - `DenseArrayRecorder(shape, dtype=None, order="C", preallocate=True)` —
    `filepath` and `overwrite` are **gone** from the signature.
  - `DenseArrayRecorder.to_group(group: h5py.Group, name: str) -> None`
  - `DenseArrayRecorder.from_payload(array, attrs) -> DenseArrayRecorder`
    (classmethod)
  - `ArrayRecorder` ABC now declares `to_group` / `from_payload` instead of
    `to_disk` / `from_disk`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unittests/handle_results/test_array_recorders.py`:

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np

from blond.handle_results.array_recorders import DenseArrayRecorder
from blond.handle_results.hdf5_io import (
    create_results_file,
    open_results_file,
    read_group_payload,
)


class TestDenseArrayRecorder(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.stem = str(Path(self._tmp.name) / "run")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _written_recorder(self) -> DenseArrayRecorder:
        recorder = DenseArrayRecorder(shape=(4, 2), dtype=float)
        recorder.write(np.array([1.0, 2.0]))
        recorder.write(np.array([3.0, 4.0]))
        return recorder

    def test_roundtrip_preserves_values_and_write_idx(self) -> None:
        recorder = self._written_recorder()
        with create_results_file(self.stem) as file:
            recorder.to_group(file.create_group("obs"), "values")
        with open_results_file(self.stem) as file:
            payload = read_group_payload(file["obs"])
        loaded = DenseArrayRecorder.from_payload(*payload["values"])
        np.testing.assert_array_equal(loaded._memory, recorder._memory)
        self.assertEqual(loaded._write_idx, 2)
        np.testing.assert_array_equal(
            loaded.get_valid_entries(), recorder.get_valid_entries()
        )

    def test_roundtrip_preserves_dtype(self) -> None:
        recorder = DenseArrayRecorder(shape=(2,), dtype=np.float32)
        recorder.write(np.float32(1.5))
        with create_results_file(self.stem) as file:
            recorder.to_group(file.create_group("obs"), "values")
        with open_results_file(self.stem) as file:
            payload = read_group_payload(file["obs"])
        loaded = DenseArrayRecorder.from_payload(*payload["values"])
        self.assertEqual(loaded._memory.dtype, np.float32)

    def test_write_with_mask_sets_nan_outside_mask(self) -> None:
        recorder = DenseArrayRecorder(shape=(1, 3), dtype=float)
        mask = np.array([True, False, True])
        recorder.write(np.array([1.0, 2.0]), mask=mask)
        np.testing.assert_array_equal(
            recorder.get_valid_entries()[0],
            np.array([1.0, np.nan, 2.0]),
        )

    def test_constructor_takes_no_filepath(self) -> None:
        with self.assertRaises(TypeError):
            DenseArrayRecorder(filepath="somewhere", shape=(1,))
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m pytest -v tests/unittests/handle_results/test_array_recorders.py`
Expected: FAIL — `TypeError` on the `DenseArrayRecorder(shape=...)` calls
(`filepath` is still a required positional argument) and
`AttributeError: ... has no attribute 'to_group'`.

- [ ] **Step 3: Rewrite `array_recorders.py`**

In the `ArrayRecorder` ABC, replace the abstract `to_disk` / `from_disk` with:

```python
    @abstractmethod  # pragma: no cover
    def to_group(self, group: h5py.Group, name: str) -> None:
        """
        Write the internal array into an HDF5 group.

        Parameters
        ----------
        group
            Open HDF5 group that receives the dataset.
        name
            Name of the dataset inside the group.
        """
        pass

    @classmethod
    @abstractmethod  # pragma: no cover
    def from_payload(
        cls,
        array: NumpyArray,
        attrs: dict[str, Any],
    ) -> ArrayRecorder:
        """
        Rebuild a recorder from a migrated HDF5 payload.

        Parameters
        ----------
        array
            Array as read from the results file.
        attrs
            Dataset attributes as read from the results file.

        Returns
        -------
        recorder
            Recorder holding the loaded array.
        """
        pass
```

In `DenseArrayRecorder`:

- Drop `filepath` and `overwrite` from `__init__`; drop the
  `os.path.exists` warning block. The new signature is
  `__init__(self, shape, dtype=None, order="C", preallocate=True)`.
- Delete `filepath_array`, `filepath_attributes`, `purge_from_disk`,
  `to_disk`, `from_disk`.
- Delete the now-unused `json`, `os.path`, `warnings` and `isfile` imports.
  Keep `numpy` (still used by `write` and the allocation).
- Add:

```python
    def to_group(self, group: h5py.Group, name: str) -> None:
        dataset = group.create_dataset(name, data=self._memory)
        dataset.attrs[ATTR_WRITE_IDX] = self._write_idx

    @classmethod
    def from_payload(
        cls,
        array: NumpyArray,
        attrs: dict[str, Any],
    ) -> DenseArrayRecorder:
        recorder = cls(shape=array.shape, dtype=array.dtype)
        recorder._memory = array
        recorder._write_idx = int(attrs[ATTR_WRITE_IDX])
        return recorder
```

`write` and `get_valid_entries` are unchanged.

- [ ] **Step 4: Run the tests and confirm GREEN**

Run: `python -m pytest -v tests/unittests/handle_results/test_array_recorders.py`
Expected: all PASS.

`python -m pytest tests/unittests/handle_results/` will still fail at this
point — the observables have not been ported yet. That is expected and is
fixed by Task 3.

- [ ] **Step 5: Delete the obsolete recorder tests**

Remove the `TestDenseArrayRecorder` class from
`tests/unittests/handle_results/test_observables.py` (currently lines
94-118). It tests `filepath=` and the overwrite warning, both of which no
longer exist.

- [ ] **Step 6: Run pre-commit and commit**

```bash
pre-commit run --all-files
git add blond/handle_results/array_recorders.py \
        tests/unittests/handle_results/test_array_recorders.py \
        tests/unittests/handle_results/test_observables.py
git commit -m "Moved DenseArrayRecorder from .npy files to HDF5 groups

The recorder no longer owns a filepath; the enclosing HDF5 group owns
the location. Replaced to_disk/from_disk with to_group/from_payload and
dropped the per-recorder overwrite flag, which is now a file-level
concern."
```

---

### Task 3: Observables write one group per observable

**Files:**
- Modify: `blond/handle_results/observables.py:121-197` (`__init__`,
  `rename`, `to_disk`, `from_disk`) and every `DenseArrayRecorder(...)` call
  site in the file (lines ~434-459, 780-800, 954-971, 1126-1135, 1257, 1419,
  1531, 1634-1639, 1770-1775, 1869)
- Modify: `blond/handle_results/observables_as_elements.py:110-124, 280-298,
  471-484` (the `DenseArrayRecorder(...)` call sites)
- Test: `tests/unittests/handle_results/test_observables.py`
- Test: `tests/unittests/handle_results/test_observables_as_elements.py`

**Interfaces:**
- Consumes: `DenseArrayRecorder.to_group` / `from_payload` (Task 2);
  `create_results_file`, `open_results_file`, `read_group_payload`,
  `migrate_payload`, `read_format_version`, `ATTR_OBSERVABLE_CLASS`,
  `ResultsFormatError` (Task 1).
- Produces:
  - `ObservablesBaseClass.__init__(folder=None, group_name=None, **kwargs)`
    with the public attribute `group_name: str`.
  - `ObservablesBaseClass.to_disk(group: h5py.Group | None = None,
    overwrite: bool = True) -> None`
  - `ObservablesBaseClass.from_disk(group: h5py.Group | None = None) -> None`
  - `ObservablesBaseClass.rename(new_group_name: str) -> None` — sets the
    **group** name.
  - `ObservablesBaseClass.purge_from_disk(verbose: bool = True) -> None` —
    deletes the standalone file only.
  - `ObservablesBaseClass.dataset_names() -> dict[str, str]` mapping dataset
    name to recorder attribute name.

- [ ] **Step 1: Write the failing tests**

In `tests/unittests/handle_results/test_observables.py`, replace the body of
`TestObservables.test_from_disk` and add the new cases. `ObservablesHelper`
at the top of that file currently overrides `to_disk` / `from_disk` with
no-ops — **delete those two overrides** so the helper exercises the real
implementation.

`ObservablesHelper` holds **no** `DenseArrayRecorder`, so `get_recorders()`
returns an empty list and every round-trip assertion below would be vacuous.
Add a second helper next to it that actually records something:

```python
class RecordingObservablesHelper(ObservablesOncePerTurnBase):
    def on_run_simulation(self, simulation, beam, n_turns, **kwargs) -> None:
        super().on_run_simulation(
            simulation=simulation, beam=beam, n_turns=n_turns, **kwargs
        )
        self._values = DenseArrayRecorder(
            shape=(self._calc_n_entries(n_turns), 2)
        )

    def _update(self) -> None:
        self._values.write(np.array([1.0, 2.0]))
```

`TestObservablesHdf5` below uses `RecordingObservablesHelper`, not
`ObservablesHelper`. Its single dataset is named `values`.

```python
class TestObservablesHdf5(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = self._tmp.name + "/"
        self.observables = RecordingObservablesHelper(
            each_turn_i=1,
            folder=self.folder,
        )
        self.observables.on_init_simulation(simulation=simulation)
        self.observables.on_run_simulation(
            simulation=simulation, beam=beam, n_turns=100
        )
        self.observables.update()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_default_name_is_class_name(self) -> None:
        self.assertEqual(
            self.observables.name, "RecordingObservablesHelper"
        )

    def test_explicit_name_is_used(self) -> None:
        observables = RecordingObservablesHelper(
            each_turn_i=1, folder=self.folder, group_name="beam1"
        )
        self.assertEqual(observables.name, "beam1")

    def test_standalone_roundtrip(self) -> None:
        before = {
            attribute: recorder.get_valid_entries().copy()
            for attribute, recorder in self.observables.get_recorders()
        }
        self.observables.to_disk()
        self.observables.from_disk()
        for attribute, expected in before.items():
            np.testing.assert_array_equal(
                getattr(self.observables, attribute).get_valid_entries(),
                expected,
            )

    def test_standalone_writes_single_h5_file(self) -> None:
        self.observables.to_disk()
        written = sorted(
            path.name for path in Path(self.folder).iterdir()
        )
        self.assertEqual(written, ["last.h5"])

    def test_no_overwrite_refuses_existing_file(self) -> None:
        self.observables.to_disk()
        with self.assertRaises(FileExistsError):
            self.observables.to_disk(overwrite=False)

    def test_group_records_observable_class(self) -> None:
        self.observables.to_disk()
        with h5py.File(Path(self.folder) / "last.h5", "r") as file:
            self.assertEqual(
                file["RecordingObservablesHelper"].attrs[
                    "observable_class"
                ],
                "RecordingObservablesHelper",
            )

    def test_missing_dataset_raises_and_names_it(self) -> None:
        self.observables.to_disk()
        dataset_name = next(iter(self.observables.dataset_names()))
        with h5py.File(Path(self.folder) / "last.h5", "r+") as file:
            del file["RecordingObservablesHelper"][dataset_name]
        with self.assertRaises(ResultsFormatError) as context:
            self.observables.from_disk()
        self.assertIn(dataset_name, str(context.exception))

    def test_extra_dataset_warns_and_is_ignored(self) -> None:
        self.observables.to_disk()
        with h5py.File(Path(self.folder) / "last.h5", "r+") as file:
            group = file["RecordingObservablesHelper"]
            dataset = group.create_dataset("from_the_future", data=[1.0])
            dataset.attrs["write_idx"] = 1
        with self.assertWarns(UserWarning):
            self.observables.from_disk()

    def test_class_mismatch_raises(self) -> None:
        self.observables.to_disk()
        with h5py.File(Path(self.folder) / "last.h5", "r+") as file:
            file["RecordingObservablesHelper"].attrs["observable_class"] = "Other"
        with self.assertRaises(ResultsFormatError):
            self.observables.from_disk()

    def test_purge_from_disk_removes_file(self) -> None:
        self.observables.to_disk()
        self.observables.purge_from_disk(verbose=False)
        self.assertEqual(list(Path(self.folder).iterdir()), [])

    def test_rename_changes_group_name(self) -> None:
        self.observables.rename("beam1")
        self.observables.to_disk()
        with h5py.File(Path(self.folder) / "last.h5", "r") as file:
            self.assertEqual(list(file.keys()), ["beam1"])
```

Add the imports this needs at the top of the file: `tempfile`, `h5py`,
`from pathlib import Path`, and
`from blond.handle_results.hdf5_io import ResultsFormatError`.

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m pytest -v tests/unittests/handle_results/test_observables.py -k Hdf5`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument
'name'` and `AttributeError: ... has no attribute 'dataset_names'`.

- [ ] **Step 3: Port `ObservablesBaseClass`**

In `blond/handle_results/observables.py`:

```python
    def __init__(
        self,
        folder: str | None = None,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        folder = folder if folder is not None else ""
        if len(folder) > 0:
            assert folder.endswith("/") or folder.endswith("\\")
        self.common_filepath = folder + "last"
        self.name = name if name is not None else type(self).__name__
        logger.info(
            f"Will save {self} to {self.common_filepath}.h5"
            f" in group {self.name}."
        )
```

(The `folder if folder is not None else ""` line also fixes a live bug: the
declared default is `None`, and `len(None)` raises `TypeError`. Only
subclasses passing `folder=""` masked it.)

Replace `rename`:

```python
    def rename(self, new_name: str) -> None:
        """
        Change the HDF5 group name this observable is stored under.

        Parameters
        ----------
        new_name
            New group name.

        Notes
        -----
        This has no effect on files that are already saved to the disk.
        """
        self.name = new_name
        logger.info(f"Changed group name of {self} to {self.name}.")
```

Add the dataset-name mapping and the two I/O methods:

```python
    def dataset_names(self) -> dict[str, str]:
        """
        Map HDF5 dataset name to recorder attribute name.

        Returns
        -------
        dataset_names
            Mapping of dataset name to the attribute holding the recorder.
        """
        return {
            attribute.removeprefix("_"): attribute
            for attribute, _recorder in self.get_recorders()
        }

    def to_disk(
        self,
        group: h5py.Group | None = None,
        overwrite: bool = True,
    ) -> None:
        """
        Save data to disk.

        Parameters
        ----------
        group
            Open HDF5 group to write into. When omitted, a standalone
            results file is created at ``<common_filepath>.h5``.
        overwrite
            Whether an existing standalone file may be replaced. Ignored
            when `group` is given.

        Raises
        ------
        FileExistsError
            If the standalone file exists and `overwrite` is False.
        """
        if group is None:
            with create_results_file(
                self.common_filepath, overwrite=overwrite
            ) as file:
                self.to_disk(file.create_group(self.name))
            logger.info(f"Saved {self} to {self.common_filepath}.h5")
            return
        group.attrs[ATTR_OBSERVABLE_CLASS] = type(self).__name__
        for dataset_name, attribute in self.dataset_names().items():
            getattr(self, attribute).to_group(group, dataset_name)

    def from_disk(self, group: h5py.Group | None = None) -> None:
        """
        Load data from disk.

        Parameters
        ----------
        group
            Open HDF5 group to read from. When omitted, the standalone
            results file at ``<common_filepath>.h5`` is used.

        Raises
        ------
        ResultsFormatError
            If the group was written by a different observable class or
            does not contain every expected dataset.
        """
        if group is None:
            with open_results_file(self.common_filepath) as file:
                if self.name not in file:
                    raise ResultsFormatError(
                        f"{self.common_filepath}.h5 has no group"
                        f" '{self.name}'. It contains {list(file.keys())}."
                    )
                self.from_disk(file[self.name])
            return

        stored_class = group.attrs.get(ATTR_OBSERVABLE_CLASS)
        if stored_class != type(self).__name__:
            raise ResultsFormatError(
                f"Group '{group.name}' holds data of"
                f" '{stored_class}', but is being loaded into"
                f" {type(self).__name__}."
            )
        expected = self.dataset_names()
        payload = migrate_payload(
            read_group_payload(group),
            from_version=read_format_version(group.file),
        )
        missing = sorted(set(expected) - set(payload))
        if len(missing) > 0:
            raise ResultsFormatError(
                f"Group '{group.name}' is missing the datasets"
                f" {missing}."
            )
        extra = sorted(set(payload) - set(expected))
        if len(extra) > 0:
            warnings.warn(
                f"Ignoring unknown datasets {extra} in group"
                f" '{group.name}'.",
                UserWarning,
                stacklevel=2,
            )
        for dataset_name, attribute in expected.items():
            array, attrs = payload[dataset_name]
            expected_trailing = getattr(self, attribute)._memory.shape[1:]
            if array.shape[1:] != expected_trailing:
                warnings.warn(
                    f"Dataset '{dataset_name}' has shape {array.shape},"
                    f" but this observable expects trailing dimensions"
                    f" {expected_trailing}. Loading the stored data"
                    f" anyway; the simulation configuration may differ.",
                    UserWarning,
                    stacklevel=2,
                )
            setattr(
                self,
                attribute,
                DenseArrayRecorder.from_payload(array, attrs),
            )

    def purge_from_disk(self, verbose: bool = True) -> None:
        """
        Delete the standalone results file of this observable.

        Parameters
        ----------
        verbose
            Whether to print the removal message.
        """
        filepath = results_filepath(self.common_filepath)
        if filepath.is_file():
            filepath.unlink()
            if verbose:
                print(f"Removed {filepath}")
```

Imports to add at the top of `observables.py`. The file already has
`import warnings` (line 15), `import logging` and
`from __future__ import annotations`, so annotations are postponed and
`h5py` only needs to be imported for typing:

```python
if TYPE_CHECKING:  # pragma: no cover
    import h5py
```

plus the runtime import:

```python
from blond.handle_results.hdf5_io import (
    ATTR_OBSERVABLE_CLASS,
    ResultsFormatError,
    create_results_file,
    migrate_payload,
    open_results_file,
    read_format_version,
    read_group_payload,
    results_filepath,
)
```

- [ ] **Step 4: Drop the filepath argument from every recorder call site**

In `observables.py` and `observables_as_elements.py`, remove the first
argument from every `DenseArrayRecorder(...)` construction and keep the
shape as a keyword. Every one of these:

```python
        self._hist2d = DenseArrayRecorder(
            f"{self.common_filepath}_hist2d",
            shape,
        )
```

becomes:

```python
        self._hist2d = DenseArrayRecorder(shape=shape)
```

Do this for all call sites listed under **Files** above. The recorder
attribute name alone now determines the dataset name, so
`self._rms_emittance` is stored as `rms_emittance` — the old
`_emittance_stat` suffix disappears.

- [ ] **Step 5: Run the observable tests and confirm GREEN**

```bash
python -m pytest -v tests/unittests/handle_results/
```

Expected: all PASS. Fix the round-trip tests in
`test_observables_as_elements.py` and the remaining `test_from_disk` methods
in `test_observables.py` the same way as `TestObservablesHdf5` — they call
`to_disk()` / `from_disk()` with no arguments, which still works, but any
assertion touching `filepath_array` or `purge_from_disk` on a *recorder*
must move to the observable.

- [ ] **Step 6: Run pre-commit and commit**

```bash
pre-commit run --all-files
git add blond/handle_results/ tests/unittests/handle_results/
git commit -m "Saved observables as one HDF5 group each

Each observable now writes one group holding one dataset per recorder,
named after the recorder attribute. Group names default to the class
name and can be set with the new group_name= argument. Loading validates the
observable class and the dataset set, and reports missing datasets by
name instead of failing late.

Also fixed ObservablesBaseClass rejecting its own documented default of
folder=None, which raised TypeError in len(folder)."
```

---

### Task 4: One file per simulation

**Files:**
- Modify: `blond/core/simulation/simulation.py:1435-1502` (`save_results`)
  and `:1504-1610` (`load_results`)
- Test: `tests/unittests/core/simulation/test_simulation.py:362-388`
  (`test_load_results`)

**Interfaces:**
- Consumes: `ObservablesBaseClass.to_disk(group)` / `.from_disk(group)` and
  `.name` (Task 3); `create_results_file`, `open_results_file`,
  `results_filepath`, `ResultsFormatError` (Task 1).
- Produces:
  - `Simulation.save_results(observe=(), common_name=None,
    overwrite=True) -> Path` — returns the written file path.
  - `Simulation.load_results(beams, n_turns=None, observe=(),
    common_name=None) -> None`
  - Module-level helper `_resolve_group_names(observe) ->
    list[tuple[ObservablesOncePerTurnBase, str]]`

- [ ] **Step 1: Write the failing tests**

In `tests/unittests/core/simulation/test_simulation.py`, replace
`test_load_results` and add the multi-observable cases:

```python
    def test_save_results_writes_single_file(self):
        observation = BeamObservationOncePerTurn(each_turn_i=10)
        kwargs = dict(beams=(self.beam,), n_turns=10, observe=(observation,))
        self.simulation.run_simulation(**kwargs)
        with tempfile.TemporaryDirectory() as folder:
            stem = str(Path(folder) / "run")
            filepath = self.simulation.save_results(
                observe=(observation,), common_name=stem
            )
            self.assertEqual(filepath, Path(stem + ".h5"))
            self.assertEqual(
                [path.name for path in Path(folder).iterdir()], ["run.h5"]
            )

    def test_load_results(self):
        observation = BeamObservationOncePerTurn(each_turn_i=10)
        kwargs = dict(beams=(self.beam,), n_turns=10, observe=(observation,))
        self.simulation.run_simulation(**kwargs)
        de_before_save = observation.dEs.copy()
        with tempfile.TemporaryDirectory() as folder:
            stem = str(Path(folder) / "run")
            self.simulation.save_results(
                observe=(observation,), common_name=stem
            )
            self.simulation.load_results(**kwargs, common_name=stem)
        np.testing.assert_almost_equal(de_before_save, observation.dEs)

    def test_two_observables_share_one_file(self):
        first = BeamObservationOncePerTurn(each_turn_i=10, group_name="beam1")
        second = BeamObservationOncePerTurn(each_turn_i=10, group_name="beam2")
        kwargs = dict(
            beams=(self.beam,), n_turns=10, observe=(first, second)
        )
        self.simulation.run_simulation(**kwargs)
        with tempfile.TemporaryDirectory() as folder:
            stem = str(Path(folder) / "run")
            self.simulation.save_results(
                observe=(first, second), common_name=stem
            )
            with h5py.File(stem + ".h5", "r") as file:
                self.assertEqual(
                    sorted(file.keys()), ["beam1", "beam2"]
                )

    def test_colliding_group_names_warn_and_suffix(self):
        first = BeamObservationOncePerTurn(each_turn_i=10)
        second = BeamObservationOncePerTurn(each_turn_i=10)
        kwargs = dict(
            beams=(self.beam,), n_turns=10, observe=(first, second)
        )
        self.simulation.run_simulation(**kwargs)
        with tempfile.TemporaryDirectory() as folder:
            stem = str(Path(folder) / "run")
            with self.assertWarns(UserWarning):
                self.simulation.save_results(
                    observe=(first, second), common_name=stem
                )
            with h5py.File(stem + ".h5", "r") as file:
                self.assertEqual(
                    sorted(file.keys()),
                    [
                        "BeamObservationOncePerTurn",
                        "BeamObservationOncePerTurn_1",
                    ],
                )

    def test_save_results_without_observables_raises(self):
        with self.assertRaises(ValueError):
            self.simulation.save_results(observe=())
```

Add `tempfile`, `h5py` and `from pathlib import Path` to that file's imports.

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m pytest -v tests/unittests/core/simulation/test_simulation.py -k results`
Expected: FAIL — `save_results` returns `None`, so
`self.assertEqual(filepath, Path(...))` fails, and the collision test finds
one group instead of two.

- [ ] **Step 3: Implement the single-file save/load**

`simulation.py` already imports `warnings` (line 22) and defines
`logger` (line 83). Add the imports from `blond.handle_results.hdf5_io`
(`ResultsFormatError`, `create_results_file`, `open_results_file`,
`results_filepath`) and `from pathlib import Path`, then add near the
other module-level helpers:

```python
def _resolve_group_names(
    observe: tuple[ObservablesOncePerTurnBase, ...],
) -> list[tuple[ObservablesOncePerTurnBase, str]]:
    """
    Assign a unique HDF5 group name to each observable.

    Parameters
    ----------
    observe
        Observables that are saved or loaded together.

    Returns
    -------
    resolved
        List of (observable, group name) in the given order.
    """
    resolved: list[tuple[ObservablesOncePerTurnBase, str]] = []
    used: set[str] = set()
    for observable in observe:
        group_name = observable.group_name
        if group_name in used:
            suffix = 1
            while f"{group_name}_{suffix}" in used:
                suffix += 1
            warnings.warn(
                f"Two observables both want the group '{group_name}';"
                f" storing the second as '{group_name}_{suffix}'."
                f" Pass group_name= to give them stable, meaningful names.",
                UserWarning,
                stacklevel=3,
            )
            group_name = f"{group_name}_{suffix}"
        used.add(group_name)
        resolved.append((observable, group_name))
    return resolved
```

`save_results` becomes:

```python
        if len(observe) == 0:
            raise ValueError(
                "save_results needs at least one observable in observe=."
            )
        stem = (
            common_name
            if common_name is not None
            else observe[0].common_filepath
        )
        filepath = results_filepath(stem)
        with create_results_file(stem, overwrite=overwrite) as file:
            for observable, group_name in _resolve_group_names(observe):
                observable.to_disk(file.create_group(group_name))
        logger.info(f"Saved results of {len(observe)} observables to {filepath}")
        return filepath
```

`load_results` keeps its `self.finalize(...)` call — that allocates the
recorders and is what makes the expected dataset set known — then:

```python
        if len(observe) == 0:
            raise ValueError(
                "load_results needs at least one observable in observe=."
            )
        stem = (
            common_name
            if common_name is not None
            else observe[0].common_filepath
        )
        with open_results_file(stem) as file:
            for observable, group_name in _resolve_group_names(observe):
                if group_name not in file:
                    raise ResultsFormatError(
                        f"{results_filepath(stem)} has no group"
                        f" '{group_name}'. It contains"
                        f" {list(file.keys())}."
                    )
                observable.from_disk(file[group_name])
```

Update both docstrings: `save_results` now documents the `overwrite`
parameter, a `Returns` section for the path, and a `Raises` section for
`ValueError`; `load_results` documents `ResultsFormatError`. The docstring
examples that mention saved files must say `<common_name>.h5`, not the old
per-array names. `numpydoc-validation` and the `-W` doc build both check
these.

- [ ] **Step 4: Run the tests and confirm GREEN**

Run: `python -m pytest -v tests/unittests/core/simulation/test_simulation.py -k results`
Expected: all PASS.

- [ ] **Step 5: Run pre-commit and commit**

```bash
pre-commit run --all-files
git add blond/core/simulation/simulation.py \
        tests/unittests/core/simulation/test_simulation.py
git commit -m "Saved a whole simulation to one HDF5 file

save_results now opens a single file and writes one group per
observable, returning the path it wrote. Colliding group names are
suffixed with a warning pointing at the group_name= argument. common_name is
now the file stem rather than a per-array filename prefix."
```

---

### Task 5: Example, docs and full-suite verification

**Files:**
- Modify: `blond/examples/scripts/EX_19_Observable_as_element.py:104`
- Modify: `blond/handle_results/__init__.py` (module docstring mentions)
- Test: the whole suite

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: no new API.

- [ ] **Step 1: Check the example still describes reality**

`EX_19_Observable_as_element.py:104` calls `beam_logger_element.to_disk()`
with no arguments — the standalone path, which still works unchanged. Add a
comment naming the file it produces so the example stays self-explanatory:

```python
    # writes one HDF5 file holding one group per observable
    beam_logger_element.to_disk()
```

- [ ] **Step 2: Run the example**

Run: `python blond/examples/scripts/EX_19_Observable_as_element.py`
Expected: exits 0 and leaves a `last.h5` next to the script's output folder.
Delete the produced file afterwards so it is not committed.

- [ ] **Step 3: Grep for stragglers**

```bash
grep -rn "npy\|filepath_array\|filepath_attributes" \
    blond/handle_results/ blond/core/simulation/ \
    blond/examples/ tests/unittests/handle_results/
```

Expected: no hits. Any hit is a missed call site — fix it before continuing.

- [ ] **Step 4: Run the full unit-test suite**

```bash
python -m pytest -v tests/unittests/
```

Expected: all PASS. `pytest-randomly` randomizes the order; if a failure
appears only under some seeds, reproduce it with
`--randomly-seed=<N>` and suspect leaked global state rather than a flaky
test.

- [ ] **Step 5: Build the docs**

```bash
cd docs && bash create_docs.sh
```

Expected: exits 0. The build runs with `-W`, so a single bad docstring or
broken cross-reference fails it. If `create_doc_blond_main_objects.py`
crashes naming an unlinked class, add that class to its
`ASSIGNED_CATEGORIES` dict.

- [ ] **Step 6: Run pre-commit and commit**

```bash
pre-commit run --all-files
git add blond/examples/scripts/EX_19_Observable_as_element.py \
        blond/handle_results/__init__.py
git commit -m "Updated example and docs for the HDF5 results format"
```

---

## Deferred (explicitly out of scope)

- **gzip compression on datasets.** Decided to defer; enabling it later is a
  non-breaking change to the writer.
- **Physical units as dataset attributes.** Would make files genuinely
  self-describing for non-Python readers.
- The `.npz` input loaders in
  `blond/specifics/muon_collider/beam_preparation.py:49,81` and the
  `.npy`/`.npz` test fixtures. Those are inputs and pinned reference data,
  not result saving.
