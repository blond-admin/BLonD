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
