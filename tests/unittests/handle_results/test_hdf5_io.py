import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from blond.handle_results.hdf5_io import (
    ATTR_FORMAT_VERSION,
    FORMAT_VERSION,
    GroupPayload,
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
        array, attrs = payload.datasets["mean_dt"]
        np.testing.assert_array_equal(array, np.arange(4.0))
        self.assertEqual(int(attrs["write_idx"]), 3)

    def test_reads_group_attributes_and_name(self) -> None:
        with create_results_file(self.stem) as file:
            group = file.create_group("obs")
            group.attrs["observable_class"] = "BunchStatistics"
        with open_results_file(self.stem) as file:
            payload = read_group_payload(file["obs"])
        self.assertEqual(payload.group_name, "obs")
        self.assertEqual(payload.attrs["observable_class"], "BunchStatistics")


def _payload(**datasets) -> GroupPayload:
    """
    Build a `GroupPayload` from ``name=array`` keyword arguments.

    Parameters
    ----------
    **datasets
        Arrays keyed by dataset name.

    Returns
    -------
    GroupPayload
        Payload holding those datasets.
    """
    return GroupPayload(
        datasets={name: (array, {}) for name, array in datasets.items()},
        attrs={},
        group_name="obs",
    )


class TestMigratePayload(unittest.TestCase):
    def test_no_migration_when_versions_match(self) -> None:
        payload = GroupPayload(
            datasets={"a": (np.zeros(2), {"write_idx": 1})},
            attrs={},
            group_name="obs",
        )
        result = migrate_payload(
            payload, from_version=FORMAT_VERSION, migrations={}
        )
        self.assertEqual(list(result.datasets), ["a"])

    def test_applies_chain_in_order(self) -> None:
        calls: list[int] = []

        def step_1(payload: GroupPayload) -> GroupPayload:
            calls.append(1)
            payload.datasets = {"b": payload.datasets["a"]}
            return payload

        def step_2(payload: GroupPayload) -> GroupPayload:
            calls.append(2)
            payload.datasets = {"c": payload.datasets["b"]}
            return payload

        result = migrate_payload(
            _payload(a=np.zeros(2)),
            from_version=1,
            to_version=3,
            migrations={1: step_1, 2: step_2},
        )
        self.assertEqual(calls, [1, 2])
        self.assertEqual(list(result.datasets), ["c"])

    def test_migration_sees_group_name_and_attributes(self) -> None:
        seen: list[tuple[str, str]] = []

        def rename_in_one_group(payload: GroupPayload) -> GroupPayload:
            seen.append(
                (payload.group_name, payload.attrs["observable_class"])
            )
            if payload.attrs["observable_class"] == "BunchStatistics":
                payload.attrs["observable_class"] = "BeamStatistics"
                payload.datasets["mean_dt_new"] = payload.datasets.pop(
                    "mean_dt"
                )
            return payload

        payload = GroupPayload(
            datasets={"mean_dt": (np.zeros(2), {})},
            attrs={"observable_class": "BunchStatistics"},
            group_name="beam1",
        )
        result = migrate_payload(
            payload,
            from_version=0,
            to_version=1,
            migrations={0: rename_in_one_group},
        )
        self.assertEqual(seen, [("beam1", "BunchStatistics")])
        self.assertEqual(list(result.datasets), ["mean_dt_new"])
        self.assertEqual(result.attrs["observable_class"], "BeamStatistics")

    def test_missing_migration_step_raises(self) -> None:
        with self.assertRaises(ResultsFormatError):
            migrate_payload(
                _payload(a=np.zeros(2)),
                from_version=1,
                to_version=3,
                migrations={1: lambda payload: payload},
            )

    def test_newer_file_version_raises(self) -> None:
        with self.assertRaises(ResultsFormatError):
            migrate_payload(
                _payload(a=np.zeros(2)),
                from_version=FORMAT_VERSION + 1,
                to_version=FORMAT_VERSION,
                migrations={},
            )
