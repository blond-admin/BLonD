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
