import tempfile
import unittest
from pathlib import Path

import numpy
import numpy as np

from blond.handle_results.array_recorders import DenseArrayRecorder
from blond.handle_results.hdf5_io import (
    create_results_file,
    open_results_file,
    read_group_payload,
)


class TestDenseArrayRecorder(unittest.TestCase):
    def setUp(self) -> None:
        self.dense_array_recorder = DenseArrayRecorder(
            shape=(20, 10),
            dtype=np.float32,
            order="C",
        )

    def test_init(self):
        no_pre = DenseArrayRecorder(
            shape=(20, 10),
            dtype=np.float32,
            order="C",
            preallocate=False,
        )

        pre = DenseArrayRecorder(
            shape=(20, 10),
            dtype=np.float32,
            order="C",
            preallocate=True,
        )

        np.testing.assert_array_equal(pre._memory, no_pre._memory)

    def test_get_valid_entries(self):
        self.assertEqual(
            0, self.dense_array_recorder.get_valid_entries().shape[0]
        )
        self.dense_array_recorder.write(np.arange(10))
        self.assertEqual(
            1, self.dense_array_recorder.get_valid_entries().shape[0]
        )
        self.dense_array_recorder.write(np.arange(10))
        self.assertEqual(
            2, self.dense_array_recorder.get_valid_entries().shape[0]
        )

    def test_write(self):
        newdata = np.linspace(10, 20, 10, dtype=np.float32)
        self.dense_array_recorder.write(newdata)
        numpy.testing.assert_array_equal(
            self.dense_array_recorder.get_valid_entries()[0, :], newdata
        )

    def test_write_with_numpy_mask(self):
        mask = np.array(
            [True, False, True, False, True, False, True, False, True, False],
            dtype=bool,
        )
        newdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        self.dense_array_recorder.write(newdata, mask=mask)
        result = self.dense_array_recorder.get_valid_entries()[0]
        np.testing.assert_array_equal(result[mask], newdata)
        self.assertTrue(np.all(np.isnan(result[~mask])))

    def test_write_with_cupy_mask(self):
        numpy_mask = np.array(
            [True, False, True, False, True, False, True, False, True, False],
            dtype=bool,
        )

        class _MockCupyArray:
            device = "cuda:0"

            def __init__(self, arr):
                self._arr = arr

            def get(self):
                return self._arr

        cupy_mask = _MockCupyArray(numpy_mask)
        newdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        self.dense_array_recorder.write(newdata, mask=cupy_mask)

        result = self.dense_array_recorder.get_valid_entries()[0]

        np.testing.assert_array_equal(result[numpy_mask], newdata)

        self.assertTrue(np.all(np.isnan(result[~numpy_mask])))

    def test_constructor_takes_no_filepath(self) -> None:
        with self.assertRaises(TypeError):
            DenseArrayRecorder(filepath="somewhere", shape=(1,))


class TestDenseArrayRecorderHDF5(unittest.TestCase):
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
        loaded = DenseArrayRecorder.from_payload(*payload.datasets["values"])
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
        loaded = DenseArrayRecorder.from_payload(*payload.datasets["values"])
        self.assertEqual(loaded._memory.dtype, np.float32)
