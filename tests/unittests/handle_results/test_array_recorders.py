import unittest

import numpy
import numpy as np

from blond.handle_results.array_recorders import (
    DenseArrayRecorder,
)
from blond.handle_results.helpers import callers_relative_path


class TestDenseArrayRecorder(unittest.TestCase):
    def setUp(self):
        self.dense_array_recorder = DenseArrayRecorder(
            filepath=callers_relative_path("deleteme", 1),
            shape=(20, 10),
            dtype=np.float32,
            order="C",
            overwrite=True,
        )

    def tearDown(self):
        self.dense_array_recorder.purge_from_disk()

    def test_init(self):
        no_pre = DenseArrayRecorder(
            filepath=callers_relative_path("deleteme", 1),
            shape=(20, 10),
            dtype=np.float32,
            order="C",
            overwrite=True,
            preallocate=False,
        )

        pre = DenseArrayRecorder(
            filepath=callers_relative_path("deleteme", 1),
            shape=(20, 10),
            dtype=np.float32,
            order="C",
            overwrite=True,
            preallocate=True,
        )

        np.testing.assert_array_equal(pre._memory, no_pre._memory)

    def test_to_disk_from_disk(self):
        self.dense_array_recorder.to_disk()
        reloaded = DenseArrayRecorder.from_disk(
            filepath=self.dense_array_recorder.filepath
        )

        np.testing.assert_array_equal(
            self.dense_array_recorder._memory,
            reloaded._memory,
        )

        self.assertEqual(
            self.dense_array_recorder._write_idx,
            reloaded._write_idx,
        )
        self.assertEqual(
            self.dense_array_recorder.filepath_array,
            reloaded.filepath_array,
        )
        self.assertEqual(
            self.dense_array_recorder.filepath_attributes,
            reloaded.filepath_attributes,
        )
        self.assertEqual(
            self.dense_array_recorder.overwrite,
            reloaded.overwrite,
        )

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

    def test_overwrite(self):
        rec1 = DenseArrayRecorder(
            filepath=callers_relative_path("deleteme2", 1),
            shape=(20, 10),
            dtype=np.float32,
            order="C",
            overwrite=True,
        )
        rec1.to_disk()  # creates deleteme2
        rec2 = DenseArrayRecorder(
            filepath=callers_relative_path("deleteme2", 1),
            shape=(20, 10),
            dtype=np.float32,
            order="C",
            overwrite=False,
        )
        with self.assertRaises(AssertionError):
            rec2.to_disk()
        rec1.purge_from_disk(verbose=False)


if __name__ == "__main__":
    unittest.main()
