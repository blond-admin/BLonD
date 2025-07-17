import unittest

import numpy
import numpy as np

from blond3.handle_results.array_recorders import (
    DenseArrayRecorder,
    ChunkedArrayRecorder,
)


class TestChunkedArrayRecorder(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.chunked_array_recorder = ChunkedArrayRecorder()

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp


class TestDenseArrayRecorder(unittest.TestCase):
    def setUp(self):
        self.dense_array_recorder = DenseArrayRecorder(
            filepath="deleteme",
            shape=(20, 10),
            dtype=np.float32,
            order="C",
            overwrite=True,
        )

    def tearDown(self):
        self.dense_array_recorder.purge_from_disk()

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

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
        self.assertEqual(0, self.dense_array_recorder.get_valid_entries().shape[0])
        self.dense_array_recorder.write(np.arange(10))
        self.assertEqual(1, self.dense_array_recorder.get_valid_entries().shape[0])
        self.dense_array_recorder.write(np.arange(10))
        self.assertEqual(2, self.dense_array_recorder.get_valid_entries().shape[0])

    def test_write(self):
        newdata = np.linspace(10, 20, 10, dtype=np.float32)
        self.dense_array_recorder.write(newdata)
        numpy.testing.assert_array_equal(
            self.dense_array_recorder.get_valid_entries()[0, :], newdata
        )


if __name__ == "__main__":
    unittest.main()
