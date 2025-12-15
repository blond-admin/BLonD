import unittest

import numpy as np

from blond.generals.arrays_ import _read_only


class TestCallables(unittest.TestCase):
    def test_read_only(self):
        array = np.ones(10)
        array2 = _read_only(array)
        with self.assertRaises(ValueError):
            array2[1] = 2
        with self.assertRaises(ValueError):
            array2[1] += 1

    @unittest.skip("`_read_only` doesnt work on Cupy (2025)")  # TODO
    def test_read_only_gpu(self):
        try:
            import cupy
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
        array = cupy.ones(10)
        array2 = _read_only(array)
        with self.assertRaises(ValueError):
            array2[1] = 2
        with self.assertRaises(ValueError):
            array2[1] += 1
