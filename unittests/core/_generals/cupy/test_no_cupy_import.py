import unittest

import numpy as np

from blond.generals.cupy.no_cupy_import import (
    _AsarrayOverrideManager,
    is_cupy_array,
)


class TestCallables(unittest.TestCase):
    def test_is_cupy_array_cpu(self):
        self.assertFalse(is_cupy_array(np.ones(10)))

    def test_is_cupy_array_gpu(self):
        try:
            import cupy as cp  # type: ignore
        except ModuleNotFoundError:
            self.skipTest("Cupy not available")
        self.assertTrue(is_cupy_array(cp.ones(10)))


class TestAsarrayOverrideManager(unittest.TestCase):
    def setUp(self):
        self.manger = _AsarrayOverrideManager()

    def test_asarray_override(self):
        try:
            import cupy as cp  # type: ignore
        except ModuleNotFoundError:
            self.skipTest("Cupy not available")

        arr_gpu = cp.ones(10)
        arr_cpu = self.manger.asarray_override(arr_gpu)
        arr_cpu2 = self.manger.asarray_override(
            arr_gpu
        )  # should trigger cache
        self.assertEqual(arr_cpu.ctypes.data, arr_cpu2.ctypes.data)


if __name__ == "__main__":
    unittest.main()
