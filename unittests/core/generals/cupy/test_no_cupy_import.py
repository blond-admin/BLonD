import unittest

import numpy as np
import pytest

from blond.generals.cupy.no_cupy_import import (
    _AsarrayOverrideManager,
    copy_to_cpu,
    is_cupy_array,
)


class TestCallables(unittest.TestCase):
    def test_is_cupy_array_cpu(self):
        self.assertFalse(is_cupy_array(np.ones(10)))

    @pytest.mark.cupy
    def test_is_cupy_array_gpu(self):
        try:
            import cupy as cp  # type: ignore
        except ModuleNotFoundError:
            self.skipTest("Cupy not available")
        self.assertTrue(is_cupy_array(cp.ones(10)))

    def test_copy_to_cpu1(self):
        array1 = np.ones(10)
        array2 = copy_to_cpu(array1)
        self.assertTrue(array1 is not array2)

    @pytest.mark.cupy
    def test_copy_to_cpu2(self):
        try:
            import cupy as cp
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(str(exc))
        array1 = cp.ones(10)
        array2 = copy_to_cpu(array1)
        self.assertTrue(array1 is not array2)


class TestAsarrayOverrideManager(unittest.TestCase):
    def setUp(self):
        self.manger = _AsarrayOverrideManager()

    @pytest.mark.cupy
    def test_asarray_override(self):
        try:
            import cupy as cp  # type: ignore
        except ModuleNotFoundError:
            self.skipTest("Cupy not available")

        arr_gpu = cp.ones(10)
        arr_cpu = self.manger.asarray_override(arr_gpu)
        np.testing.assert_array_equal(arr_cpu, np.ones(10))

        arr_cpu = self.manger.array_override(arr_gpu)
        np.testing.assert_array_equal(arr_cpu, np.ones(10))


if __name__ == "__main__":
    unittest.main()
