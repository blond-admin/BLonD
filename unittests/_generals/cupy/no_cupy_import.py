import unittest

import numpy as np
from matplotlib import pyplot as plt

from blond._generals.cupy.no_cupy_import import is_cupy_array


class TestFunctions(unittest.TestCase):
    def test_allow_plotting(self) -> None:
        try:
            import cupy as cp  # type: ignore
        except ImportError as exc:
            unittest.skip(str(exc))
        from blond import AllowPlotting

        # demo of AllowPlotting
        array = cp.array([1, 2, 23])
        array2 = cp.array([1, 2, 25])
        plt.figure()
        from blond._core.backends.backend import Cupy32Bit, backend

        backend.change_backend(Cupy32Bit)
        with AllowPlotting():
            # would crash without AllowPlotting
            # TypeError: Implicit conversion to a NumPy array is not allowed. Please use `.get()` to construct a NumPy array explicitly.
            plt.plot(array)
            plt.plot(array2)

        plt.close()

    def test_is_cupy_array(self):
        try:
            import cupy as cp  # type: ignore
            from numba import cuda
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        self.assertRaises(
            TypeError, lambda: is_cupy_array(cuda.to_device(np.ones(10)))
        )
        self.assertEqual(is_cupy_array(cp.ones(10)), True)

        self.assertEqual(is_cupy_array(np.ones(10)), False)
        self.assertEqual(is_cupy_array([1, 2, 3]), False)
        self.assertEqual(is_cupy_array("Not an array"), False)
