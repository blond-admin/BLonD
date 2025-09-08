import unittest

from matplotlib import pyplot as plt


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
