import unittest

import matplotlib.pyplot as plt
import numpy as np
import pytest

from blond.generals.array_helpers import is_linspace_like


class TestCallables(unittest.TestCase):
    def test_is_linspace_like(self):
        self.assertTrue(is_linspace_like(np.linspace(-1e-12, 1e-12, 10)))
        self.assertTrue(is_linspace_like(np.linspace(-1e12, 1e12, 10)))
        not_linspace = np.linspace(-1e12, 1e12, 10)
        not_linspace[3] = 1.1e12
        self.assertFalse(is_linspace_like(not_linspace))

    @pytest.mark.cupy
    def test_is_linspace_like_gpu(self) -> None:
        try:
            import cupy as cp  # type: ignore
        except ModuleNotFoundError:
            self.skipTest("Cupy not available")
        self.assertTrue(is_linspace_like(cp.linspace(-1e-12, 1e-12, 10)))
        self.assertTrue(is_linspace_like(cp.linspace(-1e12, 1e12, 10)))
        not_linspace = cp.linspace(-1e12, 1e12, 10)
        not_linspace[3] = 1.1e12
        self.assertFalse(is_linspace_like(not_linspace))

    def test_bug_fftfeq_f64(self):
        array = np.fft.rfftfreq(1182720, d=1.950553260576804e-11).astype(
            np.float64
        )
        self.assertTrue(is_linspace_like(array))

    def test_bug_fftfeq_f32(self):
        array = np.fft.rfftfreq(1_182_720, d=1.950553260576804e-11).astype(
            np.float32
        )
        DEV_DRAW = True  # TODO set false
        if DEV_DRAW:
            plt.plot(np.abs(np.diff(array)))
            plt.show()

        self.assertTrue(is_linspace_like(array))


if __name__ == "__main__":
    unittest.main()
