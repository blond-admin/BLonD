import unittest

import numpy as np
import pytest

from blond.generals.arrays_ import is_linspace_like


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


if __name__ == "__main__":
    unittest.main()
