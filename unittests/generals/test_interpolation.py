import unittest

import numpy as np
from matplotlib import pyplot as plt

from blond.generals.interpolation import (
    interp_linear,
    interp_makima,
    interp_pchip,
)

test_x = np.linspace(1, 7, 256, endpoint=True)
test_xp = np.linspace(1, 7, 7, endpoint=True)
test_fp = np.array([-1, -1, -1, 0, 1, 1, 1], dtype=float)


class TestInterpLinear(unittest.TestCase):
    def test_executes(self):
        x = np.linspace(0, 1, 10)
        xp = np.linspace(-1, 2, 512)
        fp = np.random.default_rng(0).random(512)
        interp_linear(x, xp, fp)

    def test_monotonic(self):
        f = interp_linear(test_x, test_xp, test_fp)
        g = np.diff(f)
        np.testing.assert_array_equal(g >= 0, True)


class TestInterpMakima(unittest.TestCase):
    def test_executes(self):
        x = np.linspace(0, 1, 10)
        xp = np.linspace(-1, 2, 512)
        fp = np.random.default_rng(0).random(512)
        interp_makima(x, xp, fp)

    def test_monotonic(self):
        f = interp_makima(test_x, test_xp, test_fp)
        g = np.diff(f)
        np.testing.assert_array_equal(g >= 0, True)


class TestInterpPchip(unittest.TestCase):
    def test_executes(self):
        x = np.linspace(0, 1, 10)
        xp = np.linspace(-1, 2, 512)
        fp = np.random.default_rng(0).random(512)
        interp_pchip(x, xp, fp)

    def test_monotonic(self):
        f = interp_pchip(test_x, test_xp, test_fp)
        g = np.diff(f)
        np.testing.assert_array_equal(g >= 0, True)


if __name__ == "__main__":
    unittest.main()
