import unittest

import numpy as np

from blond.core.backends.numba.fastmath import fast_sin


class TestCallables(unittest.TestCase):
    def test_sin(self):
        xs = np.linspace(-10, 10, 100)
        ys1 = np.sin(xs)
        ys2 = np.array([fast_sin(x) for x in xs])
        np.testing.assert_allclose(ys1, ys2)
