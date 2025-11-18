import unittest

import numpy as np

from blond._core.backends.numba.callables import enforce_precision


class TestCallables(unittest.TestCase):
    def test_enforce_precision(self):
        for floattype in (np.float32, np.float64):

            @enforce_precision(floattype)
            def foo(a, b):
                return a + b[0]

            res1 = foo(10.0, np.ones(10, dtype=floattype))
            self.assertEqual(type(res1), floattype)


if __name__ == "__main__":
    unittest.main()
