from unittest import TestCase

import numpy as np

from blond._core.backends.numba.callables import enforce_precision


class TestCallables(TestCase):
    def test_enforce_precision(self):
        for floattype in (np.float32, np.float64):

            @enforce_precision(floattype)
            def foo(a, b):
                return a + b

            res1 = foo(10.0, 20.0)
            self.assertEqual(type(res1), floattype)
