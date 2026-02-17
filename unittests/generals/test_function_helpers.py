import unittest

import numpy as np

from blond.generals.function_helpers import (
    UnevenArraySizes,
    raise_on_uneven_array_sizes,
)


class Test_raise_on_uneven_array_sizes(unittest.TestCase):
    def test_raise_on_uneven_array_sizes(self):
        test_tuple = [
            np.array([20.0, 20.0, 1.0, 10.0, 1.0]),
            np.array([5.0, 2.0]),
            np.array([50e6, 40e6, 1e6, 10e6, 5e6]),
        ]
        with self.assertRaisesRegex(
            expected_exception=UnevenArraySizes,
            expected_regex="Input sequences of more than one element have different lengths.",
        ):
            raise_on_uneven_array_sizes(test_tuple)
