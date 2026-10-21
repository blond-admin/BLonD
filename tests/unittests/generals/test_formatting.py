import unittest

import numpy as np

from blond.generals.formatting_ import pretty_string, si_format
from blond.testing.backend_testing import BLonDTestCase


class TestPrettyString(BLonDTestCase):
    def test_array_is_summarized(self):
        formatted = pretty_string(np.array([1.0, 3.0, 2.0]))
        self.assertIn("min=1.0", formatted)
        self.assertIn("max=3.0", formatted)
        self.assertIn("shape=(3,)", formatted)

    def test_zero_dimensional_array(self):
        self.assertIn("shape=()", pretty_string(np.array(10)))

    def test_non_array_is_returned_unchanged(self):
        value = {"a": 1}
        self.assertIs(pretty_string(value), value)

    def test_reexported_from_ring_elements(self):
        from blond.core.ring import beam_physics_relevant_elements

        self.assertIs(
            beam_physics_relevant_elements.pretty_string, pretty_string
        )


class TestSiFormat(BLonDTestCase):
    def test_kilo(self):
        self.assertEqual(si_format(1e3), "1.00k")

    def test_zero(self):
        self.assertEqual(si_format(0), "0")


if __name__ == "__main__":
    unittest.main()
