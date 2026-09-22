import unittest

from blond.generals.iterables_ import _as_tuple, all_equal
from blond.testing.backend_testing import BLonDTestCase


class TestFunctions(BLonDTestCase):
    def test_all_equal_tuple(self):
        for iterable_type in (tuple, list, set):
            for comparison_type in (int, float, str):
                a = comparison_type(1)
                b = comparison_type(2)
                self.assertEqual(
                    False,
                    all_equal(iterable_type((a, b))),
                    msg=f"{iterable_type=} {comparison_type=}",
                )
                self.assertEqual(
                    True,
                    all_equal(iterable_type((a, a))),
                    msg=f"{iterable_type=} {comparison_type=}",
                )

    def test_all_equal_empty(self):
        self.assertTrue(all_equal([]))

    def test_as_tuple_propagates_type_error_while_iterating(self):
        """A `TypeError` raised *during* iteration is not swallowed.

        Only a non-iterable input may be wrapped into a 1-tuple.
        """

        class BrokenIterable:
            def __iter__(self):
                yield 1
                raise TypeError("raised while iterating")

        with self.assertRaises(TypeError):
            _as_tuple(BrokenIterable())


if __name__ == "__main__":
    unittest.main()
