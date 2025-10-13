import unittest

from blond._generals._iterables import all_equal


class TestFunctions(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
