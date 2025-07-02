import unittest

from blond3._core.helpers import int_from_float_with_warning


class TestFunctions(unittest.TestCase):
    def test_int_from_float_with_warning(self):
        with self.assertWarns(Warning):
            int_from_float_with_warning(1.2, 2)

    @unittest.skip
    def test_find_instances_with_method(self):
        # TODO: implement test for `find_instances_with_method`
        find_instances_with_method(root=None, method_name=None)

    @unittest.skip
    def test_float_or_array_typesafe(self):
        # TODO: implement test for `float_or_array_typesafe`
        float_or_array_typesafe(something=None, dtype=None)

    @unittest.skip
    def test_safe_index(self):
        # TODO: implement test for `safe_index`
        safe_index(x=None, idx=None)

    @unittest.skip
    def test_walk(self):
        # TODO: implement test for `walk`
        walk(obj=None)


if __name__ == "__main__":
    unittest.main()
