import unittest

from blond.core.helpers import (
    find_instances_with_method,
    int_from_float_with_warning,
)


class TestFunctions(unittest.TestCase):
    def test_int_from_float_with_warning(self):
        with self.assertWarns(Warning):
            int_from_float_with_warning(1.2, 2)

    def test_int_from_float_with_exception(self):
        with self.assertRaises(TypeError):
            int_from_float_with_warning(type(int_from_float_with_warning), 2)
    def test_find_instances_with_method(self):
        class Test:
            def __init__(self):
                self.a = 1

            def to_be_found(self):
                pass

        test = Test()
        found = find_instances_with_method(
            root=test, method_name="to_be_found"
        )
        self.assertEqual(found.pop(), test)

    def test_find_instances_with_recursion_attribute(self):
        class Foo:
            def __init__(self, subclass, subclass2):
                self.a = 1
                self.subclass = subclass
                self.subclass2 = subclass2

            def to_be_found(self):
                pass

        class Bar:
            def __init__(self):
                self.a = 1

            def to_be_found(self):
                pass

        class Car:
            def __init__(self):
                self.a = 1

            def not_found(self):
                pass

        bar = Bar()
        car = Car()
        foo = Foo(bar, car)
        found = find_instances_with_method(root=foo, method_name="to_be_found")
        self.assertTrue(foo in found)
        self.assertTrue(bar in found)
        self.assertTrue(len(found) == 2)

    def test_find_instances_with_recursion_tuple(self):
        class Foo:
            def __init__(self, subclasses):
                self.a = 1
                self.subclasses = subclasses

            def to_be_found(self):
                pass

        class Bar:
            def __init__(self):
                self.a = 1

            def to_be_found(self):
                pass

        class Car:
            def __init__(self):
                self.a = 1

            def not_found(self):
                pass

        bar = Bar()
        car = Car()
        foo = Foo((bar, car)) # a tuple
        found = find_instances_with_method(root=foo,
                                           method_name="to_be_found")
        self.assertTrue(foo in found)
        self.assertTrue(bar in found)
        self.assertTrue(len(found) == 2)

    def test_find_instances_with_method2(self):
        class Test1:
            def __init__(self):
                self.a = 1

            def to_be_found(self):
                pass

        class Problem:
            def __getattribute__(self, name):
                raise Exception()

        class Test2:
            skip_find_instances_attributes = ["problem"]

            def __init__(self, test1):
                self.a = test1
                self.problem = Problem()

            def not_found(self):
                pass

        test1 = Test1()
        test2 = Test2(test1=test1)
        found = find_instances_with_method(
            root=test2, method_name="to_be_found"
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found.pop(), test1)

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
