import unittest

from blond.utils.legacy_support import _handle_legacy_kwargs


class TestHandleLegacyKwargs(unittest.TestCase):
    def test_handle_legacy_kwargs_function(self):
        @_handle_legacy_kwargs({"old": "new"})
        def execute(new, untouched):
            return new + untouched

        execute(1, 1)  # not crashing
        execute(1, untouched=1)  # not crashing
        execute(new=1, untouched=1)  # not crashing
        self.assertWarns(
            DeprecationWarning, lambda: execute(old=1, untouched=1)
        )

    def test_handle_legacy_kwargs_class(self):
        class Foo(object):
            @_handle_legacy_kwargs({"old": "new"})
            def __init__(self, new, untouched):
                self.new = new
                self.untouched = untouched

        Foo(1, 1)  # not crashing
        Foo(1, untouched=1)  # not crashing
        Foo(new=1, untouched=1)  # not crashing
        self.assertWarns(DeprecationWarning, lambda: Foo(old=1, untouched=1))

    def test_handle_legacy_kwargs_function_overloaded(self):
        @_handle_legacy_kwargs({"old": "new", "Unused": "unused"})
        def execute(new):
            return new

        execute(
            2,
        )  # not crashing
        execute(
            new=1,
        )  # not crashing
        self.assertWarns(DeprecationWarning, lambda: execute(old=1))


if __name__ == "__main__":
    unittest.main()
