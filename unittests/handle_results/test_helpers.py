import unittest

from blond.handle_results.helpers import (
    callers_relative_path,
    filesafe_datetime,
)


class TestFunctions(unittest.TestCase):
    def test_filesafe_datetime(self):
        datetime = filesafe_datetime()
        print(datetime)

    def test_callers_relative_path(self):
        def save(file):
            abspath = callers_relative_path(file, stacklevel=1)
            return abspath

        expected = "/unittests/handle_results/test"
        self.assertEqual(
            expected,
            (
                save("test")[-len(expected) :].replace("\\", "/")
            ),  # replacement for windows like paths
        )
