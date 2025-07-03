import unittest

from blond3.handle_results.helpers import filesafe_datetime, \
    callers_relative_path


class TestFunctions(unittest.TestCase):
    def test_filesafe_datetime(self):
        datetime = filesafe_datetime()
        print(datetime)

    def test_callers_relative_path(self):
        def save(file):
            abspath = callers_relative_path(file,stacklevel=1)
            return abspath

        expected = "/unittests_blond3/handle_results/test"
        self.assertEqual(expected, save("test")[-len(expected):])