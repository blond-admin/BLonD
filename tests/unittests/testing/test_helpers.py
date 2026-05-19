import unittest

from blond.testing import pytest_active


class TestCallables(unittest.TestCase):
    def test_pytest_active(self):
        self.assertTrue(pytest_active())


if __name__ == "__main__":
    unittest.main()
