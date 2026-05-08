import unittest

from blond.generals.exceptions_ import BLonDException, UnevenArraySizes


class TestExceptions(unittest.TestCase):
    def test_BLonDException(self):
        BLonDException()

    def test_UnevenArraySizes(self):
        UnevenArraySizes()
