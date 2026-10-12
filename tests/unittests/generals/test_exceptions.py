import unittest

from blond.generals.exceptions_ import BLonDException, UnevenArraySizes
from blond.testing.backend_testing import BLonDTestCase


class TestExceptions(BLonDTestCase):
    def test_BLonDException(self):
        BLonDException()

    def test_UnevenArraySizes(self):
        UnevenArraySizes()
