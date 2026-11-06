import unittest

from blond.testing.backend_testing import BLonDTestCase


class TestProtocols(BLonDTestCase):
    def test_no_crash(self):
        from blond.generals import protocols  # NOQA
