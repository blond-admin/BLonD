import unittest


class TestProtocols(unittest.TestCase):
    def test_no_crash(self):
        from blond.generals import protocols  # NOQA
