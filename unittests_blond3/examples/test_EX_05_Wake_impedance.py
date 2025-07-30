import unittest

from blond3._core.backends.backend import backend, Numpy64Bit, Numpy32Bit

class TestEX05StationaryMultistation(unittest.TestCase):
    def setUp(self):
        backend.change_backend(Numpy64Bit)
    def tearDown(self):
        backend.change_backend(Numpy32Bit)
    def test_executable(self):
        from blond3.examples import EX_05_Stationary_multistation  # NOQA
        # will run the
        # full script. just checking if it crashes
