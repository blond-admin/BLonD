import unittest

from blond._core.backends.backend import Numpy32Bit, Numpy64Bit, backend


class TestEX03RfNoise(unittest.TestCase):
    def setUp(self):
        backend.change_backend(Numpy64Bit)

    def tearDown(self):
        backend.change_backend(Numpy32Bit)

    def test_executable(self):
        from blond.examples import EX_03_RFnoise  # NOQA will run the

        # full script. just checking if it crashes
