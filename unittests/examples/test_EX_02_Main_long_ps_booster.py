import unittest

from blond._core.backends.backend import Numpy32Bit, Numpy64Bit, backend


class TestEX02MainLongPsBooster(unittest.TestCase):
    def setUp(self):
        backend.change_backend(Numpy64Bit)

    def tearDown(self):
        backend.change_backend(Numpy32Bit)

    def test_executable(self):
        from blond.examples import EX_02_Main_long_ps_booster  # NOQA will run the

        # full script. just checking if it crashes
