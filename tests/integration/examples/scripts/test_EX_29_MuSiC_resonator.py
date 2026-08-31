import unittest

import pytest

from blond.core.backends.backend import Numpy64Bit, backend


class TestEX_29_MuSiC_resonator(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("python")

    @pytest.mark.backend_mutation
    def test_executable_python(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("python")
        from blond.examples.scripts import EX_29_MuSiC_resonator

        # full script, just checking it does not crash
        EX_29_MuSiC_resonator.main()

    @pytest.mark.backend_mutation
    def test_executable_cpp(self):
        backend.change_backend(Numpy64Bit)
        try:
            backend.set_specials("cpp")
        except (FileNotFoundError, OSError):
            self.skipTest("cpp backend not available")
        from blond.examples.scripts import EX_29_MuSiC_resonator

        # full script, just checking it does not crash
        EX_29_MuSiC_resonator.main()
