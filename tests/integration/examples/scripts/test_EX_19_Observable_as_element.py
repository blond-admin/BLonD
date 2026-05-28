import unittest

import pytest

from blond.core.backends.backend import (
    Cupy64Bit,
    Numpy64Bit,
    backend,
)


class TestEX_19_Observable_as_element(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import EX_19_Observable_as_element

        # full script. just checking if it crashes
        EX_19_Observable_as_element.main()

    @pytest.mark.backend_mutation
    def test_executable_cuda64(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import EX_19_Observable_as_element

        # full script. just checking if it crashes
        EX_19_Observable_as_element.main()
        backend.zeros(100)
