import unittest

import pytest

from blond.core.backends.backend import (
    Cupy64Bit,
    Numpy64Bit,
    backend,
)


class TestEX_12_Wake_impedance_pooled(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import EX_12_Wake_impedance_pooled

        # full script. just checking if it crashes
        EX_12_Wake_impedance_pooled.main(n_turns=2)

    @pytest.mark.backend_mutation
    def test_executable_cuda64(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import EX_12_Wake_impedance_pooled

        # full script. just checking if it crashes
        EX_12_Wake_impedance_pooled.main(n_turns=2)
        backend.zeros(100)
