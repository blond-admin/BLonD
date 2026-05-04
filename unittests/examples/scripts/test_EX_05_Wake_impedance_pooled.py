import unittest

import pytest

from blond.core.backends.backend import (
    Cupy32Bit,
    Cupy64Bit,
    Numpy32Bit,
    Numpy64Bit,
    backend,
)
from blond.examples.scripts.EX_01_Acceleration_no_beam import n_turns


class TestEX_05_Wake_impedance_pooled(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test_executable_numba32(self):
        backend.change_backend(Numpy32Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import EX_05_Wake_impedance_pooled

        # full script. just checking if it crashes
        EX_05_Wake_impedance_pooled.main(n_turns=2)

    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import EX_05_Wake_impedance_pooled

        # full script. just checking if it crashes
        EX_05_Wake_impedance_pooled.main(n_turns=2)

    @pytest.mark.backend_mutation
    def test_executable_cuda32(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy32Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import EX_05_Wake_impedance_pooled

        # full script. just checking if it crashes

        EX_05_Wake_impedance_pooled.main(n_turns=2)
        backend.zeros(100)

    @pytest.mark.backend_mutation
    def test_executable_cuda64(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import EX_05_Wake_impedance_pooled

        # full script. just checking if it crashes
        EX_05_Wake_impedance_pooled.main(n_turns=2)
        backend.zeros(100)
