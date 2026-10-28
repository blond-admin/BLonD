import unittest

import pytest

pytest.importorskip("xtrack")

from blond.core.backends.backend import (  # noqa: E402
    Cupy64Bit,
    Numpy64Bit,
    backend,
)


class TestEX_30_Xsuite_in_BLonD(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import EX_30_Xsuite_in_BLonD

        # full script. just checking if it crashes
        EX_30_Xsuite_in_BLonD.main(
            run_n_turns=10,
            n_macroparticles=100,
        )

    @pytest.mark.backend_mutation
    def test_executable_cuda64(self):
        try:
            import cupy  # type: ignore # noqa: F401
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import EX_30_Xsuite_in_BLonD

        # full script. just checking if it crashes
        EX_30_Xsuite_in_BLonD.main(
            run_n_turns=10,
            n_macroparticles=100,
        )
        backend.zeros(100)
