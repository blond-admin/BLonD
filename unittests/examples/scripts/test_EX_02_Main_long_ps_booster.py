import unittest

import pytest

from blond.core.backends.backend import (
    Cupy32Bit,
    Cupy64Bit,
    Numpy32Bit,
    Numpy64Bit,
    backend,
)
from blond.testing.helpers import assert_runtime_below_threshold


class TestEX_02_Main_long_ps_booster(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test_executable_numba32(self):
        backend.change_backend(Numpy32Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import EX_02_Main_long_ps_booster

        # full script. just checking if it crashes
        assert_runtime_below_threshold(EX_02_Main_long_ps_booster.main, 30)

    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import EX_02_Main_long_ps_booster

        # full script. just checking if it crashes
        assert_runtime_below_threshold(EX_02_Main_long_ps_booster.main, 30)

    @pytest.mark.backend_mutation
    def test_executable_cuda32(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy32Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import EX_02_Main_long_ps_booster

        # full script. just checking if it crashes

        assert_runtime_below_threshold(EX_02_Main_long_ps_booster.main, 30)
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
        from blond.examples.scripts import EX_02_Main_long_ps_booster

        # full script. just checking if it crashes
        assert_runtime_below_threshold(EX_02_Main_long_ps_booster.main, 30)
        backend.zeros(100)
