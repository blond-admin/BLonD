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


class TestEX_01_Acceleration(unittest.TestCase):
    @pytest.mark.backend_mutation
    @pytest.mark.mpi
    def test_executable_numba32(self):
        backend.change_backend(Numpy32Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import (
            EX_01_Acceleration_revolution_time,  # NOQA will run the
        )

        assert_runtime_below_threshold(
            EX_01_Acceleration_revolution_time.main, 30
        )

        # full script. just checking if it crashes

    @pytest.mark.backend_mutation
    @pytest.mark.mpi
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import (
            EX_01_Acceleration_revolution_time,  # NOQA will run the
        )

        assert_runtime_below_threshold(
            EX_01_Acceleration_revolution_time.main, 30
        )

        # full script. just checking if it crashes

    @pytest.mark.backend_mutation
    @pytest.mark.mpi
    def test_executable_cuda32(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy32Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import (
            EX_01_Acceleration_revolution_time,  # NOQA will run the
        )

        assert_runtime_below_threshold(
            EX_01_Acceleration_revolution_time.main, 30
        )
        backend.zeros(100)

        # full script. just checking if it crashes

    @pytest.mark.backend_mutation
    @pytest.mark.mpi
    def test_executable_cuda64(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import (
            EX_01_Acceleration_revolution_time,  # NOQA will run the
        )

        assert_runtime_below_threshold(
            EX_01_Acceleration_revolution_time.main, 30
        )
        backend.zeros(100)

        # full script. just checking if it crashes
