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


class TestEX_01_Acceleration_no_beam(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test_executable_numba32(self):
        backend.change_backend(Numpy32Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import (
            EX_01_Acceleration_no_beam,  # NOQA will run
        )

        # the
        EX_01_Acceleration_no_beam.n_turns = 10  # for testing
        EX_01_Acceleration_no_beam.main()
        assert_runtime_below_threshold(EX_01_Acceleration_no_beam.main, 30)

        # full script. just checking if it crashes

    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import (
            EX_01_Acceleration_no_beam,  # NOQA will run
        )

        # the
        EX_01_Acceleration_no_beam.n_turns = 10  # for testing
        EX_01_Acceleration_no_beam.main()
        assert_runtime_below_threshold(EX_01_Acceleration_no_beam.main, 30)

        # full script. just checking if it crashes

    @pytest.mark.backend_mutation
    def test_executable_cuda32(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy32Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import (
            EX_01_Acceleration_no_beam,  # NOQA will run
        )

        # the
        EX_01_Acceleration_no_beam.n_turns = 10  # for testing
        assert_runtime_below_threshold(
            EX_01_Acceleration_no_beam.main,
            threshold=30,
            repeat=10,
            matrix_size=(2**13, 2**13),
        )
        backend.zeros(100)

        # full script. just checking if it crashes

    @pytest.mark.backend_mutation
    def test_executable_cuda64(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import (
            EX_01_Acceleration_no_beam,  # NOQA will run
        )

        # the
        EX_01_Acceleration_no_beam.n_turns = 10  # for testing
        assert_runtime_below_threshold(
            EX_01_Acceleration_no_beam.main,
            threshold=30,
            repeat=10,
            matrix_size=(2**13, 2**13),
        )
        backend.zeros(100)

        # full script. just checking if it crashes
