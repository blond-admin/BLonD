import unittest

import pytest

pytest.importorskip("xtrack")

from blond.core.backends.backend import (  # noqa: E402
    Cupy64Bit,
    Numpy64Bit,
    backend,
)


class TestEX_29_BLonD_in_Xsuite(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import EX_29_BLonD_in_Xsuite

        # full script. just checking if it crashes
        EX_29_BLonD_in_Xsuite.main(
            run_n_turns=10,
            n_macroparticles=100,
        )

    @pytest.mark.backend_mutation
    def test_turns_recorded_schedule_numba64(self):
        # The existing executable test runs 10 turns while a profile is
        # recorded every 100 turns, so the chunked tracking loop body -- and
        # its ``turns_recorded.append(turn)`` line -- never executes. Drive a
        # profiling period shorter than the run so the loop actually runs, and
        # assert the recorded turns follow the schedule: turn 0, then every
        # ``each_turn_i_profile``-th turn, then the trailing remainder up to
        # ``run_n_turns``.
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import EX_29_BLonD_in_Xsuite

        turns_recorded = EX_29_BLonD_in_Xsuite.main(
            run_n_turns=10,
            n_macroparticles=100,
            each_turn_i_profile=3,
        )
        self.assertEqual(turns_recorded, [0, 3, 6, 9, 10])

    @pytest.mark.backend_mutation
    def test_executable_cuda64(self):
        try:
            import cupy  # type: ignore # noqa: F401
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import EX_29_BLonD_in_Xsuite

        # full script. just checking if it crashes
        EX_29_BLonD_in_Xsuite.main(
            run_n_turns=10,
            n_macroparticles=100,
        )
        backend.zeros(100)
