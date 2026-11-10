import subprocess
import sys
import unittest

import pytest

from blond.testing.backend_testing import BLonDTestCase

# `time.time_ns` frozen: two processes starting in the same nanosecond, the
# limit of MPI ranks or array jobs that start within a few microseconds.
_FIRST_SEED_WITH_FROZEN_CLOCK = (
    "import time; time.time_ns = lambda: 0; "
    "from blond.core.backends.cuda import callables; "
    "print(next(callables._quantum_excitation_seed_counter))"
)


@pytest.mark.cupy
class TestQuantumExcitationSeeds(BLonDTestCase):
    def setUp(self) -> None:
        try:
            import cupy  # type: ignore # NOQA: F401
        except ModuleNotFoundError:
            self.skipTest("Cupy not available")

    def _first_seed_of_new_process(self) -> int:
        result = subprocess.run(
            [sys.executable, "-c", _FIRST_SEED_WITH_FROZEN_CLOCK],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        return int(result.stdout.split()[-1])

    def test_processes_started_together_get_different_seeds(self) -> None:
        """Seeds count up per launch, so equal starts mean shared noise."""
        seeds = {self._first_seed_of_new_process() for _ in range(2)}
        self.assertEqual(len(seeds), 2)


if __name__ == "__main__":
    unittest.main()
