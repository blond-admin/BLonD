import unittest

import pytest

from blond.core.backends.backend import (
    Cupy64Bit,
    Numpy64Bit,
    backend,
)
from blond.testing.backend_testing import BLonDTestCase

pytest.importorskip("pydantic")


class TestEX_32_Simulate_cycle_from_json(BLonDTestCase):
    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import EX_32_Simulate_cycle_from_json

        EX_32_Simulate_cycle_from_json.n_turns = 100
        EX_32_Simulate_cycle_from_json.n_macroparticles = 100
        EX_32_Simulate_cycle_from_json.main()

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
        from blond.examples.scripts import EX_32_Simulate_cycle_from_json

        EX_32_Simulate_cycle_from_json.n_turns = 100
        EX_32_Simulate_cycle_from_json.n_macroparticles = 100
        EX_32_Simulate_cycle_from_json.main()
        backend.zeros(100)

        # full script. just checking if it crashes


if __name__ == "__main__":
    unittest.main()
