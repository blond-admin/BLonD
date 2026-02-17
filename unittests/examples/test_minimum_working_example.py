import unittest

import pytest
from blond.core.backends.backend import (
    Cupy32Bit,
    Cupy64Bit,
    Numpy32Bit,
    Numpy64Bit,
    backend,
)


class Test_minimum_working_example(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test_executable_numba32(self):
        backend.change_backend(Numpy32Bit)
        backend.set_specials("numba")
        from blond.examples import minimum_working_example  # NOQA will run the

        minimum_working_example.n_turns = 100
        minimum_working_example.n_macroparticles = 100
        minimum_working_example.main()

        # full script. just checking if it crashes

    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples import minimum_working_example  # NOQA will run the

        minimum_working_example.n_turns = 100
        minimum_working_example.n_macroparticles = 100
        minimum_working_example.main()

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
        from blond.examples import minimum_working_example  # NOQA will run the

        minimum_working_example.n_turns = 100
        minimum_working_example.n_macroparticles = 100
        minimum_working_example.main()
        backend.zeros(
            100
        )  # TODO document everywhere reason: Force cupy to raise error on corrupt memory.

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
        from blond.examples import minimum_working_example  # NOQA will run the

        minimum_working_example.n_turns = 100
        minimum_working_example.n_macroparticles = 100
        minimum_working_example.main()
        backend.zeros(100)

        # full script. just checking if it crashes


if __name__ == "__main__":
    unittest.main()
