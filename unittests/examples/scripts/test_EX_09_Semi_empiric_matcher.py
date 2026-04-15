import unittest

import pytest

from blond.core.backends.backend import (
    Cupy32Bit,
    Cupy64Bit,
    Numpy32Bit,
    Numpy64Bit,
    backend,
)


class TestEX_09_Semi_empiric_matcher(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test_executable_numba32(self):
        backend.change_backend(Numpy32Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import EX_09_Semi_empiric_matcher

        # full script. just checking if it crashes
        (
            EX_09_Semi_empiric_matcher.increment_intensity_effects_until_iteration_i
        ) = 2
        EX_09_Semi_empiric_matcher.main()

    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import EX_09_Semi_empiric_matcher

        # full script. just checking if it crashes
        (
            EX_09_Semi_empiric_matcher.increment_intensity_effects_until_iteration_i
        ) = 2
        EX_09_Semi_empiric_matcher.main()

    @pytest.mark.backend_mutation
    def test_executable_cuda32(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy32Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import EX_09_Semi_empiric_matcher

        # full script. just checking if it crashes

        (
            EX_09_Semi_empiric_matcher.increment_intensity_effects_until_iteration_i
        ) = 2
        EX_09_Semi_empiric_matcher.main()
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
        from blond.examples.scripts import EX_09_Semi_empiric_matcher

        # full script. just checking if it crashes
        (
            EX_09_Semi_empiric_matcher.increment_intensity_effects_until_iteration_i
        ) = 2
        EX_09_Semi_empiric_matcher.main()
        backend.zeros(100)
