import unittest

import matplotlib.pyplot as plt
import pytest

from blond.core.backends.backend import (
    Cupy64Bit,
    Numpy64Bit,
    backend,
)


class TestEX_18_Filamentation_matcher(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import (
            EX_18_Filamentation_matcher,  # NOQA will run the
        )

        EX_18_Filamentation_matcher.animate = False
        EX_18_Filamentation_matcher.n_iter = 2
        # full script. just checking if it crashes
        EX_18_Filamentation_matcher.main()
        plt.close()

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
            EX_18_Filamentation_matcher,  # NOQA will run the
        )

        EX_18_Filamentation_matcher.animate = False
        EX_18_Filamentation_matcher.n_iter = 2
        # full script. just checking if it crashes
        EX_18_Filamentation_matcher.main()
        plt.close()
        backend.zeros(100)
