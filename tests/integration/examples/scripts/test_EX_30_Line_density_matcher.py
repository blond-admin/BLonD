import unittest

import matplotlib.pyplot as plt
import pytest

from blond.core.backends.backend import (
    Cupy64Bit,
    Numpy64Bit,
    backend,
)


class TestEX_30_Line_density_matcher(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import (
            EX_30_Line_density_matcher,  # NOQA will run the
        )

        EX_30_Line_density_matcher.N_TURNS = 20
        EX_30_Line_density_matcher.N_MACROPARTICLES = 2_000
        EX_30_Line_density_matcher.N_POINTS_GRID = 300
        EX_30_Line_density_matcher.N_POINTS_ABEL = 2_000
        # full script (incl. intensity effects). just checking crashes
        EX_30_Line_density_matcher.main()
        plt.close("all")

    @pytest.mark.backend_mutation
    def test_executable_cuda64(self):
        try:
            import cupy  # type: ignore # NOQA
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import (
            EX_30_Line_density_matcher,  # NOQA will run the
        )

        EX_30_Line_density_matcher.N_TURNS = 20
        EX_30_Line_density_matcher.N_MACROPARTICLES = 2_000
        EX_30_Line_density_matcher.N_POINTS_GRID = 300
        EX_30_Line_density_matcher.N_POINTS_ABEL = 2_000
        # full script (incl. intensity effects). just checking crashes
        EX_30_Line_density_matcher.main()
        plt.close("all")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
