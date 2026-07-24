import unittest

import matplotlib.pyplot as plt
import pytest

from blond.core.backends.backend import (
    Cupy64Bit,
    Numpy64Bit,
    backend,
)


class TestEX_32_Self_consistent_multibunch_matcher(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import (
            EX_32_Self_consistent_multibunch_matcher,  # NOQA will run the
        )

        EX_32_Self_consistent_multibunch_matcher.N_TURNS = 20
        EX_32_Self_consistent_multibunch_matcher.N_MACROPARTICLES_PER_BUNCH = (
            1_000
        )
        EX_32_Self_consistent_multibunch_matcher.N_POINTS_GRID = 300
        # full script (incl. intensity effects). just checking crashes
        EX_32_Self_consistent_multibunch_matcher.main()
        plt.close("all")
        # also cover the periodic-conditions branch
        EX_32_Self_consistent_multibunch_matcher.PERIODIC_CONDITIONS = True
        EX_32_Self_consistent_multibunch_matcher.main()
        EX_32_Self_consistent_multibunch_matcher.PERIODIC_CONDITIONS = False
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
            EX_32_Self_consistent_multibunch_matcher,  # NOQA will run the
        )

        EX_32_Self_consistent_multibunch_matcher.N_TURNS = 20
        EX_32_Self_consistent_multibunch_matcher.N_MACROPARTICLES_PER_BUNCH = (
            1_000
        )
        EX_32_Self_consistent_multibunch_matcher.N_POINTS_GRID = 300
        # full script (incl. intensity effects). just checking crashes
        EX_32_Self_consistent_multibunch_matcher.main()
        plt.close("all")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
