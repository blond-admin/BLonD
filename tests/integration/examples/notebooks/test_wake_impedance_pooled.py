import unittest

import matplotlib
import pytest

from blond.core.backends.backend import (
    Cupy64Bit,
    Numpy64Bit,
    backend,
)
from blond.testing.backend_testing import BLonDTestCase

matplotlib.use("agg")


class TestWakeImpedancePooled(BLonDTestCase):
    def run_notebook(self):
        from blond.examples.notebooks import wake_impedance_pooled

        wake_impedance_pooled.app.run()

    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        self.skipTest("Too slow.")  # TODO activate
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")

        self.run_notebook()

    @pytest.mark.backend_mutation
    def test_executable_cuda64(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")

        self.run_notebook()


if __name__ == "__main__":
    unittest.main()
