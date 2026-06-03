import importlib
import os.path
import unittest

import pytest

from blond.core.backends.backend import (
    Cupy64Bit,
    Numpy64Bit,
    backend,
)
from blond.examples import notebooks
from blond.testing.notebooks import ipynb_to_py

NOTEBOOK_DIR = os.path.dirname(notebooks.__file__)
import matplotlib

matplotlib.use("agg")


class TestTutorialSemiEmpiricMatcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # the notebook is converted to a python file
        # so that debugging is easily possible
        ipynb_to_py(
            os.path.join(
                NOTEBOOK_DIR,
                "wake_impedance_pooled.ipynb",
            ),
            os.path.join(
                NOTEBOOK_DIR,
                "wake_impedance_pooled.py",
            ),
        )

    @classmethod
    def tearDownClass(cls):
        os.remove(
            os.path.join(
                NOTEBOOK_DIR,
                "wake_impedance_pooled.py",
            )
        )

    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        self.skipTest("Too slow.")  # TODO activate
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")

        # this file is only created by `setUpClass` and deleted by `tearDownClass`
        from blond.examples.notebooks import wake_impedance_pooled  # NOQA

        importlib.reload(
            wake_impedance_pooled
        )  # make sure the script is executed

    @pytest.mark.backend_mutation
    def test_executable_cuda64(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")

        # this file is only created by `setUpClass` and deleted by `tearDownClass`tearDownClass`
        from blond.examples.notebooks import wake_impedance_pooled  # NOQA

        importlib.reload(
            wake_impedance_pooled
        )  # make sure the script is executed
