import importlib
import os.path
import unittest

import pytest

from blond.core.backends.backend import (
    Cupy32Bit,
    Cupy64Bit,
    Numpy32Bit,
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
                "tutorial_SemiEmpiricMatcher.ipynb",
            ),
            os.path.join(
                NOTEBOOK_DIR,
                "tutorial_SemiEmpiricMatcher.py",
            ),
        )

    @classmethod
    def tearDownClass(cls):
        os.remove(
            os.path.join(
                NOTEBOOK_DIR,
                "tutorial_SemiEmpiricMatcher.py",
            )
        )

    @pytest.mark.backend_mutation
    def test_executable_numba32(self):
        self.skipTest("Too slow.")
        backend.change_backend(Numpy32Bit)
        backend.set_specials("numba")

        # this file is only created by `setUpClass` and deleted by `tearDownClass`
        from blond.examples.notebooks import (
            tutorial_SemiEmpiricMatcher,  # NOQA
        )

        importlib.reload(
            tutorial_SemiEmpiricMatcher
        )  # make sure the script is executed

    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")

        # this file is only created by `setUpClass` and deleted by `tearDownClass`
        from blond.examples.notebooks import (
            tutorial_SemiEmpiricMatcher,  # NOQA
        )

        importlib.reload(
            tutorial_SemiEmpiricMatcher
        )  # make sure the script is executed

    @pytest.mark.backend_mutation
    def test_executable_cuda32(self):
        self.skipTest("Too slow.")
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy32Bit)
        backend.set_specials("cuda")

        # this file is only created by `setUpClass` and deleted by `tearDownClass`
        from blond.examples.notebooks import (
            tutorial_SemiEmpiricMatcher,  # NOQA
        )

        importlib.reload(
            tutorial_SemiEmpiricMatcher
        )  # make sure the script is executed

    @pytest.mark.backend_mutation
    def test_executable_cuda64(self):
        self.skipTest("Too slow.")
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")

        # this file is only created by `setUpClass` and deleted by `tearDownClass`tearDownClass`
        from blond.examples.notebooks import (
            tutorial_SemiEmpiricMatcher,  # NOQA
        )

        importlib.reload(
            tutorial_SemiEmpiricMatcher
        )  # make sure the script is executed
