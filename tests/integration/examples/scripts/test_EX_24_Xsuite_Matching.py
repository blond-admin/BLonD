import unittest

import pytest

from blond.core.backends.backend import (
    Cupy64Bit,
    Numpy64Bit,
    backend,
)

# NOTE: catch broad ``Exception`` rather than ``ImportError``. This import runs
# at pytest collection time, and on some Python versions the installed xsuite
# stack fails to import with a non-ImportError (e.g.
# ``TypeError: typing.LiteralString is not subscriptable``). A narrow guard lets
# that propagate and aborts collection of the whole suite instead of skipping.
try:
    import xpart

    HAS_XSUITE = True
except Exception:
    HAS_XSUITE = False


@unittest.skipUnless(HAS_XSUITE, "XSUITE is not available")
class TestEX_24_Xsuite_Matching(unittest.TestCase):
    def setUp(self):
        try:
            import xpart
        except Exception as exception:
            self.skipTest(str(exception))

    @pytest.mark.backend_mutation
    @unittest.skip("Too slow")
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import EX_24_Xsuite_Matching

        # full script. just checking if it crashes
        EX_24_Xsuite_Matching.main()

    @pytest.mark.backend_mutation
    def test_executable_cuda64(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import EX_24_Xsuite_Matching

        # full script. just checking if it crashes
        EX_24_Xsuite_Matching.main()
        backend.zeros(100)
