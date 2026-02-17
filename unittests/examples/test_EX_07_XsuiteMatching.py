import unittest

import pytest
from blond.core.backends.backend import (
    Cupy32Bit,
    Cupy64Bit,
    Numpy32Bit,
    Numpy64Bit,
    backend,
)

try:
    import xpart

    HAS_XSUITE = True
except ImportError:
    HAS_XSUITE = False


@unittest.skipUnless(HAS_XSUITE, "XSUITE is not available")
class TestEX_07_Xsuite_Matching(unittest.TestCase):
    def setUp(self):
        try:
            import xpart
        except ModuleNotFoundError as exception:
            self.skipTest(str(exception))

    @pytest.mark.backend_mutation
    @unittest.skip("Too slow")
    def test_executable_numba32(self):
        backend.change_backend(Numpy32Bit)
        backend.set_specials("numba")
        from blond.examples import EX_07_Xsuite_Matching  # NOQA will run the

        # full script. just checking if it crashes
        EX_07_Xsuite_Matching.main()

    @pytest.mark.backend_mutation
    @unittest.skip("Too slow")
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples import EX_07_Xsuite_Matching  # NOQA will run the

        # full script. just checking if it crashes
        EX_07_Xsuite_Matching.main()

    @pytest.mark.backend_mutation
    @unittest.skip("Too slow")
    def test_executable_cuda32(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy32Bit)
        backend.set_specials("cuda")
        from blond.examples import EX_07_Xsuite_Matching  # NOQA will run the

        # full script. just checking if it crashes

        EX_07_Xsuite_Matching.main()
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
        from blond.examples import EX_07_Xsuite_Matching  # NOQA will run the

        # full script. just checking if it crashes
        EX_07_Xsuite_Matching.main()
        backend.zeros(100)
