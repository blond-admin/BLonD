import unittest

import pytest


class TestEX_Xsuite_LHC_Xsuite_base(unittest.TestCase):
    @pytest.mark.backend_mutation
    def test_executable_numba32(self):
        from blond.core.backends.backend import (
            Numpy32Bit,
            backend,
        )

        self.skipTest("Too slow")
        backend.change_backend(Numpy32Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import EX_Xsuite_LHC_Xsuite_base

        # full script. just checking if it crashes
        EX_Xsuite_LHC_Xsuite_base.main()

    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        from blond.core.backends.backend import (
            Numpy64Bit,
            backend,
        )

        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import EX_Xsuite_LHC_Xsuite_base

        # full script. just checking if it crashes
        EX_Xsuite_LHC_Xsuite_base.main()

    @pytest.mark.backend_mutation
    def test_executable_cuda32(self):
        from blond.core.backends.backend import (
            Cupy32Bit,
            backend,
        )

        self.skipTest("Too slow")
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy32Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import EX_Xsuite_LHC_Xsuite_base

        # full script. just checking if it crashes

        EX_Xsuite_LHC_Xsuite_base.main()
        backend.zeros(100)

    @pytest.mark.backend_mutation
    def test_executable_cuda64(self):
        from blond.core.backends.backend import (
            Cupy64Bit,
            backend,
        )

        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import EX_Xsuite_LHC_Xsuite_base

        # full script. just checking if it crashes
        EX_Xsuite_LHC_Xsuite_base.main()
        backend.zeros(100)
