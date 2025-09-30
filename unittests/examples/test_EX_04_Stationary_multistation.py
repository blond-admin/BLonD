import unittest

from blond._core.backends.backend import (
    Cupy32Bit,
    Cupy64Bit,
    Numpy32Bit,
    Numpy64Bit,
    backend,
)


class TestEX_04_Stationary_multistation(unittest.TestCase):
    def test_executable_numba32(self):
        backend.change_backend(Numpy32Bit)
        backend.set_specials("numba")
        from blond.examples import EX_04_Stationary_multistation  # NOQA will run the

        # full script. just checking if it crashes
        EX_04_Stationary_multistation.main()

    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples import EX_04_Stationary_multistation  # NOQA will run the

        # full script. just checking if it crashes
        EX_04_Stationary_multistation.main()

    def test_executable_cuda32(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy32Bit)
        backend.set_specials("cuda")
        from blond.examples import EX_04_Stationary_multistation  # NOQA will run the

        # full script. just checking if it crashes

        EX_04_Stationary_multistation.main()
        backend.zeros(100)

    def test_executable_cuda64(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")
        from blond.examples import EX_04_Stationary_multistation  # NOQA will run the

        # full script. just checking if it crashes
        EX_04_Stationary_multistation.main()
        backend.zeros(100)
