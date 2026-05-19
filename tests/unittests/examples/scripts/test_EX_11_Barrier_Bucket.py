import unittest

import pytest

from blond.core.backends.backend import (
    Cupy32Bit,
    Cupy64Bit,
    Numpy32Bit,
    Numpy64Bit,
    backend,
)
from blond.testing.backend_testing import skip_if_no_cupy


class TestEX_11_Barrier_Bucket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.init_backend = type(backend)
        cls.init_specials = backend.specials_mode

    @classmethod
    def tearDownClass(cls):
        backend.change_backend(cls.init_backend)
        backend.set_specials(cls.init_specials)

    @pytest.mark.backend_mutation
    def test_executable_numba32(self):
        backend.change_backend(Numpy32Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import (
            EX_11_Barrier_Bucket,
        )

        # full script. just checking if it crashes
        EX_11_Barrier_Bucket.main(
            run_n_turns=10,
            n_macroparticles=100,
        )

    @pytest.mark.backend_mutation
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        from blond.examples.scripts import (
            EX_11_Barrier_Bucket,
        )

        # full script. just checking if it crashes
        EX_11_Barrier_Bucket.main(
            run_n_turns=10,
            n_macroparticles=100,
        )

    @pytest.mark.backend_mutation
    @skip_if_no_cupy
    def test_executable_cuda32(self):
        backend.change_backend(Cupy32Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import (
            EX_11_Barrier_Bucket,
        )

        # full script. just checking if it crashes
        EX_11_Barrier_Bucket.main(
            run_n_turns=10,
            n_macroparticles=100,
        )

    @pytest.mark.backend_mutation
    @skip_if_no_cupy
    def test_executable_cuda64(self):
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")
        from blond.examples.scripts import (
            EX_11_Barrier_Bucket,
        )

        # full script. just checking if it crashes
        EX_11_Barrier_Bucket.main(
            run_n_turns=10,
            n_macroparticles=100,
        )
