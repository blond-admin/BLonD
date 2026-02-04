import unittest

import numpy as np
import pytest

from blond import Beam
from blond.core.backends.backend import (
    Cupy32Bit,
    Cupy64Bit,
    Numpy32Bit,
    Numpy64Bit,
    backend,
)


class TestEX_01_Acceleration_match_density(unittest.TestCase):
    def _execute(self):
        from blond.examples import EX_01_Acceleration_match_density  # NOQA

        EX_01_Acceleration_match_density.N_TURNS = 10
        EX_01_Acceleration_match_density.animate_fitting = False
        EX_01_Acceleration_match_density.plot_result = False
        bunch: Beam = EX_01_Acceleration_match_density.main()
        fetch_new_pinned_values = False
        if fetch_new_pinned_values:
            print("dt_mean =", bunch._dt.mean())
            print("dt_std =", bunch._dt.std())
            print("dE_mean =", bunch._dE.mean())
            print("dE_std =", bunch._dE.std())
        # pinned values
        dt_mean = 8.340270167873549e-10
        dt_std = 1.1025231519466652e-10
        dE_mean = -215614.70425667908
        dE_std = 34575081.67125967
        rtol = 1e-2 if backend.float == np.float32 else 1e-12

        np.testing.assert_allclose(bunch._dt.mean(), dt_mean, rtol=rtol)
        np.testing.assert_allclose(bunch._dt.std(), dt_std, rtol=rtol)
        np.testing.assert_allclose(bunch._dE.mean(), dE_mean, rtol=rtol)
        np.testing.assert_allclose(bunch._dE.std(), dE_std, rtol=rtol)

    @pytest.mark.backend_mutation
    @pytest.mark.mpi
    def test_executable_numba32(self):
        backend.change_backend(Numpy32Bit)
        backend.set_specials("numba")

        self._execute()

    @pytest.mark.backend_mutation
    @pytest.mark.mpi
    def test_executable_numba64(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")
        self._execute()

    @pytest.mark.backend_mutation
    @pytest.mark.mpi
    def test_executable_cuda32(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy32Bit)
        backend.set_specials("cuda")

        self._execute()

        backend.zeros(100)  # make sure that cupy is still working,
        # previous memory
        # violations would crash this command

    @pytest.mark.backend_mutation
    @pytest.mark.mpi
    def test_executable_cuda64(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        backend.change_backend(Cupy64Bit)
        backend.set_specials("cuda")

        self._execute()

        backend.zeros(100)  # make sure that cupy is still working,
        # previous memory
        # violations would crash this command
