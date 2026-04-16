import unittest

import numpy as np
import pytest

from blond import Cupy64Bit, Numpy32Bit, Numpy64Bit, backend, copy_to_cpu
from blond.core.backends.mpi_distributed.callables import (
    rms_emittance,
)
from blond.generals.distributed.distributed_array import DistributedArray
from blond.generals.distributed.helpers import mpi_is_distributed

is_distributed = mpi_is_distributed()
not_distributed = not is_distributed


class TestCallables(unittest.TestCase):
    @unittest.skipIf(is_distributed, "Runs only without `mpirun`")
    def test_rms_wo_mpi(self):
        dt = DistributedArray(
            backend.random.normal(loc=0, scale=1, size=512).astype(
                backend.float
            )
        )
        dE = DistributedArray(
            backend.random.normal(loc=0, scale=1, size=512).astype(
                backend.float
            )
        )
        mean_dt = np.mean(copy_to_cpu(dt.array_local), dtype=float)
        mean_dE = np.mean(copy_to_cpu(dE.array_local), dtype=float)
        centered_dt = copy_to_cpu(dt.array_local) - mean_dt
        centered_dE = copy_to_cpu(dE.array_local) - mean_dE
        rms_expected = float(
            np.sqrt(
                np.average(centered_dt**2) * np.average(centered_dE**2)
                - (np.average(centered_dt * centered_dE)) ** 2
            )
        )
        rms = rms_emittance(dt=dt, dE=dE)
        self.assertAlmostEqual(rms_expected, rms, rtol=1e-7)

    @pytest.mark.mpi
    @unittest.skipIf(not_distributed, "Runs only with `mpirun`")
    def test_rms_mpi(self):
        rng = np.random.default_rng(0)
        dt = DistributedArray(rng.normal(loc=0, scale=1, size=512))
        dE = DistributedArray(rng.normal(loc=0, scale=1, size=512))
        mean_dt = np.mean(dt.array_local)
        mean_dE = np.mean(dE.array_local)
        centered_dt = dt.array_local - mean_dt
        centered_dE = dE.array_local - mean_dE
        rms_expected = np.sqrt(
            np.average(centered_dt**2) * np.average(centered_dE**2)
            - (np.average(centered_dt * centered_dE)) ** 2
        )
        dt.mpi_scatter()
        dE.mpi_scatter()
        self.assertLess(dt.local_size, 512)
        rms = rms_emittance(dt=dt, dE=dE)
        self.assertAlmostEqual(rms_expected, rms)

    @pytest.mark.mpi
    @unittest.skipIf(not_distributed, "Runs only with `mpirun`")
    def test_rms_mpi_cuda(self):
        try:
            import cupy as cp
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
        backend.change_backend(Cupy64Bit)

        cp.random.seed(0)
        dt = DistributedArray(cp.random.normal(loc=0, scale=1, size=512))
        dE = DistributedArray(cp.random.normal(loc=0, scale=1, size=512))
        mean_dt = cp.mean(dt.array_local)
        mean_dE = cp.mean(dE.array_local)
        centered_dt = dt.array_local - mean_dt
        centered_dE = dE.array_local - mean_dE
        rms_expected = cp.sqrt(
            cp.average(centered_dt**2) * cp.average(centered_dE**2)
            - (cp.average(centered_dt * centered_dE)) ** 2
        )
        dt.mpi_scatter()
        dE.mpi_scatter()
        self.assertLess(dt.local_size, 512)
        rms = rms_emittance(dt=dt, dE=dE)
        self.assertAlmostEqual(rms_expected, rms)
        backend.change_backend(Numpy64Bit)
