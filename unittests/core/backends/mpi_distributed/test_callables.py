import unittest

import numpy as np

from blond.core.backends.mpi_distributed.callables import (
    mpi_is_active,
    rms_emittance,
)
from blond.generals.distribted.distributed_array import DistributedArray

mpi_active = mpi_is_active()
mpi_inactive = not mpi_active


class TestCallables(unittest.TestCase):
    @unittest.skipIf(mpi_active, "Runs with `mpirun`")
    def test_rms_wo_mpi(self):
        rng = np.random.default_rng(0)
        dt = DistributedArray(rng.normal(loc=0, scale=1, size=512))
        dE = DistributedArray(rng.normal(loc=0, scale=1, size=512))
        rms_expected = np.sqrt(
            np.average(dt.array_local**2) * np.average(dE.array_local**2)
            - (np.average(dt.array_local * dE.array_local)) ** 2
        )
        rms = rms_emittance(dt=dt, dE=dE)
        self.assertAlmostEqual(rms_expected, rms)

    @unittest.skipIf(mpi_inactive, "Runs without `mpirun`")
    def test_rms_mpi(self):
        rng = np.random.default_rng(0)
        dt = DistributedArray(rng.normal(loc=0, scale=1, size=512))
        dE = DistributedArray(rng.normal(loc=0, scale=1, size=512))
        rms_expected = np.sqrt(
            np.average(dt.array_local**2) * np.average(dE.array_local**2)
            - (np.average(dt.array_local * dE.array_local)) ** 2
        )
        dt.mpi_scatter()
        dE.mpi_scatter()
        self.assertLess(dt.local_size, 512)
        rms = rms_emittance(dt=dt, dE=dE)
        self.assertAlmostEqual(rms_expected, rms)

    @unittest.skipIf(mpi_inactive, "Runs without `mpirun`")
    def test_rms_mpi_cuda(self):
        try:
            import cupy as cp
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))

        cp.random.seed(0)
        dt = DistributedArray(cp.random.normal(loc=0, scale=1, size=512))
        dE = DistributedArray(cp.random.normal(loc=0, scale=1, size=512))
        rms_expected = float(
            cp.sqrt(
                cp.average(dt.array_local**2) * cp.average(dE.array_local**2)
                - (cp.average(dt.array_local * dE.array_local)) ** 2
            )
        )
        dt.mpi_scatter()
        dE.mpi_scatter()
        self.assertLess(dt.local_size, 512)
        rms = rms_emittance(dt=dt, dE=dE)
        self.assertAlmostEqual(rms_expected, rms)
