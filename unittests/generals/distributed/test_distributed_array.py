import unittest

import numpy as np

from blond import backend
from blond.core.backends.mpi_distributed.callables import (
    mpi_is_active,
)
from blond.generals.distributed.distributed_array import DistributedArray

mpi_active = mpi_is_active()
mpi_inactive = not mpi_active


class TestDistributedArray(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.array = rng.normal(loc=0, scale=1.0, size=128)
        self.distributed_array = DistributedArray(self.array.copy())

    def test_local_size(self):
        self.assertEqual(self.distributed_array.local_size, 128)
        if mpi_active:
            self.distributed_array.mpi_scatter()

        if mpi_active:
            self.assertEqual(
                self.distributed_array.local_size, 64
            )  # assumes `mpirun -n 2`
            self.assertTrue(self.distributed_array.is_distributed)
            self.assertEqual(self.distributed_array.global_size, 128)
        else:
            self.assertEqual(self.distributed_array.local_size, 128)
            self.assertFalse(self.distributed_array.is_distributed)
            self.assertEqual(self.distributed_array.global_size, 128)

    def _call_test(self, func, func_name):
        expected = func(self.array)
        if mpi_active:
            self.distributed_array.mpi_scatter()
        actual = getattr(self.distributed_array, func_name)()
        np.testing.assert_almost_equal(expected, actual)

    def test_min(self):
        self._call_test(np.min, "min")

    def test_max(self):
        self._call_test(np.max, "max")

    def test_mean(self):
        self._call_test(np.mean, "mean")

    def test_std(self):
        self._call_test(np.std, "std")

    def test_sum(self):
        self._call_test(np.sum, "sum")

    def test_histogram(self):
        expected, _ = np.histogram(self.array, bins=8)
        if mpi_active:
            self.distributed_array.mpi_scatter()
        actual = self.distributed_array.histogram(bins=8)
        np.testing.assert_allclose(expected, actual)

    def test_histogram_with_out(self):
        expected, _ = np.histogram(self.array, bins=8)
        if mpi_active:
            self.distributed_array.mpi_scatter()
        actual = np.zeros_like(expected, dtype=backend.float)
        self.distributed_array.histogram(bins=8, out=actual)
        np.testing.assert_allclose(expected, actual)

    def test_barrier(self):
        if mpi_active:
            self.distributed_array.array_local = (
                self.distributed_array.array_local[:64]
            )
            self.distributed_array.barrier()
            self.assertEqual(
                self.distributed_array.global_size, 2 * 64
            )  # assumes `mpirun -n 2`
