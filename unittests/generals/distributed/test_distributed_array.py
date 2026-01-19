import sys
import unittest
from unittest.mock import patch

import numpy as np


class TestDistributedArray(unittest.TestCase):
    def setUp(self):
        from blond.generals.distributed.distributed_array import (
            DistributedArray,
        )

        rng = np.random.default_rng(0)
        self.array = rng.normal(loc=0, scale=1.0, size=128)
        self.distributed_array = DistributedArray(self.array.copy())

    def test_local_size(self):
        from blond.core.backends.mpi_distributed.callables import (
            mpi_is_active,
        )

        mpi_active = mpi_is_active()

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
        from blond.core.backends.mpi_distributed.callables import (
            mpi_is_active,
        )

        mpi_active = mpi_is_active()

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
        from blond.core.backends.mpi_distributed.callables import (
            mpi_is_active,
        )

        mpi_active = mpi_is_active()

        expected, _ = np.histogram(self.array, bins=8)
        if mpi_active:
            self.distributed_array.mpi_scatter()
        actual = self.distributed_array.histogram(bins=8)
        np.testing.assert_allclose(expected, actual)

    def test_histogram_with_out(self):
        from blond import backend
        from blond.core.backends.mpi_distributed.callables import (
            mpi_is_active,
        )

        mpi_active = mpi_is_active()

        expected, _ = np.histogram(self.array, bins=8)
        if mpi_active:
            self.distributed_array.mpi_scatter()
        actual = np.zeros_like(expected, dtype=backend.float)
        self.distributed_array.histogram(bins=8, out=actual)
        np.testing.assert_allclose(expected, actual)

    def test_barrier(self):
        from blond.core.backends.mpi_distributed.callables import (
            mpi_is_active,
        )

        mpi_active = mpi_is_active()

        if mpi_active:
            self.distributed_array.array_local = (
                self.distributed_array.array_local[:64]
            )
            self.distributed_array.barrier()
            self.assertEqual(
                self.distributed_array.global_size, 2 * 64
            )  # assumes `mpirun -n 2`


class TestDistributedArrayNoMPI(unittest.TestCase):
    def test_no_mpi(self):
        with patch.dict(sys.modules, {"mpi4py": None}):
            sys.modules.pop(
                "blond.generals.distributed.distributed_array", None
            )
            from blond.generals.distributed.distributed_array import (
                DistributedArray,
            )

            rng = np.random.default_rng(0)
            self.array = rng.normal(loc=0, scale=1.0, size=128)
            distributed_array = DistributedArray(self.array.copy())
            self.assertFalse(distributed_array.is_distributed)
            self.assertEqual(distributed_array.rank, 0)
            self.assertEqual(distributed_array.size, 1)


if __name__ == "__main__":
    unittest.main()
