import sys
import unittest
from unittest.mock import patch

import numpy as np
import pytest

from blond.generals.distributed.helpers import mpi_barrier, mpi_is_distributed


@pytest.mark.mpi
class TestDistributedArray(unittest.TestCase):
    def setUp(self):
        from blond.generals.distributed.distributed_array import (
            DistributedArray,
        )

        rng = np.random.default_rng(0)
        self.array = rng.normal(loc=0, scale=1.0, size=128)
        # Force the global extrema onto zero-weight particles so that
        # weighted min/max tests are non-trivial.
        self.array[0] = -999.0
        self.array[1] = 999.0
        self.distributed_array = DistributedArray(self.array.copy())

        self.weights = rng.uniform(0.5, 1.5, size=128)
        self.weights[0] = 0.0  # global minimum particle is inactive
        self.weights[1] = 0.0  # global maximum particle is inactive
        self.distributed_weights = DistributedArray(self.weights.copy())

    def test_local_size(self):
        mpi_active = mpi_is_distributed()

        self.assertEqual(self.distributed_array.local_size, 128)
        if mpi_active:
            self.distributed_array.mpi_scatter()

        if mpi_active:
            self.assertEqual(
                self.distributed_array.local_size, 64
            )  # assumes `mpirun -n 2`
            self.assertTrue(self.distributed_array._is_distributed)
            self.assertEqual(self.distributed_array.global_size, 128)
        else:
            self.assertEqual(self.distributed_array.local_size, 128)
            self.assertFalse(self.distributed_array._is_distributed)
            self.assertEqual(self.distributed_array.global_size, 128)

    def _call_test(self, func, func_name):
        mpi_active = mpi_is_distributed()

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
        mpi_active = mpi_is_distributed()

        expected, _ = np.histogram(self.array, bins=8)
        if mpi_active:
            self.distributed_array.mpi_scatter()
        actual = self.distributed_array.histogram(bins=8)
        np.testing.assert_allclose(expected, actual)

    def test_histogram_with_out(self):
        from blond import backend

        mpi_active = mpi_is_distributed()

        expected, _ = np.histogram(self.array, bins=8)
        if mpi_active:
            self.distributed_array.mpi_scatter()
        actual = np.zeros_like(expected, dtype=backend.float)
        self.distributed_array.histogram(bins=8, out=actual)
        np.testing.assert_allclose(expected, actual)

    # ------------------------------------------------------------------
    # Weighted statistics
    # ------------------------------------------------------------------

    def _call_test_weighted(self, func, func_name):
        """Run func(array, weights) as reference; compare to DistributedArray."""
        mpi_active = mpi_is_distributed()

        expected = func(self.array, self.weights)
        if mpi_active:
            self.distributed_array.mpi_scatter()
            self.distributed_weights.mpi_scatter()
        actual = getattr(self.distributed_array, func_name)(
            weights=self.distributed_weights
        )
        np.testing.assert_almost_equal(expected, actual)

    def test_min_weighted(self):
        self._call_test_weighted(lambda x, w: np.min(x[w > 0]), "min")

    def test_max_weighted(self):
        self._call_test_weighted(lambda x, w: np.max(x[w > 0]), "max")

    def test_min_weighted_differs_from_unweighted(self):
        """Particles with weight 0 must be excluded from the minimum."""
        mpi_active = mpi_is_distributed()
        if mpi_active:
            self.distributed_array.mpi_scatter()
            self.distributed_weights.mpi_scatter()
        weighted_min = self.distributed_array.min(
            weights=self.distributed_weights
        )
        unweighted_min = self.distributed_array.min()
        self.assertGreater(weighted_min, unweighted_min)

    def test_max_weighted_differs_from_unweighted(self):
        """Particles with weight 0 must be excluded from the maximum."""
        mpi_active = mpi_is_distributed()
        if mpi_active:
            self.distributed_array.mpi_scatter()
            self.distributed_weights.mpi_scatter()
        weighted_max = self.distributed_array.max(
            weights=self.distributed_weights
        )
        unweighted_max = self.distributed_array.max()
        self.assertLess(weighted_max, unweighted_max)

    def test_mean_weighted(self):
        self._call_test_weighted(
            lambda x, w: np.sum(x * w) / np.sum(w), "mean"
        )

    def test_std_weighted(self):
        def weighted_std(x, w):
            mean = np.sum(w * x) / np.sum(w)
            variance = np.sum(w * (x - mean) ** 2) / np.sum(w)
            return np.sqrt(variance)

        self._call_test_weighted(weighted_std, "std")

    def test_sum_weighted(self):
        self._call_test_weighted(lambda x, w: np.sum(x * w), "sum")

    def test_histogram_weighted(self):
        mpi_active = mpi_is_distributed()

        expected, _ = np.histogram(self.array, bins=8, weights=self.weights)
        if mpi_active:
            self.distributed_array.mpi_scatter()
            self.distributed_weights.mpi_scatter()
        actual = self.distributed_array.histogram(
            bins=8, weights=self.distributed_weights
        )
        np.testing.assert_allclose(expected, actual)

    def test_histogram_weighted_with_out(self):
        from blond import backend

        mpi_active = mpi_is_distributed()

        expected, _ = np.histogram(self.array, bins=8, weights=self.weights)
        if mpi_active:
            self.distributed_array.mpi_scatter()
            self.distributed_weights.mpi_scatter()
        out = np.zeros(8, dtype=backend.float)
        self.distributed_array.histogram(
            bins=8, out=out, weights=self.distributed_weights
        )
        np.testing.assert_allclose(expected, out)

    def test_histogram_weighted_uniform_equals_unweighted(self):
        """Uniform weights of 1 must reproduce the unweighted histogram."""
        from blond.generals.distributed.distributed_array import (
            DistributedArray,
        )

        mpi_active = mpi_is_distributed()

        uniform_weights = DistributedArray(np.ones(128))
        if mpi_active:
            self.distributed_array.mpi_scatter()
            uniform_weights.mpi_scatter()

        unweighted = self.distributed_array.histogram(bins=8)
        weighted = self.distributed_array.histogram(
            bins=8, weights=uniform_weights
        )
        np.testing.assert_allclose(weighted, unweighted)

    def test_barrier(self):
        mpi_active = mpi_is_distributed()

        if mpi_active:
            self.distributed_array.array_local = (
                self.distributed_array.array_local[:64]
            )
            mpi_barrier()
            self.assertEqual(
                self.distributed_array.global_size, 2 * 64
            )  # assumes `mpirun -n 2`


@pytest.mark.mpi
class TestDistributedArrayNoMPI(unittest.TestCase):
    def test_no_mpi(self):
        with patch.dict(sys.modules, {"mpi4py": None}):
            # trigger new import
            sys.modules.pop(
                "blond.generals.distributed.distributed_array", None
            )
            from blond.generals.distributed.distributed_array import (
                DistributedArray,
            )

            rng = np.random.default_rng(0)
            self.array = rng.normal(loc=0, scale=1.0, size=128)
            distributed_array = DistributedArray(self.array.copy())
            self.assertFalse(distributed_array._is_distributed)
            self.assertEqual(distributed_array._rank, 0)
            self.assertEqual(distributed_array._size, 1)


if __name__ == "__main__":
    unittest.main()
