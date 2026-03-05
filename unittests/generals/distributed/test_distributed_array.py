import sys
import unittest
from unittest.mock import patch

import numpy as np
import pytest

from blond.generals.cupy.no_cupy_import import is_cupy_array
from blond.generals.distributed.helpers import mpi_barrier, mpi_is_distributed


@pytest.mark.mpi
class TestDistributedArray(unittest.TestCase):
    def setUp(self):
        from blond.generals.distributed.distributed_array import (
            DistributedArray,
        )

        rng = np.random.default_rng(0)
        self.array = rng.normal(loc=0, scale=1.0, size=128)
        self.distributed_array = DistributedArray(self.array.copy())

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

    def test_copy_as_numpy(self):
        array = self.distributed_array.copy_as_numpy()
        assert array.device == "cpu"

    def test_copy_as_cupy(self):
        try:
            import cupy  # type: ignore
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            self.skipTest(str(exc))
        array = self.distributed_array.copy_as_cupy()
        assert is_cupy_array(array)

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

    def test_histogram_sparse_left_edged(self) -> None:
        from blond.generals.distributed.distributed_array import (
            DistributedArray,
        )

        mpi_active = mpi_is_distributed()

        if mpi_active:
            particles_x = []
            # mark all left and right edges, left edge should result 2, right 1
            for left_edge in (-12, -12 + 2 * 8, -12 + 4 * 8):
                for _ in range(2):  # so hist_y counts two
                    particles_x.append(left_edge)
            for right_edge in (-12 + 4, -12 + 2 * 8 + 4, -12 + 4 * 8 + 4):
                for _ in range(1):  # so hist_y counts one
                    particles_x.append(right_edge)
            da = DistributedArray(np.array(particles_x, float))
            mpi_barrier()

            bins_per_profile = 4
            n_profiles = 3
            array_write = np.ones(bins_per_profile * n_profiles, dtype=float)
            filling_pattern = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
            bucket_index_to_memory_index = np.array(
                [0, 0, 4, 4, 8, 8],
                dtype=np.int32,
            )

            for _ in range(
                10
            ):  # not 1 to see if result is accumulated (shouldn't be)
                result_direct = da.histogram_sparse(
                    out=array_write,
                    first_left_cut=-12,
                    left_cut_distance=8,
                    cut_width=4,
                    bins_per_profile=bins_per_profile,
                    n_active_profiles=n_profiles,
                    filling_pattern=filling_pattern,
                    bucket_index_to_memory_index=bucket_index_to_memory_index,
                )
            result_indirect = array_write
            expected_single_node = np.array(
                [
                    2.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            )
            mpi_nodes = 2  # expect `mpirun -n 2`
            expected_combined = mpi_nodes * expected_single_node

            np.testing.assert_allclose(result_direct, expected_combined)
            np.testing.assert_allclose(result_indirect, expected_combined)


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
