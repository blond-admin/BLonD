import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from blond import backend, copy_to_cpu
from blond.core.backends.mpi_distributed.distributed_array import (
    DistributedArray,
    concatenate,
    local_sizes_match,
    scatter_from_root,
)
from blond.core.backends.mpi_distributed.helpers import (
    mpi_barrier,
    mpi_is_distributed,
)
from blond.generals.cupy_.no_cupy_import import is_cupy_array
from blond.generals.exceptions_ import ArrayPrecisionError
from blond.testing.backend_testing import BLonDTestCase, skip_if_no_cupy


@pytest.mark.mpi
class TestDistributedArray(BLonDTestCase):
    def setUp(self):
        from blond.core.backends.mpi_distributed.distributed_array import (
            DistributedArray,
        )

        rng = np.random.default_rng(0)
        self.array = np.astype(
            rng.normal(loc=0, scale=1.0, size=128), backend.float
        )
        self.distributed_array = DistributedArray(
            scatter_from_root(backend.array(self.array.copy()))
        )

    def test_local_size(self):
        if mpi_is_distributed():
            self.assertEqual(
                self.distributed_array.local_size, 64
            )  # assumes `mpirun -n 2`
            self.assertTrue(self.distributed_array._is_distributed)
            self.assertEqual(self.distributed_array.global_size, 128)
        else:
            self.assertEqual(self.distributed_array.local_size, 128)
            self.assertFalse(self.distributed_array._is_distributed)
            self.assertEqual(self.distributed_array.global_size, 128)

    def test_global_size_read_on_root_only_after_scatter(self):
        """Reading the size after creating it needs no other rank."""
        if self.distributed_array._rank == 0:
            self.assertEqual(self.distributed_array.global_size, 128)
        mpi_barrier()

    def test_global_size_read_on_root_only_after_assignment(self):
        """Assigning reduces the size, so reading it needs no rank."""
        self.distributed_array.array_local = (
            self.distributed_array.array_local[:16]
        )
        if self.distributed_array._rank == 0:
            n_ranks = self.distributed_array._size
            self.assertEqual(self.distributed_array.global_size, 16 * n_ranks)
        mpi_barrier()

    def test_scatter_of_data_on_root_only(self):
        """Only rank 0 holds the data; the other ranks pass None."""
        on_root = self.distributed_array._rank == 0
        chunk = scatter_from_root(
            backend.array(self.array.copy()) if on_root else None
        )
        self.assertEqual(DistributedArray(chunk).global_size, 128)

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
        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")
        expected = func(self.array)
        actual = getattr(self.distributed_array, func_name)()
        np.testing.assert_almost_equal(actual, expected, 11)

    def test_min(self):
        self._call_test(np.min, "min")

    def test_max(self):
        self._call_test(np.max, "max")

    def test_mean(self):
        self._call_test(np.mean, "mean")

    def test_std(self):
        self._call_test(np.std, "std")

    def test_sum(self):
        self._call_test(lambda x: float(np.sum(x)), "sum")

    def test_histogram(self):
        expected, _ = np.histogram(self.array, bins=8)
        actual = self.distributed_array.histogram(bins=8)
        np.testing.assert_allclose(expected, copy_to_cpu(actual))

    def test_histogram_cache_hit(self):
        self.distributed_array.histogram(bins=8)  # cache miss
        actual = self.distributed_array.histogram(bins=8)  # cache hit
        expected, _ = np.histogram(self.array, bins=8)
        np.testing.assert_allclose(expected, copy_to_cpu(actual))

    def test_histogram_with_out(self):
        from blond import backend

        expected, _ = np.histogram(self.array, bins=8)
        actual = backend.zeros_like(expected, dtype=backend.float)
        self.distributed_array.histogram(bins=8, out=actual)
        np.testing.assert_allclose(expected, copy_to_cpu(actual))

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
                10  # not 1 to see if result is accumulated (shouldn't be)
            ):
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

    def test_concatenate(self):
        mpi_active = mpi_is_distributed()

        if mpi_active:
            array_1 = DistributedArray(
                scatter_from_root(backend.array([1, 2, 3, 4, 5, 6]))
            )
            array_2 = DistributedArray(
                scatter_from_root(backend.array([7, 8, 9, 10, 11, 12]))
            )

            array_3 = concatenate(array_1, array_2)

            self.assertEqual(array_3.global_size, 12)
            self.assertEqual(array_3.min(), 1)
            self.assertEqual(array_3.max(), 12)

    @skip_if_no_cupy
    def test_concatenate_errors(self):
        array_1 = DistributedArray(
            np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        )
        array_2 = DistributedArray(
            np.array([7, 8, 9, 10, 11, 12], dtype=np.float64)
        )

        with self.assertRaises(ArrayPrecisionError):
            concatenate(array_1, array_2)

        import cupy as cp

        array_1 = DistributedArray(cp.array([1, 2, 3, 4, 5, 6]))
        array_2 = DistributedArray(np.array([7, 8, 9, 10, 11, 12]))

        with self.assertRaises(TypeError):
            concatenate(array_1, array_2)

    def test_gather(self):
        in_array = np.array([1, 2, 3, 4, 5, 6])
        array = DistributedArray(scatter_from_root(in_array))
        out_array = array.mpi_gather()

        mpi_active = mpi_is_distributed()
        if mpi_active:
            rank = array._comm.Get_rank()
            if rank != 0:
                return

        np.testing.assert_equal(in_array, out_array)


@pytest.mark.mpi
class TestDistributedArrayNoMPI(BLonDTestCase):
    def test_no_mpi(self):
        with patch.dict(sys.modules, {"mpi4py": None}):
            # trigger new import; `distributed_array` takes MPI from `helpers`
            for module in ("helpers", "distributed_array"):
                sys.modules.pop(
                    f"blond.core.backends.mpi_distributed.{module}", None
                )
            from blond.core.backends.mpi_distributed.distributed_array import (
                DistributedArray,
            )

            rng = np.random.default_rng(0)
            self.array = rng.normal(loc=0, scale=1.0, size=128)
            distributed_array = DistributedArray(self.array.copy())
            self.assertFalse(distributed_array._is_distributed)
            self.assertEqual(distributed_array._rank, 0)
            self.assertEqual(distributed_array._size, 1)


class _StdCancellationCases:
    """Inputs where a one-pass ``sum(x**2)/n - mean**2`` breaks down.

    Shared by the MPI and non-MPI test classes below, so the same cases
    run whether or not the array is scattered.
    """

    def _std_of(self, array: np.ndarray) -> float:
        distributed_array = DistributedArray(
            scatter_from_root(backend.array(array, dtype=backend.float))
        )
        return distributed_array.std()

    def test_std_constant_array_is_zero(self):
        # One-pass variance rounds slightly negative here and
        # ``math.sqrt`` raises ``ValueError: math domain error``.
        std = self._std_of(np.full(1000, 0.1))
        self.assertAlmostEqual(std, 0.0, delta=1e-15)

    def test_std_small_spread_large_offset(self):
        rng = np.random.default_rng(42)
        array = rng.normal(loc=1e-3, scale=1e-12, size=10_000)
        expected = float(np.std(array))
        std = self._std_of(array)
        self.assertAlmostEqual(std, expected, delta=1e-6 * expected)


class TestStdCancellationWithoutMPI(_StdCancellationCases, BLonDTestCase):
    def setUp(self):
        if mpi_is_distributed():
            self.skipTest("Runs only without `mpirun`")


@pytest.mark.mpi
class TestStdCancellationMPI(_StdCancellationCases, BLonDTestCase):
    def setUp(self):
        if not mpi_is_distributed():
            self.skipTest("Runs only with `mpirun`")


if __name__ == "__main__":
    unittest.main()


@pytest.mark.mpi
class TestScatterRootFailure(BLonDTestCase):
    """A failure on rank 0 must not strand the other ranks.

    `scatter_from_root` prepares the chunks in an ``if rank == 0`` block. If
    that raises, rank 0 never reaches ``comm.scatter`` while every other
    rank is already blocked inside it, and the run hangs forever instead
    of reporting the error.
    """

    @unittest.skipUnless(mpi_is_distributed(), "Runs only with `mpirun`")
    def test_root_failure_raises_on_every_rank(self):
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        # only 1D arrays can be scattered; rank 0 must say so to everyone
        with self.assertRaises(Exception) as caught:
            scatter_from_root(backend.zeros((4, 4), dtype=backend.float))

        if rank == 0:
            self.assertIsInstance(caught.exception, ValueError)
        else:
            self.assertIn("Rank 0 failed", str(caught.exception))

    @unittest.skipUnless(mpi_is_distributed(), "Runs only with `mpirun`")
    def test_scatter_keeps_dtype_and_uses_backend_arrays(self):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        array = backend.arange(13, dtype=np.int32)
        local = scatter_from_root(array if comm.Get_rank() == 0 else None)

        self.assertEqual(local.dtype, np.int32)
        self.assertIs(type(local), type(backend.zeros(1)))
        np.testing.assert_array_equal(
            copy_to_cpu(local),
            np.array_split(np.arange(13), comm.Get_size())[comm.Get_rank()],
        )

    @unittest.skipUnless(mpi_is_distributed(), "Runs only with `mpirun`")
    def test_scatter_of_none_gives_none_on_every_rank(self):
        self.assertIsNone(scatter_from_root(None))


class TestMinMaxWithoutElements(BLonDTestCase):
    """`min`/`max` of an array without elements raise, as NumPy does."""

    def setUp(self):
        if mpi_is_distributed():
            self.skipTest("Runs only without `mpirun`")

    def test_min_and_max_of_empty_array_raise(self):
        empty = DistributedArray(backend.zeros(0, dtype=backend.float))
        for reduction in ("min", "max"):
            with (
                self.subTest(reduction=reduction),
                self.assertRaises(ValueError),
            ):
                getattr(empty, reduction)()


@pytest.mark.mpi
class TestMinMaxEmptyRank(BLonDTestCase):
    """A rank without particles (e.g. all lost) must not strand the others.

    `min`/`max` used to reduce the local array before the `allreduce`: an
    empty rank raised there while every other rank waited in the
    `allreduce` forever.
    """

    def setUp(self):
        if not mpi_is_distributed():
            self.skipTest("Runs only with `mpirun`")
        from mpi4py import MPI

        self.rank = MPI.COMM_WORLD.Get_rank()

    def test_min_and_max_skip_empty_rank(self):
        # `mpirun -n 2`: rank 1 holds no elements
        local = np.zeros(0) if self.rank == 1 else np.array([3.0, -2.0, 7.0])
        distributed_array = DistributedArray(
            backend.array(local, dtype=backend.float)
        )
        self.assertEqual(distributed_array.min(), -2.0)
        self.assertEqual(distributed_array.max(), 7.0)

    def test_min_and_max_raise_on_every_rank_without_elements(self):
        empty = DistributedArray(backend.zeros(0, dtype=backend.float))
        for reduction in ("min", "max"):
            with (
                self.subTest(reduction=reduction),
                self.assertRaises(ValueError),
            ):
                getattr(empty, reduction)()


@pytest.mark.mpi
@pytest.mark.cupy
class TestDistributedArrayCupyMPI(BLonDTestCase):
    """Histograms of GPU arrays must not hand device memory to MPI."""

    def setUp(self):
        if not mpi_is_distributed():
            self.skipTest("Only with MPI")

    @skip_if_no_cupy
    def test_histogram_of_cupy_array(self):
        import cupy as cp

        from blond import Cupy64Bit, Numpy64Bit

        backend.change_backend(Cupy64Bit)
        self.addCleanup(backend.change_backend, Numpy64Bit)
        array = np.random.default_rng(0).normal(size=128)
        expected, _ = np.histogram(array, bins=8)
        distributed_array = DistributedArray(
            scatter_from_root(cp.asarray(array))
        )
        actual = distributed_array.histogram(bins=8)
        self.assertTrue(is_cupy_array(actual))
        np.testing.assert_allclose(copy_to_cpu(actual), expected)


class TestReductionsCollectives(BLonDTestCase):
    """Each reduction costs one buffer `Allreduce` over the ranks.

    mpi4py's lowercase `allreduce` pickles its Python object and sends a
    length first; called ~10 times per turn, that dominated the MPI cost
    of a turn on GPUs.
    """

    def setUp(self):
        if mpi_is_distributed():
            self.skipTest("Emulates two ranks, runs only without `mpirun`")
        self.values = np.array([3.0, -2.0, 7.0, 0.5])

    def _reduce(self, reduction: str, warm_global_size: bool = True):
        """`reduction` of the values on two ranks, and its collectives."""
        from blond.testing.mocks import identical_mpi_ranks

        with identical_mpi_ranks(n_ranks=2) as fake_mpi:
            array = DistributedArray(backend.array(self.values))
            if warm_global_size:
                array.global_size  # noqa: B018
            fake_mpi.COMM_WORLD.reset_mock()
            result = getattr(array, reduction)()
        comm = fake_mpi.COMM_WORLD
        self.assertEqual(comm.allreduce.call_count, 0, "pickled collective")
        return result, comm.Allreduce.call_count

    def test_global_size_is_reduced_once(self):
        from blond.testing.mocks import identical_mpi_ranks

        with identical_mpi_ranks(n_ranks=2) as fake_mpi:
            array = DistributedArray(backend.array(self.values))
            self.assertEqual(array.global_size, 8)
            self.assertEqual(array.global_size, 8)
            self.assertEqual(fake_mpi.COMM_WORLD.Allreduce.call_count, 1)
            array.array_local = array.array_local[:1]
            self.assertEqual(array.global_size, 2)
        self.assertEqual(fake_mpi.COMM_WORLD.allreduce.call_count, 0)

    def test_assignment_reduces_global_size(self):
        """Every rank assigns together, so reading needs no other rank."""
        from blond.testing.mocks import identical_mpi_ranks

        with identical_mpi_ranks(n_ranks=2) as fake_mpi:
            array = DistributedArray(backend.array(self.values))
            array.array_local = array.array_local[:1]
            comm = fake_mpi.COMM_WORLD
            n_after_assignment = comm.Allreduce.call_count
            self.assertEqual(array.global_size, 2)
            self.assertEqual(comm.Allreduce.call_count, n_after_assignment)

    def test_concatenate_global_size(self):
        from blond.testing.mocks import identical_mpi_ranks

        with identical_mpi_ranks(n_ranks=2):
            array_1 = DistributedArray(backend.array(self.values))
            array_2 = DistributedArray(backend.array(self.values[:1]))
            joined = concatenate(array_1, array_2)
            self.assertEqual(joined.global_size, 10)

    def test_local_sizes_match_in_one_collective(self):
        from blond.testing.mocks import identical_mpi_ranks

        with identical_mpi_ranks(n_ranks=2) as fake_mpi:
            four = DistributedArray(backend.array(self.values))
            also_four = DistributedArray(backend.array(self.values))
            three = DistributedArray(backend.array(self.values[:3]))
            comm = fake_mpi.COMM_WORLD
            comm.reset_mock()
            self.assertTrue(local_sizes_match(four, also_four, four))
            self.assertFalse(local_sizes_match(four, three))
            self.assertEqual(comm.Allreduce.call_count, 2)
            # MAX: a mismatch on any rank is a mismatch on every rank
            for call in comm.Allreduce.call_args_list:
                self.assertIs(call.kwargs["op"], fake_mpi.MAX)

    def test_local_sizes_match_without_mpi(self):
        four = DistributedArray(backend.array(self.values))
        three = DistributedArray(backend.array(self.values[:3]))
        self.assertTrue(local_sizes_match(four, four))
        self.assertFalse(local_sizes_match(three, four))

    def test_min_max_in_one_collective(self):
        result, n_collectives = self._reduce("min_max")
        self.assertEqual(result, (-2.0, 7.0))
        self.assertEqual(n_collectives, 1)

    def test_one_collective_per_reduction(self):
        values = self.values
        expected = {
            "min": (values.min(), 1),
            "max": (values.max(), 1),
            "sum": (2 * values.sum(), 1),  # two ranks
            "mean": (values.mean(), 1),
            "std": (values.std(), 2),  # two passes: mean, then spread
        }
        for reduction, (value, n_collectives) in expected.items():
            with self.subTest(reduction=reduction):
                result, n_calls = self._reduce(reduction)
                self.assertAlmostEqual(result, value, places=12)
                self.assertEqual(n_calls, n_collectives)

    def test_min_max_of_empty_array_raises(self):
        empty = DistributedArray(backend.zeros(0, dtype=backend.float))
        with self.assertRaises(ValueError):
            empty.min_max()


class _FakeNonRootComm:
    """Stands in for ``MPI.COMM_WORLD`` as seen from rank 1 of 2.

    Coverage is only collected on rank 0 of an ``mpirun``, so the
    receiving side of `scatter_from_root` is checked here without MPI.
    """

    def __init__(self, global_array: np.ndarray | None, root_error=None):
        self._global_array = global_array
        self._root_error = root_error

    def Get_size(self) -> int:
        return 2

    def Get_rank(self) -> int:
        return 1

    def bcast(self, obj, root: int):
        if self._global_array is None:
            return None, self._root_error
        header = (self._global_array.size, self._global_array.dtype.str)
        return header, self._root_error

    def Recv(self, buffer: np.ndarray, source: int) -> None:
        buffer[:] = np.array_split(self._global_array, 2)[1]


class TestScatterFromRoot(BLonDTestCase):
    """`scatter_from_root` turns rank 0's array into local chunks."""

    def setUp(self):
        if mpi_is_distributed():
            self.skipTest("Fakes the MPI ranks, runs only without `mpirun`")

    def _on_rank_1(self, comm: _FakeNonRootComm):
        """Run `scatter_from_root` as rank 1, which holds None."""
        fake_mpi = SimpleNamespace(COMM_WORLD=comm)
        with patch(
            "blond.core.backends.mpi_distributed.distributed_array.MPI",
            fake_mpi,
        ):
            return scatter_from_root(None)

    def test_without_mpi_returns_the_array(self):
        array = backend.array([1.0, 2.0, 3.0])
        self.assertIs(scatter_from_root(array), array)

    def test_without_mpi_none_stays_none(self):
        self.assertIsNone(scatter_from_root(None))

    def test_non_root_rank_receives_its_chunk(self):
        global_array = np.arange(13, dtype=np.int32)
        local = self._on_rank_1(_FakeNonRootComm(global_array))
        self.assertEqual(local.dtype, np.int32)
        self.assertIs(type(local), type(backend.zeros(1)))
        np.testing.assert_array_equal(
            copy_to_cpu(local), np.array_split(global_array, 2)[1]
        )

    def test_non_root_rank_gets_none_if_root_has_none(self):
        self.assertIsNone(self._on_rank_1(_FakeNonRootComm(None)))

    def test_non_root_rank_raises_rank_zero_error(self):
        comm = _FakeNonRootComm(
            np.arange(3), root_error="ValueError: bad input"
        )
        with self.assertRaisesRegex(RuntimeError, "bad input"):
            self._on_rank_1(comm)


class TestNoNoneArray(BLonDTestCase):
    def test_none_is_not_an_array(self):
        """Ranks hold their chunk; root-only data is not distributed."""
        with self.assertRaisesRegex(TypeError, "scatter_from_root"):
            DistributedArray(None)
