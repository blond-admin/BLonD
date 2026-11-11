import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from blond import Cupy64Bit, Numpy64Bit, backend, copy_to_cpu
from blond.core.backends.mpi_distributed.callables import (
    global_sum_array,
    rms_emittance,
)
from blond.core.backends.mpi_distributed.distributed_array import (
    DistributedArray,
)
from blond.core.backends.mpi_distributed.helpers import mpi_is_distributed
from blond.testing.backend_testing import BLonDTestCase, skip_if_no_cupy

is_distributed = mpi_is_distributed()
not_distributed = not is_distributed


def _mpi_library_reports_cuda_support() -> bool:
    """
    Ask the loaded MPI library whether it is CUDA-aware.

    mpi4py does not wrap this query, so it is called via ctypes. Only
    Open MPI provides `MPIX_Query_cuda_support`; with any other MPI
    library the answer is False, and the CUDA-aware tests skip.
    """
    import ctypes

    try:
        query = getattr(ctypes.CDLL(None), "MPIX_Query_cuda_support", None)
    except (OSError, TypeError):  # e.g. Windows has no `CDLL(None)`
        return False
    return query is not None and query() == 1


class TestCallables(BLonDTestCase):
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
        self.assertAlmostEqual(rms_expected, rms, places=6)

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
        self.addCleanup(backend.change_backend, Numpy64Bit)

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
        self.assertAlmostEqual(float(copy_to_cpu(rms_expected)), rms)


def _two_pass_rms_emittance(dt: np.ndarray, dE: np.ndarray) -> float:
    """Reference RMS emittance from the centred covariance matrix."""
    covariance = np.cov(dt, dE, bias=True)
    return float(np.sqrt(max(np.linalg.det(covariance), 0.0)))


class _RmsEmittanceCancellationCases:
    """Inputs where one-pass second moments lose all significant digits.

    Shared by the MPI and non-MPI test classes below, so the same cases
    run whether or not the arrays are scattered.
    """

    def _rms_emittance_of(self, dt: np.ndarray, dE: np.ndarray) -> float:
        dt_distributed = DistributedArray(
            backend.array(dt, dtype=backend.float)
        )
        dE_distributed = DistributedArray(
            backend.array(dE, dtype=backend.float)
        )
        dt_distributed.mpi_scatter()
        dE_distributed.mpi_scatter()
        return rms_emittance(dt=dt_distributed, dE=dE_distributed)

    def test_rms_emittance_large_dt_offset(self):
        # A 1 ms offset on a 0.3 ns bunch: sum(dt**2)/n and mean(dt)**2
        # agree to ~13 digits and their difference is mostly rounding.
        rng = np.random.default_rng(7)
        n_macroparticles = 100_000
        dt = 1e-3 + rng.normal(scale=0.3e-9, size=n_macroparticles)
        dE = rng.normal(scale=5e8, size=n_macroparticles)
        expected = _two_pass_rms_emittance(dt, dE)
        rms = self._rms_emittance_of(dt, dE)
        self.assertAlmostEqual(rms, expected, delta=1e-8 * expected)

    def test_rms_emittance_zero_for_correlated_beam(self):
        # Perfectly correlated dt/dE span zero phase-space area. The
        # one-pass determinant rounds negative here and gives NaN.
        dt_centred = np.linspace(-1e-9, 1e-9, 1001)
        dt = dt_centred + 1e-6
        dE = 2e17 * dt_centred + 1e6
        rms = self._rms_emittance_of(dt, dE)
        scale = float(np.std(dt) * np.std(dE))
        self.assertFalse(np.isnan(rms))
        self.assertAlmostEqual(rms, 0.0, delta=1e-6 * scale)


class TestRmsEmittanceCancellationWithoutMPI(
    _RmsEmittanceCancellationCases, BLonDTestCase
):
    def setUp(self):
        if is_distributed:
            self.skipTest("Runs only without `mpirun`")


@pytest.mark.mpi
class TestRmsEmittanceCancellationMPI(
    _RmsEmittanceCancellationCases, BLonDTestCase
):
    def setUp(self):
        if not_distributed:
            self.skipTest("Runs only with `mpirun`")


class TestRmsEmittanceRounding(BLonDTestCase):
    def test_zero_emittance_beam_is_not_nan(self):
        import warnings

        dt_host = np.linspace(1e-9, 2e-9, 1001)
        for slope in np.linspace(0.5, 10, 96):
            for offset in (0.0, 1e6):
                dt = DistributedArray(
                    backend.array(dt_host, dtype=backend.float)
                )
                dE = DistributedArray(
                    backend.array(
                        slope * dt_host + offset, dtype=backend.float
                    )
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    emittance = rms_emittance(dt, dE)

                self.assertFalse(np.isnan(emittance), (slope, offset))
                self.assertGreaterEqual(emittance, 0.0)
                self.assertEqual(
                    [
                        w
                        for w in caught
                        if issubclass(w.category, RuntimeWarning)
                    ],
                    [],
                )


class TestGlobalSumArray(BLonDTestCase):
    @unittest.skipIf(is_distributed, "Runs only without `mpirun`")
    def test_without_mpi_returns_input_unchanged(self):
        array = np.arange(6.0).reshape(2, 3)
        result = global_sum_array(array)
        self.assertIs(result, array)
        np.testing.assert_array_equal(result, np.arange(6.0).reshape(2, 3))

    @pytest.mark.mpi
    @unittest.skipIf(not_distributed, "Runs only with `mpirun`")
    def test_sums_over_ranks_in_place(self):
        from blond.core.backends.mpi_distributed.helpers import (
            MPI_RANK,
            MPI_SIZE,
        )

        array = np.full((2, 3), float(MPI_RANK))
        result = global_sum_array(array)
        self.assertIs(result, array)
        expected = sum(range(MPI_SIZE))
        np.testing.assert_array_equal(result, np.full((2, 3), expected))


@pytest.mark.cupy
class TestGlobalSumArrayCupy(BLonDTestCase):
    """A CuPy array is reduced on the host, as MPI may not be CUDA-aware."""

    @skip_if_no_cupy
    def test_mpi_receives_host_memory(self):
        import cupy as cp

        from blond.core.backends.mpi_distributed import callables

        received = []

        def fake_allreduce(_in_place, buffer, op):  # two identical ranks
            received.append(type(buffer))
            buffer *= 2

        fake_mpi = MagicMock()
        fake_mpi.COMM_WORLD.Allreduce.side_effect = fake_allreduce
        array = cp.arange(6.0).reshape(2, 3)
        with (
            patch.object(callables, "MPI", fake_mpi),
            patch.object(callables, "mpi_is_distributed", return_value=True),
        ):
            result = callables.global_sum_array(array)
        self.assertEqual(received, [np.ndarray])
        self.assertIs(result, array)
        np.testing.assert_array_equal(
            copy_to_cpu(result), 2 * np.arange(6.0).reshape(2, 3)
        )

    def _sum_with_fake_mpi(self, array, cuda_aware: bool):
        """Run `global_sum_array` as one of two identical MPI ranks."""
        from blond.core.backends.mpi_distributed import callables

        events = []

        def fake_allreduce(_in_place, buffer, op):
            events.append(("allreduce", buffer))
            buffer *= 2

        fake_mpi = MagicMock()
        fake_mpi.COMM_WORLD.Allreduce.side_effect = fake_allreduce
        environ = {"BLOND_MPI_CUDA_AWARE": str(cuda_aware)}
        with (
            patch.object(callables, "MPI", fake_mpi),
            patch.object(callables, "mpi_is_distributed", return_value=True),
            patch.dict(os.environ, environ),
        ):
            result = callables.global_sum_array(array)
        return result, events

    @skip_if_no_cupy
    def test_host_buffer_is_pinned_and_reused(self):
        import cupy as cp

        from blond.core.backends.mpi_distributed import callables

        first = cp.ones(16)
        second = cp.ones(16)
        _, events_first = self._sum_with_fake_mpi(first, cuda_aware=False)
        _, events_second = self._sum_with_fake_mpi(second, cuda_aware=False)
        buffer = events_first[0][1]
        self.assertIs(events_second[0][1], buffer)
        attributes = cp.cuda.runtime.pointerGetAttributes(buffer.ctypes.data)
        self.assertEqual(attributes.type, cp.cuda.runtime.memoryTypeHost)
        np.testing.assert_array_equal(copy_to_cpu(second), np.full(16, 2.0))
        other_shape, _ = self._sum_with_fake_mpi(cp.ones(8), cuda_aware=False)
        np.testing.assert_array_equal(
            copy_to_cpu(other_shape), np.full(8, 2.0)
        )
        self.assertIsNot(callables._pinned_host_buffer(cp.ones(8)), buffer)

    @skip_if_no_cupy
    def test_cuda_aware_mpi_receives_device_memory_after_sync(self):
        import cupy as cp

        array = cp.arange(6.0)
        stream = MagicMock()
        events = []
        stream.synchronize.side_effect = lambda: events.append(("sync",))
        with patch("cupy.cuda.get_current_stream", return_value=stream):
            result, allreduce_events = self._sum_with_fake_mpi(
                array, cuda_aware=True
            )
        self.assertEqual(stream.synchronize.call_count, 1)
        self.assertEqual(len(allreduce_events), 1)
        self.assertIs(allreduce_events[0][1], array)
        self.assertIs(result, array)
        np.testing.assert_array_equal(copy_to_cpu(result), 2 * np.arange(6.0))

    @pytest.mark.mpi
    @unittest.skipIf(not_distributed, "Runs only with `mpirun`")
    @skip_if_no_cupy
    def test_sums_cupy_array_over_ranks(self):
        import cupy as cp

        from blond.core.backends.mpi_distributed.helpers import (
            MPI_RANK,
            MPI_SIZE,
        )

        array = cp.full((2, 3), float(MPI_RANK))
        result = global_sum_array(array)
        self.assertIs(result, array)
        expected = np.full((2, 3), float(sum(range(MPI_SIZE))))
        np.testing.assert_array_equal(copy_to_cpu(result), expected)


@pytest.mark.mpi
@pytest.mark.cupy
class TestGlobalSumArrayCudaAwareMPI(BLonDTestCase):
    """The CUDA-aware path with a real MPI library, where it is one."""

    def setUp(self):
        if not_distributed:
            self.skipTest("Runs only with `mpirun`")
        if not _mpi_library_reports_cuda_support():
            self.skipTest(
                "MPI library does not report CUDA support"
                " (`MPIX_Query_cuda_support`)"
            )

    @skip_if_no_cupy
    def test_sums_device_arrays_directly(self):
        import cupy as cp

        from blond.core.backends.mpi_distributed import callables
        from blond.core.backends.mpi_distributed.helpers import (
            MPI_RANK,
            MPI_SIZE,
        )

        array = cp.full((2, 3), float(MPI_RANK))
        no_host_copy = patch.object(
            callables,
            "_pinned_host_buffer",
            side_effect=AssertionError("took the host-copy path"),
        )
        environ = {"BLOND_MPI_CUDA_AWARE": "True"}
        with no_host_copy, patch.dict(os.environ, environ):
            result = callables.global_sum_array(array)
        self.assertIs(result, array)
        expected = np.full((2, 3), float(sum(range(MPI_SIZE))))
        np.testing.assert_array_equal(copy_to_cpu(result), expected)


@unittest.skipIf(is_distributed, "Emulates two ranks, runs only without MPI")
class TestCallablesCollectives(BLonDTestCase):
    """Scalar reductions batch into few buffer `Allreduce` calls."""

    def _run_on_two_ranks(self, function, *arrays):
        """`function` of `arrays` on two identical ranks, and collectives."""
        from blond.testing.mocks import identical_mpi_ranks

        with identical_mpi_ranks(n_ranks=2) as fake_mpi:
            distributed = [DistributedArray(backend.array(a)) for a in arrays]
            # assigning an array reduces its size, outside of `function`
            fake_mpi.COMM_WORLD.reset_mock()
            result = function(*distributed)
        comm = fake_mpi.COMM_WORLD
        self.assertEqual(comm.allreduce.call_count, 0, "pickled collective")
        return result, comm.Allreduce.call_count

    def test_rms_emittance_in_two_collectives(self):
        rng = np.random.default_rng(1)
        dt = rng.normal(scale=1e-9, size=100)
        dE = 1e8 * dt + rng.normal(scale=1e6, size=100)
        rms, n_collectives = self._run_on_two_ranks(rms_emittance, dt, dE)
        # the two means, the three second moments; size known already
        self.assertEqual(n_collectives, 2)
        expected = _two_pass_rms_emittance(dt, dE)
        self.assertAlmostEqual(rms, expected, delta=1e-12 * expected)

    def test_count_active_macroparticles_in_one_collective(self):
        from blond.core.backends.flags import BeamFlags
        from blond.core.backends.mpi_distributed.callables import (
            count_active_macroparticles,
        )

        flags = np.array(
            [BeamFlags.ACTIVE.value, BeamFlags.LOST.value] * 3, np.int32
        )
        count, n_collectives = self._run_on_two_ranks(
            count_active_macroparticles, flags
        )
        self.assertEqual(count, 6)
        self.assertIsInstance(count, int)
        self.assertEqual(n_collectives, 1)
