import unittest

import numpy as np
import pytest

from blond import Cupy64Bit, Numpy64Bit, backend, copy_to_cpu
from blond.core.backends.mpi_distributed.callables import (
    rms_emittance,
)
from blond.core.backends.mpi_distributed.distributed_array import (
    DistributedArray,
)
from blond.core.backends.mpi_distributed.helpers import mpi_is_distributed
from blond.testing.backend_testing import BLonDTestCase

is_distributed = mpi_is_distributed()
not_distributed = not is_distributed


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
