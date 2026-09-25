import unittest
import unittest.mock

import numpy as np
import pytest

from blond import Cupy64Bit, Numpy64Bit, backend, copy_to_cpu
from blond.core.backends.mpi_distributed.callables import (
    phase_space_moments,
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


class TestPhaseSpaceMoments(unittest.TestCase):
    """
    Every phase-space statistic of a beam from one set of sums.

    Means, RMS sizes and the RMS emittance all derive from the same five
    sums over the particles, so they are computed together, in one fused
    pass of the active backend's ``phase_space_sums`` kernel, instead of
    once per statistic.
    """

    def setUp(self):
        rng = np.random.default_rng(11)
        self.dt_host = rng.normal(loc=2.0e-9, scale=3.0e-11, size=4096)
        self.dE_host = 0.4 * (self.dt_host - 2.0e-9) * 1.0e17 + rng.normal(
            loc=1.0e5, scale=2.0e6, size=4096
        )
        self.dt = DistributedArray(
            backend.array(self.dt_host, dtype=backend.float)
        )
        self.dE = DistributedArray(
            backend.array(self.dE_host, dtype=backend.float)
        )

    @unittest.skipIf(is_distributed, "Runs only without `mpirun`")
    def test_the_statistics_match_numpy(self):
        moments = phase_space_moments(dt=self.dt, dE=self.dE)

        self.assertEqual(moments.n_macroparticles, 4096)
        np.testing.assert_allclose(
            moments.mean_dt, np.mean(self.dt_host), rtol=1e-12
        )
        np.testing.assert_allclose(
            moments.mean_dE, np.mean(self.dE_host), rtol=1e-12
        )
        np.testing.assert_allclose(
            moments.sigma_dt, np.std(self.dt_host), rtol=1e-9
        )
        np.testing.assert_allclose(
            moments.sigma_dE, np.std(self.dE_host), rtol=1e-12
        )
        covariance = np.cov(self.dt_host, self.dE_host, bias=True)
        np.testing.assert_allclose(
            moments.rms_emittance,
            np.sqrt(np.linalg.det(covariance)),
            rtol=1e-9,
        )

    def test_the_emittance_is_the_one_rms_emittance_returns(self):
        # Two separate calls: a threaded backend may combine its partial
        # sums in another order each time, so equal to rounding, not bits.
        np.testing.assert_allclose(
            phase_space_moments(dt=self.dt, dE=self.dE).rms_emittance,
            rms_emittance(dt=self.dt, dE=self.dE),
            rtol=1e-12,
        )

    def test_one_fused_pass_over_the_particles(self):
        calls = []
        specials = backend.specials
        for name in (
            "phase_space_sums",
            "sum_1d_array",
            "dot_product_1d_array",
        ):
            original = getattr(specials, name)

            def counted(*args, _name=name, _original=original):
                calls.append(_name)
                return _original(*args)

            patcher = unittest.mock.patch.object(specials, name, counted)
            patcher.start()
            self.addCleanup(patcher.stop)

        phase_space_moments(dt=self.dt, dE=self.dE)

        self.assertEqual(calls, ["phase_space_sums"])
