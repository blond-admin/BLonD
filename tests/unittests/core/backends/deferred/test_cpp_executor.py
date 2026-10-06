import ctypes as ct

import numpy as np
import pytest

from blond.core.backends.backend import Numpy64Bit, backend
from blond.core.backends.cpp.callables import DEFERRED_CHUNK_SIZE, c_index_t
from blond.core.backends.deferred import kernel_call_records as records
from blond.core.backends.deferred.kernel_call_queue import KernelCallQueue
from blond.testing.backend_testing import BLonDTestCase


def _pack(*kernel_calls: records.KernelCallArgs) -> np.ndarray:
    queue = KernelCallQueue()
    for kernel_call in kernel_calls:
        queue.append(kernel_call)
    return queue.buffer[: queue.n_bytes].copy()


KICK = dict(
    voltage=8e3, omega_rf=2e7, phi_rf=0.1, charge=1.0, acceleration_kick=12.0
)
DRIFT = dict(T=1e-6, eta_0=0.01, beta=0.9, energy=2e9)


@pytest.mark.backend_mutation
class TestCppExecutor(BLonDTestCase):
    def setUp(self) -> None:
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")
        self.eager = backend.specials
        self.library = self.eager._library

    def tearDown(self) -> None:
        backend.set_specials("python")

    def _execute(self, batch, dt, dE, chunk_size=DEFERRED_CHUNK_SIZE) -> None:
        self.library.execute_kernel_call_batch(
            ct.c_void_p(batch.ctypes.data),
            ct.c_size_t(batch.size),
            ct.c_int(-1),  # no counting record
            c_index_t(0),
            ct.c_void_p(dt.ctypes.data),
            ct.c_void_p(dE.ctypes.data),
            c_index_t(len(dt)),
            c_index_t(chunk_size),
        )

    def test_kick_then_drift_matches_eager(self) -> None:
        rng = np.random.default_rng(1)
        for n in (1, 7, 1000, 100003):
            for chunk_size in (1, 64, 4096, 10**9):
                dt = rng.uniform(-1e-9, 1e-9, n)
                dE = rng.uniform(-1e6, 1e6, n)
                dt_eager, dE_eager = dt.copy(), dE.copy()
                self.eager.kick_single_harmonic(
                    dt=dt_eager, dE=dE_eager, **KICK
                )
                self.eager.drift_simple(dt=dt_eager, dE=dE_eager, **DRIFT)
                batch = _pack(
                    records.KickSingleHarmonicArgs(**KICK),
                    records.DriftSimpleArgs(**DRIFT),
                )
                self._execute(batch, dt, dE, chunk_size)
                np.testing.assert_allclose(dE, dE_eager, rtol=1e-12)
                np.testing.assert_allclose(dt, dt_eager, rtol=1e-12)

    def test_zero_macroparticles(self) -> None:  # Review Focus 4
        dt, dE = np.empty(0), np.empty(0)
        batch = _pack(records.DriftSimpleArgs(**DRIFT))
        self._execute(batch, dt, dE)  # must not crash

    def test_args_sizes_match_dtypes(self) -> None:
        for args_type in records.KERNEL_CALL_ARGS:
            self.assertEqual(
                self.library.kernel_call_args_size(args_type.kernel_id()),
                args_type.args_dtype().itemsize,
            )
        self.assertEqual(self.library.kernel_call_args_size(999), 0)


class TestDeferredChunkSize(BLonDTestCase):
    def test_chunk_fits_hyperthread_share_of_l1(self) -> None:
        # All records of a batch are applied to one chunk before the
        # next, so its dt/dE must stay L1-resident. Two hyperthreads
        # share one L1d, and 32 KiB is the smallest common L1d size, so
        # each thread's chunk gets at most 16 KiB of 64-bit dt + dE.
        chunk_bytes = 2 * DEFERRED_CHUNK_SIZE * np.dtype(np.float64).itemsize
        self.assertLessEqual(chunk_bytes, 32 * 1024 // 2)
