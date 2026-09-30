import ctypes as ct

import numpy as np
import pytest

from blond.core.backends.backend import Numpy64Bit, backend
from blond.core.backends.cpp.callables import c_index_t
from blond.core.backends.deferred import kernel_call_records as records
from blond.testing.backend_testing import BLonDTestCase


def _pack(*kernel_calls: tuple[str, dict]) -> np.ndarray:
    parts = []
    for method, values in kernel_calls:
        kernel = records.KERNELS_BY_SPECIALS_METHOD[method]
        item = np.zeros((), dtype=kernel.record_dtype)
        kernel.pack(item, values, [])
        parts.append(item.tobytes())
    return np.frombuffer(b"".join(parts), dtype=np.uint8).copy()


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

    def _execute(self, batch, dt, dE, chunk_size=4096) -> None:
        self.library.execute_kernel_call_batch(
            ct.c_void_p(batch.ctypes.data),
            ct.c_size_t(batch.size),
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
                    ("kick_single_harmonic", KICK),
                    ("drift_simple", DRIFT),
                )
                self._execute(batch, dt, dE, chunk_size)
                np.testing.assert_allclose(dE, dE_eager, rtol=1e-12)
                np.testing.assert_allclose(dt, dt_eager, rtol=1e-12)

    def test_zero_macroparticles(self) -> None:  # Review Focus 4
        dt, dE = np.empty(0), np.empty(0)
        batch = _pack(("drift_simple", DRIFT))
        self._execute(batch, dt, dE)  # must not crash

    def test_args_sizes_match_dtypes(self) -> None:
        for kernel in records.DEFERRABLE_KERNELS:
            self.assertEqual(
                self.library.kernel_call_args_size(kernel.kernel_id),
                kernel.args_dtype.itemsize,
            )
        self.assertEqual(self.library.kernel_call_args_size(999), 0)
