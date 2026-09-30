import threading

import numpy as np
import pytest

from blond.core.backends.backend import Numpy64Bit, Specials, backend
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.testing.backend_testing import BLonDTestCase, cupy_available

RNG = np.random.default_rng(3)


def _beam(n):
    """Beam coordinates on the active backend's device."""
    return (
        backend.array(RNG.uniform(-1e-8, 1e-8, n), dtype=backend.float),
        backend.array(RNG.uniform(-1e6, 1e6, n), dtype=backend.float),
    )


def _close(actual, expected, **tolerances):
    np.testing.assert_allclose(
        copy_to_cpu(actual), copy_to_cpu(expected), **tolerances
    )


def _equal(a, b) -> bool:
    return np.array_equal(copy_to_cpu(a), copy_to_cpu(b))


KICK = dict(
    voltage=8e3, omega_rf=2e7, phi_rf=0.1, charge=1.0, acceleration_kick=12.0
)
DRIFT = dict(T=1e-6, eta_0=0.01, beta=0.9, energy=2e9)


def _turn(specials, dt, dE, n_rf=3, n_alpha=2, n_bins=64):
    specials.kick_single_harmonic(dt=dt, dE=dE, **KICK)
    specials.kick_multi_harmonic(
        dt=dt,
        dE=dE,
        voltage=np.linspace(1e3, 2e3, n_rf),
        omega_rf=np.linspace(1e7, 3e7, n_rf),
        phi_rf=np.linspace(0, 1, n_rf),
        charge=1.0,
        n_rf=n_rf,
        acceleration_kick=5.0,
    )
    specials.drift_simple(dt=dt, dE=dE, **DRIFT)
    specials.drift_like_line_segment(dt=dt, dE=dE, **DRIFT)
    specials.drift_exact(
        dt=dt,
        dE=dE,
        T=1e-6,
        alpha_0=0.01,
        higher_alpha=np.linspace(1e-3, 2e-3, n_alpha),
        beta=0.9,
        energy=2e9,
    )
    # Host arrays for the inlined RF/alpha values (the ABC's NumpyArray),
    # device arrays for the profile-sized interpolated kick inputs.
    specials.kick_interpolated(
        dt=dt,
        dE=dE,
        voltage=backend.array(np.sin(np.linspace(0, 3, n_bins)) * 1e3),
        bin_centers=backend.array(np.linspace(-1e-8, 1e-8, n_bins)),
        charge=1.0,
        acceleration_kick=3.0,
    )


@pytest.mark.backend_mutation
class TestCppDeferredSpecials(BLonDTestCase):
    mode = "cpp_deferred"
    eager_mode = "cpp"

    def setUp(self) -> None:
        backend.change_backend(Numpy64Bit)
        backend.set_specials(self.eager_mode)
        self.eager = backend.specials
        backend.set_specials(self.mode)
        self.deferred = backend.specials
        # `backend.specials` is the class itself for cpp/cpp_deferred but
        # an instance for other modes (e.g. cuda_deferred); resolve the
        # deferred specials class once so both cases work uniformly.
        self.deferred_class = (
            self.deferred
            if isinstance(self.deferred, type)
            else type(self.deferred)
        )

    def tearDown(self) -> None:
        self.deferred.flush()
        backend.set_specials("python")

    def _assert_matches_eager(self, **turn_kwargs) -> None:
        for n in (1, 7, 1000, 100003):
            dt, dE = _beam(n)
            dt_eager, dE_eager = backend.copy(dt), backend.copy(dE)
            for _ in range(3):
                _turn(self.eager, dt_eager, dE_eager, **turn_kwargs)
                _turn(self.deferred, dt, dE, **turn_kwargs)
            self.deferred.flush()
            _close(dt, dt_eager, rtol=1e-11, atol=0)
            _close(dE, dE_eager, rtol=1e-11, atol=1e-6)

    def test_matches_eager(self) -> None:
        self._assert_matches_eager()

    def test_more_than_32_harmonics(self) -> None:
        self._assert_matches_eager(n_rf=40)

    def test_drift_exact_coefficient_counts(self) -> None:
        for n_alpha in (0, 1, 4, 5, 9):  # 9 falls back to eager
            self._assert_matches_eager(n_alpha=n_alpha)

    def _assert_batch_matches_eager(self, calls) -> None:
        dt, dE = _beam(100003)
        dt_eager, dE_eager = backend.copy(dt), backend.copy(dE)
        for specials, dt_, dE_ in (
            (self.eager, dt_eager, dE_eager),
            (self.deferred, dt, dE),
        ):
            for method, kwargs in calls:
                getattr(specials, method)(dt=dt_, dE=dE_, **kwargs)
        self.deferred.flush()
        _close(dt, dt_eager, rtol=1e-11, atol=0)
        _close(dE, dE_eager, rtol=1e-11, atol=1e-6)

    def test_kick_only_batch(self) -> None:
        # Writes dE only: the CUDA executor skips storing dt.
        self._assert_batch_matches_eager([("kick_single_harmonic", KICK)] * 2)

    def test_drift_only_batch(self) -> None:
        # Writes dt only: the CUDA executor skips storing dE.
        self._assert_batch_matches_eager(
            [("drift_simple", DRIFT), ("drift_like_line_segment", DRIFT)]
        )

    def test_queued_until_flush(self) -> None:
        dt, dE = _beam(10)
        before = backend.copy(dE)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **KICK)
        self.assertTrue(_equal(dE, before))
        self.deferred.flush()
        self.assertFalse(_equal(dE, before))

    def test_unqueued_kernel_flushes_first(self) -> None:
        dt, dE = _beam(10)
        before = backend.copy(dE)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **KICK)
        total = self.deferred.sum_1d_array(dE)  # not deferrable: flushes
        self.assertEqual(self.deferred.kernel_call_queue.n_bytes, 0)
        self.assertFalse(_equal(dE, before))
        self.assertAlmostEqual(
            total, float(np.sum(copy_to_cpu(dE))), delta=1e-3
        )

    def test_kernel_on_a_slice_of_the_beam(self) -> None:  # Review Focus 2
        dt, dE = _beam(100)
        dt_eager, dE_eager = backend.copy(dt), backend.copy(dE)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **KICK)
        self.deferred.drift_simple(dt=dt[:50], dE=dE[:50], **DRIFT)
        self.deferred.flush()
        self.eager.kick_single_harmonic(dt=dt_eager, dE=dE_eager, **KICK)
        self.eager.drift_simple(dt=dt_eager[:50], dE=dE_eager[:50], **DRIFT)
        _close(dt, dt_eager, rtol=1e-12)
        _close(dE, dE_eager, rtol=1e-12)

    def test_voltage_mutated_after_queue(self) -> None:  # Review Focus 1
        dt, dE = _beam(1000)
        dt_eager, dE_eager = backend.copy(dt), backend.copy(dE)
        voltage = backend.array(np.sin(np.linspace(0, 3, 64)) * 1e3)
        bins = backend.array(np.linspace(-1e-8, 1e-8, 64))
        self.eager.kick_interpolated(
            dt=dt_eager,
            dE=dE_eager,
            voltage=backend.copy(voltage),
            bin_centers=bins,
            charge=1.0,
            acceleration_kick=0.0,
        )
        self.deferred.kick_interpolated(
            dt=dt,
            dE=dE,
            voltage=voltage,
            bin_centers=bins,
            charge=1.0,
            acceleration_kick=0.0,
        )
        voltage[:] = 0.0  # caller reuses its buffer before the flush
        self.deferred.flush()
        _close(dE, dE_eager, rtol=1e-12)

    def test_failed_flush_clears_queue(self) -> None:  # Review Focus 3
        dt, dE = _beam(10)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **KICK)
        queue = self.deferred.kernel_call_queue
        original = self.deferred_class._execute_batch

        def failing(*args):
            raise RuntimeError("boom")

        self.deferred_class._execute_batch = staticmethod(failing)
        try:
            with self.assertRaises(RuntimeError):
                self.deferred.flush()
        finally:
            self.deferred_class._execute_batch = staticmethod(original)
        self.assertEqual(queue.n_bytes, 0)
        self.assertEqual(queue.keep_alive, [])

    def test_scalars_captured_at_enqueue(self) -> None:
        dt, dE = _beam(10)
        dt_e, dE_e = backend.copy(dt), backend.copy(dE)
        kick = dict(KICK)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **kick)
        kick["voltage"] = 0.0
        self.deferred.flush()
        self.eager.kick_single_harmonic(dt=dt_e, dE=dE_e, **KICK)
        _close(dE, dE_e, rtol=1e-12)

    def test_positional_arguments(self) -> None:
        dt, dE = _beam(10)
        dt_e, dE_e = backend.copy(dt), backend.copy(dE)
        self.deferred.drift_simple(dt, dE, 1e-6, 0.01, 0.9, 2e9)
        self.deferred.flush()
        self.eager.drift_simple(dt_e, dE_e, 1e-6, 0.01, 0.9, 2e9)
        _close(dt, dt_e, rtol=1e-12)

    def test_queues_are_per_thread(self) -> None:
        results = {}

        def worker(key):
            dt, dE = _beam(10000)
            dt_e, dE_e = backend.copy(dt), backend.copy(dE)
            for _ in range(20):
                _turn(self.deferred, dt, dE)
                _turn(self.eager, dt_e, dE_e)
            self.deferred.flush()
            results[key] = (
                np.allclose(copy_to_cpu(dt), copy_to_cpu(dt_e), rtol=1e-11),
                np.allclose(
                    copy_to_cpu(dE), copy_to_cpu(dE_e), rtol=1e-11, atol=1e-6
                ),
            )

        threads = [
            threading.Thread(target=worker, args=(k,)) for k in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(results, {k: (True, True) for k in range(4)})

    def test_every_specials_method_is_deferred_or_wrapped(self) -> None:
        for name, value in vars(Specials).items():
            if name.startswith("_") or not isinstance(value, staticmethod):
                continue
            self.assertIn(name, vars(self.deferred_class), name)

    def test_switching_specials_flushes(self) -> None:
        dt, dE = _beam(10)
        before = backend.copy(dE)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **KICK)
        backend.set_specials(self.eager_mode)
        self.assertFalse(_equal(dE, before))


@pytest.mark.cupy
@pytest.mark.backend_mutation
class TestCudaDeferredSpecials(TestCppDeferredSpecials):
    """Every cpp_deferred test, rerun on the GPU."""

    mode = "cuda_deferred"
    eager_mode = "cuda"

    def setUp(self) -> None:
        if not cupy_available:
            self.skipTest("CuPy is not available")
        from blond.core.backends.backend import Cupy64Bit

        backend.change_backend(Cupy64Bit)
        backend.set_specials(self.eager_mode)
        self.eager = backend.specials
        backend.set_specials(self.mode)
        self.deferred = backend.specials
        self.deferred_class = type(self.deferred)

    def tearDown(self) -> None:
        self.deferred.flush()
        backend.change_backend(Numpy64Bit)
        backend.set_specials("python")

    def test_batch_larger_than_capacity_is_split(self) -> None:
        from blond.core.backends.cuda.callables import (
            KERNEL_CALL_BATCH_CAPACITY_BYTES,
        )

        # 40 harmonics are 2 multi-harmonic records (~1.6 KB) per turn,
        # so three turns queue more than one launch holds.
        dt, dE = _beam(1000)
        dt_e, dE_e = backend.copy(dt), backend.copy(dE)
        for _ in range(3):
            _turn(self.deferred, dt, dE, n_rf=40)
            _turn(self.eager, dt_e, dE_e, n_rf=40)
        self.assertGreater(
            self.deferred.kernel_call_queue.n_bytes,
            KERNEL_CALL_BATCH_CAPACITY_BYTES,
        )
        self.deferred.flush()
        _close(dt, dt_e, rtol=1e-11, atol=0)
        _close(dE, dE_e, rtol=1e-11, atol=1e-6)

    def test_split_batch_ranges(self) -> None:
        from blond.core.backends.cuda.callables import _split_batch
        from blond.core.backends.deferred.kernel_call_records import (
            KERNELS_BY_SPECIALS_METHOD as KERNELS,
        )

        multi = KERNELS["kick_multi_harmonic"]  # 800 B records
        single = KERNELS["kick_single_harmonic"]  # 48 B records
        self.assertEqual(
            _split_batch([multi, single, multi, single], 1690),
            [(0, 1648, [multi, single, multi]), (1648, 1696, [single])],
        )

    def test_split_batch_exact_fit(self) -> None:
        from blond.core.backends.cuda.callables import _split_batch
        from blond.core.backends.deferred.kernel_call_records import (
            KERNELS_BY_SPECIALS_METHOD as KERNELS,
        )

        multi = KERNELS["kick_multi_harmonic"]
        drift = KERNELS["drift_simple"]
        self.assertEqual(
            _split_batch([multi, multi, drift], 1600),
            [(0, 1600, [multi, multi]), (1600, 1640, [drift])],
        )

    def test_queue_remembers_the_kernel_of_every_record(self) -> None:
        dt, dE = _beam(10)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **KICK)
        self.deferred.drift_simple(dt=dt, dE=dE, **DRIFT)
        self.assertEqual(
            [
                k.specials_method
                for k in self.deferred.kernel_call_queue.kernels
            ],
            ["kick_single_harmonic", "drift_simple"],
        )

    def test_zero_macroparticles(self) -> None:  # Review Focus 4
        dt, dE = backend.zeros(0), backend.zeros(0)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **KICK)
        self.deferred.flush()
        self.assertEqual(self.deferred.kernel_call_queue.n_bytes, 0)

    def test_store_flags(self) -> None:
        from blond.core.backends.cuda.callables import (
            STORE_DE,
            STORE_DT,
            _store_flags,
        )
        from blond.core.backends.deferred.kernel_call_records import (
            KERNELS_BY_SPECIALS_METHOD as KERNELS,
        )

        self.assertEqual(
            _store_flags([KERNELS["kick_interpolated"]]), STORE_DE
        )
        self.assertEqual(
            _store_flags([KERNELS["drift_simple"], KERNELS["drift_exact"]]),
            STORE_DT,
        )
        self.assertEqual(
            _store_flags(
                [KERNELS["kick_single_harmonic"], KERNELS["drift_simple"]]
            ),
            STORE_DT | STORE_DE,
        )
        self.assertEqual(_store_flags([]), 0)

    def test_specials_mode_is_tracked(self) -> None:
        self.assertEqual(backend.specials_mode, self.mode)
