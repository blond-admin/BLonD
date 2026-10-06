import threading

import numpy as np
import pytest

from blond.core.backends.backend import Numpy64Bit, Specials, backend
from blond.core.backends.deferred.kernel_call_records import (
    KERNEL_CALL_BATCH_CAPACITY_BYTES,
    MAX_RF_HARMONICS_PER_RECORD,
)
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
# Inside the ±1e-8 s of `_beam`, so some particles fall outside.
CUTS = dict(start=-0.8e-8, stop=0.9e-8)


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
            # Beyond 32 harmonics the eager CUDA kick sums them in several
            # launches, a record in one: dE differs by rounding, which the
            # relativistic drifts' sqrt(1 + x) - 1 turns into ~T * eps in
            # dt (T = 1e-6 s), however close to zero dt is.
            _close(dt, dt_eager, rtol=1e-11, atol=1e-20)
            _close(dE, dE_eager, rtol=1e-11, atol=1e-6)

    def test_matches_eager(self) -> None:
        self._assert_matches_eager()

    def test_more_than_32_harmonics(self) -> None:
        self._assert_matches_eager(n_rf=40)

    def test_harmonic_counts(self) -> None:
        # Records hold exactly n_rf harmonics, split at
        # MAX_RF_HARMONICS_PER_RECORD; the eager CUDA kernel splits at 32.
        max_rf = MAX_RF_HARMONICS_PER_RECORD
        for n_rf in (0, 1, 2, 4, 5, 31, 32, 33, 64, max_rf, max_rf + 1):
            with self.subTest(n_rf=n_rf):
                self._assert_matches_eager(n_rf=n_rf)

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

    def test_drift_records_with_distinct_parameters(self) -> None:
        # The CUDA executor prepares each record's factors once per
        # launch; distinct beta/energy per record catch factors applied
        # to the wrong record. Enough records to span several launches.
        calls = []
        for k in range(30):
            beta = 0.5 + 0.015 * k
            energy = 1e9 * (1.0 + 0.3 * k)
            calls += [
                ("kick_single_harmonic", KICK),
                ("drift_simple", {**DRIFT, "beta": beta, "energy": energy}),
                (
                    "drift_like_line_segment",
                    {**DRIFT, "beta": beta + 0.01, "energy": 2 * energy},
                ),
                (
                    "drift_exact",
                    dict(
                        T=1e-6,
                        alpha_0=0.01,
                        higher_alpha=np.array([1e-3, 2e-3]),
                        beta=beta + 0.02,
                        energy=3 * energy,
                    ),
                ),
            ]
        self._assert_batch_matches_eager(calls)

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

    def test_kernel_call_queue_is_the_calling_threads(self) -> None:
        dt, dE = _beam(10)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **KICK)
        main_queue = self.deferred.kernel_call_queue
        seen = {}

        def worker():
            seen["queue"] = self.deferred.kernel_call_queue
            seen["n_bytes"] = seen["queue"].n_bytes

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        self.assertIs(self.deferred.kernel_call_queue, main_queue)
        self.assertIs(self.deferred_class.kernel_call_queue, main_queue)
        self.assertIsNot(seen["queue"], main_queue)
        self.assertEqual(seen["n_bytes"], 0)
        self.assertGreater(main_queue.n_bytes, 0)

    def test_every_specials_method_is_deferred_or_wrapped(self) -> None:
        for name, value in vars(Specials).items():
            if name.startswith("_") or not isinstance(value, staticmethod):
                continue
            self.assertIn(name, vars(self.deferred_class), name)

    # ----------------------------------------------------------- histogram

    def _histogram_matches_eager(self, dt, dE, n_bins, calls=()) -> None:
        """Run ``calls`` then ``histogram`` of dt eagerly and deferred."""
        dt_eager, dE_eager = backend.copy(dt), backend.copy(dE)
        results = []
        for specials, dt_, dE_ in (
            (self.eager, dt_eager, dE_eager),
            (self.deferred, dt, dE),
        ):
            for method, kwargs in calls:
                getattr(specials, method)(dt=dt_, dE=dE_, **kwargs)
            hist_y = backend.ones(n_bins, dtype=backend.float)
            specials.histogram(array_read=dt_, array_write=hist_y, **CUTS)
            results.append(hist_y)
        # Counts are integers: no tolerance.
        np.testing.assert_array_equal(
            copy_to_cpu(results[1]), copy_to_cpu(results[0])
        )
        _close(dt, dt_eager, rtol=1e-11, atol=0)
        _close(dE, dE_eager, rtol=1e-11, atol=1e-6)

    def test_histogram_matches_eager(self) -> None:
        for n in (1, 7, 1000, 100003):
            for n_bins in (1, 64, 1000):
                with self.subTest(n=n, n_bins=n_bins):
                    dt, dE = _beam(n)
                    self._histogram_matches_eager(dt, dE, n_bins)

    def test_histogram_after_a_turn_bins_the_final_dt(self) -> None:
        dt, dE = _beam(100003)
        calls = [
            ("kick_single_harmonic", KICK),
            ("drift_simple", DRIFT),
            ("kick_single_harmonic", KICK),
            ("drift_simple", DRIFT),
        ]
        self._histogram_matches_eager(dt, dE, 1000, calls)

    def test_histogram_edges(self) -> None:
        # Exactly cut_left and cut_right count in the first and last bin;
        # values that scale to n_bins but lie below cut_right count in the
        # last bin; values just outside either cut are dropped.
        left, right = CUTS["start"], CUTS["stop"]
        n_bins = 1000
        width = (right - left) / n_bins
        values = np.array(
            [
                left,
                right,
                np.nextafter(right, left),
                np.nextafter(left, -1.0),
                left - 0.5 * width,
                np.nextafter(right, 1.0),
                -1e30,
                1e30,
                0.0,
            ]
        )
        dt = backend.array(np.tile(values, 1001), dtype=backend.float)
        dE = backend.zeros(dt.size, dtype=backend.float)
        self._histogram_matches_eager(
            dt, dE, n_bins, [("kick_single_harmonic", KICK)]
        )

    def test_histogram_flushes_the_queue(self) -> None:
        # The histogram is the last record of a batch: queuing it runs the
        # batch, so whoever reads hist_y next sees the finished histogram.
        dt, dE = _beam(1000)
        before = backend.copy(dE)
        hist_y = backend.ones(64, dtype=backend.float)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **KICK)
        self.deferred.histogram(array_read=dt, array_write=hist_y, **CUTS)
        self.assertEqual(self.deferred.kernel_call_queue.n_bytes, 0)
        self.assertFalse(_equal(dE, before))
        self.assertEqual(
            float(np.sum(copy_to_cpu(hist_y))),
            float(
                np.sum(
                    (copy_to_cpu(dt) >= CUTS["start"])
                    & (copy_to_cpu(dt) <= CUTS["stop"])
                )
            ),
        )

    def test_histogram_runs_in_the_batch_it_ends(self) -> None:
        from blond.core.backends.deferred.kernel_call_records import (
            DriftSimpleArgs,
            HistogramArgs,
            KickSingleHarmonicArgs,
        )

        dt, dE = _beam(1000)
        hist_y = backend.ones(64, dtype=backend.float)
        batches = []
        original = self.deferred_class._execute_batch

        def recording(buffer, n_bytes, args_types, *rest):
            batches.append(list(args_types))
            original(buffer, n_bytes, args_types, *rest)

        self.deferred_class._execute_batch = staticmethod(recording)
        try:
            self.deferred.kick_single_harmonic(dt=dt, dE=dE, **KICK)
            self.deferred.drift_simple(dt=dt, dE=dE, **DRIFT)
            self.deferred.histogram(array_read=dt, array_write=hist_y, **CUTS)
        finally:
            self.deferred_class._execute_batch = staticmethod(original)
        self.assertEqual(
            batches,
            [[KickSingleHarmonicArgs, DriftSimpleArgs, HistogramArgs]],
        )

    def test_histogram_of_another_array_runs_after_the_batch(self) -> None:
        # Only the queued dt is binned in the batch; any other array (dE,
        # a copy of dt) is binned eagerly, after the batch has run.
        from blond.core.backends.deferred.kernel_call_records import (
            KickSingleHarmonicArgs,
        )

        dt, dE = _beam(1000)
        dE_eager = backend.copy(dE)
        self.eager.kick_single_harmonic(dt=dt, dE=dE_eager, **KICK)
        energy_cuts = dict(start=-1e6, stop=1.5e6)
        for array, eager_array in ((dE, dE_eager), (backend.copy(dt), dt)):
            with self.subTest(array_is_dE=array is dE):
                batches = []
                original = self.deferred_class._execute_batch

                def recording(buffer, n_bytes, args_types, *rest):
                    batches.append(list(args_types))
                    original(buffer, n_bytes, args_types, *rest)

                self.deferred_class._execute_batch = staticmethod(recording)
                try:
                    if array is dE:
                        self.deferred.kick_single_harmonic(
                            dt=dt, dE=dE, **KICK
                        )
                    else:  # dt itself is unchanged by a kick
                        self.deferred.kick_single_harmonic(
                            dt=dt, dE=backend.copy(dE), **KICK
                        )
                    hist_y = backend.ones(64, dtype=backend.float)
                    self.deferred.histogram(
                        array_read=array, array_write=hist_y, **energy_cuts
                    )
                finally:
                    self.deferred_class._execute_batch = staticmethod(original)
                self.assertEqual(batches, [[KickSingleHarmonicArgs]])
                expected = backend.zeros(64, dtype=backend.float)
                self.eager.histogram(
                    array_read=eager_array, array_write=expected, **energy_cuts
                )
                np.testing.assert_array_equal(
                    copy_to_cpu(hist_y), copy_to_cpu(expected)
                )

    def test_histogram_of_another_beam(self) -> None:
        # The histogram's beam is not the queued one: the queued batch
        # runs first, on its own beam.
        dt, dE = _beam(1000)
        dt_other, dE_other = _beam(500)
        dE_eager = backend.copy(dE)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **KICK)
        self._histogram_matches_eager(dt_other, dE_other, 64)
        self.eager.kick_single_harmonic(dt=dt, dE=dE_eager, **KICK)
        _close(dE, dE_eager, rtol=1e-12)

    def test_histogram_with_many_bins(self) -> None:
        # More bins than the CUDA executor holds in shared memory.
        dt, dE = _beam(100003)
        self._histogram_matches_eager(
            dt, dE, 20_000, [("drift_simple", DRIFT)]
        )

    def test_histogram_of_no_particles(self) -> None:
        dt, dE = backend.zeros(0), backend.zeros(0)
        hist_y = backend.ones(16, dtype=backend.float)
        self.deferred.histogram(array_read=dt, array_write=hist_y, **CUTS)
        np.testing.assert_array_equal(copy_to_cpu(hist_y), np.zeros(16))

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

    def test_chunk_sizes(self) -> None:
        self.skipTest("the chunk size exists on the cpp executor only")

    def test_batch_larger_than_capacity_is_split(self) -> None:
        # 100 harmonics are a 2432 B record per turn, so three turns
        # queue more than one launch holds.
        dt, dE = _beam(1000)
        dt_e, dE_e = backend.copy(dt), backend.copy(dE)
        for _ in range(3):
            _turn(self.deferred, dt, dE, n_rf=100)
            _turn(self.eager, dt_e, dE_e, n_rf=100)
        self.assertGreater(
            self.deferred.kernel_call_queue.n_bytes,
            KERNEL_CALL_BATCH_CAPACITY_BYTES,
        )
        # Binned in the last launch, the counting record's position
        # counted from that launch's first record.
        hist_y = backend.ones(64, dtype=backend.float)
        expected = backend.zeros(64, dtype=backend.float)
        self.deferred.histogram(array_read=dt, array_write=hist_y, **CUTS)
        self.eager.histogram(array_read=dt_e, array_write=expected, **CUTS)
        np.testing.assert_array_equal(
            copy_to_cpu(hist_y), copy_to_cpu(expected)
        )
        _close(dt, dt_e, rtol=1e-11, atol=0)
        _close(dE, dE_e, rtol=1e-11, atol=1e-6)

    def test_multi_harmonic_turns_fit_one_launch(self) -> None:
        # A two-harmonic kick is an 80 B record, so ten turns of kick and
        # drift fit one launch instead of three with 800 B records.
        from blond.core.backends.cuda.callables import _split_batch

        dt, dE = _beam(10)
        for _ in range(10):
            self.deferred.kick_multi_harmonic(
                dt=dt,
                dE=dE,
                voltage=np.array([1e3, 2e3]),
                omega_rf=np.array([1e7, 3e7]),
                phi_rf=np.array([0.0, 1.0]),
                charge=1.0,
                n_rf=2,
                acceleration_kick=5.0,
            )
            self.deferred.drift_simple(dt=dt, dE=dE, **DRIFT)
        queue = self.deferred.kernel_call_queue
        self.assertEqual(queue.n_bytes, 10 * (80 + 40))
        self.assertEqual(
            len(
                _split_batch(
                    queue.args_types,
                    queue.record_sizes,
                    KERNEL_CALL_BATCH_CAPACITY_BYTES,
                )
            ),
            1,
        )

    def test_batch_parameter_holds_the_launch_bytes(self) -> None:
        # A view of the queue's buffer where a whole batch struct fits
        # after `start`, else a zero-padded copy.
        from blond.core.backends.cuda.callables import (
            _KERNEL_CALL_BATCH_DTYPE,
            _batch_parameter,
        )

        capacity = _KERNEL_CALL_BATCH_DTYPE.itemsize
        buffer = (np.arange(capacity + 100) % 251 + 1).astype(np.uint8)
        for start, end in ((0, 48), (100, 100 + capacity), (200, 300)):
            with self.subTest(start=start, end=end):
                parameter = _batch_parameter(buffer, start, end)
                # A 0-d array, not an `np.void` scalar: CuPy 13 rejects
                # `np.void` launch arguments and CuPy 14.0 asserts their
                # size fits 32 bytes.
                self.assertIsInstance(parameter, np.ndarray)
                self.assertEqual(parameter.shape, ())
                self.assertEqual(parameter.dtype, _KERNEL_CALL_BATCH_DTYPE)
                raw = np.frombuffer(parameter.tobytes(), dtype=np.uint8)
                np.testing.assert_array_equal(
                    raw[: end - start], buffer[start:end]
                )
                if start + capacity > buffer.size:
                    self.assertFalse(raw[end - start :].any())

    def test_split_batch_ranges(self) -> None:
        from blond.core.backends.cuda.callables import _split_batch
        from blond.core.backends.deferred.kernel_call_records import (
            KickMultiHarmonicArgs as Multi,
        )
        from blond.core.backends.deferred.kernel_call_records import (
            KickSingleHarmonicArgs as Single,  # 48 B records
        )

        # Records of one kernel differ in size: 32 harmonics are 800 B,
        # two are 80 B.
        self.assertEqual(
            _split_batch(
                [Multi, Single, Multi, Single, Multi],
                [800, 48, 800, 48, 80],
                1690,
            ),
            [
                (0, 1648, [Multi, Single, Multi]),
                (1648, 1776, [Single, Multi]),
            ],
        )

    def test_split_batch_exact_fit(self) -> None:
        from blond.core.backends.cuda.callables import _split_batch
        from blond.core.backends.deferred.kernel_call_records import (
            DriftSimpleArgs as Drift,  # 40 B records
        )
        from blond.core.backends.deferred.kernel_call_records import (
            KickMultiHarmonicArgs as Multi,
        )

        self.assertEqual(
            _split_batch([Multi, Multi, Drift], [800, 800, 40], 1600),
            [(0, 1600, [Multi, Multi]), (1600, 1640, [Drift])],
        )

    def test_queue_remembers_the_kernel_of_every_record(self) -> None:
        from blond.core.backends.deferred.kernel_call_records import (
            DriftSimpleArgs,
            KickSingleHarmonicArgs,
        )

        dt, dE = _beam(10)
        self.deferred.kick_single_harmonic(dt=dt, dE=dE, **KICK)
        self.deferred.drift_simple(dt=dt, dE=dE, **DRIFT)
        self.assertEqual(
            self.deferred.kernel_call_queue.args_types,
            [KickSingleHarmonicArgs, DriftSimpleArgs],
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
            DriftExactArgs,
            DriftSimpleArgs,
            KickInterpolatedArgs,
            KickSingleHarmonicArgs,
        )

        self.assertEqual(_store_flags([KickInterpolatedArgs]), STORE_DE)
        self.assertEqual(
            _store_flags([DriftSimpleArgs, DriftExactArgs]), STORE_DT
        )
        self.assertEqual(
            _store_flags([KickSingleHarmonicArgs, DriftSimpleArgs]),
            STORE_DT | STORE_DE,
        )
        self.assertEqual(_store_flags([]), 0)

    def test_specials_mode_is_tracked(self) -> None:
        self.assertEqual(backend.specials_mode, self.mode)

    def test_every_tail_tile_width_matches_eager(self) -> None:
        # The executor covers the beam past its last whole sweep of full
        # tiles with the halvings of the tile, as many as that tail needs
        # particles per thread. One beam size per tail length (1 to 8
        # particles per thread, the last one short), plus a beam of whole
        # sweeps and one below a particle per thread.
        from blond.core.backends.cuda.callables import (
            _deferred_block_size,
            grid_size,
        )

        n_threads = grid_size[0] * _deferred_block_size[0]
        particles_per_thread = 8
        sweep = particles_per_thread * n_threads
        sizes = [sweep + tail * n_threads - 5 for tail in range(1, 9)]
        sizes += [2 * sweep, n_threads - 3, 100007]
        for n in sizes:
            with self.subTest(n_macroparticles=n):
                dt, dE = _beam(n)
                dt_eager, dE_eager = backend.copy(dt), backend.copy(dE)
                for specials, dt_, dE_ in (
                    (self.eager, dt_eager, dE_eager),
                    (self.deferred, dt, dE),
                ):
                    for _ in range(2):
                        specials.kick_single_harmonic(dt=dt_, dE=dE_, **KICK)
                        specials.drift_simple(dt=dt_, dE=dE_, **DRIFT)
                self.deferred.flush()
                _close(dt, dt_eager, rtol=1e-11, atol=0)
                _close(dE, dE_eager, rtol=1e-11, atol=1e-6)

    def test_histogram_every_tail_tile_width(self) -> None:
        # Lanes past the last particle hold padding (dt = 0, inside the
        # cuts) and must not be counted.
        from blond.core.backends.cuda.callables import (
            _deferred_block_size,
            grid_size,
        )

        n_threads = grid_size[0] * _deferred_block_size[0]
        sweep = 8 * n_threads
        sizes = [sweep + tail * n_threads - 5 for tail in range(1, 9)]
        sizes += [2 * sweep, n_threads - 3, 1]
        for n in sizes:
            with self.subTest(n_macroparticles=n):
                dt, dE = _beam(n)
                self._histogram_matches_eager(
                    dt, dE, 64, [("drift_simple", DRIFT)]
                )
