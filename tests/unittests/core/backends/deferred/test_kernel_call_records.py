import dataclasses
import inspect
import struct
import subprocess
import sys
import textwrap

import numpy as np

from blond.core.backends.backend import Specials
from blond.core.backends.deferred import kernel_call_records as records
from blond.core.backends.deferred.kernel_call_queue import KernelCallQueue
from blond.core.backends.deferred.kernel_call_records import (
    DriftExactArgs,
    DriftLikeLineSegmentArgs,
    KernelCallArgs,
    KickMultiHarmonicArgs,
)
from blond.core.backends.python.callables import PythonSpecials
from blond.testing.backend_testing import BLonDTestCase

_N_PARTICLES = 50
_N_BINS = 16
# One call per deferrable kernel, with arguments that move the particles.
_EAGER_CALLS = {
    "kick_single_harmonic": dict(
        voltage=8e3,
        omega_rf=2e7,
        phi_rf=0.1,
        charge=1.0,
        acceleration_kick=12.0,
    ),
    "kick_multi_harmonic": dict(
        voltage=np.array([1e3, 2e3]),
        omega_rf=np.array([1e7, 3e7]),
        phi_rf=np.array([0.0, 1.0]),
        charge=1.0,
        n_rf=2,
        acceleration_kick=5.0,
    ),
    "drift_simple": dict(T=1e-6, eta_0=0.01, beta=0.9, energy=2e9),
    "drift_like_line_segment": dict(T=1e-6, eta_0=0.01, beta=0.9, energy=2e9),
    "drift_exact": dict(
        T=1e-6,
        alpha_0=0.01,
        higher_alpha=np.array([1e-3, 2e-3]),
        beta=0.9,
        energy=2e9,
    ),
    "kick_interpolated": dict(
        voltage=np.sin(np.linspace(0, 3, _N_BINS)) * 1e3,
        bin_centers=np.linspace(-1e-8, 1e-8, _N_BINS),
        charge=1.0,
        acceleration_kick=3.0,
    ),
    # Bins dt, which it reads as `array_read`: arguments from (dt, dE).
    "histogram": lambda dt, dE: dict(
        array_read=dt,
        array_write=np.zeros(_N_BINS),
        start=-0.8e-8,
        stop=0.9e-8,
    ),
}


def _multi_harmonic_arguments(n_rf: int) -> dict:
    return dict(
        n_rf=n_rf,
        voltage=np.linspace(1e3, 2e3, n_rf),
        omega_rf=np.linspace(1e7, 3e7, n_rf),
        phi_rf=np.linspace(0.0, 1.0, n_rf),
        charge=1.0,
        acceleration_kick=5.0,
    )


# Run under `python -O`, where asserts are stripped: every field must still
# be required and written, or a reused queue buffer would silently keep a
# stale value from an earlier record. It queues one call of every
# deferrable kernel through the deferred specials' own queuing methods,
# i.e. the fast packing path, and decodes the records with their dtypes.
_OPTIMIZED_PACKING_CHECK = textwrap.dedent(
    """
    import numpy as np

    from blond.core.backends.deferred import kernel_call_records as records
    from blond.core.backends.deferred.kernel_call_queue import (
        KernelCallQueue,
        make_deferred_specials,
    )
    from blond.core.backends.python.callables import PythonSpecials

    def fail(message):  # not `assert`: -O strips it
        raise SystemExit(message)

    queue = KernelCallQueue()
    queue.buffer[:] = 0xFF  # stale bytes of earlier records
    queue.append(
        records.DriftSimpleArgs(T=1.0, eta_0=2.0, beta=3.0, energy=4.0)
    )
    dtype = records.DriftSimpleArgs.record_dtype()
    args = queue.buffer[: queue.n_bytes].view(dtype)[0]["args"]
    if args.tolist() != (1.0, 2.0, 3.0, 4.0):
        fail(f"stale record fields: {args}")
    try:
        records.DriftSimpleArgs(T=1.0, eta_0=2.0, beta=3.0)
    except TypeError:
        print("missing field rejected")

    table = np.arange(8.0)

    class Eager(PythonSpecials):
        @staticmethod
        def _build_voltage_kick_table(
            voltage, bin_centers, charge, acceleration_kick
        ):
            return table

    deferred = make_deferred_specials(Eager, lambda *args: None)
    dt, dE = np.zeros(4), np.zeros(4)
    deferred.kernel_call_queue.buffer[:] = 0xFF
    deferred.kick_single_harmonic(dt, dE, 1.0, 2.0, 3.0, 4.0, 5.0)
    deferred.kick_multi_harmonic(
        dt, dE, np.array([6.0, 7.0]), np.array([8.0, 9.0]),
        np.array([10.0, 11.0]), 12.0, 2, 13.0,
    )
    deferred.drift_simple(dt, dE, 14.0, 15.0, 16.0, 17.0)
    deferred.drift_like_line_segment(dt, dE, 18.0, 19.0, 20.0, 21.0)
    deferred.drift_exact(dt, dE, 22.0, 23.0, np.array([24.0]), 25.0, 26.0)
    deferred.kick_interpolated(dt, dE, np.zeros(4), np.zeros(4), 1.0, 27.0)
    expected = {
        records.KickSingleHarmonicArgs: (1.0, 2.0, 3.0, 4.0, 5.0),
        records.KickMultiHarmonicArgs: (12.0, 13.0, 2, 0),
        records.DriftSimpleArgs: (14.0, 15.0, 16.0, 17.0),
        records.DriftLikeLineSegmentArgs: (18.0, 19.0, 20.0, 21.0),
        records.DriftExactArgs: (
            22.0, 23.0, 25.0, 26.0, 1, 0, (24.0,) + (0.0,) * 7
        ),
        records.KickInterpolatedArgs: (table.ctypes.data, 8, 27.0),
    }
    buffer = deferred.kernel_call_queue.buffer
    offset = 0
    for args_type, size in zip(
        deferred.kernel_call_queue.args_types,
        deferred.kernel_call_queue.record_sizes,
    ):
        fixed = args_type.record_dtype()
        record = buffer[offset : offset + fixed.itemsize].view(fixed)[0]
        if record["kernel_id"] != args_type.kernel_id():
            fail(f"{args_type.__name__}: kernel id {record['kernel_id']}")
        if record["record_size_bytes"] != size:
            fail(f"{args_type.__name__}: size {record['record_size_bytes']}")
        values = tuple(
            record["args"][name].tolist()
            if record["args"][name].ndim == 0
            else tuple(record["args"][name].tolist())
            for name in fixed["args"].names
        )
        # Padding members are unnamed; read them from the raw bytes.
        raw = buffer[offset + 8 : offset + fixed.itemsize]
        if args_type is records.KickMultiHarmonicArgs:
            values = values + (int(raw[20:24].view(np.int32)[0]),)
            columns = buffer[offset + fixed.itemsize : offset + size]
            if columns.view(np.float64).tolist() != [
                6.0, 7.0, 8.0, 9.0, 10.0, 11.0
            ]:
                fail(f"harmonics: {columns.view(np.float64)}")
        if args_type is records.DriftExactArgs:
            values = values[:5] + (int(raw[36:40].view(np.int32)[0]),)
            values = values + (tuple(record["args"]["higher_alpha"]),)
        if values != expected[args_type]:
            fail(f"{args_type.__name__}: {values} != {expected[args_type]}")
        offset += size
    if offset != deferred.kernel_call_queue.n_bytes or offset == 0:
        fail(f"decoded {offset} of {deferred.kernel_call_queue.n_bytes} B")
    try:
        deferred.drift_simple(dt, dE, 14.0, 15.0, 16.0)
    except TypeError:
        print("missing argument rejected")
    try:
        deferred.drift_exact(dt, dE, 1.0, 1.0, np.ones(9), 1.0, 1.0)
    except Exception:
        fail("more than MAX_HIGHER_ALPHA must run eagerly, not raise")
    """
)


def _reference_record(args: KernelCallArgs) -> bytes:
    """Record bytes packed through the numpy dtypes: the test oracle."""
    record = np.zeros(args.record_size_bytes(), dtype=np.uint8)
    args.pack_into(record, [])
    return record.tobytes()


def _fast_record(args: KernelCallArgs, keep_alive: list) -> bytes:
    """Record bytes of the fast packer, written over stale bytes."""
    buffer = np.full(
        records.KERNEL_CALL_BATCH_CAPACITY_BYTES + 64, 0xFF, dtype=np.uint8
    )
    values = [getattr(args, f.name) for f in dataclasses.fields(args)]
    size = type(args).record_packer()(
        memoryview(buffer), 16, keep_alive, *values
    )
    return buffer[16 : 16 + size].tobytes()


def _random_records(rng: np.random.Generator):
    """Many records of every kernel, with awkward but valid values."""
    specials = np.array([0.0, -0.0, np.inf, -np.inf, 1e-308, -1e308])

    def real():
        if rng.random() < 0.2:
            return float(rng.choice(specials))
        kind = rng.integers(4)
        value = rng.normal() * 10.0 ** rng.integers(-12, 12)
        # Python floats and ints, and numpy scalars of several types.
        return (
            float(value),
            np.float64(value),
            np.float32(value),
            int(value) if abs(value) < 2**53 else float(value),
        )[kind]

    for _ in range(20):
        yield records.KickSingleHarmonicArgs(*(real() for _ in range(5)))
        yield records.DriftSimpleArgs(*(real() for _ in range(4)))
        yield records.DriftLikeLineSegmentArgs(*(real() for _ in range(4)))
        yield records.KickInterpolatedArgs(
            voltage_kick_table=rng.normal(size=2 * rng.integers(2, 300)),
            acceleration_kick=real(),
        )
        yield records.HistogramArgs(
            array_write=np.zeros(rng.integers(1, 300)),
            start=real(),
            stop=real(),
        )
    for n_rf in range(records.MAX_RF_HARMONICS_PER_RECORD + 1):
        columns = [rng.normal(size=n_rf) * 1e6 for _ in range(3)]
        if n_rf % 3 == 1:
            columns[1] = columns[1].tolist()  # a list column
        if n_rf % 3 == 2:
            columns[2] = columns[2].astype(np.float32)  # cast on packing
        yield records.KickMultiHarmonicArgs(
            charge=real(), acceleration_kick=real(), harmonics=tuple(columns)
        )
    for n_alpha in range(records.MAX_HIGHER_ALPHA + 1):
        higher_alpha = rng.normal(size=n_alpha) * 1e-3
        yield records.DriftExactArgs(
            T=real(),
            alpha_0=real(),
            beta=real(),
            energy=real(),
            n_alpha=n_alpha,
            higher_alpha=(
                higher_alpha.tolist() if n_alpha % 2 else higher_alpha
            ),
        )


class TestRecordPacker(BLonDTestCase):
    """The fast packer against the numpy-dtype reference packing."""

    def test_bytes_match_the_reference_for_every_kernel(self) -> None:
        rng = np.random.default_rng(11)
        seen = set()
        for args in _random_records(rng):
            seen.add(type(args))
            keep_alive = []
            with self.subTest(args=args):
                self.assertEqual(
                    _fast_record(args, keep_alive), _reference_record(args)
                )
        self.assertEqual(seen, set(records.KERNEL_CALL_ARGS))

    def test_padding_is_zeroed_over_stale_bytes(self) -> None:
        # The kick's n_rf is followed by 4 padding bytes, drift_exact's
        # n_alpha too: they must not keep bytes of an earlier record.
        (kick,) = KickMultiHarmonicArgs.from_specials_call(
            _multi_harmonic_arguments(2), eager_specials=None
        )
        record = _fast_record(kick, [])
        self.assertEqual(record[8 + 20 : 8 + 24], bytes(4))
        (drift,) = DriftExactArgs.from_specials_call(
            dict(
                T=1.0,
                alpha_0=2.0,
                beta=3.0,
                energy=4.0,
                higher_alpha=np.array([5.0]),
            ),
            eager_specials=None,
        )
        record = _fast_record(drift, [])
        self.assertEqual(record[8 + 36 : 8 + 40], bytes(4))

    def test_input_arrays_are_kept_alive(self) -> None:
        table = np.arange(6.0)
        keep_alive = []
        _fast_record(
            records.KickInterpolatedArgs(
                voltage_kick_table=table, acceleration_kick=1.0
            ),
            keep_alive,
        )
        self.assertEqual(len(keep_alive), 1)
        self.assertIs(keep_alive[0], table)

    def test_too_many_inline_values_fail_loudly(self) -> None:
        # Not an assert: it must fail under `python -O` as well.
        args = records.DriftExactArgs(
            T=1.0,
            alpha_0=1.0,
            beta=1.0,
            energy=1.0,
            n_alpha=9,
            higher_alpha=np.ones(records.MAX_HIGHER_ALPHA + 1),
        )
        with self.assertRaises(struct.error):
            _fast_record(args, [])

    def test_record_beyond_one_launch_fails_loudly(self) -> None:
        n_rf = records.MAX_RF_HARMONICS_PER_RECORD + 1
        args = KickMultiHarmonicArgs(
            charge=1.0,
            acceleration_kick=0.0,
            harmonics=(np.ones(n_rf), np.ones(n_rf), np.ones(n_rf)),
        )
        with self.assertRaises(ValueError):
            _fast_record(args, [])

    def test_missing_value_is_rejected(self) -> None:
        packer = records.DriftSimpleArgs.record_packer()
        buffer = memoryview(np.zeros(64, dtype=np.uint8))
        with self.assertRaises(TypeError):
            packer(buffer, 0, [], 1.0, 2.0, 3.0)


class TestKernelCallRecords(BLonDTestCase):
    def test_packing_is_complete_under_python_optimize(self) -> None:
        result = subprocess.run(
            [sys.executable, "-O", "-c", _OPTIMIZED_PACKING_CHECK],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("missing field rejected", result.stdout)
        self.assertIn("missing argument rejected", result.stdout)

    def test_header_is_current(self) -> None:
        with open(records.HEADER_PATH) as file:
            on_disk = file.read()
        self.assertEqual(
            on_disk,
            records.generate_header(),
            "kernel_call_records.h is stale; run `python -m "
            "blond.core.backends.deferred.kernel_call_records`",
        )

    def test_specials_method_from_class_name(self) -> None:
        self.assertEqual(
            DriftLikeLineSegmentArgs.specials_method(),
            "drift_like_line_segment",
        )
        for args_type in records.KERNEL_CALL_ARGS:
            self.assertTrue(
                hasattr(Specials, args_type.specials_method()),
                args_type.__name__,
            )

    def test_unknown_kernel_is_rejected_at_definition(self) -> None:
        with self.assertRaises(TypeError):

            class NoSuchKernelArgs(KernelCallArgs):  # noqa: F841
                pass

    def test_default_fields_are_specials_arguments(self) -> None:
        # Kernels without their own field_values_from_specials_call are
        # filled from the kwargs by field name.
        default = KernelCallArgs.field_values_from_specials_call.__func__
        for args_type in records.KERNEL_CALL_ARGS:
            hook = args_type.field_values_from_specials_call.__func__
            if hook is not default:
                continue
            parameters = inspect.signature(
                getattr(Specials, args_type.specials_method())
            ).parameters
            for field in dataclasses.fields(args_type):
                self.assertIn(field.name, parameters, args_type.__name__)

    def test_every_field_has_a_record_field_marker(self) -> None:
        for args_type in records.KERNEL_CALL_ARGS:
            names = [name for name, _ in args_type.record_fields()]
            self.assertEqual(
                names, [f.name for f in dataclasses.fields(args_type)]
            )

    def test_dtypes_are_8_byte_padded(self) -> None:
        for args_type in records.KERNEL_CALL_ARGS:
            record = args_type.record_dtype()
            self.assertEqual(record.itemsize % 8, 0)
            self.assertEqual(
                record.itemsize,
                records.HEADER_DTYPE.itemsize
                + args_type.args_dtype().itemsize,
            )

    def test_kick_multi_harmonic_layout(self) -> None:
        # Fixed part: charge, acceleration_kick, n_rf and 4 bytes padding;
        # the harmonics trail it as voltage, omega_rf and phi_rf columns.
        dtype = KickMultiHarmonicArgs.args_dtype()
        self.assertEqual(dtype.itemsize, 24)
        self.assertEqual(dtype.fields["charge"][1], 0)
        self.assertEqual(dtype.fields["acceleration_kick"][1], 8)
        self.assertEqual(dtype.fields["n_rf"][1], 16)
        _, columns = KickMultiHarmonicArgs.trailing_field()
        self.assertEqual(columns.columns, ("voltage", "omega_rf", "phi_rf"))
        self.assertEqual(columns.row_nbytes, 24)

    def test_kick_multi_harmonic_record_holds_only_its_harmonics(
        self,
    ) -> None:
        for n_rf in (0, 1, 2, 4, 5):
            with self.subTest(n_rf=n_rf):
                queue = KernelCallQueue()
                (record,) = KickMultiHarmonicArgs.from_specials_call(
                    _multi_harmonic_arguments(n_rf), eager_specials=None
                )
                queue.append(record)
                size = records.HEADER_DTYPE.itemsize + 24 + 24 * n_rf
                self.assertEqual(queue.n_bytes, size)
                self.assertEqual(queue.record_sizes, [size])
                header = queue.buffer[:8].view(records.HEADER_DTYPE)[0]
                self.assertEqual(header["record_size_bytes"], size)

    def test_kick_multi_harmonic_record_contents(self) -> None:
        arguments = _multi_harmonic_arguments(3)
        queue = KernelCallQueue()
        queue.buffer[:] = 0xFF  # stale bytes of earlier records
        (record,) = KickMultiHarmonicArgs.from_specials_call(
            arguments, eager_specials=None
        )
        queue.append(record)
        fixed_end = 8 + KickMultiHarmonicArgs.args_dtype().itemsize
        args = queue.buffer[8:fixed_end].view(
            KickMultiHarmonicArgs.args_dtype()
        )[0]
        self.assertEqual(args["n_rf"], 3)
        self.assertEqual(args["charge"], arguments["charge"])
        self.assertEqual(
            args["acceleration_kick"], arguments["acceleration_kick"]
        )
        columns = queue.buffer[fixed_end : queue.n_bytes].view(np.float64)
        for position, name in enumerate(("voltage", "omega_rf", "phi_rf")):
            np.testing.assert_array_equal(
                columns[3 * position : 3 * (position + 1)], arguments[name]
            )

    def test_max_harmonics_per_record_fill_one_cuda_launch(self) -> None:
        # The per-record limit is what one CUDA launch can hold, not a
        # tuning knob.
        max_rf = records.MAX_RF_HARMONICS_PER_RECORD
        fixed = KickMultiHarmonicArgs.record_dtype().itemsize
        self.assertLessEqual(
            fixed + 24 * max_rf, records.KERNEL_CALL_BATCH_CAPACITY_BYTES
        )
        self.assertGreater(
            fixed + 24 * (max_rf + 1),
            records.KERNEL_CALL_BATCH_CAPACITY_BYTES,
        )

    def test_harmonics_beyond_one_record_are_split(self) -> None:
        max_rf = records.MAX_RF_HARMONICS_PER_RECORD
        for n_rf, counts in (
            (0, [0]),
            (max_rf, [max_rf]),
            (max_rf + 1, [max_rf, 1]),
            (2 * max_rf + 5, [max_rf, max_rf, 5]),
        ):
            with self.subTest(n_rf=n_rf):
                kernel_calls = KickMultiHarmonicArgs.from_specials_call(
                    _multi_harmonic_arguments(n_rf), eager_specials=None
                )
                queue = KernelCallQueue()
                for kernel_call in kernel_calls:
                    queue.append(kernel_call)
                self.assertEqual(
                    queue.record_sizes,
                    [8 + 24 + 24 * count for count in counts],
                )
                # acceleration_kick goes into the last record only.
                self.assertEqual(
                    [call.acceleration_kick for call in kernel_calls],
                    [0.0] * (len(counts) - 1) + [5.0],
                )

    def test_kernel_ids_are_positions(self) -> None:
        for position, args_type in enumerate(records.KERNEL_CALL_ARGS):
            self.assertEqual(args_type.kernel_id(), position)

    def test_misspelt_field_fails_at_construction(self) -> None:
        with self.assertRaises(TypeError):
            records.DriftSimpleArgs(T=1.0, eta0=1.0, beta=1.0, energy=1.0)

    def test_digest_is_sha256(self) -> None:
        self.assertEqual(len(records.header_digest()), 64)

    def test_write_flags_match_the_eager_kernels(self) -> None:
        # The CUDA executor stores only the coordinates a batch writes,
        # so a wrong flag silently drops a kernel's result there.
        self.assertEqual(
            set(_EAGER_CALLS), set(records.ARGS_BY_SPECIALS_METHOD)
        )
        rng = np.random.default_rng(5)
        for args_type in records.KERNEL_CALL_ARGS:
            dt = rng.uniform(-1e-8, 1e-8, _N_PARTICLES)
            dE = rng.uniform(-1e6, 1e6, _N_PARTICLES)
            dt_before, dE_before = dt.copy(), dE.copy()
            call = _EAGER_CALLS[args_type.specials_method()]
            arguments = (
                call(dt, dE) if callable(call) else dict(dt=dt, dE=dE, **call)
            )
            getattr(PythonSpecials, args_type.specials_method())(**arguments)
            with self.subTest(kernel=args_type.__name__):
                self.assertEqual(
                    args_type.writes_dt,
                    not np.array_equal(dt, dt_before),
                )
                self.assertEqual(
                    args_type.writes_dE,
                    not np.array_equal(dE, dE_before),
                )

    def test_drift_exact_runs_eagerly_for_a_device_higher_alpha(self) -> None:
        # `CudaSpecials.drift_exact` still accepts a device `higher_alpha`
        # as a compatibility path (it copies it to host itself). The
        # deferred record inlines the coefficients into the record, which
        # would either crash or silently misread device memory, so this
        # case must fall back to running eagerly instead of asserting.
        class _FakeDeviceArray:
            def __init__(self, data: np.ndarray) -> None:
                self._data = data
                self.device = "cuda:0"

            def __len__(self) -> int:
                return len(self._data)

        arguments = dict(
            T=1e-6,
            alpha_0=0.01,
            higher_alpha=_FakeDeviceArray(np.array([1e-3, 2e-3])),
            beta=0.9,
            energy=2e9,
        )
        self.assertIsNone(
            DriftExactArgs.from_specials_call(arguments, eager_specials=None)
        )

    def test_drift_exact_still_queues_a_host_higher_alpha(self) -> None:
        arguments = dict(
            T=1e-6,
            alpha_0=0.01,
            higher_alpha=np.array([1e-3, 2e-3]),
            beta=0.9,
            energy=2e9,
        )
        records = DriftExactArgs.from_specials_call(
            arguments, eager_specials=None
        )
        self.assertIsNotNone(records)
        self.assertEqual(len(records), 1)

    def test_kick_multi_harmonic_rejects_mismatched_lengths(self) -> None:
        # Eager `CudaSpecials.kick_multi_harmonic` asserts
        # `len(voltage) == len(omega_rf) == len(phi_rf) == n_rf`; the
        # deferred record must reject the same mismatch instead of
        # silently slicing to the wrong length.
        arguments = dict(
            n_rf=2,
            voltage=np.array([1e3, 2e3, 3e3]),
            omega_rf=np.array([1e7, 3e7]),
            phi_rf=np.array([0.0, 1.0]),
            charge=1.0,
            acceleration_kick=5.0,
        )
        with self.assertRaises(AssertionError):
            KickMultiHarmonicArgs.from_specials_call(
                arguments, eager_specials=None
            )

    def test_counting_kernel_cannot_write(self) -> None:
        with self.assertRaisesRegex(TypeError, "cannot write"):

            class HistogramArgs(KernelCallArgs):  # noqa: F841
                writes_dt = True
                writes_dE = False
                counts_across_particles = True

    def test_missing_write_flags_are_rejected_at_definition(self) -> None:
        with self.assertRaisesRegex(TypeError, "writes_dE"):

            class DriftSimpleArgs(KernelCallArgs):  # noqa: F841
                writes_dt = True

        with self.assertRaisesRegex(TypeError, "writes_dt"):

            class KickSingleHarmonicArgs(KernelCallArgs):  # noqa: F841
                writes_dE = True
