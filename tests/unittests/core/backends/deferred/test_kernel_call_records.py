import dataclasses
import inspect
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
# stale value from an earlier record.
_OPTIMIZED_PACKING_CHECK = textwrap.dedent(
    """
    from blond.core.backends.deferred import kernel_call_records as records
    from blond.core.backends.deferred.kernel_call_queue import (
        KernelCallQueue,
    )

    queue = KernelCallQueue()
    queue.buffer[:] = 0xFF  # stale bytes of earlier records
    queue.append(
        records.DriftSimpleArgs(T=1.0, eta_0=2.0, beta=3.0, energy=4.0)
    )
    dtype = records.DriftSimpleArgs.record_dtype()
    args = queue.buffer[: queue.n_bytes].view(dtype)[0]["args"]
    if args.tolist() != (1.0, 2.0, 3.0, 4.0):  # not `assert`: -O strips it
        raise SystemExit(f"stale record fields: {args}")
    try:
        records.DriftSimpleArgs(T=1.0, eta_0=2.0, beta=3.0)
    except TypeError:
        print("missing field rejected")
    """
)


class TestKernelCallRecords(BLonDTestCase):
    def test_packing_is_complete_under_python_optimize(self) -> None:
        result = subprocess.run(
            [sys.executable, "-O", "-c", _OPTIMIZED_PACKING_CHECK],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("missing field rejected", result.stdout)

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
        # Kernels without their own from_specials_call are filled from
        # the kwargs by field name.
        default = KernelCallArgs.from_specials_call.__func__
        for args_type in records.KERNEL_CALL_ARGS:
            if args_type.from_specials_call.__func__ is not default:
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
            getattr(PythonSpecials, args_type.specials_method())(
                dt=dt, dE=dE, **_EAGER_CALLS[args_type.specials_method()]
            )
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

    def test_missing_write_flags_are_rejected_at_definition(self) -> None:
        with self.assertRaisesRegex(TypeError, "writes_dE"):

            class DriftSimpleArgs(KernelCallArgs):  # noqa: F841
                writes_dt = True

        with self.assertRaisesRegex(TypeError, "writes_dt"):

            class KickSingleHarmonicArgs(KernelCallArgs):  # noqa: F841
                writes_dE = True
