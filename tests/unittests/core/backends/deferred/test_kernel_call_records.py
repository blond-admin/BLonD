import inspect

import numpy as np

from blond.core.backends.backend import Specials
from blond.core.backends.deferred import kernel_call_records as records
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


KERNELS = records.KERNELS_BY_SPECIALS_METHOD


class TestKernelCallRecords(BLonDTestCase):
    def test_kernel_ids_are_positions(self) -> None:
        # kernel_call_records.h numbers KernelId the same way.
        for position, kernel in enumerate(records.DEFERRABLE_KERNELS):
            self.assertEqual(kernel.kernel_id, position)
        self.assertEqual(len(KERNELS), len(records.DEFERRABLE_KERNELS))

    def test_every_kernel_is_a_specials_method(self) -> None:
        for kernel in records.DEFERRABLE_KERNELS:
            self.assertTrue(
                hasattr(Specials, kernel.specials_method),
                kernel.specials_method,
            )

    def test_default_fields_are_specials_arguments(self) -> None:
        # Kernels without their own build_records are filled from the
        # kwargs by field name.
        for kernel in records.DEFERRABLE_KERNELS:
            if kernel.build_records is not None:
                continue
            parameters = inspect.signature(
                getattr(Specials, kernel.specials_method)
            ).parameters
            for name in kernel.args_dtype.names:
                self.assertIn(name, parameters, kernel.specials_method)

    def test_record_is_header_then_args(self) -> None:
        for kernel in records.DEFERRABLE_KERNELS:
            record = kernel.record_dtype
            self.assertEqual(record.itemsize % 8, 0)
            self.assertEqual(
                record.fields["args"][1], records.HEADER_DTYPE.itemsize
            )
            self.assertEqual(
                record.itemsize,
                records.HEADER_DTYPE.itemsize + kernel.args_dtype.itemsize,
            )

    def test_int32_fields_are_padded_like_c(self) -> None:
        # `align=True` pads the int32 before the next double as the C
        # compiler does; without it every later field would be shifted.
        multi = KERNELS["kick_multi_harmonic"].args_dtype
        self.assertEqual(multi.fields["voltage"][1], 8)
        self.assertEqual(multi.fields["voltage"][0].shape, (32,))
        self.assertEqual(multi.itemsize, 792)
        exact = KERNELS["drift_exact"].args_dtype
        self.assertEqual(exact.fields["higher_alpha"][1], 40)
        self.assertEqual(exact.itemsize, 104)

    def test_pack_requires_exactly_the_fields(self) -> None:
        kernel = KERNELS["drift_simple"]
        record = np.zeros((), dtype=kernel.record_dtype)
        for values in (
            dict(T=1.0, eta0=1.0, beta=1.0, energy=1.0),
            dict(T=1.0, beta=1.0, energy=1.0),
        ):
            with self.assertRaises(AssertionError):
                kernel.pack(record, values, [])

    def test_pack_zeroes_unused_inline_slots(self) -> None:
        kernel = KERNELS["drift_exact"]
        record = np.full((), 7, dtype=kernel.record_dtype)
        values = kernel.records(_EAGER_CALLS["drift_exact"], None)[0]
        kernel.pack(record, values, [])
        self.assertEqual(record["kernel_id"], kernel.kernel_id)
        self.assertEqual(record["record_size_bytes"], record.dtype.itemsize)
        np.testing.assert_array_equal(
            record["args"]["higher_alpha"], [1e-3, 2e-3, 0, 0, 0, 0, 0, 0]
        )

    def test_pack_keeps_pointed_arrays_alive(self) -> None:
        kernel = KERNELS["kick_interpolated"]
        table = np.arange(6.0)
        record = np.zeros((), dtype=kernel.record_dtype)
        keep_alive = []
        kernel.pack(
            record,
            dict(
                voltage_kick_table=table,
                voltage_kick_table_length=table.size,
                acceleration_kick=0.0,
            ),
            keep_alive,
        )
        self.assertEqual(
            record["args"]["voltage_kick_table"], table.ctypes.data
        )
        self.assertIs(keep_alive[0], table)

    def test_digest_is_sha256(self) -> None:
        self.assertEqual(len(records.header_digest()), 64)

    def test_write_flags_match_the_eager_kernels(self) -> None:
        # The CUDA executor stores only the coordinates a batch writes,
        # so a wrong flag silently drops a kernel's result there.
        self.assertEqual(set(_EAGER_CALLS), set(KERNELS))
        rng = np.random.default_rng(5)
        for kernel in records.DEFERRABLE_KERNELS:
            dt = rng.uniform(-1e-8, 1e-8, _N_PARTICLES)
            dE = rng.uniform(-1e6, 1e6, _N_PARTICLES)
            dt_before, dE_before = dt.copy(), dE.copy()
            getattr(PythonSpecials, kernel.specials_method)(
                dt=dt, dE=dE, **_EAGER_CALLS[kernel.specials_method]
            )
            with self.subTest(kernel=kernel.specials_method):
                self.assertEqual(
                    kernel.writes_dt, not np.array_equal(dt, dt_before)
                )
                self.assertEqual(
                    kernel.writes_dE, not np.array_equal(dE, dE_before)
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
            _EAGER_CALLS["drift_exact"],
            higher_alpha=_FakeDeviceArray(np.array([1e-3, 2e-3])),
        )
        self.assertIsNone(KERNELS["drift_exact"].records(arguments, None))

    def test_drift_exact_still_queues_a_host_higher_alpha(self) -> None:
        values = KERNELS["drift_exact"].records(
            _EAGER_CALLS["drift_exact"], None
        )
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0]["n_alpha"], 2)

    def test_drift_exact_runs_eagerly_beyond_max_higher_alpha(self) -> None:
        arguments = dict(
            _EAGER_CALLS["drift_exact"],
            higher_alpha=np.ones(records.MAX_HIGHER_ALPHA + 1),
        )
        self.assertIsNone(KERNELS["drift_exact"].records(arguments, None))

    def test_kick_multi_harmonic_splits_harmonics(self) -> None:
        n_rf = records.MAX_RF_HARMONICS_PER_RECORD + 3
        arguments = dict(
            n_rf=n_rf,
            voltage=np.ones(n_rf),
            omega_rf=np.ones(n_rf),
            phi_rf=np.ones(n_rf),
            charge=1.0,
            acceleration_kick=5.0,
        )
        values = KERNELS["kick_multi_harmonic"].records(arguments, None)
        self.assertEqual([v["n_rf"] for v in values], [32, 3])
        self.assertEqual([v["acceleration_kick"] for v in values], [0, 5])

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
            KERNELS["kick_multi_harmonic"].records(arguments, None)
