import dataclasses
import inspect

from blond.core.backends.backend import Specials
from blond.core.backends.deferred import kernel_call_records as records
from blond.core.backends.deferred.kernel_call_records import (
    DriftLikeLineSegmentArgs,
    KernelCallArgs,
    KickMultiHarmonicArgs,
)
from blond.testing.backend_testing import BLonDTestCase


class TestKernelCallRecords(BLonDTestCase):
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
        dtype = KickMultiHarmonicArgs.args_dtype()
        self.assertEqual(dtype.fields["n_rf"][1], 0)
        self.assertEqual(dtype.fields["voltage"][1], 8)  # 4 bytes padding
        self.assertEqual(dtype.fields["voltage"][0].shape, (32,))

    def test_kernel_ids_are_positions(self) -> None:
        for position, args_type in enumerate(records.KERNEL_CALL_ARGS):
            self.assertEqual(args_type.kernel_id(), position)

    def test_misspelt_field_fails_at_construction(self) -> None:
        with self.assertRaises(TypeError):
            records.DriftSimpleArgs(T=1.0, eta0=1.0, beta=1.0, energy=1.0)

    def test_digest_is_sha256(self) -> None:
        self.assertEqual(len(records.header_digest()), 64)
