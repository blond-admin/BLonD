# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Layout of deferred kernel call records -- the single source of truth.

A deferred specials (``cpp_deferred``, ``cuda_deferred``) turns every call
of a deferrable kernel into a *kernel call record*: a `HEADER_DTYPE`
header followed by that kernel's ``Args`` struct. The records of one
flush form a *batch*, which the backend's executor applies to the
particles in a single fused pass.

Each kernel's ``Args`` is a frozen dataclass below, e.g. `DriftSimpleArgs`.
Its annotated fields (`Real`, `Int32`, `InputArray`, ...) fix the layout:
``kernel_call_records.h`` is generated from them, with C structs of the
same names, and the numpy dtypes and the `struct` formats the queue packs
records with (`KernelCallArgs.record_packer`) are derived from them.
A last `TrailingColumnsField` (e.g. the harmonics of
`KickMultiHarmonicArgs`) makes the record variable-length: its columns
follow the fixed struct, and ``record_size_bytes`` in the header covers
them.

Notes
-----
To add a deferrable kernel:

1. Add a ``<Kernel>Args(KernelCallArgs)`` dataclass; its name must be the
   `Specials` method in CamelCase, and it must set `writes_dt` and
   `writes_dE` to the coordinates the kernel modifies. Override
   `field_values_from_specials_call` only if the method's arguments need
   transforming.
   Append it to `KERNEL_CALL_ARGS`.
2. Regenerate the header:
   ``python -m blond.core.backends.deferred.kernel_call_records``.
3. Add one overload per backend: ``apply_to_chunk(const <Kernel>Args&,
   ...)`` in ``cpp/particle_kernels.h`` and ``apply_to_particle(const
   <Kernel>Args&, <Factors>, ...)`` in ``cuda/kernels.cu``, plus a
   ``prepare(const <Kernel>Args&)`` returning ``<Factors>`` if the kernel
   has loop-invariant factors (``NoFactors`` otherwise). A missing
   overload does not compile.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    get_args,
    get_type_hints,
)

import numpy as np

from blond.core.backends.backend import INDEX_DTYPE, Specials
from blond.generals.cupy_.no_cupy_import import is_cupy_array

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping
    from typing import Self

HEADER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kernel_call_records.h"
)
# The CUDA executor takes the batch by value in its kernel parameters,
# which are limited to 4096 bytes before CUDA 12.1 / Volta; the other
# 32 bytes are its remaining parameters (`execute_kernel_call_batch`).
KERNEL_CALL_BATCH_CAPACITY_BYTES = 4096 - 32
MAX_HIGHER_ALPHA = 8

# Records are laid out for 64-bit reals and indices; the generated header
# static_asserts the same on the C++/CUDA side.
_REAL = np.dtype(np.float64)
_INT32 = np.dtype(np.int32)
_INDEX = np.dtype(INDEX_DTYPE)
_POINTER = np.dtype(np.uintp)
assert _INDEX.itemsize == _REAL.itemsize, "records assume a 64-bit index_t"


def address_of(array: Any) -> int:
    """
    Return the data address of a host or device array.

    Parameters
    ----------
    array
        NumPy or CuPy array.

    Returns
    -------
    int
        Host address for NumPy, device address for CuPy.
    """
    return array.data.ptr if is_cupy_array(array) else array.ctypes.data


# ---------------------------------------------------------------- fields


class RecordField(ABC):
    """How one field of a `KernelCallArgs` is laid out and packed."""

    @abstractmethod
    def members(self, name: str) -> list[tuple[str, str, np.dtype]]:
        """
        Return the C struct members this field becomes.

        Parameters
        ----------
        name
            The dataclass field name.

        Returns
        -------
        list
            ``(C declaration, numpy field name, numpy dtype)`` per member.
        """

    def pack(
        self, packed: Any, name: str, value: Any, keep_alive: list
    ) -> None:
        """
        Write ``value`` into the packed record (reference packing).

        Parameters
        ----------
        packed
            Structured view of the record's ``Args``.
        name
            The dataclass field name.
        value
            The field's value.
        keep_alive
            Arrays the batch references, to hold until the flush.
        """
        packed[name] = value

    def packer_source(self, name: str) -> tuple[list[str], list[str]]:
        """
        Return the source that packs this field in `record_packer`.

        The fast packer is generated Python code: it runs the statements,
        then packs the values of the expressions, one per scalar of the
        field's `members`, with one `struct.Struct.pack_into`. The field
        value is the local variable ``name``; ``keep_alive`` is in scope,
        and so is every name of `packer_globals`.

        Parameters
        ----------
        name
            The dataclass field name.

        Returns
        -------
        tuple
            ``(statements, value expressions)``.
        """
        return [], [name]

    def packer_globals(self, name: str) -> dict[str, Any]:
        """
        Return the global names `packer_source` refers to.

        Parameters
        ----------
        name
            The dataclass field name.

        Returns
        -------
        dict
            Name to object; none by default.
        """
        return {}


@dataclass(frozen=True)
class RealField(RecordField):
    """A ``real_t`` scalar."""

    def members(  # NOQA: D102
        self, name: str
    ) -> list[tuple[str, str, np.dtype]]:
        return [(f"real_t {name};", name, _REAL)]


@dataclass(frozen=True)
class Int32Field(RecordField):
    """A ``std::int32_t`` scalar; overflowing values raise when packed."""

    def members(  # NOQA: D102
        self, name: str
    ) -> list[tuple[str, str, np.dtype]]:
        return [(f"std::int32_t {name};", name, _INT32)]

    def pack(  # NOQA: D102
        self, packed: Any, name: str, value: Any, keep_alive: list
    ) -> None:
        packed[name] = np.int32(value)  # OverflowError instead of wrapping


@dataclass(frozen=True)
class IndexField(RecordField):
    """An ``index_t`` scalar, e.g. a particle count."""

    def members(  # NOQA: D102
        self, name: str
    ) -> list[tuple[str, str, np.dtype]]:
        return [(f"index_t {name};", name, _INDEX)]


@dataclass(frozen=True)
class InputArrayField(RecordField):
    """
    A backend array the batch reads: pointer plus ``<name>_length``.

    The array is kept alive until the flush. `from_specials_call` must
    hand over an array nobody writes before then (e.g. a fresh table).
    """

    def members(  # NOQA: D102
        self, name: str
    ) -> list[tuple[str, str, np.dtype]]:
        return [
            (f"const real_t *{name};", name, _POINTER),
            (f"index_t {name}_length;", f"{name}_length", _INDEX),
        ]

    def pack(  # NOQA: D102
        self, packed: Any, name: str, value: Any, keep_alive: list
    ) -> None:
        assert value.dtype == _REAL and value.flags.c_contiguous
        packed[name] = address_of(value)
        packed[f"{name}_length"] = value.size
        keep_alive.append(value)

    def packer_source(  # NOQA: D102
        self, name: str
    ) -> tuple[list[str], list[str]]:
        return (
            [
                f"assert {name}.dtype == _REAL and {name}.flags.c_contiguous",
                f"keep_alive.append({name})",
            ],
            [f"_address_of({name})", f"{name}.size"],
        )


@dataclass(frozen=True)
class InlineRealArrayField(RecordField):
    """
    Up to ``max_length`` reals copied into the record.

    Takes host arrays only: reading a device array would sync on every
    call. Unused slots are zeroed; how many are used is a separate
    `Int32` field of the kernel (e.g. ``n_rf``).
    """

    max_length: int

    def members(  # NOQA: D102
        self, name: str
    ) -> list[tuple[str, str, np.dtype]]:
        return [
            (
                f"real_t {name}[{self.max_length}];",
                name,
                np.dtype((_REAL, (self.max_length,))),
            )
        ]

    def pack(  # NOQA: D102
        self, packed: Any, name: str, value: Any, keep_alive: list
    ) -> None:
        assert not is_cupy_array(value), f"`{name}` must be a host array"
        n_values = len(value)
        assert n_values <= self.max_length
        packed[name][:n_values] = value
        packed[name][n_values:] = 0.0

    def packer_source(  # NOQA: D102
        self, name: str
    ) -> tuple[list[str], list[str]]:
        # Zeros fill the unused slots; more than `max_length` values leave
        # none and overflow the struct, which raises even under -O.
        return (
            [
                f"assert not _is_cupy_array({name}), "
                f"'`{name}` must be a host array'"
            ],
            [f"*_host_values({name})", f"*_{name}_zeros[len({name}) :]"],
        )

    def packer_globals(self, name: str) -> dict[str, Any]:  # NOQA: D102
        return {f"_{name}_zeros": (0.0,) * self.max_length}


@dataclass(frozen=True)
class TrailingColumnsField(RecordField):
    """
    Variable-length columns of reals after the fixed ``Args``.

    The record ends with the value's columns one after the other, as
    many reals each as the value has rows, so its ``record_size_bytes``
    grows with them instead of reserving a maximum. In the fixed struct
    the field is only its ``count``, an ``std::int32_t``; the generated
    ``<name>_of(args)`` returns a ``view_struct`` of one pointer per
    column. It must be the last field; reals keep the next record
    8-byte aligned.

    Columns rather than interleaved rows: a loop over the rows then
    reads each column contiguously, which GCC vectorises without
    shuffles, and the layout matches the eager CUDA kick's
    ``RFParamsBatch``.

    The value is one host array per column, in order and of equal
    length: reading a device array would sync on every call.
    """

    count: str
    """Name of the ``std::int32_t`` count member of the fixed struct."""
    view_struct: str
    """C name of the struct of column pointers `<name>_of` returns."""
    columns: tuple[str, ...]
    """The columns, in layout order."""
    max_length_constant: str
    """C and Python name of the most rows one record may hold."""

    @property
    def row_nbytes(self) -> int:
        """
        Return the bytes one row adds to the record.

        Returns
        -------
        int
            One real per column.
        """
        return len(self.columns) * _REAL.itemsize

    def members(  # NOQA: D102
        self, name: str
    ) -> list[tuple[str, str, np.dtype]]:
        return [(f"std::int32_t {self.count};", self.count, _INT32)]

    def pack(  # NOQA: D102
        self, packed: Any, name: str, value: Any, keep_alive: list
    ) -> None:
        assert len(value) == len(self.columns)
        assert all(len(column) == len(value[0]) for column in value)
        assert not any(is_cupy_array(column) for column in value), (
            f"`{name}` must be host arrays"
        )
        packed[self.count] = np.int32(len(value[0]))

    def trailing_nbytes(self, value: Any) -> int:
        """
        Return the bytes the columns add after the fixed ``Args``.

        Parameters
        ----------
        value
            The field's value.

        Returns
        -------
        int
            Row count times `row_nbytes`.
        """
        return len(value[0]) * self.row_nbytes

    def pack_trailing(self, trailing: Any, value: Any) -> None:
        """
        Write the columns after the fixed ``Args`` struct.

        Parameters
        ----------
        trailing
            The record's ``uint8`` bytes after the fixed struct,
            `trailing_nbytes` long.
        value
            The field's value.
        """
        reals = trailing.view(_REAL)
        n_rows = len(value[0])
        for position, column in enumerate(value):
            reals[position * n_rows : (position + 1) * n_rows] = column

    def packer_source(  # NOQA: D102
        self, name: str
    ) -> tuple[list[str], list[str]]:
        return (
            [
                f"assert len({name}) == {len(self.columns)}",
                f"assert len(set(map(len, {name}))) == 1",
                f"assert not any(map(_is_cupy_array, {name})), "
                f"'`{name}` must be host arrays'",
            ],
            [f"len({name}[0])"],
        )

    def write_trailing(
        self, view: memoryview, position: int, value: Any
    ) -> None:
        """
        Write the columns from byte ``position`` on (fast packing).

        Parameters
        ----------
        view
            ``memoryview`` of the batch bytes.
        position
            Byte offset of the first column, right after the fixed
            ``Args``.
        value
            The field's value. A column of another length than the first
            does not fit its slot and raises, also under ``python -O``.
        """
        n_bytes = len(value[0]) * _REAL.itemsize
        for column in value:
            view[position : position + n_bytes] = np.asarray(
                column, dtype=_REAL
            ).tobytes()
            position += n_bytes


Real = Annotated[float, RealField()]
Int32 = Annotated[int, Int32Field()]
Index = Annotated[int, IndexField()]
InputArray = Annotated[Any, InputArrayField()]
RfHarmonics = Annotated[
    Any,
    TrailingColumnsField(
        count="n_rf",
        view_struct="RfHarmonics",
        columns=("voltage", "omega_rf", "phi_rf"),
        max_length_constant="MAX_RF_HARMONICS_PER_RECORD",
    ),
]
HigherAlphas = Annotated[Any, InlineRealArrayField(MAX_HIGHER_ALPHA)]


# ------------------------------------------------------------ the kernels


@dataclass(frozen=True, eq=False)
class KernelCallArgs:
    """
    Parameters of one deferred kernel call; a subclass is one kernel.

    The subclass is named ``<Kernel>Args`` after the `Specials` method in
    CamelCase, and the generated C struct has the same name. Its fields,
    annotated with `Real`, `Int32`, ..., are the struct members in order.
    """

    writes_dt: ClassVar[bool]
    """Whether the kernel modifies ``dt``; required on every subclass."""
    writes_dE: ClassVar[bool]
    """Whether the kernel modifies ``dE``; required on every subclass."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Reject a subclass that is not a complete kernel description.

        Parameters
        ----------
        **kwargs
            Forwarded to `object.__init_subclass__`.

        Raises
        ------
        TypeError
            If the class is not named ``<Kernel>Args`` after a `Specials`
            method, or does not set `writes_dt` and `writes_dE`.
        """
        super().__init_subclass__(**kwargs)
        if not cls.__name__.endswith("Args") or not hasattr(
            Specials, cls.specials_method()
        ):
            raise TypeError(
                f"{cls.__name__} must be named <Kernel>Args after a "
                f"Specials method; Specials has no "
                f"{cls.specials_method()!r}."
            )
        # The CUDA executor stores only the coordinates a batch writes,
        # so a missing flag must not default to either value.
        for flag in ("writes_dt", "writes_dE"):
            if not isinstance(getattr(cls, flag, None), bool):
                raise TypeError(
                    f"{cls.__name__} must set `{flag}` to True or False: "
                    "whether the kernel modifies that coordinate."
                )

    @classmethod
    def specials_method(cls) -> str:
        """
        Name of the `Specials` method, e.g. ``"drift_simple"``.

        Returns
        -------
        str
            The snake_case form of `kernel_id_name`.
        """
        return re.sub(r"(?<!^)(?=[A-Z])", "_", cls.kernel_id_name()).lower()

    @classmethod
    def kernel_id_name(cls) -> str:
        """
        Name of the ``KernelId`` enumerator, e.g. ``"DriftSimple"``.

        Returns
        -------
        str
            The class name without the ``Args`` suffix.
        """
        return cls.__name__.removesuffix("Args")

    @classmethod
    def kernel_id(cls) -> int:
        """
        Value of ``KernelId``: the position in `KERNEL_CALL_ARGS`.

        Returns
        -------
        int
            The kernel's id in every record header.
        """
        return _KERNEL_IDS[cls]

    @classmethod
    def record_fields(cls) -> tuple[tuple[str, RecordField], ...]:
        """
        ``(name, RecordField)`` of every field, in layout order.

        Returns
        -------
        tuple
            One ``(field name, RecordField marker)`` per dataclass field.
        """
        return _record_fields(cls)

    @classmethod
    def args_dtype(cls) -> np.dtype:
        """
        Numpy dtype with the offsets of the C ``Args`` struct.

        Returns
        -------
        np.dtype
            Structured dtype, padded to 8 bytes.
        """
        return _args_dtype(cls)

    @classmethod
    def record_dtype(cls) -> np.dtype:
        """
        Numpy dtype of a whole record: header, then ``args``.

        Returns
        -------
        np.dtype
            Structured dtype with ``kernel_id``, ``record_size_bytes``
            and ``args``.
        """
        return _record_dtype(cls)

    @classmethod
    def trailing_field(cls) -> tuple[str, TrailingColumnsField] | None:
        """
        ``(name, TrailingColumnsField)`` of the last field, if it is one.

        Returns
        -------
        tuple or None
            The trailing array field, or None for a fixed-size record.
        """
        return _trailing_field(cls)

    @classmethod
    def max_trailing_length(cls) -> int:
        """
        Most trailing elements one record may hold.

        As many as keep the record within one CUDA launch
        (`KERNEL_CALL_BATCH_CAPACITY_BYTES`).

        Returns
        -------
        int
            The limit; 0 for a fixed-size record.
        """
        trailing = cls.trailing_field()
        if trailing is None:
            return 0
        free_bytes = KERNEL_CALL_BATCH_CAPACITY_BYTES - (
            cls.record_dtype().itemsize
        )
        return free_bytes // trailing[1].row_nbytes

    def record_size_bytes(self) -> int:
        """
        Return the bytes of this record: header, ``Args``, trailing array.

        Returns
        -------
        int
            A multiple of 8.
        """
        dtype, _, trailing = _packing(type(self))
        if trailing is None:
            return dtype.itemsize
        name, record_field = trailing
        return dtype.itemsize + record_field.trailing_nbytes(
            getattr(self, name)
        )

    @classmethod
    def record_packer(cls) -> Callable[..., int]:
        """
        Return the fast packer of this kernel's records.

        ``packer(view, offset, keep_alive, *field_values)`` writes one
        whole record -- header, ``Args`` including its padding, trailing
        columns -- into the ``memoryview`` ``view`` at byte ``offset`` and
        returns its size. ``field_values`` are the dataclass fields in
        order, so no dataclass instance is needed. It is generated once
        from the `record_dtype` layout with one `struct.Struct`, and
        writes exactly the bytes of `pack_into`, the reference packing.
        A missing or surplus value raises `TypeError`, too many inline
        values `struct.error`, a record beyond one CUDA launch
        `ValueError` -- also under ``python -O``.

        Returns
        -------
        Callable
            The packer; ``view`` needs `KERNEL_CALL_BATCH_CAPACITY_BYTES`
            free bytes at ``offset``.
        """
        return _record_packer(cls)

    def pack_into(self, record: Any, keep_alive: list) -> None:
        """
        Write the whole record into ``record`` through the numpy dtypes.

        The reference packing: the queue packs with the equivalent
        `record_packer`, which is several times faster, and tests hold it
        to the same bytes. Unlike it, this leaves padding bytes as they
        were.

        Parameters
        ----------
        record
            ``uint8`` array of exactly `record_size_bytes`.
        keep_alive
            Arrays the batch references, to hold until the flush.
        """
        dtype, fields, trailing = _packing(type(self))
        item = record[: dtype.itemsize].view(dtype)[0]
        item["kernel_id"] = _KERNEL_IDS[type(self)]
        item["record_size_bytes"] = record.size
        packed = item["args"]
        for name, record_field in fields:
            record_field.pack(packed, name, getattr(self, name), keep_alive)
        if trailing is not None:
            name, record_field = trailing
            record_field.pack_trailing(
                record[dtype.itemsize :], getattr(self, name)
            )

    @classmethod
    def field_values_from_specials_call(
        cls, arguments: Mapping[str, Any], eager_specials: Any
    ) -> list[tuple] | None:
        """
        Return the field values of the records of one `Specials` call.

        Override this only if the method's arguments need transforming;
        the deferred specials pass the arguments of a kernel that does
        not straight to its `record_packer`.

        Parameters
        ----------
        arguments
            The call's arguments by name, defaults applied.
        eager_specials
            The eager specials class, for backend-specific helpers.

        Returns
        -------
        list or None
            One tuple of field values, in field order, per record to
            queue, or None to run the call eagerly. By default one record,
            each field from the argument of the same name.
        """
        return [tuple(arguments[name] for name, _ in cls.record_fields())]

    @classmethod
    def from_specials_call(
        cls, arguments: Mapping[str, Any], eager_specials: Any
    ) -> list[Self] | None:
        """
        Build the records of one `Specials` call as dataclasses.

        Parameters
        ----------
        arguments
            The call's arguments by name, defaults applied.
        eager_specials
            The eager specials class, for backend-specific helpers.

        Returns
        -------
        list or None
            The records of `field_values_from_specials_call`, or None to
            run the call eagerly.
        """
        records = cls.field_values_from_specials_call(
            arguments, eager_specials
        )
        if records is None:
            return None
        return [cls(*values) for values in records]


@dataclass(frozen=True, eq=False)
class KickSingleHarmonicArgs(KernelCallArgs):
    """`Specials.kick_single_harmonic`."""

    writes_dt = False
    writes_dE = True

    voltage: Real
    omega_rf: Real
    phi_rf: Real
    charge: Real
    acceleration_kick: Real


@dataclass(frozen=True, eq=False)
class KickMultiHarmonicArgs(KernelCallArgs):
    """
    `Specials.kick_multi_harmonic`, its harmonics trailing the record.

    ``harmonics`` is ``(voltage, omega_rf, phi_rf)``, stored as three
    columns of ``n_rf`` reals after the fixed fields, so a record is
    ``32 + 24 * n_rf`` bytes.
    """

    writes_dt = False
    writes_dE = True

    charge: Real
    acceleration_kick: Real
    harmonics: RfHarmonics

    @classmethod
    def field_values_from_specials_call(
        cls, arguments: Mapping[str, Any], eager_specials: Any
    ) -> list[tuple]:
        """
        Split the harmonics over records of `MAX_RF_HARMONICS_PER_RECORD`.

        ``acceleration_kick`` goes into the last record only, and one
        record is always queued, so ``n_rf == 0`` still applies it -- as
        ``CudaSpecials.kick_multi_harmonic`` splits its launches.

        Parameters
        ----------
        arguments
            The call's arguments by name.
        eager_specials
            Unused.

        Returns
        -------
        list
            The field values of one record per
            `MAX_RF_HARMONICS_PER_RECORD` harmonics.
        """
        n_rf = int(arguments["n_rf"])
        voltage = arguments["voltage"]
        omega_rf = arguments["omega_rf"]
        phi_rf = arguments["phi_rf"]
        assert len(voltage) == len(omega_rf) == len(phi_rf) == n_rf
        per_record = _max_trailing_length(cls)
        charge = arguments["charge"]
        acceleration_kick = arguments["acceleration_kick"]
        if n_rf <= per_record and (
            len(voltage) == len(omega_rf) == len(phi_rf) == n_rf
        ):  # one record of the whole arrays, without slicing them
            return [(charge, acceleration_kick, (voltage, omega_rf, phi_rf))]
        records = []
        for first in range(0, max(n_rf, 1), per_record):
            last = min(first + per_record, n_rf)
            records.append(
                (
                    charge,
                    acceleration_kick if last == n_rf else 0.0,
                    (
                        voltage[first:last],
                        omega_rf[first:last],
                        phi_rf[first:last],
                    ),
                )
            )
        return records


@dataclass(frozen=True, eq=False)
class DriftSimpleArgs(KernelCallArgs):
    """`Specials.drift_simple`."""

    writes_dt = True
    writes_dE = False

    T: Real
    eta_0: Real
    beta: Real
    energy: Real


@dataclass(frozen=True, eq=False)
class DriftLikeLineSegmentArgs(KernelCallArgs):
    """`Specials.drift_like_line_segment`."""

    writes_dt = True
    writes_dE = False

    T: Real
    eta_0: Real
    beta: Real
    energy: Real


@dataclass(frozen=True, eq=False)
class DriftExactArgs(KernelCallArgs):
    """`Specials.drift_exact`, up to `MAX_HIGHER_ALPHA` coefficients."""

    writes_dt = True
    writes_dE = False

    T: Real
    alpha_0: Real
    beta: Real
    energy: Real
    n_alpha: Int32
    higher_alpha: HigherAlphas

    @classmethod
    def field_values_from_specials_call(
        cls, arguments: Mapping[str, Any], eager_specials: Any
    ) -> list[tuple] | None:
        """
        Inline the higher-order alphas; more than 8 runs eagerly.

        The polynomial cannot be split over records like harmonics. A
        device ``higher_alpha`` also runs eagerly: `CudaSpecials.
        drift_exact` accepts it as a compatibility path and copies it to
        host itself, but inlining it into the record here would read
        device memory as a host buffer.

        Parameters
        ----------
        arguments
            The call's arguments by name.
        eager_specials
            Unused.

        Returns
        -------
        list or None
            The field values of one record, or None beyond
            `MAX_HIGHER_ALPHA` coefficients or for a device `higher_alpha`.
        """
        higher_alpha = arguments["higher_alpha"]
        n_alpha = len(higher_alpha)
        if n_alpha > MAX_HIGHER_ALPHA or is_cupy_array(higher_alpha):
            return None
        return [
            (
                arguments["T"],
                arguments["alpha_0"],
                arguments["beta"],
                arguments["energy"],
                n_alpha,
                higher_alpha,
            )
        ]


@dataclass(frozen=True, eq=False)
class KickInterpolatedArgs(KernelCallArgs):
    """
    `Specials.kick_interpolated`, dense profiles only.

    ``voltage_kick_table`` is ``[first_bin_center, inverse_bin_width,
    (slope, offset) * n_bins]`` with ``charge`` and ``acceleration_kick``
    folded into the pairs; both backends build it the same way.
    """

    writes_dt = False
    writes_dE = True

    voltage_kick_table: InputArray
    acceleration_kick: Real

    @classmethod
    def field_values_from_specials_call(
        cls, arguments: Mapping[str, Any], eager_specials: Any
    ) -> list[tuple] | None:
        """
        Build the voltage-kick table when the call is queued.

        The table touches no particles, so building it before the flush
        is exact; it is a fresh array owned by the queue.

        Parameters
        ----------
        arguments
            The call's arguments by name.
        eager_specials
            Provides ``_build_voltage_kick_table``.

        Returns
        -------
        list or None
            The field values of one record, or None for sparse profiles.
        """
        if arguments["first_left_cut"] is not None:
            return None
        acceleration_kick = arguments["acceleration_kick"]
        table = eager_specials._build_voltage_kick_table(
            voltage=arguments["voltage"],
            bin_centers=arguments["bin_centers"],
            charge=arguments["charge"],
            acceleration_kick=acceleration_kick,
        )
        return [(table, acceleration_kick)]


KERNEL_CALL_ARGS: tuple[type[KernelCallArgs], ...] = (
    KickSingleHarmonicArgs,
    KickMultiHarmonicArgs,
    DriftSimpleArgs,
    DriftLikeLineSegmentArgs,
    DriftExactArgs,
    KickInterpolatedArgs,
)
ARGS_BY_SPECIALS_METHOD = {
    args_type.specials_method(): args_type for args_type in KERNEL_CALL_ARGS
}
_KERNEL_IDS = {args_type: i for i, args_type in enumerate(KERNEL_CALL_ARGS)}

HEADER_DTYPE = np.dtype(
    {
        "names": ["kernel_id", "record_size_bytes"],
        "formats": [np.uint32, np.uint32],
        "offsets": [0, 4],
        "itemsize": 8,
    }
)


# ---------------------------------------------------------------- layout


@cache
def _record_fields(
    args_type: type[KernelCallArgs],
) -> tuple[tuple[str, RecordField], ...]:
    hints = get_type_hints(args_type, include_extras=True)
    fields = []
    for field in dataclasses.fields(args_type):
        markers = [
            m
            for m in get_args(hints[field.name])
            if isinstance(m, RecordField)
        ]
        if len(markers) != 1:
            raise TypeError(
                f"{args_type.__name__}.{field.name} needs one field "
                "annotation such as Real, Int32 or InputArray"
            )
        fields.append((field.name, markers[0]))
    return tuple(fields)


@cache
def _trailing_field(
    args_type: type[KernelCallArgs],
) -> tuple[str, TrailingColumnsField] | None:
    fields = args_type.record_fields()
    trailing = [
        position
        for position, (_, record_field) in enumerate(fields)
        if isinstance(record_field, TrailingColumnsField)
    ]
    if not trailing:
        return None
    if trailing != [len(fields) - 1]:
        raise TypeError(
            f"{args_type.__name__} may have one TrailingColumnsField only, "
            "as its last field"
        )
    name, record_field = fields[-1]
    assert isinstance(record_field, TrailingColumnsField)
    return name, record_field


@cache
def _packing(
    args_type: type[KernelCallArgs],
) -> tuple[
    np.dtype,
    tuple[tuple[str, RecordField], ...],
    tuple[str, TrailingColumnsField] | None,
]:
    """
    Return everything packing a record of ``args_type`` looks up.

    Parameters
    ----------
    args_type
        The kernel's ``Args`` class.

    Returns
    -------
    tuple
        Its `_record_dtype`, `_record_fields` and `_trailing_field`,
        in one cached lookup per record.
    """
    return (
        _record_dtype(args_type),
        _record_fields(args_type),
        _trailing_field(args_type),
    )


@cache
def _layout(
    args_type: type[KernelCallArgs],
) -> tuple[tuple[tuple[str, str | None, np.dtype | None, int], ...], int]:
    """
    Return ``(members, size)`` of the C struct.

    Every member is aligned to its element size and the struct is padded
    to 8 bytes with explicit ``std::int32_t padding_<k>`` members, so C
    and numpy never depend on compiler padding.

    Parameters
    ----------
    args_type
        The kernel's ``Args`` class.

    Returns
    -------
    tuple
        ``((declaration, numpy name or None, dtype or None, offset), ...)``
        and the struct size in bytes.
    """
    members: list[tuple[str, str | None, np.dtype | None, int]] = []
    offset = 0

    def pad_to(alignment: int) -> None:
        nonlocal offset
        if offset % alignment:
            members.append(
                (f"std::int32_t padding_{len(members)};", None, None, offset)
            )
            offset += 4

    for name, record_field in args_type.record_fields():
        for declaration, numpy_name, dtype in record_field.members(name):
            pad_to(dtype.base.itemsize)
            members.append((declaration, numpy_name, dtype, offset))
            offset += dtype.itemsize
    pad_to(8)
    return tuple(members), offset


@cache
def _args_dtype(args_type: type[KernelCallArgs]) -> np.dtype:
    members, size = _layout(args_type)
    named = [m for m in members if m[1] is not None]
    return np.dtype(
        {
            "names": [m[1] for m in named],
            "formats": [m[2] for m in named],
            "offsets": [m[3] for m in named],
            "itemsize": size,
        }
    )


@cache
def _record_dtype(args_type: type[KernelCallArgs]) -> np.dtype:
    args = _args_dtype(args_type)
    return np.dtype(
        {
            "names": ["kernel_id", "record_size_bytes", "args"],
            "formats": [np.uint32, np.uint32, args],
            "offsets": [0, 4, HEADER_DTYPE.itemsize],
            "itemsize": HEADER_DTYPE.itemsize + args.itemsize,
        }
    )


@cache
def _max_trailing_length(args_type: type[KernelCallArgs]) -> int:
    return args_type.max_trailing_length()


def _host_values(value: Any) -> Any:
    """
    Return the values of a host array as Python scalars.

    They unpack into a `struct` ~10x faster than the numpy scalars that
    iterating the array yields.

    Parameters
    ----------
    value
        A host array, or any other sequence of numbers.

    Returns
    -------
    Any
        A list for a numpy array, else ``value`` itself.
    """
    return value.tolist() if isinstance(value, np.ndarray) else value


# `struct` codes of the scalar record members, by numpy kind and size.
_STRUCT_CODES = {
    ("f", 8): "d",
    ("i", 4): "i",
    ("i", 8): "q",
    ("u", 4): "I",
    ("u", 8): "Q",
}


def _struct_code(dtype: np.dtype) -> str:
    if dtype.subdtype is not None:
        base, shape = dtype.subdtype
        return f"{int(np.prod(shape))}{_struct_code(base)}"
    return _STRUCT_CODES[(dtype.kind, dtype.itemsize)]


def _record_struct(args_type: type[KernelCallArgs]) -> struct.Struct:
    """
    Return the `struct.Struct` of the fixed part of a record.

    Header and ``Args`` in the offsets of `_record_dtype`; padding is
    ``x``, which `struct` writes as zero bytes.

    Parameters
    ----------
    args_type
        The kernel's ``Args`` class.

    Returns
    -------
    struct.Struct
        Native byte order, no implicit alignment.

    Raises
    ------
    TypeError
        If the format does not reproduce the dtype's size.
    """
    record = _record_dtype(args_type)
    members = [HEADER_DTYPE.fields[name][1::-1] for name in HEADER_DTYPE.names]
    members += [
        (HEADER_DTYPE.itemsize + offset, dtype)
        for _, numpy_name, dtype, offset in _layout(args_type)[0]
        if numpy_name is not None  # padding: a gap filled with "x"
    ]
    codes = []
    position = 0
    for offset, dtype in members:
        if offset > position:
            codes.append(f"{offset - position}x")
        codes.append(_struct_code(dtype))
        position = offset + dtype.itemsize
    if record.itemsize > position:
        codes.append(f"{record.itemsize - position}x")
    packer = struct.Struct("=" + "".join(codes))
    if packer.size != record.itemsize:
        raise TypeError(
            f"{args_type.__name__}: struct {packer.format!r} is "
            f"{packer.size} B, the record dtype {record.itemsize} B"
        )
    return packer


@cache
def _record_packer(args_type: type[KernelCallArgs]) -> Callable[..., int]:
    """
    Generate the fast packer of `KernelCallArgs.record_packer`.

    Python source like ``dataclasses`` generates ``__init__``: the field
    names are the parameters, every `RecordField` contributes its
    `RecordField.packer_source`, and one ``pack_into`` of
    `_record_struct` writes header, fields and padding.

    Parameters
    ----------
    args_type
        The kernel's ``Args`` class.

    Returns
    -------
    Callable
        ``packer(view, offset, keep_alive, *field_values) -> size``.
    """
    fields = args_type.record_fields()
    names = [name for name, _ in fields]
    # The generated code's own names start with "_", or are keep_alive.
    assert all(
        name.isidentifier() and not name.startswith("_") for name in names
    )
    assert "keep_alive" not in names
    fixed_size = _record_dtype(args_type).itemsize
    assert fixed_size <= KERNEL_CALL_BATCH_CAPACITY_BYTES
    namespace: dict[str, Any] = {
        "_pack_into": _record_struct(args_type).pack_into,
        "_address_of": address_of,
        "_is_cupy_array": is_cupy_array,
        "_host_values": _host_values,
        "_REAL": _REAL,
    }
    statements: list[str] = []
    values: list[str] = []
    for name, record_field in fields:
        field_statements, field_values = record_field.packer_source(name)
        statements += field_statements
        values += field_values
        namespace.update(record_field.packer_globals(name))
    trailing = args_type.trailing_field()
    if trailing is None:
        size = str(fixed_size)
        write_trailing: list[str] = []
    else:
        name, columns = trailing
        namespace["_trailing"] = columns
        statements += [
            f"_size = {fixed_size} + len({name}[0]) * {columns.row_nbytes}",
            f"if _size > {KERNEL_CALL_BATCH_CAPACITY_BYTES}:",
            "    raise ValueError(",
            f"        f'a {{_size}} B {args_type.__name__} record exceeds one '",
            f"        'launch, {KERNEL_CALL_BATCH_CAPACITY_BYTES} B'",
            "    )",
        ]
        size = "_size"
        write_trailing = [
            f"_trailing.write_trailing(_view, _offset + {fixed_size}, {name})"
        ]
    lines = [
        f"def pack_{args_type.__name__}("
        f"_view, _offset, keep_alive, {', '.join(names)}):",
        *(f"    {line}" for line in statements),
        f"    _pack_into(_view, _offset, {_KERNEL_IDS[args_type]}, {size}, "
        f"{', '.join(values)})",
        *(f"    {line}" for line in write_trailing),
        f"    return {size}",
    ]
    exec("\n".join(lines), namespace)  # noqa: S102
    return namespace[f"pack_{args_type.__name__}"]


MAX_RF_HARMONICS_PER_RECORD = KickMultiHarmonicArgs.max_trailing_length()
"""Most harmonics one record holds: as many as fit one CUDA launch."""


# ---------------------------------------------------------------- header


def _copyright_lines() -> list[str]:
    notice = os.path.join(
        os.path.dirname(HEADER_PATH),
        "..",
        "..",
        "..",
        "..",
        "dev_tools",
        "copyright_notice.txt",
    )
    with open(notice) as file:
        return [re.sub(r"^#", "//", line.rstrip("\n")) for line in file]


def generate_header() -> str:
    """
    Return the text of ``kernel_call_records.h``.

    Returns
    -------
    str
        The header, byte for byte as it must be on disk.
    """
    lines = [
        *_copyright_lines(),
        "",
        "// GENERATED by `python -m "
        "blond.core.backends.deferred.kernel_call_records`",
        "// from the KernelCallArgs dataclasses in kernel_call_records.py.",
        "// Do not edit.",
        "//",
        "// Include after `real_t` and `index_t` are defined:",
        "// blond_common.h on the C++ side, kernels.cu on the CUDA side.",
        "",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "#ifdef __CUDACC__",
        "#define BLOND_HOST_DEVICE __host__ __device__",
        "#else",
        "#define BLOND_HOST_DEVICE",
        "#endif",
        "",
        'static_assert(sizeof(real_t) == 8, "records assume 64-bit real_t");',
        'static_assert(sizeof(index_t) == 8, "records assume 64-bit index_t");',
        "",
        "// Layout checks of the structs below, one line each.",
        "#define BLOND_CHECK_SIZE(type, size) \\",
        '  static_assert(sizeof(type) == (size), "regenerate this header")',
        "#define BLOND_CHECK_OFFSET(type, member, offset) \\",
        "  static_assert(offsetof(type, member) == (offset), \\",
        '                "regenerate this header")',
        "",
        "enum class KernelId : std::uint32_t {",
        *(
            f"  {args_type.kernel_id_name()} = {args_type.kernel_id()},"
            for args_type in KERNEL_CALL_ARGS
        ),
        "};",
        f"constexpr int KERNEL_COUNT = {len(KERNEL_CALL_ARGS)};",
        "constexpr std::size_t KERNEL_CALL_BATCH_CAPACITY_BYTES = "
        f"{KERNEL_CALL_BATCH_CAPACITY_BYTES};",
        "",
        "struct KernelCallHeader {",
        "  KernelId kernel_id;",
        "  // header + Args + trailing elements, multiple of 8",
        "  std::uint32_t record_size_bytes;",
        "};",
        "",
        "// Plain C arrays: the layout is fixed by the numpy dtypes.",
        "// NOLINTBEGIN(*-avoid-c-arrays)",
    ]
    for args_type in KERNEL_CALL_ARGS:
        members, size = _layout(args_type)
        struct = args_type.__name__
        trailing = args_type.trailing_field()
        if trailing is not None:
            name, columns = trailing
            lines.append(
                f"constexpr int {columns.max_length_constant} = "
                f"{args_type.max_trailing_length()};"
            )
            lines.append(
                f"// Followed by {len(columns.columns)} columns of "
                f"`{columns.count}` reals, see `{name}_of`."
            )
        lines.append(f"struct {struct} {{")
        lines.extend(f"  {member[0]}" for member in members)
        lines.append("};")
        lines.append(f"BLOND_CHECK_SIZE({struct}, {size});")
        for _declaration, numpy_name, _, offset in members:
            if numpy_name is not None:
                lines.append(
                    f"BLOND_CHECK_OFFSET({struct}, {numpy_name}, {offset});"
                )
        lines.append("")
    lines += [
        "// A macro, so the CUDA side can initialise a __device__ array",
        "// from the same list (kernels.cu).",
        "#define KERNEL_CALL_ARGS_SIZES_INITIALIZER \\",
        "  { \\",
        *(
            f"    sizeof({args_type.__name__}), \\"
            for args_type in KERNEL_CALL_ARGS
        ),
        "  }",
        "constexpr std::uint32_t KERNEL_CALL_ARGS_SIZES[KERNEL_COUNT] =",
        "    KERNEL_CALL_ARGS_SIZES_INITIALIZER;",
        "// NOLINTEND(*-avoid-c-arrays)",
        "",
        "// The records are packed back to back in a byte buffer, hence the",
        "// casts from the header to its Args and to the next header.",
        "// NOLINTBEGIN(*-reinterpret-cast,*-pointer-arithmetic)",
        "template <class Args>",
        "BLOND_HOST_DEVICE inline const Args &",
        "record_args(const KernelCallHeader *record) {",
        "  return *reinterpret_cast<const Args *>(",
        "      reinterpret_cast<const char *>(record) + "
        "sizeof(KernelCallHeader));",
        "}",
        "",
        "BLOND_HOST_DEVICE inline const KernelCallHeader *",
        "next_record(const KernelCallHeader *record) {",
        "  return reinterpret_cast<const KernelCallHeader *>(",
        "      reinterpret_cast<const char *>(record) + "
        "record->record_size_bytes);",
        "}",
    ]
    for args_type in KERNEL_CALL_ARGS:
        trailing = args_type.trailing_field()
        if trailing is None:
            continue
        name, columns = trailing
        count = columns.count
        lines += [
            "",
            f"// The trailing columns of a {args_type.__name__} record.",
            f"struct {columns.view_struct} {{",
            *(f"  const real_t *{column};" for column in columns.columns),
            "};",
            f"BLOND_HOST_DEVICE inline {columns.view_struct}",
            f"{name}_of(const {args_type.__name__} &args) {{",
            "  const auto *first = reinterpret_cast<const real_t *>"
            "(&args + 1);",
            "  return {"
            + ", ".join(
                "first"
                if k == 0
                else f"first + args.{count}"
                if k == 1
                else f"first + {k} * args.{count}"
                for k in range(len(columns.columns))
            )
            + "};",
            "}",
        ]
    lines += [
        "// NOLINTEND(*-reinterpret-cast,*-pointer-arithmetic)",
        "",
        "// The only switch over KernelId. `visitor(args)` resolves to the",
        "// backend's overload for that Args type; a missing overload does",
        "// not compile. The pragma lets a host-only visitor instantiate it",
        "// under nvcc without a __host__ __device__ mismatch warning.",
        "#ifdef __CUDACC__",
        "#pragma nv_exec_check_disable",
        "#endif",
        "template <class Visitor>",
        "BLOND_HOST_DEVICE inline void",
        "visit_kernel_call(const KernelCallHeader *record, "
        "const Visitor &visitor) {",
        "  switch (record->kernel_id) {",
    ]
    for args_type in KERNEL_CALL_ARGS:
        lines += [
            f"  case KernelId::{args_type.kernel_id_name()}:",
            f"    visitor(record_args<{args_type.__name__}>(record));",
            "    break;",
        ]
    lines += ["  }", "}", ""]
    return "\n".join(lines)


def header_digest() -> str:
    """
    Return the SHA-256 of the header on disk.

    Folded into the compiled-library cache keys of the cpp and cuda
    backends, whose own source hashes do not cover this folder.

    Returns
    -------
    str
        Hex digest of ``kernel_call_records.h``.
    """
    with open(HEADER_PATH, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    with open(HEADER_PATH, "w") as file:
        file.write(generate_header())
