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
same names, and the numpy dtypes the queue packs are derived from them.

Notes
-----
To add a deferrable kernel:

1. Add a ``<Kernel>Args(KernelCallArgs)`` dataclass; its name must be the
   `Specials` method in CamelCase, and it must set `writes_dt` and
   `writes_dE` to the coordinates the kernel modifies. Override
   `from_specials_call` only if the method's arguments need transforming.
   Append it to `KERNEL_CALL_ARGS`.
2. Regenerate the header:
   ``python -m blond.core.backends.deferred.kernel_call_records``.
3. Add one overload per backend: ``apply_to_chunk(const <Kernel>Args&,
   ...)`` in ``cpp/particle_kernels.h`` and ``apply_to_particle(const
   <Kernel>Args&, ...)`` in ``cuda/kernels.cu``. A missing overload does
   not compile.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Self,
    get_args,
    get_type_hints,
)

import numpy as np

from blond.core.backends.backend import INDEX_DTYPE, Specials
from blond.generals.cupy_.no_cupy_import import is_cupy_array

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

HEADER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kernel_call_records.h"
)
# The CUDA executor takes the batch by value in its kernel parameters,
# which are limited to 4096 bytes before CUDA 12.1 / Volta; the other
# 32 bytes are its remaining parameters (`execute_kernel_call_batch`).
KERNEL_CALL_BATCH_CAPACITY_BYTES = 4096 - 32
MAX_RF_HARMONICS_PER_RECORD = 32
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
        Write ``value`` into the packed record.

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


Real = Annotated[float, RealField()]
Int32 = Annotated[int, Int32Field()]
Index = Annotated[int, IndexField()]
InputArray = Annotated[Any, InputArrayField()]
RfParameters = Annotated[
    Any, InlineRealArrayField(MAX_RF_HARMONICS_PER_RECORD)
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
    def from_specials_call(
        cls, arguments: Mapping[str, Any], eager_specials: Any
    ) -> list[Self] | None:
        """
        Build the records of one `Specials` call.

        Parameters
        ----------
        arguments
            The call's arguments by name, defaults applied.
        eager_specials
            The eager specials class, for backend-specific helpers.

        Returns
        -------
        list or None
            The records to queue, or None to run the call eagerly.
            By default one record, each field from the argument of the
            same name.
        """
        return [
            cls(**{f.name: arguments[f.name] for f in dataclasses.fields(cls)})
        ]


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
    """`Specials.kick_multi_harmonic`, 32 harmonics per record."""

    writes_dt = False
    writes_dE = True

    n_rf: Int32
    voltage: RfParameters
    omega_rf: RfParameters
    phi_rf: RfParameters
    charge: Real
    acceleration_kick: Real

    @classmethod
    def from_specials_call(
        cls, arguments: Mapping[str, Any], eager_specials: Any
    ) -> list[Self]:
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
            One record per 32 harmonics.
        """
        n_rf = int(arguments["n_rf"])
        voltage = arguments["voltage"]
        omega_rf = arguments["omega_rf"]
        phi_rf = arguments["phi_rf"]
        assert len(voltage) == len(omega_rf) == len(phi_rf) == n_rf
        records = []
        for first in range(0, max(n_rf, 1), MAX_RF_HARMONICS_PER_RECORD):
            last = min(first + MAX_RF_HARMONICS_PER_RECORD, n_rf)
            records.append(
                cls(
                    n_rf=last - first,
                    voltage=arguments["voltage"][first:last],
                    omega_rf=arguments["omega_rf"][first:last],
                    phi_rf=arguments["phi_rf"][first:last],
                    charge=arguments["charge"],
                    acceleration_kick=(
                        arguments["acceleration_kick"] if last == n_rf else 0.0
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
    def from_specials_call(
        cls, arguments: Mapping[str, Any], eager_specials: Any
    ) -> list[Self] | None:
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
            One record, or None beyond `MAX_HIGHER_ALPHA` coefficients or
            for a device `higher_alpha`.
        """
        higher_alpha = arguments["higher_alpha"]
        if len(higher_alpha) > MAX_HIGHER_ALPHA:
            return None
        if is_cupy_array(higher_alpha):
            return None
        return [
            cls(
                T=arguments["T"],
                alpha_0=arguments["alpha_0"],
                beta=arguments["beta"],
                energy=arguments["energy"],
                n_alpha=len(higher_alpha),
                higher_alpha=higher_alpha,
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
    def from_specials_call(
        cls, arguments: Mapping[str, Any], eager_specials: Any
    ) -> list[Self] | None:
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
            One record, or None for sparse profiles.
        """
        if arguments["first_left_cut"] is not None:
            return None
        table = eager_specials._build_voltage_kick_table(
            voltage=arguments["voltage"],
            bin_centers=arguments["bin_centers"],
            charge=arguments["charge"],
            acceleration_kick=arguments["acceleration_kick"],
        )
        return [
            cls(
                voltage_kick_table=table,
                acceleration_kick=arguments["acceleration_kick"],
            )
        ]


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
        "  std::uint32_t record_size_bytes; // header + Args, multiple of 8",
        "};",
        "",
        "// Plain C arrays: the layout is fixed by the numpy dtypes.",
        "// NOLINTBEGIN(*-avoid-c-arrays)",
    ]
    for args_type in KERNEL_CALL_ARGS:
        members, size = _layout(args_type)
        struct = args_type.__name__
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
