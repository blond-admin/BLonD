# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Deferrable kernels and the layout of their kernel call records.

A deferred specials (``cpp_deferred``, ``cuda_deferred``) turns every call
of a deferrable kernel into a *kernel call record*: a `HEADER_DTYPE`
header followed by that kernel's ``Args`` struct. The records of one
flush form a *batch*, which the backend's executor applies to the
particles in a single fused pass.

The ``Args`` structs are written by hand in ``kernel_call_records.h``.
Each `DeferrableKernel` below mirrors one of them as a numpy dtype with
``align=True``, which lays fields out by the same rules as the C
compiler. Two checks keep both sides in step: the cpp and cuda backends
compare every compiled ``sizeof(Args)`` with its dtype when loading, and
the deferred-vs-eager tests catch reordered fields and kernel ids.

Notes
-----
To add a deferrable kernel:

1. Add its ``struct <Kernel>Args``, its ``KernelId`` and its ``case`` in
   ``visit_kernel_call`` to ``kernel_call_records.h``.
2. Add a `DeferrableKernel` with the same fields, in the same order, to
   `DEFERRABLE_KERNELS`, at the position of its ``KernelId``.
3. Add one overload per backend: ``apply_to_chunk(const <Kernel>Args&,
   ...)`` in ``cpp/particle_kernels.h`` and ``apply_to_particle(const
   <Kernel>Args&, ...)`` in ``cuda/kernels.cu``. A missing overload does
   not compile.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from blond.core.backends.backend import INDEX_DTYPE
from blond.generals.cupy_.no_cupy_import import is_cupy_array

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

HEADER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kernel_call_records.h"
)
MAX_RF_HARMONICS_PER_RECORD = 32
MAX_HIGHER_ALPHA = 8

HEADER_DTYPE = np.dtype(
    [("kernel_id", np.uint32), ("record_size_bytes", np.uint32)]
)

# Records hold 64-bit reals and indices, as kernel_call_records.h
# static_asserts on the C++/CUDA side.
_REAL = np.float64
_POINTER = np.uintp


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


@dataclass(frozen=True)
class DeferrableKernel:
    """A `Specials` method whose calls can be queued as records."""

    kernel_id: int
    """Value of its ``KernelId`` in ``kernel_call_records.h``."""
    specials_method: str
    """Name of the `Specials` method, e.g. ``"drift_simple"``."""
    args_dtype: np.dtype
    """
    Mirror of its C ``Args`` struct, built with ``align=True``.

    Subarray fields are inline host arrays with unused slots zeroed;
    ``uintp`` fields point at a backend array the batch reads.
    """
    writes_dt: bool
    """Whether the kernel modifies ``dt``; CUDA stores only written ones."""
    writes_dE: bool
    """Whether the kernel modifies ``dE``; CUDA stores only written ones."""
    build_records: (
        Callable[[Mapping[str, Any], Any], list[dict[str, Any]] | None] | None
    ) = None
    """
    ``build_records(arguments, eager_specials)`` returning the field
    values of each record to queue, or None to run the call eagerly.

    None means one record, each field from the argument of the same name.
    """

    @cached_property
    def record_dtype(self) -> np.dtype:
        """
        Structured dtype of one record, the header followed by ``args``.

        Returns
        -------
        np.dtype
            Structured dtype with ``kernel_id``, ``record_size_bytes``
            and ``args``.
        """
        return np.dtype(
            [
                ("kernel_id", np.uint32),
                ("record_size_bytes", np.uint32),
                ("args", self.args_dtype),
            ],
            align=True,
        )

    def records(
        self, arguments: Mapping[str, Any], eager_specials: Any
    ) -> list[dict[str, Any]] | None:
        """
        Return the field values of the records of one `Specials` call.

        Parameters
        ----------
        arguments
            The call's arguments by name, defaults applied.
        eager_specials
            The eager specials class, for backend-specific helpers.

        Returns
        -------
        list or None
            One ``{field: value}`` per record, or None to run the call
            eagerly.
        """
        if self.build_records is not None:
            return self.build_records(arguments, eager_specials)
        return [{name: arguments[name] for name in self.args_dtype.names}]

    def pack(
        self, record: Any, values: Mapping[str, Any], keep_alive: list
    ) -> None:
        """
        Write one record into ``record``, a scalar of `record_dtype`.

        Parameters
        ----------
        record
            Where to write the record.
        values
            ``{field: value}`` of the ``Args`` struct, as from `records`.
        keep_alive
            Arrays the record points at, to hold until the flush.
        """
        assert set(values) == set(self.args_dtype.names), values.keys()
        record["kernel_id"] = self.kernel_id
        record["record_size_bytes"] = self.record_dtype.itemsize
        args = record["args"]
        for name, value in values.items():
            field_dtype = self.args_dtype.fields[name][0]
            if field_dtype.subdtype is not None:
                # Reading a device array here would sync on every call.
                assert not is_cupy_array(value), f"`{name}` must be on host"
                n_values = len(value)
                args[name][:n_values] = value
                args[name][n_values:] = 0.0
            elif field_dtype == _POINTER:
                assert value.dtype == _REAL and value.flags.c_contiguous
                args[name] = address_of(value)
                keep_alive.append(value)
            else:
                args[name] = value


def _kick_multi_harmonic_records(
    arguments: Mapping[str, Any], eager_specials: Any
) -> list[dict[str, Any]]:
    # Splits the harmonics over records of MAX_RF_HARMONICS_PER_RECORD.
    # `acceleration_kick` goes into the last record only, and one record
    # is always queued, so `n_rf == 0` still applies it -- as
    # `CudaSpecials.kick_multi_harmonic` splits its launches.
    n_rf = int(arguments["n_rf"])
    voltage = arguments["voltage"]
    omega_rf = arguments["omega_rf"]
    phi_rf = arguments["phi_rf"]
    assert len(voltage) == len(omega_rf) == len(phi_rf) == n_rf
    records = []
    for first in range(0, max(n_rf, 1), MAX_RF_HARMONICS_PER_RECORD):
        last = min(first + MAX_RF_HARMONICS_PER_RECORD, n_rf)
        records.append(
            {
                "n_rf": last - first,
                "voltage": voltage[first:last],
                "omega_rf": omega_rf[first:last],
                "phi_rf": phi_rf[first:last],
                "charge": arguments["charge"],
                "acceleration_kick": (
                    arguments["acceleration_kick"] if last == n_rf else 0.0
                ),
            }
        )
    return records


def _drift_exact_records(
    arguments: Mapping[str, Any], eager_specials: Any
) -> list[dict[str, Any]] | None:
    # The polynomial cannot be split over records like harmonics, so more
    # than MAX_HIGHER_ALPHA coefficients run eagerly. A device
    # `higher_alpha` also runs eagerly: `CudaSpecials.drift_exact` accepts
    # it as a compatibility path, but inlining it into the record would
    # read device memory as a host buffer.
    higher_alpha = arguments["higher_alpha"]
    if len(higher_alpha) > MAX_HIGHER_ALPHA or is_cupy_array(higher_alpha):
        return None
    return [
        {
            "T": arguments["T"],
            "alpha_0": arguments["alpha_0"],
            "beta": arguments["beta"],
            "energy": arguments["energy"],
            "n_alpha": len(higher_alpha),
            "higher_alpha": higher_alpha,
        }
    ]


def _kick_interpolated_records(
    arguments: Mapping[str, Any], eager_specials: Any
) -> list[dict[str, Any]] | None:
    # Sparse profiles run eagerly. The voltage-kick table touches no
    # particles, so building it when the call is queued is exact; it is a
    # fresh array nobody writes before the flush. Its layout is
    # `[first_bin_center, inverse_bin_width, (slope, offset) * n_bins]`
    # with `charge` and `acceleration_kick` folded into the pairs.
    if arguments["first_left_cut"] is not None:
        return None
    table = eager_specials._build_voltage_kick_table(
        voltage=arguments["voltage"],
        bin_centers=arguments["bin_centers"],
        charge=arguments["charge"],
        acceleration_kick=arguments["acceleration_kick"],
    )
    return [
        {
            "voltage_kick_table": table,
            "voltage_kick_table_length": table.size,
            "acceleration_kick": arguments["acceleration_kick"],
        }
    ]


_DRIFT_FIELDS = [
    ("T", _REAL),
    ("eta_0", _REAL),
    ("beta", _REAL),
    ("energy", _REAL),
]

DEFERRABLE_KERNELS: tuple[DeferrableKernel, ...] = (
    DeferrableKernel(
        kernel_id=0,
        specials_method="kick_single_harmonic",
        args_dtype=np.dtype(
            [
                ("voltage", _REAL),
                ("omega_rf", _REAL),
                ("phi_rf", _REAL),
                ("charge", _REAL),
                ("acceleration_kick", _REAL),
            ],
            align=True,
        ),
        writes_dt=False,
        writes_dE=True,
    ),
    DeferrableKernel(
        kernel_id=1,
        specials_method="kick_multi_harmonic",
        args_dtype=np.dtype(
            [
                ("n_rf", np.int32),
                ("voltage", _REAL, MAX_RF_HARMONICS_PER_RECORD),
                ("omega_rf", _REAL, MAX_RF_HARMONICS_PER_RECORD),
                ("phi_rf", _REAL, MAX_RF_HARMONICS_PER_RECORD),
                ("charge", _REAL),
                ("acceleration_kick", _REAL),
            ],
            align=True,
        ),
        writes_dt=False,
        writes_dE=True,
        build_records=_kick_multi_harmonic_records,
    ),
    DeferrableKernel(
        kernel_id=2,
        specials_method="drift_simple",
        args_dtype=np.dtype(_DRIFT_FIELDS, align=True),
        writes_dt=True,
        writes_dE=False,
    ),
    DeferrableKernel(
        kernel_id=3,
        specials_method="drift_like_line_segment",
        args_dtype=np.dtype(_DRIFT_FIELDS, align=True),
        writes_dt=True,
        writes_dE=False,
    ),
    DeferrableKernel(
        kernel_id=4,
        specials_method="drift_exact",
        args_dtype=np.dtype(
            [
                ("T", _REAL),
                ("alpha_0", _REAL),
                ("beta", _REAL),
                ("energy", _REAL),
                ("n_alpha", np.int32),
                ("higher_alpha", _REAL, MAX_HIGHER_ALPHA),
            ],
            align=True,
        ),
        writes_dt=True,
        writes_dE=False,
        build_records=_drift_exact_records,
    ),
    DeferrableKernel(
        kernel_id=5,
        specials_method="kick_interpolated",
        args_dtype=np.dtype(
            [
                ("voltage_kick_table", _POINTER),
                ("voltage_kick_table_length", INDEX_DTYPE),
                ("acceleration_kick", _REAL),
            ],
            align=True,
        ),
        writes_dt=False,
        writes_dE=True,
        build_records=_kick_interpolated_records,
    ),
)
KERNELS_BY_SPECIALS_METHOD = {
    kernel.specials_method: kernel for kernel in DEFERRABLE_KERNELS
}


def header_digest() -> str:
    """
    Return the SHA-256 of ``kernel_call_records.h``.

    Folded into the compiled-library cache keys of the cpp and cuda
    backends, whose own source hashes do not cover this folder.

    Returns
    -------
    str
        Hex digest of ``kernel_call_records.h``.
    """
    with open(HEADER_PATH, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()
