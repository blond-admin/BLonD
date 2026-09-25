# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Deferred C++ specials: queue per-particle kernels, run them chunked.

The eager kernels each stream the whole beam through memory. Here the
per-particle kernels that ``deferred.cpp`` lists are only queued; when a
result is needed the queue is flushed to ``deferred_execute``, which runs
all queued kernels on one cache-sized chunk of the beam before moving to
the next chunk.

The queue is flushed

- by a queued ``histogram`` of the queued beam, which is the last op of
  the flush, as its result is needed right away,
- before any other kernel runs (it may read what the queue writes),
- before a kernel on another beam is queued,
- by an explicit `flush`, e.g. at the end of the main loop.

Every Python thread has its own queue, so simulations in separate threads
do not mix. Python code that reads or writes ``dt``/``dE`` directly between
two flushes sees stale data; nothing guards against that yet.

Parameters are packed by the names the C++ op publishes (``scalars()`` and
``arrays()`` in ``particle_ops.h``), which are the argument names of the
`Specials` method. Their order is therefore only written down in C++.
"""

from __future__ import annotations

import ctypes as ct
import os
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from blond.core.backends.backend import Specials

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from ctypes import CDLL

    from numpy.typing import NDArray as NumpyArray

#: Particles per chunk, per thread. Chunks of ``dt`` and ``dE`` must stay
#: in the core's cache while every queued kernel runs on them.
DEFAULT_CHUNK_SIZE = int(os.environ.get("BLOND_DEFERRED_CHUNK_SIZE", "4096"))

#: Suffix marking an array the op writes; it is not copied when queued.
_OUTPUT_MARK = ":out"


class _CompiledOp:
    """
    An op of the compiled library: its id and parameter names.

    Parameters
    ----------
    libblond
        The loaded C++ library.
    op_id
        Index of the op in ``BLOND_PARTICLE_OPS``.
    """

    def __init__(self, libblond: CDLL, op_id: int) -> None:
        self.op_id = op_id
        self.scalars = tuple(
            libblond.deferred_op_scalars(ct.c_int(op_id)).decode().split()
        )
        arrays = libblond.deferred_op_arrays(ct.c_int(op_id)).decode()
        # (name, copy when queued)
        self.arrays = tuple(
            (name.removesuffix(_OUTPUT_MARK), not name.endswith(_OUTPUT_MARK))
            for name in arrays.split()
        )


def _load_compiled_ops(libblond: CDLL) -> dict[str, _CompiledOp]:
    for function in (
        libblond.deferred_op_name,
        libblond.deferred_op_scalars,
        libblond.deferred_op_arrays,
    ):
        function.restype = ct.c_char_p
    libblond.deferred_execute.restype = ct.c_int
    return {
        libblond.deferred_op_name(ct.c_int(op_id)).decode(): _CompiledOp(
            libblond, op_id
        )
        for op_id in range(libblond.deferred_n_ops())
    }


def make_deferred_specials(  # NOQA: PLR0915
    cpp_specials: type[Specials],
    libblond: CDLL,
    get_pointer: Callable[[NumpyArray], ct.c_void_p],
    floattype: type[np.float64],
) -> type[Specials]:
    """
    Build the deferred variant of `cpp_specials`.

    Parameters
    ----------
    cpp_specials
        The eager `CppSpecials`, whose kernels run everything not queued.
    libblond
        The loaded C++ library holding ``deferred_execute``.
    get_pointer
        The cached array-to-pointer helper of `cpp_specials`.
    floattype
        Float type of the particle coordinates.

    Returns
    -------
    DeferredCppSpecials
        Subclass of `cpp_specials` with a queue per thread.
    """
    compiled_ops = _load_compiled_ops(libblond)
    settings = {"chunk_size": DEFAULT_CHUNK_SIZE}

    class _Queue(threading.local):
        """Kernels of this thread waiting for a flush, on one beam."""

        def __init__(self) -> None:
            self.clear()

        def clear(self) -> None:
            self.dt: NumpyArray | None = None
            self.dE: NumpyArray | None = None
            self.op_ids: list[int] = []
            self.scalars: list[float] = []
            self.pointers: list[int] = []
            # arrays the queued pointers point into, kept alive
            self.keep_alive: list[NumpyArray] = []

        def holds(self, dt: NumpyArray, dE: NumpyArray) -> bool:
            """
            Whether ops on `dt`/`dE` may join the queue.

            Parameters
            ----------
            dt
                Time coordinates the op works on.
            dE
                Energy coordinates the op works on.

            Returns
            -------
            bool
                True if the queue is empty or holds the same arrays.
            """
            return self.dt is None or (
                _address(dt) == _address(self.dt)
                and _address(dE) == _address(self.dE)
                and len(dt) == len(self.dt)
            )

    queue = _Queue()

    def _address(array: NumpyArray) -> int:
        return get_pointer(array).value

    def _check(*arrays: NumpyArray) -> None:
        # same validation as the eager wrappers, stripped by `python -O`
        for array in arrays:
            assert array.dtype == floattype, f"{array.dtype=}"
            assert array.flags.c_contiguous

    def _enqueue(name: str, dt: NumpyArray, dE: NumpyArray, **params: Any):
        """
        Queue op `name` on `dt`/`dE`, packing `params` by name.

        Parameters
        ----------
        name
            The `Specials` method the op implements.
        dt
            Time coordinates the op works on.
        dE
            Energy coordinates the op works on.
        **params
            Every parameter the C++ op names, by name.
        """
        if not queue.holds(dt, dE):
            DeferredCppSpecials.flush()
        if queue.dt is None:
            queue.dt, queue.dE = dt, dE
        op = compiled_ops[name]
        queue.op_ids.append(op.op_id)
        queue.scalars.extend([params[scalar] for scalar in op.scalars])
        for array_name, copy in op.arrays:
            array = params[array_name]
            if copy:
                # the caller may overwrite its buffer before the queue runs
                array = np.array(array, dtype=floattype, copy=True)
            queue.pointers.append(_address(array))
            queue.keep_alive.append(array)

    def _flushing(name: str) -> staticmethod:
        eager = getattr(cpp_specials, name)

        def flush_then_run(*args, **kwargs):
            DeferredCppSpecials.flush()
            return eager(*args, **kwargs)

        flush_then_run.__name__ = name
        flush_then_run.__doc__ = eager.__doc__
        return staticmethod(flush_then_run)

    class DeferredCppSpecials(cpp_specials):
        """`CppSpecials` that queue per-particle kernels (see module)."""

        @staticmethod
        def flush() -> None:
            """Run all kernels this thread queued, chunk by chunk."""
            if not queue.op_ids:
                return
            op_ids = np.array(queue.op_ids, dtype=np.int32)
            scalars = np.array(queue.scalars, dtype=np.float64)
            pointers = np.array(queue.pointers, dtype=np.uintp)
            dt, dE = queue.dt, queue.dE
            status = libblond.deferred_execute(
                get_pointer(dt),
                get_pointer(dE),
                ct.c_int64(len(dt)),
                ct.c_int(len(op_ids)),
                ct.c_void_p(op_ids.ctypes.data),
                ct.c_void_p(scalars.ctypes.data),
                ct.c_void_p(pointers.ctypes.data),
                ct.c_int64(settings["chunk_size"]),
            )
            queue.clear()
            if status != 0:  # pragma: no cover - ids come from the library
                raise RuntimeError("deferred_execute: unknown op id.")

        @staticmethod
        def n_pending() -> int:
            """
            Return the number of kernels this thread queued.

            Returns
            -------
            int
                Kernels waiting for the next flush.
            """
            return len(queue.op_ids)

        @staticmethod
        def get_chunk_size() -> int:
            """
            Return the particles per chunk and thread.

            Returns
            -------
            int
                The chunk size `flush` uses.
            """
            return settings["chunk_size"]

        @staticmethod
        def set_chunk_size(chunk_size: int) -> None:
            """
            Set the particles per chunk and thread; flushes first.

            Parameters
            ----------
            chunk_size
                Particles per chunk and thread, at least 1.
            """
            if chunk_size < 1:
                raise ValueError(f"{chunk_size=} must be positive.")
            DeferredCppSpecials.flush()
            settings["chunk_size"] = int(chunk_size)

        @staticmethod
        def compiled_ops() -> dict[str, tuple[str, ...]]:
            """
            Return the queueable ops of the library with their parameters.

            Returns
            -------
            dict[str, tuple[str, ...]]
                Parameter names, scalars then arrays, per op name.
            """
            return {
                name: op.scalars + tuple(array for array, _ in op.arrays)
                for name, op in compiled_ops.items()
            }

        @staticmethod
        def kick_single_harmonic(
            dt: NumpyArray,
            dE: NumpyArray,
            voltage: float,
            omega_rf: float,
            phi_rf: float,
            charge: float,
            acceleration_kick: float,
        ) -> None:
            _check(dt, dE)
            _enqueue(
                "kick_single_harmonic",
                dt,
                dE,
                voltage=voltage,
                omega_rf=omega_rf,
                phi_rf=phi_rf,
                charge=charge,
                acceleration_kick=acceleration_kick,
            )

        @staticmethod
        def kick_multi_harmonic(
            dt: NumpyArray,
            dE: NumpyArray,
            voltage: NumpyArray,
            omega_rf: NumpyArray,
            phi_rf: NumpyArray,
            charge: float,
            n_rf: int,
            acceleration_kick: float,
        ) -> None:
            _check(dt, dE, voltage, omega_rf, phi_rf)
            _enqueue(
                "kick_multi_harmonic",
                dt,
                dE,
                voltage=voltage,
                omega_rf=omega_rf,
                phi_rf=phi_rf,
                charge=charge,
                n_rf=n_rf,
                acceleration_kick=acceleration_kick,
            )

        @staticmethod
        def drift_simple(
            dt: NumpyArray,
            dE: NumpyArray,
            T: float,
            eta_0: float,
            beta: float,
            energy: float,
        ) -> None:
            _check(dt, dE)
            _enqueue(
                "drift_simple",
                dt,
                dE,
                T=T,
                eta_0=eta_0,
                beta=beta,
                energy=energy,
            )

        @staticmethod
        def drift_like_line_segment(
            dt: NumpyArray,
            dE: NumpyArray,
            T: float,
            eta_0: float,
            beta: float,
            energy: float,
        ) -> None:
            _check(dt, dE)
            _enqueue(
                "drift_like_line_segment",
                dt,
                dE,
                T=T,
                eta_0=eta_0,
                beta=beta,
                energy=energy,
            )

        @staticmethod
        def kick_interpolated(
            dt: NumpyArray,
            dE: NumpyArray,
            voltage: NumpyArray,
            bin_centers: NumpyArray,
            charge: float,
            acceleration_kick: float,
            first_left_cut: float | None = None,
            left_cut_distance: float | None = None,
            cut_width: float | None = None,
            bins_per_profile: int | None = None,
            filling_pattern: NumpyArray | None = None,
            bucket_index_to_memory_index: NumpyArray | None = None,
        ) -> None:
            if first_left_cut is not None:
                # sparse profiles are not queued
                DeferredCppSpecials.flush()
                cpp_specials.kick_interpolated(
                    dt=dt,
                    dE=dE,
                    voltage=voltage,
                    bin_centers=bin_centers,
                    charge=charge,
                    acceleration_kick=acceleration_kick,
                    first_left_cut=first_left_cut,
                    left_cut_distance=left_cut_distance,
                    cut_width=cut_width,
                    bins_per_profile=bins_per_profile,
                    filling_pattern=filling_pattern,
                    bucket_index_to_memory_index=bucket_index_to_memory_index,
                )
                return
            _check(dt, dE, voltage, bin_centers)
            _enqueue(
                "kick_interpolated",
                dt,
                dE,
                voltage=voltage,
                bin_centers=bin_centers,
                n_bins=len(bin_centers),
                charge=charge,
                acceleration_kick=acceleration_kick,
            )

        @staticmethod
        def histogram(
            array_read: NumpyArray,
            array_write: NumpyArray,
            start: float,
            stop: float,
        ) -> None:
            queued_beam = queue.dt is not None
            reads_dt = queued_beam and queue.holds(array_read, queue.dE)
            reads_dE = queued_beam and queue.holds(queue.dt, array_read)
            if not (reads_dt or reads_dE):
                DeferredCppSpecials.flush()
                cpp_specials.histogram(array_read, array_write, start, stop)
                return
            _check(array_read, array_write)
            _enqueue(
                "histogram",
                queue.dt,
                queue.dE,
                array_write=array_write,
                start=start,
                stop=stop,
                n_bins=len(array_write),
                reads_dE=float(reads_dE),
            )
            # the caller reads the profile next
            DeferredCppSpecials.flush()

    for name in dir(cpp_specials):
        if name.startswith("_") or name == "flush" or name in compiled_ops:
            continue
        if callable(getattr(cpp_specials, name)):
            setattr(DeferredCppSpecials, name, _flushing(name))

    return DeferredCppSpecials
