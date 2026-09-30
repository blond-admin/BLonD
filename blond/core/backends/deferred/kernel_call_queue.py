# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Queue of deferred kernel calls, shared by ``cpp_deferred``/``cuda_deferred``.

`make_deferred_specials` derives deferred specials from eager ones: each
deferrable kernel (`KERNEL_CALL_ARGS`) packs a kernel call record into
a per-thread `KernelCallQueue` instead of running; every other method
flushes the queue and then runs eagerly. A flush hands the batch to the
backend's ``execute_batch``, which applies it in one fused pass.
"""

from __future__ import annotations

import functools
import inspect
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from blond.core.backends.backend import Specials, backend
from blond.core.backends.deferred.kernel_call_records import (
    ARGS_BY_SPECIALS_METHOD,
    KernelCallArgs,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

_INITIAL_CAPACITY_BYTES = 4096


class KernelCallQueue(threading.local):
    """
    Kernel call records of one Python thread, waiting for the next flush.

    Per thread because one specials object serves every simulation in the
    process, and simulations may run in separate Python threads (ctypes
    releases the GIL). The queue is bound to one beam's ``dt``/``dE``.
    """

    def __init__(self) -> None:
        self.buffer = np.zeros(_INITIAL_CAPACITY_BYTES, dtype=np.uint8)
        self.n_bytes = 0
        self.args_types: list[type[KernelCallArgs]] = []
        self.record_sizes: list[int] = []
        self.keep_alive: list[Any] = []
        self.dt: Any = None
        self.dE: Any = None

    def holds(self, dt: Any, dE: Any) -> bool:
        """
        Return whether the queued records belong to this beam.

        Parameters
        ----------
        dt, dE
            Beam coordinate arrays of the next call.

        Returns
        -------
        bool
            True for the very same array objects.
        """
        return self.dt is dt and self.dE is dE

    def bind(self, dt: Any, dE: Any) -> None:
        """
        Bind the empty queue to a beam.

        Parameters
        ----------
        dt, dE
            Beam coordinate arrays all queued records will act on.
        """
        assert self.n_bytes == 0, "bind only an empty queue"
        assert dt.dtype == backend.float and dE.dtype == backend.float
        assert dt.flags.c_contiguous and dE.flags.c_contiguous
        self.dt, self.dE = dt, dE

    def append(self, args: KernelCallArgs) -> None:
        """
        Pack one kernel call record at the end of the batch.

        Parameters
        ----------
        args
            The kernel call; its class fixes the record layout, its
            trailing array (if any) the record size.
        """
        size = args.record_size_bytes()
        self._reserve(size)
        args.pack_into(
            self.buffer[self.n_bytes : self.n_bytes + size], self.keep_alive
        )
        self.n_bytes += size
        self.args_types.append(type(args))
        self.record_sizes.append(size)

    def clear(self) -> None:
        """Drop all records and array references; keep the buffer."""
        self.n_bytes = 0
        self.args_types = []
        self.record_sizes = []
        self.keep_alive = []
        self.dt = self.dE = None

    def _reserve(self, size: int) -> None:
        needed = self.n_bytes + size
        if needed > self.buffer.size:
            grown = np.zeros(max(2 * self.buffer.size, needed), np.uint8)
            grown[: self.n_bytes] = self.buffer[: self.n_bytes]
            self.buffer = grown


def make_deferred_specials(
    eager_specials: type,
    execute_batch: Callable[
        [np.ndarray, list[type[KernelCallArgs]], list[int], Any, Any], None
    ],
) -> type:
    """
    Derive deferred specials from eager ones.

    Parameters
    ----------
    eager_specials
        The eager specials class, e.g. ``CppSpecials``.
    execute_batch
        ``execute_batch(batch, args_types, record_sizes, dt, dE)``
        applying the batch bytes, in one fused pass where possible, to
        the beam; ``args_types`` and ``record_sizes`` hold the ``Args``
        class and byte size of every record.

    Returns
    -------
    type
        A subclass of ``eager_specials`` named ``Deferred<name>``.
    """
    queue = KernelCallQueue()
    namespace: dict[str, Any] = {"kernel_call_queue": queue}

    # `deferred_class` is created below; `flush` only runs once it exists,
    # and looks `_execute_batch` up at call time so it can be replaced.
    def flush() -> None:
        """Run all queued kernel calls of this thread."""
        if queue.n_bytes == 0:
            return
        batch = queue.buffer[: queue.n_bytes]
        try:
            deferred_class._execute_batch(
                batch, queue.args_types, queue.record_sizes, queue.dt, queue.dE
            )
        finally:
            queue.clear()  # never re-apply a batch, even after an error

    namespace["flush"] = staticmethod(flush)
    namespace["_execute_batch"] = staticmethod(execute_batch)

    for name, value in vars(Specials).items():
        if name.startswith("_") or name == "flush":
            continue
        if not isinstance(value, staticmethod):
            continue
        eager_method = getattr(eager_specials, name)
        if name in ARGS_BY_SPECIALS_METHOD:
            method = _queuing_method(
                ARGS_BY_SPECIALS_METHOD[name],
                eager_method,
                eager_specials,
                queue,
                flush,
            )
        else:
            method = _flushing_method(eager_method, flush)
        namespace[name] = staticmethod(method)

    deferred_class = type(
        f"Deferred{eager_specials.__name__}", (eager_specials,), namespace
    )
    return deferred_class


def _flushing_method(eager_method: Callable, flush: Callable) -> Callable:
    @functools.wraps(eager_method)
    def flush_then_call(*args: Any, **kwargs: Any) -> Any:
        flush()
        return eager_method(*args, **kwargs)

    return flush_then_call


def _queuing_method(
    args_type: type[KernelCallArgs],
    eager_method: Callable,
    eager_specials: type,
    queue: KernelCallQueue,
    flush: Callable,
) -> Callable:
    signature = inspect.signature(eager_method)

    @functools.wraps(eager_method)
    def queue_kernel_call(*args: Any, **kwargs: Any) -> None:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        kernel_calls = args_type.from_specials_call(arguments, eager_specials)
        if kernel_calls is None:  # this call must run eagerly
            flush()
            eager_method(*args, **kwargs)
            return
        dt, dE = arguments["dt"], arguments["dE"]
        if not queue.holds(dt, dE):
            flush()
            queue.bind(dt, dE)
        for kernel_call in kernel_calls:
            queue.append(kernel_call)

    return queue_kernel_call
