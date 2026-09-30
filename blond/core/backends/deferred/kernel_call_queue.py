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
    KERNEL_CALL_BATCH_CAPACITY_BYTES,
    KernelCallArgs,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

_INITIAL_CAPACITY_BYTES = 4 * 4096
# No record is larger than one CUDA launch (`record_packer` raises
# otherwise), so this much free space fits any record.
_RECORD_HEADROOM_BYTES = KERNEL_CALL_BATCH_CAPACITY_BYTES


class KernelCallQueue(threading.local):
    """
    Kernel call records of one Python thread, waiting for the next flush.

    Per thread because one specials object serves every simulation in the
    process, and simulations may run in separate Python threads (ctypes
    releases the GIL). The queue is bound to one beam's ``dt``/``dE``.
    """

    def __init__(self) -> None:
        self.buffer = np.zeros(_INITIAL_CAPACITY_BYTES, dtype=np.uint8)
        self.view = memoryview(self.buffer)
        # Records are appended while n_bytes <= packing_limit.
        self.packing_limit = self.buffer.size - _RECORD_HEADROOM_BYTES
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
        args_type = type(args)
        self.append_record(
            args_type,
            args_type.record_packer(),
            *(getattr(args, name) for name, _ in args_type.record_fields()),
        )

    def append_record(
        self,
        args_type: type[KernelCallArgs],
        packer: Callable[..., int],
        *field_values: Any,
    ) -> None:
        """
        Pack one kernel call record from its field values.

        The hot path of the deferred specials: no dataclass instance.

        Parameters
        ----------
        args_type
            The kernel's ``Args`` class.
        packer
            Its `KernelCallArgs.record_packer`.
        *field_values
            The ``Args`` fields in order.
        """
        offset = self.n_bytes
        if offset > self.packing_limit:
            self._grow()
        size = packer(self.view, offset, self.keep_alive, *field_values)
        self.n_bytes = offset + size
        self.args_types.append(args_type)
        self.record_sizes.append(size)

    def clear(self) -> None:
        """Drop all records and array references; keep the buffer."""
        self.n_bytes = 0
        self.args_types = []
        self.record_sizes = []
        self.keep_alive = []
        self.dt = self.dE = None

    def _grow(self) -> None:
        grown = np.zeros(2 * self.buffer.size, np.uint8)
        grown[: self.n_bytes] = self.buffer[: self.n_bytes]
        self.buffer = grown
        self.view = memoryview(grown)
        self.packing_limit = grown.size - _RECORD_HEADROOM_BYTES


def make_deferred_specials(
    eager_specials: type,
    execute_batch: Callable[
        [np.ndarray, int, list[type[KernelCallArgs]], list[int], Any, Any],
        None,
    ],
) -> type:
    """
    Derive deferred specials from eager ones.

    Parameters
    ----------
    eager_specials
        The eager specials class, e.g. ``CppSpecials``.
    execute_batch
        ``execute_batch(buffer, n_bytes, args_types, record_sizes, dt,
        dE)`` applying the first ``n_bytes`` of the ``uint8`` array
        ``buffer``, in one fused pass where possible, to the beam;
        ``args_types`` and ``record_sizes`` hold the ``Args`` class and
        byte size of every record. ``buffer`` is the queue's own array,
        the same object from flush to flush until it grows, so a cached
        pointer to it stays valid.

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
        n_bytes = queue.n_bytes
        if n_bytes == 0:
            return
        try:
            deferred_class._execute_batch(
                queue.buffer,
                n_bytes,
                queue.args_types,
                queue.record_sizes,
                queue.dt,
                queue.dE,
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
    """
    Generate the method queuing ``args_type`` records.

    Generated Python source, like ``dataclasses`` generates ``__init__``:
    it has the eager method's parameters, so Python itself binds the
    arguments (no `inspect.Signature.bind` per call), and hands them to
    the `KernelCallArgs.record_packer` directly. A kernel with its own
    `KernelCallArgs.field_values_from_specials_call` gets its arguments
    as a dict first.

    Parameters
    ----------
    args_type
        The kernel's ``Args`` class.
    eager_method
        The eager specials method; it runs calls the records cannot
        hold, and gives the signature.
    eager_specials
        The eager specials class, for backend-specific helpers.
    queue
        The per-thread queue.
    flush
        Runs the queue.

    Returns
    -------
    Callable
        The queuing method, with the eager method's metadata.

    Raises
    ------
    TypeError
        If the eager method takes other than plain positional-or-keyword
        parameters.
    """
    parameters = inspect.signature(eager_method).parameters
    names = list(parameters)
    if (
        any(
            parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in parameters.values()
        )
        or any(name.startswith("_") for name in names)
        or names[:2] != ["dt", "dE"]
    ):
        raise TypeError(
            f"cannot defer {eager_method.__qualname__}{parameters}: it "
            "must take dt, dE, then plain positional-or-keyword parameters"
        )

    def rebind(dt: Any, dE: Any) -> None:
        flush()
        queue.bind(dt, dE)

    namespace = {
        "_queue": queue,
        "_rebind": rebind,
        "_append": queue.append_record,
        "_args_type": args_type,
        "_packer": args_type.record_packer(),
        "_field_values": args_type.field_values_from_specials_call,
        "_eager_specials": eager_specials,
        "_eager_method": eager_method,
        "_flush": flush,
        "_defaults": {
            name: parameter.default
            for name, parameter in parameters.items()
            if parameter.default is not inspect.Parameter.empty
        },
    }
    signature = ", ".join(
        f"{name}=_defaults[{name!r}]"
        if name in namespace["_defaults"]
        else name
        for name in names
    )
    by_name = ", ".join(f"{name}={name}" for name in names)
    rebind_if_other_beam = [
        "    if _queue.dt is not dt or _queue.dE is not dE:",
        "        _rebind(dt, dE)",
    ]
    default_hook = KernelCallArgs.field_values_from_specials_call.__func__
    if args_type.field_values_from_specials_call.__func__ is default_hook:
        field_names = [name for name, _ in args_type.record_fields()]
        body = [
            *rebind_if_other_beam,
            f"    _append(_args_type, _packer, {', '.join(field_names)})",
        ]
    else:
        body = [
            "    _records = _field_values(",
            f"        dict({by_name}), _eager_specials",
            "    )",
            "    if _records is None:  # this call must run eagerly",
            "        _flush()",
            f"        return _eager_method({by_name})",
            *rebind_if_other_beam,
            "    for _values in _records:",
            "        _append(_args_type, _packer, *_values)",
        ]
    name = eager_method.__name__
    source = "\n".join([f"def {name}({signature}):", *body])
    exec(source, namespace)  # noqa: S102
    return functools.update_wrapper(namespace[name], eager_method)
