# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Beam coordinate storage that runs queued (deferred) kernel calls on access.

With ``cpp_deferred``/``cuda_deferred`` specials, per-particle kernels are
queued and only applied at the next flush. Instead of every Beam method
having to remember a flush, the Beam stores its coordinates in a
`FlushingDistributedArray`, whose data access flushes by itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from blond.core.backends.backend import backend
from blond.core.backends.mpi_distributed.distributed_array import (
    MPI,
    DistributedArray,
)

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


class FlushingDistributedArray(DistributedArray):
    """
    `DistributedArray` that runs queued kernel calls before data access.

    Reading or replacing `array_local` first flushes the active specials,
    so every inherited method (``min``, ``mean``, ``histogram``,
    ``mpi_gather``, ...) and every copy sees up-to-date data. Sizes do not
    flush: queued kernels never change the number of particles.
    Constructed like `DistributedArray`, from the local array.
    """

    @classmethod
    def from_distributed_array(
        cls, distributed: DistributedArray
    ) -> FlushingDistributedArray:
        """
        Wrap the local array of ``distributed`` without copying it.

        Parameters
        ----------
        distributed
            Plain distributed array to convert.

        Returns
        -------
        FlushingDistributedArray
            Flushing view on the same local array.
        """
        return cls(distributed.array_local)

    @property
    def array_local(self) -> NumpyArray | CupyArray:
        """
        The local array, after queued kernel calls have run.

        Returns
        -------
        array_local
            The local array data for this process.
        """
        backend.specials.flush()
        return self._array_local

    @array_local.setter
    def array_local(self, array: NumpyArray | CupyArray) -> None:
        # Queued kernel calls reference the old array: run them before
        # it is dropped.
        backend.specials.flush()
        self._array_local = array

    @property
    def array_local_without_flush(self) -> NumpyArray | CupyArray:
        """
        The local array, without running queued kernel calls.

        Only for `BeamBaseClass.kernel_call_dt` / `kernel_call_dE`, which
        pass the array to a kernel that may itself be queued.

        Returns
        -------
        array_local
            The local array data for this process, possibly stale.
        """
        return self._array_local

    @property
    def local_size(self) -> int:
        """
        Get the number of elements on the local process, without a flush.

        Returns
        -------
        int
            The number of elements on the local process.
        """
        return self._array_local.size

    @property
    def global_size(self) -> int:
        """
        Get the total number of elements across all processes.

        Returns
        -------
        int
            The total size of the distributed array across all processes.
        """
        local_size = self._array_local.size
        if self._is_distributed:
            return self._comm.allreduce(local_size, op=MPI.SUM)
        return local_size

    def __getstate__(self) -> dict[str, Any]:
        """
        Flush before ``copy``, ``deepcopy`` or pickling read the state.

        Returns
        -------
        dict
            The instance attributes.
        """
        backend.specials.flush()
        return self.__dict__


class FlushingCoordinates:
    """
    Beam attribute that always holds a `FlushingDistributedArray`.

    Assigning a plain `DistributedArray` wraps it; assigning ``None``
    marks the coordinate as not set up. Queued kernel calls run before the
    old array is replaced, since they reference it.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        """
        Store the value under ``<name>_storage`` in the instance.

        Parameters
        ----------
        owner
            The class the attribute is defined on.
        name
            The attribute name, e.g. ``_dt``.
        """
        self._storage_name = f"{name}_storage"

    def __get__(
        self, instance: Any, owner: type | None = None
    ) -> FlushingDistributedArray | None | FlushingCoordinates:
        """
        Return the stored coordinates.

        Parameters
        ----------
        instance
            The beam, or ``None`` on class access.
        owner
            The beam class.

        Returns
        -------
        FlushingDistributedArray | None
            The coordinates, or ``None`` if not set up.
        """
        if instance is None:
            return self
        return instance.__dict__.get(self._storage_name)

    def __set__(self, instance: Any, value: DistributedArray | None) -> None:
        """
        Flush, then store ``value`` as a `FlushingDistributedArray`.

        Only flushes when an old array is actually being replaced: the
        first assignment (e.g. `BeamBaseClass.__init__` setting the
        coordinate to ``None``, or setting it up for the first time) has
        nothing queued against it yet, so it must not split a pending
        batch queued for another beam.

        Parameters
        ----------
        instance
            The beam.
        value
            New coordinates, or ``None``.

        Raises
        ------
        TypeError
            If ``value`` is neither a `DistributedArray` nor ``None``.
        """
        if value is not None and not isinstance(value, DistributedArray):
            raise TypeError(
                "Beam coordinates must be a `DistributedArray`, "
                f"got {type(value).__name__}."
            )
        old_value = instance.__dict__.get(self._storage_name)
        if old_value is not None:
            backend.specials.flush()
        if value is not None and not isinstance(
            value, FlushingDistributedArray
        ):
            value = FlushingDistributedArray.from_distributed_array(value)
        instance.__dict__[self._storage_name] = value
