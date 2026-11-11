# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


"""Helper module to work with CPU/GPU arrays distributed via MPI."""

from __future__ import annotations

from math import inf, sqrt
from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import backend
from blond.core.backends.mpi_distributed.callables import (
    global_reduce,
    global_sum_array,
)
from blond.core.backends.mpi_distributed.helpers import MPI
from blond.generals.cupy_.no_cupy_import import copy_to_cpu, is_cupy_array
from blond.generals.exceptions_ import ArrayPrecisionError

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


class DistributedArray:
    """
    Initialize a DistributedArray.

    This class is intended to hold the beam dE and dt coordinates,
    which are arrays with the size up to some TB.

    Parameters
    ----------
    array
        The local array data for this process.
    """

    def __init__(self, array: NumpyArray | CupyArray):
        self.array_local = array
        if MPI is None:
            self._comm = None
            # Determine rank and size
            self._rank = 0
            self._size = 1
            self._is_distributed = False
        else:
            self._comm = MPI.COMM_WORLD

            # Determine rank and size
            self._rank = self._comm.Get_rank()
            self._size = self._comm.Get_size()
            self._is_distributed = self._size > 1

        self._histogram_local_cache: dict[int, NumpyArray | CupyArray] = {}

    @property
    def array_local(self) -> NumpyArray | CupyArray:
        """
        The array data on this rank.

        Replacing it is local to this rank (rank 0 may fill an array
        alone before `mpi_scatter`) and forgets the global size, so the
        next read of `global_size` is a collective.

        Returns
        -------
        array_local
            The array data on this rank.
        """
        return self._array_local

    @array_local.setter
    def array_local(self, array: NumpyArray | CupyArray | None) -> None:
        self._array_local = array
        self._global_size: int | None = None

    def copy_as_numpy(self) -> NumpyArray:
        """
        Get a copy of the local array, guaranteed to be in the CPU-RAM.

        Returns
        -------
        array_local_cpu
             A copy of the local array, guaranteed to be in the CPU-RAM.
        """
        # just a shortcut
        return copy_to_cpu(self.array_local)

    def copy_as_cupy(self) -> CupyArray:
        """
        Get a copy of the local array, guaranteed to be in the GPU-RAM.

        Returns
        -------
        array_local_gpu
            A copy of the local array, guaranteed to be in the GPU-RAM.
        """
        # just a shortcut

        import cupy as cp  # this will fail if cupy is not available

        return cp.array(self.array_local, copy=True)

    @property
    def is_distributed(self):
        """
        Whether the software runs with a MPI size > 1 or not.

        Returns
        -------
        is_distributed
            Whether the software runs with a MPI size > 1 or not.
        """
        return self._is_distributed

    def mpi_scatter(self) -> None:
        """
        Scatter rank 0's 1D array; each rank keeps its contiguous chunk.

        Only rank 0's array is read; the other ranks' may be None. If
        rank 0's array is None, every rank ends up with None. Chunk
        sizes and order are those of `np.array_split`. The chunks travel
        as host buffers, not pickled, so no GPU holds more than its own
        chunk and a chunk may exceed 2 GiB. Afterwards each rank holds an
        array of the active backend, with the dtype of rank 0's array.
        """
        if not self._is_distributed:
            return

        size = self._comm.Get_size()
        rank = self._comm.Get_rank()

        host_array = None
        header = None
        root_exception: Exception | None = None
        root_error: str | None = None

        if rank == 0:
            # Any failure here must be announced before the transfer
            # below: every other rank is already blocking in it, so
            # raising straight away would deadlock the whole run.
            try:
                array = self.array_local
                host_array = (
                    None
                    if array is None
                    else (
                        copy_to_cpu(array)
                        if is_cupy_array(array)
                        else np.ascontiguousarray(array)
                    )
                )
                if host_array is not None and host_array.ndim != 1:
                    raise ValueError(
                        f"Only 1D arrays can be scattered, "
                        f"got {host_array.ndim}D."
                    )
                if host_array is not None:
                    header = (host_array.size, host_array.dtype.str)
            except Exception as exc:
                root_exception = exc
                root_error = f"{type(exc).__name__}: {exc}"

        header, root_error = self._comm.bcast((header, root_error), root=0)
        if root_error is not None:
            if root_exception is not None:
                raise root_exception
            raise RuntimeError(
                f"Rank 0 failed to prepare the array for `mpi_scatter`, "
                f"so there is nothing to scatter: {root_error}"
            )

        if header is None:
            self.array_local = None
            self._global_size = 0
            return
        n_global, dtype_str = header
        chunk_sizes = [
            n_global // size + (1 if r < n_global % size else 0)
            for r in range(size)
        ]
        if rank == 0:
            start = chunk_sizes[0]
            for dest in range(1, size):
                stop = start + chunk_sizes[dest]
                self._comm.Send(host_array[start:stop], dest=dest)
                start = stop
            chunk = host_array[: chunk_sizes[0]]
        else:
            chunk = np.empty(chunk_sizes[rank], dtype=np.dtype(dtype_str))
            self._comm.Recv(chunk, source=0)
        # `backend.array` copies, so rank 0 drops the global array
        self.array_local = backend.array(chunk, dtype=chunk.dtype)
        self._global_size = n_global

    def mpi_gather(self) -> NumpyArray | CupyArray | None:
        """
        Gather the distributed data and return it as a single array.

        Gather a 1D NumPy array.
        Rank 0 owns the global array after scatter.
        Before scatter, each rank owns its local chunk.

        Returns
        -------
        array | None
            The gathered global array from all processes if ``rank==0``
            else None.
        """
        if self._is_distributed:
            gathered = self._comm.gather(self.array_local, root=0)

            if self._rank != 0:  # pragma: no cover
                return None

            array_global = backend.hstack(gathered)
        else:
            array_global = self.array_local.copy()

        return array_global

    @property
    def _array_or_empty(self) -> NumpyArray | CupyArray:
        """
        `array_local`, or an empty array on a rank holding None.

        Before a root-distributes scatter, the other ranks hold None. In
        a reduction they must take part as empty ranks: raising instead
        would leave the ranks with data waiting in the collective.

        Returns
        -------
        array
            `array_local`, or an empty array if it is None.
        """
        if self.array_local is None:
            return backend.zeros(0, dtype=backend.float)
        return self.array_local

    @property
    def local_size(self) -> int:
        """
        Get the number of elements on this rank.

        A rank holding None has no elements.

        Returns
        -------
        int
            The number of elements on this rank.
        """
        return 0 if self.array_local is None else self.array_local.size

    @property
    def global_size(self) -> int:
        """
        Get the total number of elements across all processes.

        Known without communication after `mpi_scatter` and
        `concatenate`. Otherwise, the first read after `array_local` was
        replaced is a collective, which every rank must make. A rank
        holding None counts as holding no elements.

        Returns
        -------
        int
            The total size of the distributed array across all processes.
        """
        if self._global_size is None:
            self._global_size = int(global_reduce(self.local_size)[0])
        return self._global_size

    def min_max(self) -> tuple[float, float]:
        """
        Compute the global minimum and maximum in one collective.

        A rank without elements (e.g. after all its particles were lost)
        takes part with ``+inf`` and ``-inf``, so the other ranks still get
        the extrema instead of waiting for them forever.

        Returns
        -------
        global_min
            The minimum value across all distributed array chunks.
        global_max
            The maximum value across all distributed array chunks.

        Raises
        ------
        ValueError
            On every rank, if no rank holds any element.
        """
        local = (-inf, -inf)  # negated minimum, so that MAX reduces both
        if self.local_size:
            local = (
                -float(backend.min(self.array_local)),
                float(backend.max(self.array_local)),
            )
        negative_min, global_max = global_reduce(*local, op="MAX")
        if global_max == -inf:  # same on all ranks, so is the raise
            raise ValueError(
                "`min`/`max` of an array without elements on any rank."
            )
        return -float(negative_min), float(global_max)

    def min(self) -> float:
        """
        Compute the global minimum across all processes.

        Returns
        -------
        float
            The minimum value across all distributed array chunks.

        See Also
        --------
        min_max : Both extrema, at the cost of one.
        """
        return self.min_max()[0]

    def max(self) -> float:
        """
        Compute the global maximum across all processes.

        Returns
        -------
        float
            The maximum value across all distributed array chunks.

        See Also
        --------
        min_max : Both extrema, at the cost of one.
        """
        return self.min_max()[1]

    def mean(self) -> float:
        """
        Compute the global mean across all processes.

        Returns
        -------
        float
            The mean value across all distributed array chunks.
        """
        return self.sum() / self.global_size

    def std(self) -> float:
        """
        Compute the global standard deviation across all processes.

        The variance is taken from the values centred on the global mean
        (two passes), which stays accurate when the spread is small
        compared to the mean.

        Returns
        -------
        float
            The standard deviation across all distributed array chunks.
        """
        centred = self._array_or_empty - self.mean()
        # dot(x, x) is a faster sum(x**2)
        local_sum_sq = float(backend.dot(centred, centred))
        return sqrt(global_reduce(local_sum_sq)[0] / self.global_size)

    def sum(self) -> float:
        """
        Compute the global sum across all processes.

        Returns
        -------
        float
            The sum of all values across all distributed array chunks.
        """
        local_sum = float(backend.sum(self._array_or_empty))
        return float(global_reduce(local_sum)[0])

    def histogram(
        self,
        bins,
        range: tuple[float, float] | None = None,  # noqa: A002 (np API)
        out: NumpyArray | CupyArray | None = None,
    ) -> NumpyArray | CupyArray:
        """
        Compute the global histogram across all processes.

        Parameters
        ----------
        bins : int
            The number of histogram bins.
        range : tuple[float, float]
            The (min, max) range for the histogram.
        out
            Array to write the results on.
            This is a performance option to prevent repeated array creating.

        Returns
        -------
        array
            The histogram counts across all distributed array chunks.
        """
        # Compute or retrieve local histogram
        if out is None:
            if bins not in self._histogram_local_cache:
                self._histogram_local_cache[bins] = backend.zeros(
                    bins, backend.float
                )
            array_write_local = self._histogram_local_cache[bins]
        else:
            assert out.dtype == backend.float
            array_write_local = out

        hist_range = self.min_max() if range is None else range

        backend.specials.histogram(
            array_read=self._array_or_empty,
            array_write=array_write_local,
            start=hist_range[0],
            stop=hist_range[1],
        )

        # Combine histograms from all processes
        return global_sum_array(array_write_local)

    def histogram_sparse(
        self,
        out: NumpyArray,
        first_left_cut: float,
        left_cut_distance: float,
        cut_width: float,
        bins_per_profile: int,
        n_active_profiles: int,
        filling_pattern: NumpyArray,
        bucket_index_to_memory_index: NumpyArray,
    ):
        """
        Compute the global histogram across all processes.

        Parameters
        ----------
        out
            Output histogram ``(n_filled_buckets * bins_per_profile)``.
        first_left_cut
            Start of the first histogram.
        left_cut_distance
            Distance between the start of each histogram.
        cut_width
            Distance between left and right edge of the histogram.
        bins_per_profile
            Number of bins per bucket.
        n_active_profiles
            Number of non-empty buckets.
        filling_pattern
            Filling pattern as a boolean array
            where ``True`` means filled bucket.
        bucket_index_to_memory_index
            Maps bucket index to memory index.
            For a ``filling_pattern = [1, 0, 0, 1]``
            ``bucket_index_to_memory_index = [0, 0, 0, 8]`` with
            ``bins_per_profile = 8``.
            Use `_gen_array_bucket_index_to_memory_index` to generate this.

        Returns
        -------
        array
            The histogram counts across all distributed array chunks.
        """
        # Compute or retrieve local histogram
        assert out.dtype == backend.float
        array_write_local = out

        backend.specials.histogram_sparse(
            x=self._array_or_empty,
            out=array_write_local,
            first_left_cut=first_left_cut,
            left_cut_distance=left_cut_distance,
            cut_width=cut_width,
            bins_per_profile=bins_per_profile,
            n_active_profiles=n_active_profiles,
            filling_pattern=filling_pattern,
            bucket_index_to_memory_index=bucket_index_to_memory_index,
        )

        # Combine histograms from all processes
        return global_sum_array(array_write_local)


def concatenate(
    array_1: DistributedArray, array_2: DistributedArray
) -> DistributedArray:
    """
    Concatenate two distributed arrays, return the result.

    Parameters
    ----------
    array_1
        The first array.
    array_2
        The second array, will be concatenated to the end of the first.

    Returns
    -------
    concatenated array
        The concatenated array.

    Raises
    ------
    RuntimeError
        Raised if the `is_distributed` flags of the two arrays do not
        match.
    ArrayPrecisionError
        Raised if the ``dtype`` values of the local arrays do not match.
    TypeError
        Raised if the ``type`` values of the local arrays do not match.
    """
    # Check both distributed, mismatch probably not possible
    if array_1.is_distributed != array_2.is_distributed:  # pragma: no cover
        raise RuntimeError(
            "Distributed arrays can only be joined if both"
            "or neither are distributed:\n"
            f"First distributed: {array_1.is_distributed}\n"
            f"Second distributed: {array_2.is_distributed}"
        )

    # Check same dtypes
    if array_1.array_local.dtype != array_2.array_local.dtype:
        raise ArrayPrecisionError(
            "Cannot concatenate arrays of different dtype:\n"
            f"First dtype: {array_1.array_local.dtype}\n"
            f"Second dtype: {array_2.array_local.dtype}"
        )

    # Check same array type
    if type(array_1.array_local) is not type(array_2.array_local):
        raise TypeError(
            "Cannot concatenate arrays of different types:\n"
            f"First type: {type(array_1.array_local)}\n"
            f"Second type: {type(array_2.array_local)}"
        )

    joined = DistributedArray(
        backend.concatenate((array_1.array_local, array_2.array_local))
    )
    # only sizes known already: a reduction here might run on one rank
    if array_1._global_size is not None and array_2._global_size is not None:
        joined._global_size = array_1._global_size + array_2._global_size
    return joined
