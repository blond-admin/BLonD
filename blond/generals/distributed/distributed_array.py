# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


"""Helper module to work with CPU/GPU arrays distributed via MPI."""

from __future__ import annotations

import warnings
from math import sqrt
from typing import TYPE_CHECKING

from blond.core.backends.backend import backend
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.generals.exceptions_ import ArrayPrecisionError

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    # Anything that can be combined with the local array element-wise:
    # a scalar, a raw CPU/GPU array, or another `DistributedArray`.
    Operand = "DistributedArray | NumpyArray | CupyArray | float | int"

try:
    from mpi4py import MPI
except Exception as exc:
    warnings.warn(str(exc), ImportWarning, stacklevel=1)
    MPI = None


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

    @staticmethod
    def _unwrap(other: Operand) -> NumpyArray | CupyArray | float | int:
        """
        Return the operand to combine with the local array.

        Unwraps another `DistributedArray` to its local array so that
        arithmetic acts element-wise on the local chunks; scalars and raw
        CPU/GPU arrays are passed through unchanged.

        Parameters
        ----------
        other
            Scalar, raw array or `DistributedArray` to combine with.

        Returns
        -------
        operand
            ``other.array_local`` if `other` is a `DistributedArray`,
            otherwise `other` itself.
        """
        if isinstance(other, DistributedArray):
            return other.array_local
        return other

    def __add__(self, other: Operand) -> DistributedArray:
        """
        Return ``self + other`` as a new `DistributedArray`.

        Parameters
        ----------
        other
            Scalar, raw array or `DistributedArray` to add.

        Returns
        -------
        result
            New `DistributedArray` holding the element-wise sum of the
            local arrays.
        """
        return DistributedArray(self.array_local + self._unwrap(other))

    def __radd__(self, other: Operand) -> DistributedArray:
        """
        Return ``other + self`` as a new `DistributedArray`.

        Parameters
        ----------
        other
            Scalar, raw array or `DistributedArray` to add.

        Returns
        -------
        result
            New `DistributedArray` holding the element-wise sum of the
            local arrays.
        """
        return DistributedArray(self._unwrap(other) + self.array_local)

    def __iadd__(self, other: Operand) -> DistributedArray:
        """
        Add `other` to the local array in place.

        Parameters
        ----------
        other
            Scalar, raw array or `DistributedArray` to add.

        Returns
        -------
        self
            This `DistributedArray` with its local array updated.
        """
        self.array_local += self._unwrap(other)
        return self

    def __sub__(self, other: Operand) -> DistributedArray:
        """
        Return ``self - other`` as a new `DistributedArray`.

        Parameters
        ----------
        other
            Scalar, raw array or `DistributedArray` to subtract.

        Returns
        -------
        result
            New `DistributedArray` holding the element-wise difference of
            the local arrays.
        """
        return DistributedArray(self.array_local - self._unwrap(other))

    def __rsub__(self, other: Operand) -> DistributedArray:
        """
        Return ``other - self`` as a new `DistributedArray`.

        Parameters
        ----------
        other
            Scalar, raw array or `DistributedArray` to subtract from.

        Returns
        -------
        result
            New `DistributedArray` holding the element-wise difference of
            the local arrays.
        """
        return DistributedArray(self._unwrap(other) - self.array_local)

    def __isub__(self, other: Operand) -> DistributedArray:
        """
        Subtract `other` from the local array in place.

        Parameters
        ----------
        other
            Scalar, raw array or `DistributedArray` to subtract.

        Returns
        -------
        self
            This `DistributedArray` with its local array updated.
        """
        self.array_local -= self._unwrap(other)
        return self

    def __mul__(self, other: Operand) -> DistributedArray:
        """
        Return ``self * other`` as a new `DistributedArray`.

        Parameters
        ----------
        other
            Scalar, raw array or `DistributedArray` to multiply by.

        Returns
        -------
        result
            New `DistributedArray` holding the element-wise product of the
            local arrays.
        """
        return DistributedArray(self.array_local * self._unwrap(other))

    def __rmul__(self, other: Operand) -> DistributedArray:
        """
        Return ``other * self`` as a new `DistributedArray`.

        Parameters
        ----------
        other
            Scalar, raw array or `DistributedArray` to multiply by.

        Returns
        -------
        result
            New `DistributedArray` holding the element-wise product of the
            local arrays.
        """
        return DistributedArray(self._unwrap(other) * self.array_local)

    def __imul__(self, other: Operand) -> DistributedArray:
        """
        Multiply the local array by `other` in place.

        Parameters
        ----------
        other
            Scalar, raw array or `DistributedArray` to multiply by.

        Returns
        -------
        self
            This `DistributedArray` with its local array updated.
        """
        self.array_local *= self._unwrap(other)
        return self

    def __truediv__(self, other: Operand) -> DistributedArray:
        """
        Return ``self / other`` as a new `DistributedArray`.

        Parameters
        ----------
        other
            Scalar, raw array or `DistributedArray` to divide by.

        Returns
        -------
        result
            New `DistributedArray` holding the element-wise quotient of the
            local arrays.
        """
        return DistributedArray(self.array_local / self._unwrap(other))

    def __rtruediv__(self, other: Operand) -> DistributedArray:
        """
        Return ``other / self`` as a new `DistributedArray`.

        Parameters
        ----------
        other
            Scalar, raw array or `DistributedArray` to be divided.

        Returns
        -------
        result
            New `DistributedArray` holding the element-wise quotient of the
            local arrays.
        """
        return DistributedArray(self._unwrap(other) / self.array_local)

    def __itruediv__(self, other: Operand) -> DistributedArray:
        """
        Divide the local array by `other` in place.

        Parameters
        ----------
        other
            Scalar, raw array or `DistributedArray` to divide by.

        Returns
        -------
        self
            This `DistributedArray` with its local array updated.
        """
        self.array_local /= self._unwrap(other)
        return self

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
        Scatter a 1D NumPy array.

        Scatter a 1D NumPy array.
        Rank 0 owns the global array before scatter.
        After scatter, each rank owns its local chunk.
        """
        if not self._is_distributed:
            return

        size = self._comm.Get_size()
        rank = self._comm.Get_rank()

        if rank == 0:
            # Split array into `size` chunks
            chunks = backend.array_split(self.array_local, size)
        else:  # pragma: no cover # when writing this, only rank 0 reports coverage
            chunks = None

        # Each rank receives one chunk
        self.array_local = self._comm.scatter(chunks, root=0)

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
    def local_size(self) -> int:
        """
        Get the number of elements on the local processes.

        Returns
        -------
        int
            The total size of the distributed array across all processes.
        """
        return self.array_local.size

    @property
    def global_size(self) -> int:
        """
        Get the total number of elements across all processes.

        Returns
        -------
        int
            The total size of the distributed array across all processes.
        """
        local_size = self.array_local.size

        if self._is_distributed:
            total_size = self._comm.allreduce(local_size, op=MPI.SUM)
        else:
            total_size = local_size

        return total_size

    def min(self):
        """
        Compute the global minimum across all processes.

        Returns
        -------
        float
            The minimum value across all distributed array chunks.
        """
        local_min = float(backend.min(self.array_local))

        if self._is_distributed:
            global_min = self._comm.allreduce(local_min, op=MPI.MIN)
        else:
            global_min = local_min

        return global_min

    def max(self):
        """
        Compute the global maximum across all processes.

        Returns
        -------
        float
            The maximum value across all distributed array chunks.
        """
        local_max = float(backend.max(self.array_local))

        if self._is_distributed:
            global_max = self._comm.allreduce(local_max, op=MPI.MAX)
        else:
            global_max = local_max

        return global_max

    def mean(self):
        """
        Compute the global mean across all processes.

        Returns
        -------
        float
            The mean value across all distributed array chunks.
        """
        local_sum = float(backend.sum(self.array_local))
        local_count = self.array_local.size

        if self._is_distributed:
            global_sum = self._comm.allreduce(local_sum, op=MPI.SUM)
            global_count = self._comm.allreduce(local_count, op=MPI.SUM)
        else:
            global_sum = local_sum
            global_count = local_count

        return global_sum / global_count

    def std(self):
        """
        Compute the global standard deviation across all processes.

        Returns
        -------
        float
            The standard deviation across all distributed array chunks.
        """
        # Compute local statistics
        local_sum = float(backend.sum(self.array_local))
        # self.array_local**2 with dot product for performacne
        local_sum_sq = float(backend.dot(self.array_local, self.array_local))
        local_count = self.array_local.size

        if self._is_distributed:
            # Gather global statistics
            global_sum = self._comm.allreduce(local_sum, op=MPI.SUM)
            global_sum_sq = self._comm.allreduce(local_sum_sq, op=MPI.SUM)
            global_count = self._comm.allreduce(local_count, op=MPI.SUM)
        else:
            global_sum = local_sum
            global_sum_sq = local_sum_sq
            global_count = local_count

        # Compute global variance and standard deviation
        global_mean = global_sum / global_count
        global_variance = (global_sum_sq / global_count) - (global_mean**2)

        return sqrt(global_variance)

    def sum(self):
        """
        Compute the global sum across all processes.

        Returns
        -------
        float
            The sum of all values across all distributed array chunks.
        """
        local_sum = float(backend.sum(self.array_local))

        if self._is_distributed:
            global_sum = self._comm.allreduce(local_sum, op=MPI.SUM)
        else:
            global_sum = local_sum

        return global_sum

    def histogram(
        self,
        bins,
        range: tuple[float, float] | None = None,
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

        if range is None:
            range = (self.min(), self.max())

        backend.specials.histogram(
            array_read=self.array_local,
            array_write=array_write_local,
            start=range[0],
            stop=range[1],
        )

        # Combine histograms from all processes
        if self._is_distributed:
            self._comm.Allreduce(MPI.IN_PLACE, array_write_local, op=MPI.SUM)

            return array_write_local
        else:
            return array_write_local

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
            x=self.array_local,
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
        if self._is_distributed:
            self._comm.Allreduce(MPI.IN_PLACE, array_write_local, op=MPI.SUM)

            return array_write_local
        else:
            return array_write_local


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
        Raised if the `dtype`s of the local arrays do not match.
    TypeError
        Raised if the `type`s of the local arrays do not match.
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

    return DistributedArray(
        backend.concatenate((array_1.array_local, array_2.array_local))
    )
