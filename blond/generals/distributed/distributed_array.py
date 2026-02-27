# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
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

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

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

        self._histogram_local_cache = {}

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

    def min(self, weights: DistributedArray | None = None):
        """
        Compute the global minimum across all processes.

        Parameters
        ----------
        weights
            When provided, only elements with ``weight > 0`` are considered.
            ``None`` (default) considers all elements.

        Returns
        -------
        float
            The minimum value across all (active) distributed array chunks.
        """
        if weights is None:
            local_arr = self.array_local
        else:
            local_arr = self.array_local[weights.array_local > 0]
        local_min = float(backend.min(local_arr))

        if self._is_distributed:
            global_min = self._comm.allreduce(local_min, op=MPI.MIN)
        else:
            global_min = local_min

        return global_min

    def max(self, weights: DistributedArray | None = None):
        """
        Compute the global maximum across all processes.

        Parameters
        ----------
        weights
            When provided, only elements with ``weight > 0`` are considered.
            ``None`` (default) considers all elements.

        Returns
        -------
        float
            The maximum value across all (active) distributed array chunks.
        """
        if weights is None:
            local_arr = self.array_local
        else:
            local_arr = self.array_local[weights.array_local > 0]
        local_max = float(backend.max(local_arr))

        if self._is_distributed:
            global_max = self._comm.allreduce(local_max, op=MPI.MAX)
        else:
            global_max = local_max

        return global_max

    def mean(self, weights: DistributedArray | None = None):
        """
        Compute the global mean across all processes.

        Parameters
        ----------
        weights
            Per-element weights as a :class:`DistributedArray`.  When provided, returns the
            plain array local to this rank.  When provided, returns the
            weighted mean ``sum(w * x) / sum(w)`` instead of the arithmetic
            mean.  ``None`` (default) uses the unweighted mean.

        Returns
        -------
        float
            The (weighted) mean value across all distributed array chunks.
        """
        if weights is None:
            local_sum = float(backend.sum(self.array_local))
            local_count = self.array_local.size

            if self._is_distributed:
                global_sum = self._comm.allreduce(local_sum, op=MPI.SUM)
                global_count = self._comm.allreduce(local_count, op=MPI.SUM)
            else:
                global_sum = local_sum
                global_count = local_count

            return global_sum / global_count
        else:
            weights_local = weights.array_local
            local_wx_sum = float(backend.sum(self.array_local * weights_local))
            local_w_sum = float(backend.sum(weights_local))

            if self._is_distributed:
                global_wx_sum = self._comm.allreduce(local_wx_sum, op=MPI.SUM)
                global_w_sum = self._comm.allreduce(local_w_sum, op=MPI.SUM)
            else:
                global_wx_sum = local_wx_sum
                global_w_sum = local_w_sum

            return global_wx_sum / global_w_sum

    def std(self, weights: DistributedArray | None = None):
        """
        Compute the global standard deviation across all processes.

        Parameters
        ----------
        weights
            Per-element weights as a :class:`DistributedArray`.  When provided, returns the
            plain array local to this rank.  When provided, returns the
            weighted standard deviation
            ``sqrt(sum(w * x²) / sum(w) - (sum(w * x) / sum(w))²)``.
            ``None`` (default) uses the unweighted standard deviation.

        Returns
        -------
        float
            The (weighted) standard deviation across all distributed array
            chunks.
        """
        if weights is None:
            # Compute local statistics
            local_sum = float(backend.sum(self.array_local))
            # self.array_local**2 with dot product for performacne
            local_sum_sq = float(
                backend.dot(self.array_local, self.array_local)
            )
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
        else:
            weights_local = weights.array_local
            local_w_sum = float(backend.sum(weights_local))
            local_wx_sum = float(backend.sum(self.array_local * weights_local))
            local_wx2_sum = float(
                backend.dot(self.array_local * weights_local, self.array_local)
            )

            if self._is_distributed:
                global_w_sum = self._comm.allreduce(local_w_sum, op=MPI.SUM)
                global_wx_sum = self._comm.allreduce(local_wx_sum, op=MPI.SUM)
                global_wx2_sum = self._comm.allreduce(
                    local_wx2_sum, op=MPI.SUM
                )
            else:
                global_w_sum = local_w_sum
                global_wx_sum = local_wx_sum
                global_wx2_sum = local_wx2_sum

            global_mean = global_wx_sum / global_w_sum
            global_variance = (global_wx2_sum / global_w_sum) - global_mean**2

        return sqrt(global_variance)

    def sum(self, weights: DistributedArray | None = None):
        """
        Compute the global sum across all processes.

        Parameters
        ----------
        weights
            Per-element weights as a :class:`DistributedArray`.  When provided, returns the
            plain array local to this rank.  When provided, returns the
            weighted sum ``sum(w * x)`` instead of ``sum(x)``.
            ``None`` (default) uses the unweighted sum.

        Returns
        -------
        float
            The (weighted) sum across all distributed array chunks.
        """
        if weights is None:
            local_sum = float(backend.sum(self.array_local))
        else:
            weights_local = weights.array_local
            local_sum = float(backend.sum(self.array_local * weights_local))

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
        weights: DistributedArray | None = None,
    ):
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
        weights
            Per-element weights.  May be a :class:`DistributedArray` (in which
            case ``.array_local`` is used for the local kernel call) or a plain
            numpy/cupy array already local to this rank.
            ``None`` (default) uses the fast unweighted path.

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

        if weights is None:
            backend.specials.histogram(
                array_read=self.array_local,
                array_write=array_write_local,
                start=range[0],
                stop=range[1],
            )
        else:
            weights_local = weights.array_local
            backend.specials.histogram_weighted(
                array_read=self.array_local,
                array_write=array_write_local,
                weights=weights_local,
                start=range[0],
                stop=range[1],
            )

        # Combine histograms from all processes
        if self._is_distributed:
            self._comm.Allreduce(MPI.IN_PLACE, array_write_local, op=MPI.SUM)

            return array_write_local
        else:
            return array_write_local
