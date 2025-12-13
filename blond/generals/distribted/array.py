# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


"""Helper module to work with CPU/GPU arrays distributed via MPI."""

from math import sqrt
from typing import TYPE_CHECKING

from blond import backend

# Try to import MPI, but don't fail if not available
try:
    from mpi4py import MPI

    _MPI_AVAILABLE = True
except ImportError:
    _MPI_AVAILABLE = False
    MPI = None  # type: ignore


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
        # Setup MPI communication
        self.comm = MPI.COMM_WORLD

        # Determine rank and size
        if self.comm is not None:
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.is_distributed = self.size > 1
        else:
            self.rank = 0
            self.size = 1
            self.is_distributed = False

        self._histogram_local_cache = {}

    @property
    def is_root(self) -> bool:
        """
        Check if this is the root process (rank 0).

        Returns
        -------
        bool
            Whether the current worker is the root worker.
        """
        return self.rank == 0

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

        if self.is_distributed:
            total_size = self.comm.allreduce(local_size, op=MPI.SUM)
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

        if self.is_distributed:
            global_min = self.comm.allreduce(local_min, op=MPI.MIN)
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

        if self.is_distributed:
            global_max = self.comm.allreduce(local_max, op=MPI.MAX)
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

        if self.is_distributed:
            global_sum = self.comm.allreduce(local_sum, op=MPI.SUM)
            global_count = self.comm.allreduce(local_count, op=MPI.SUM)
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

        if self.is_distributed:
            # Gather global statistics
            global_sum = self.comm.allreduce(local_sum, op=MPI.SUM)
            global_sum_sq = self.comm.allreduce(local_sum_sq, op=MPI.SUM)
            global_count = self.comm.allreduce(local_count, op=MPI.SUM)
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

        if self.is_distributed:
            global_sum = self.comm.allreduce(local_sum, op=MPI.SUM)
        else:
            global_sum = local_sum

        return global_sum

    def histogram(
        self,
        bins,
        range: tuple[float, float],
        out: NumpyArray | CupyArray | None = None,
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

        backend.specials.histogram(
            array_read=self.array_local,
            array_write=array_write_local,
            start=range[0],
            stop=range[1],
        )

        # Combine histograms from all processes
        if self.is_distributed:
            self.comm.Allreduce(MPI.IN_PLACE, array_write_local, op=MPI.SUM)

            return array_write_local
        else:
            return array_write_local

    def barrier(self):
        """
        Synchronize all processes.

        This method blocks until all processes in the communicator have called it.
        Useful for ensuring all processes reach a certain point before continuing.

        Notes
        -----
        In non-distributed mode (single process), this is a no-op.
        """
        if self.is_distributed:
            self.comm.Barrier()
