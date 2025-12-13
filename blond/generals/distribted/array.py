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
        self._histogram_local_cache = {}
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

    @classmethod
    def from_array(
        cls, array: NumpyArray | CupyArray, comm=None
    ) -> "DistributedArray":
        """
        Create a DistributedArray from an existing array.

        This is a convenience factory method equivalent to calling the
        constructor directly.

        Parameters
        ----------
        array
            The local array data for this process.
        comm : MPI.Comm, optional
            MPI communicator to use. If None, uses MPI.COMM_WORLD.
            Ignored if MPI is not available.

        Returns
        -------
        DistributedArray
            A new DistributedArray instance.

        Examples
        --------
        >>> import numpy as np
        >>> arr = DistributedArray.from_array(np.random.rand(1000))
        >>> print(arr.global_size)
        """
        instance = cls(array)
        if comm is not None and instance.is_distributed:
            instance.comm = comm
            instance.rank = comm.Get_rank()
            instance.size = comm.Get_size()
            instance.is_distributed = instance.size > 1
        return instance

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
            # Allocate global histogram array
            if out is not None:
                array_write_local = array_write_local.copy()
                global_histogram = out
            else:
                global_histogram = backend.zeros(bins, backend.float)

            # Sum up histograms from all ranks
            self.comm.Allreduce(
                array_write_local, global_histogram, op=MPI.SUM
            )

            return global_histogram
        else:
            return array_write_local

    def percentile(self, q):
        """
        Compute the global percentile across all processes.

        This method gathers all data to the root process for exact percentile
        calculation, which may be memory-intensive for large arrays.

        Parameters
        ----------
        q : float or array-like
            Percentile(s) to compute, values between 0 and 100.

        Returns
        -------
        float or array
            The percentile value(s) across all distributed array chunks.

        Notes
        -----
        This implementation gathers all distributed data to the root process,
        which may not be feasible for very large datasets (TB-scale).
        For approximate percentiles on large data, consider implementing
        streaming quantile algorithms (e.g., t-digest, GK algorithm).
        """
        if self.is_distributed:
            # Gather all data to root
            local_data = self.array_local
            if self.is_root:
                # Gather sizes from all processes
                sizes = self.comm.gather(local_data.size, root=0)
                # Prepare receive buffer
                all_data = backend.zeros(sum(sizes), dtype=local_data.dtype)
            else:
                sizes = self.comm.gather(local_data.size, root=0)
                all_data = None

            # Gather actual data
            self.comm.Gatherv(
                sendbuf=local_data,
                recvbuf=(
                    all_data,
                    sizes if self.is_root else None,
                ),
                root=0,
            )

            # Compute percentile on root
            if self.is_root:
                # Convert to numpy for percentile calculation if needed
                if hasattr(all_data, "get"):  # CuPy array
                    result = backend.percentile(all_data, q).get()
                else:
                    import numpy as np

                    result = np.percentile(all_data, q)
            else:
                result = None

            # Broadcast result to all processes
            result = self.comm.bcast(result, root=0)
            return result
        # Non-distributed case
        elif hasattr(self.array_local, "get"):  # CuPy array
            import cupy as cp

            return cp.percentile(self.array_local, q).get()
        else:
            import numpy as np

            return np.percentile(self.array_local, q)

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
