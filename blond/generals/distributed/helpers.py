# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helper functions to work with MPI."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.random import Generator as NumpyGenerator

    from blond.generals.distributed.distributed_array import DistributedArray

try:
    from mpi4py.MPI import COMM_WORLD as MPI_COMM_WORLD

    MPI_RANK = MPI_COMM_WORLD.Get_rank()
    MPI_SIZE = MPI_COMM_WORLD.Get_size()
except Exception as exc:
    warnings.warn(str(exc), ImportWarning, stacklevel=1)
    MPI_COMM_WORLD = None
    MPI_RANK = 0
    MPI_SIZE = 1


def mpi_local_size(global_size: int, warning_hint: str) -> int:
    """
    Cast the global size to an MPI-aware local size.

    Parameters
    ----------
    global_size
        Integer that defines the global size,
        e.g. the global number of macro-particles.
    warning_hint
        The variable name that is displayed in a warning,
        if `global_n` is truncated.

    Returns
    -------
    local_n
        The local array size to get the global array size.
    """
    local_n_ = int(global_size // MPI_SIZE)  # might lose the decimal places
    global_size_effective = local_n_ * MPI_SIZE
    if global_size_effective != global_size:  # if decimal places are lost
        warnings.warn(
            f"Because MPI is used, `{warning_hint}` is truncated"
            f" from {global_size} to {global_size_effective}.",
            UserWarning,
            stacklevel=1,
        )
    return local_n_


def mpi_aware_random_generator_cpu(
    seed: int | None, n_forward_per_rank: int
) -> NumpyGenerator:
    """
    Get a random generator compatible with MPI.

    Parameters
    ----------
    seed
        Random seed.
    n_forward_per_rank
        Considers that the other MPI-ranks also generate n samples.

    Returns
    -------
    random_generator_cpu
        The random generator.

    Notes
    -----
    As the Cupy random generators behave differently than the Numpy random
    generators, this routine returns only the CPU generators for consistency.
    The GPU interaction must be handled explicitly outside this function.

    Examples
    --------
    >>> from blond.core.helpers import int_from_float_with_warning
    >>> from blond.generals.distributed.helpers import (
    ...     mpi_local_size,
    ...     mpi_aware_random_generator_cpu,
    ... )
    ...
    >>> n_macroparticles = 10
    >>> local_size = mpi_local_size(
    ...     int_from_float_with_warning(n_macroparticles, warning_stacklevel=2),
    ...     warning_hint="n_macroparticles",
    ... )
    >>> random_array = mpi_aware_random_generator_cpu(
    ...     seed=None, n_forward_per_rank=local_size
    ... ).standard_normal(size=local_size)
    """
    # Generate coordinates. For reproducibility,
    # a separate random number stream is used for dt and dE

    # All ranks have the same random generator.
    random_generator_cpu = np.random.default_rng(seed)

    # Consider the fact that the other ranks also produce particles.
    random_generator_cpu.bit_generator.advance(MPI_RANK * n_forward_per_rank)

    # Cupy doesn't implement the `advance` function (2025)
    # When Cupy provides for the same random generators & `advance`,
    # this function could be extended to GPU.

    return random_generator_cpu


def distributed_arange(
    local_n: int, dtype: np.typing.DTypeLike
) -> DistributedArray:
    """
    Distributed version of `np.arange` and `cp.arange`.

    Parameters
    ----------
    local_n
        Number of elements owned by *this MPI rank*.
    dtype
        Data type of the array.

    Returns
    -------
    DistributedArray
        Globally consistent arange distributed across MPI ranks.

        Example (2 ranks):
            rank 0: [0, 1, 2]
            rank 1: [3, 4, 5]
    """
    from blond import backend
    from blond.generals.distributed.distributed_array import DistributedArray

    # Compute starting offset for this rank
    if MPI_COMM_WORLD is None:
        offset = None
    else:
        offset: int | None = MPI_COMM_WORLD.exscan(local_n)

    if offset is None:
        offset = 0

    local_ids = backend.arange(
        offset,
        offset + local_n,
        dtype=dtype,
    )

    return DistributedArray(local_ids)


def mpi_is_distributed():
    """
    Whether the software runs with a MPI size > 1 or not.

    Returns
    -------
    is_distributed
        Whether the software runs with a MPI size > 1 or not.
    """
    if MPI_COMM_WORLD is None:
        return False
    if MPI_COMM_WORLD.Get_size() > 1:
        return True


def mpi_barrier():
    """
    Synchronize all processes.

    This method blocks until all processes in the communicator have called it.
    Useful for ensuring all processes reach a certain point before continuing.

    Notes
    -----
    In non-distributed mode (single process), this is a no-op.
    """
    if mpi_is_distributed():
        MPI_COMM_WORLD.Barrier()


def mpi_is_root() -> bool:
    """
    Check if this is the root process (rank 0).

    Returns
    -------
    bool
        Whether the current worker is the root worker.
    """
    if MPI_COMM_WORLD is None:
        return True
    else:
        return MPI_COMM_WORLD.Get_rank() == 0
