# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helper functions to work with MPI."""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from types import ModuleType

    from numpy.random import Generator as NumpyGenerator

    from blond.core.backends.mpi_distributed.distributed_array import (
        DistributedArray,
    )

# Environment variables set by MPI launchers on every rank: Open MPI,
# PMIx (Open MPI 5, PRRTE, `srun --mpi=pmix`), PMI/PMI2 (MPICH/Hydra,
# Intel MPI, MS-MPI, `srun --mpi=pmi2`) and MVAPICH.
MPI_LAUNCHER_ENV_KEYS = (
    "OMPI_COMM_WORLD_SIZE",
    "PMIX_RANK",
    "PMI_RANK",
    "PMI_SIZE",
    "MV2_COMM_WORLD_SIZE",
)


def mpi_launched() -> bool:
    """
    Whether this process was started by an MPI launcher (e.g. `mpirun`).

    BLonD only initialises MPI in that case. Importing `mpi4py.MPI` calls
    `MPI_Init`, which would otherwise slow down every `import blond` and
    set up the MPI runtime (e.g. UCX) in processes that never use MPI.

    Returns
    -------
    launched
        True if an MPI launcher variable is set, unless overridden by the
        `BLOND_LOAD_MPI` environment variable (``True`` or ``False``).

    Raises
    ------
    ValueError
        If `BLOND_LOAD_MPI` is set to anything but ``True`` or ``False``.
    """
    # Normally `BLOND_LOAD_MPI` is unset and the launcher variables decide.
    # It is an escape hatch: `True` for a launcher whose variables are not
    # in `MPI_LAUNCHER_ENV_KEYS` (otherwise every rank would silently run
    # as its own serial simulation), `False` to stay serial under one.
    override = os.environ.get("BLOND_LOAD_MPI")
    if override is None:
        launched = any(key in os.environ for key in MPI_LAUNCHER_ENV_KEYS)
    elif override in ("True", "False"):
        launched = override == "True"
    else:
        raise ValueError(
            "BLOND_LOAD_MPI environment variable must be either True"
            f" or False, not {override}"
        )
    return launched


def _import_mpi() -> ModuleType | None:
    """
    Import `mpi4py.MPI` only when running under an MPI launcher.

    Returns
    -------
    mpi_module
        The `mpi4py.MPI` module, or None when not launched by MPI or
        `mpi4py` cannot be imported.
    """
    mpi_module = None
    if mpi_launched():
        try:
            from mpi4py import MPI as mpi_module
        except Exception as exc:
            warnings.warn(str(exc), ImportWarning, stacklevel=1)
    return mpi_module


MPI = _import_mpi()
if MPI is None:
    MPI_COMM_WORLD = None
    MPI_RANK = 0
    MPI_SIZE = 1
else:
    MPI_COMM_WORLD = MPI.COMM_WORLD
    MPI_RANK = MPI_COMM_WORLD.Get_rank()
    MPI_SIZE = MPI_COMM_WORLD.Get_size()


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
    >>> from blond.core.backends.mpi_distributed.helpers import (
    ...     mpi_local_size,
    ...     mpi_aware_random_generator_cpu,
    ... )
    ...
    >>> n_macroparticles = 10
    >>> local_size = mpi_local_size(
    ...     int_from_float_with_warning(n_macroparticles, warning_stacklevel=1),
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
    from blond.core.backends.backend import backend
    from blond.core.backends.mpi_distributed.distributed_array import (
        DistributedArray,
    )

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


def distributed_zeros(
    local_n: int, dtype: np.typing.DTypeLike
) -> DistributedArray:
    """
    Distributed version of `np.zeros` and `cp.zeros`.

    Parameters
    ----------
    local_n
        Number of elements owned by *this MPI rank*.
    dtype
        Data type of the array.

    Returns
    -------
    DistributedArray
        Zeros distributed across MPI ranks.

        Example (2 ranks):
            rank 0: [0, 0, 0]
            rank 1: [0, 0, 0]
    """
    from blond.core.backends.backend import backend
    from blond.core.backends.mpi_distributed.distributed_array import (
        DistributedArray,
    )

    local_ids = backend.zeros(local_n, dtype=dtype)

    return DistributedArray(local_ids)


def mpi_is_distributed() -> bool:
    """
    Whether the software runs with a MPI size > 1 or not.

    Returns
    -------
    is_distributed
        Whether the software runs with a MPI size > 1 or not.
    """
    return MPI_SIZE > 1


def mpi_barrier() -> None:
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
    return MPI_RANK == 0
