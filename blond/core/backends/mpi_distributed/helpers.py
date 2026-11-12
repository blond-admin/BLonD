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
    from collections.abc import Callable
    from types import ModuleType

    from cupy.random import RandomState as CupyRandomState  # type: ignore
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


def _node_local_rank(comm_world) -> int:
    """
    Rank of this process among the ranks on the same node.

    Asks MPI which ranks share memory, so it needs no launcher-specific
    environment variable. It is a collective, run once at import, which
    every rank does.

    Parameters
    ----------
    comm_world
        The `MPI.COMM_WORLD` communicator.

    Returns
    -------
    local_rank
        Node-local rank, from 0 to the number of ranks on the node - 1.
    """
    node_comm = comm_world.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = node_comm.Get_rank()
    node_comm.Free()
    return local_rank


MPI = _import_mpi()
if MPI is None:
    MPI_COMM_WORLD = None
    MPI_RANK = 0
    MPI_SIZE = 1
    MPI_LOCAL_RANK = 0
else:
    MPI_COMM_WORLD = MPI.COMM_WORLD
    MPI_RANK = MPI_COMM_WORLD.Get_rank()
    MPI_SIZE = MPI_COMM_WORLD.Get_size()
    MPI_LOCAL_RANK = _node_local_rank(MPI_COMM_WORLD)


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


def mpi_aware_random_generator_cpu(seed: int | None) -> NumpyGenerator:
    """
    Get a random generator compatible with MPI.

    Every rank gets its own stream of the seeded PCG64 generator, jumped
    ahead by ``rank`` times ~2**127 draws, so the ranks never share random
    numbers, however many each one draws. Rank 0 (and a serial run) gets
    exactly ``np.random.default_rng(seed)``.

    Parameters
    ----------
    seed
        Random seed.

    Returns
    -------
    random_generator_cpu
        The random generator of this rank.

    Notes
    -----
    A run on several ranks does not reproduce the serial random numbers:
    a rank cannot know how many draws the ranks before it used, since
    e.g. `standard_normal` and rejection loops take a varying number.

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
    ...     seed=None
    ... ).standard_normal(size=local_size)
    """
    # `jumped(0)` is the unjumped generator, i.e. `default_rng(seed)`.
    bit_generator = np.random.PCG64(seed).jumped(MPI_RANK)
    return np.random.Generator(bit_generator)


def mpi_aware_random_generator_gpu(seed: int | None) -> CupyRandomState:
    """
    Get a random generator compatible with MPI that draws on the GPU.

    Every rank gets its own seed, spawned from `seed`, so the ranks draw
    independent streams.

    Parameters
    ----------
    seed
        Random seed.

    Returns
    -------
    random_generator_gpu
        The CuPy random generator of this rank.

    Notes
    -----
    The same seed gives different numbers than
    `mpi_aware_random_generator_cpu`, so a beam drawn with it depends on
    the backend.
    """
    import cupy as cp

    rank_seed = np.random.SeedSequence(seed).spawn(MPI_SIZE)[MPI_RANK]
    return cp.random.RandomState(seed=int(rank_seed.generate_state(1)[0]))


def mpi_aware_random_generator(
    seed: int | None, backend_dependent_random_numbers: bool = False
) -> NumpyGenerator | CupyRandomState:
    """
    Get this rank's random generator for the active backend.

    Parameters
    ----------
    seed
        Random seed.
    backend_dependent_random_numbers
        If False (default), return `mpi_aware_random_generator_cpu`, so
        the same seed gives the same numbers on every backend. If True and
        the backend uses the GPU, return `mpi_aware_random_generator_gpu`,
        which draws directly on the GPU: faster and needs no host memory,
        but the same seed then gives different numbers than on the CPU.

    Returns
    -------
    random_generator
        The random generator of this rank.
    """
    from blond.core.backends.backend import backend

    if backend_dependent_random_numbers and backend.is_gpu:
        return mpi_aware_random_generator_gpu(seed=seed)
    return mpi_aware_random_generator_cpu(seed=seed)


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


def mpi_run_on_root(function: Callable[[], object]) -> None:
    """
    Run `function` on rank 0 only, and raise on every rank if it failed.

    Used for side effects that must happen exactly once, e.g. writing
    result files, which would collide if every rank wrote them. The other
    ranks wait until rank 0 is done, so they can read what it wrote. If
    rank 0 raises, every other rank raises too instead of waiting for it
    forever.

    Parameters
    ----------
    function
        Callable without arguments, called on rank 0 only.

    Raises
    ------
    RuntimeError
        On ranks other than 0, if `function` raised on rank 0.
        Rank 0 re-raises the original exception.

    Notes
    -----
    In non-distributed mode (single process), this just calls `function`.
    """
    if not mpi_is_distributed():
        function()
        return
    error = None
    if mpi_is_root():
        try:
            function()
        except Exception as exc:
            error = exc
    # The broadcast doubles as the synchronisation point: the other ranks
    # block until rank 0 has finished `function`.
    error_message = MPI_COMM_WORLD.bcast(
        None if error is None else repr(error), root=0
    )
    if error is not None:
        raise error
    if error_message is not None:
        raise RuntimeError(
            f"MPI rank 0 failed, so rank {MPI_RANK} stops too: {error_message}"
        )


def mpi_local_gpu_index(n_devices: int) -> int | None:
    """
    Index of the GPU this MPI rank should use.

    The ranks of a node share out its GPUs. The node-local rank is used, not the global one, which would pick
    the wrong GPU on every node but the first. If the launcher already
    gives each rank one GPU (`CUDA_VISIBLE_DEVICES`), `n_devices` is 1
    and the index is 0.

    Parameters
    ----------
    n_devices
        Number of GPUs visible to this process.

    Returns
    -------
    device_index
        Index of the GPU to use, or None without MPI, so that the
        current device stays as it is.
    """
    if not mpi_is_distributed():
        return None
    return MPI_LOCAL_RANK % n_devices


def mpi_cuda_aware() -> bool:
    """
    Whether GPU arrays may be passed to MPI directly.

    Set `BLOND_MPI_CUDA_AWARE=True` only with a CUDA-aware MPI library.
    Neither mpi4py nor the MPI standard can tell at runtime, and handing
    GPU memory to an MPI that is not CUDA-aware crashes or corrupts data.

    Returns
    -------
    cuda_aware
        True if `BLOND_MPI_CUDA_AWARE` is ``True``; False if it is
        ``False`` or unset (the default).

    Raises
    ------
    ValueError
        If `BLOND_MPI_CUDA_AWARE` is set to anything but ``True`` or
        ``False``.
    """
    value = os.environ.get("BLOND_MPI_CUDA_AWARE", "False")
    if value not in ("True", "False"):
        raise ValueError(
            "BLOND_MPI_CUDA_AWARE environment variable must be either True"
            f" or False, not {value}"
        )
    return value == "True"
