# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helper functions for beam creation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.core.beam.base import BeamBaseClass
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.generals.distributed.helpers import (
    mpi_aware_random_generator_cpu,
    mpi_local_size,
)

if TYPE_CHECKING:
    from cupy.typing import NDArray as CupyArray
    from numpy._typing import NDArray as NumpyArray

    from blond import Beam


def make_multibunch_beam(
    beam: Beam, n_times: int, t_distance: float, common_offset: float = 0.0
) -> Beam:
    """
    Add a bunch repeatedly with different time offset.

    Parameters
    ----------
    beam
        The beam object that is used as a reference.
    n_times
        Number of times the beam should be repeatedly added.
    t_distance
        Distance between each beam that is added, in [s].
    common_offset
        Offset that is applied to all added bunches equally, in[s].

    Returns
    -------
    full_beam
        Beam with many ``dt``-shifted copies of the input beam.

    Examples
    --------
    >>> from blond import make_multibunch_beam, Beam, proton
    >>>
    >>> beam = Beam(
    ...     intensity=1, particle_type=proton
    ... )
    >>> beam.setup_beam(dt=[1, 2, 3], dE=[1e3, 2e3, 3e3])
    >>> beam = make_multibunch_beam(
    ...     beam=beam,
    ...     n_times=3,
    ...     t_distance=222,
    ...     common_offset=111,
    ... )
    """
    from blond import Beam, backend  # prevent cyclic import

    assert beam.is_set_up(), (
        "Please set up beam correctly, e.g. using ``beam.setup_beam(...)``."
    )
    full_beam = Beam(
        intensity=n_times * beam.intensity,
        particle_type=beam.particle_type,
        is_counter_rotating=beam.is_counter_rotating,
    )
    # np.repeat([1,2], 2)
    # array([1, 1, 2, 2])
    full_dE = backend.repeat(beam._dE.array_local, n_times)

    full_dt = backend.repeat(beam._dt.array_local, n_times)
    for i in range(n_times):
        t_offset = t_distance * i + common_offset
        sel = slice(i, None, n_times)
        full_dt[sel] += t_offset

    full_beam.setup_beam(
        dt=full_dt,
        dE=full_dE,
        mpi_mode="all-ranks",
    )
    return full_beam


def generate_particle_coordinates(
    time_grid: NumpyArray,
    deltaE_grid: NumpyArray,
    density_grid: NumpyArray,
    n_macroparticles: int,
    seed: int | None,
) -> tuple[NumpyArray | CupyArray, NumpyArray | CupyArray]:
    """
    Fill bunch with macroparticles according to `density_distribution`.

    Parameters
    ----------
    time_grid
        2D grid of positions in time, in [s].
    deltaE_grid
        2D grid of energies, in [eV].
    density_grid
        2D grid of densities according to time vs. energy.
    n_macroparticles
        Number of macroparticles to distribute, according to the grid.
    seed
        Random seed, to make function with same seed
        always return the same value.

    Returns
    -------
    dt_local
        Particle coordinates (on the local MPI node, if MPI is active).
    dE_local
        Particle coordinates (on the local MPI node, if MPI is active).
    """
    from blond import backend  # prevent cyclic import

    # Initialise the random number generator
    # DEV NOTE (2025) It might be checked at a later time,
    # if cupy and numpy provide for the exact same random generators.
    n_macroparticles_local = mpi_local_size(
        n_macroparticles, warning_hint="n_macroparticles"
    )
    random_generator_cpu = mpi_aware_random_generator_cpu(
        seed=seed, n_forward_per_rank=n_macroparticles_local
    )
    # Generating particles randomly inside the grid cells according to the
    # provided density_grid
    indexes = random_generator_cpu.choice(
        np.arange(0, np.size(density_grid)),
        n_macroparticles_local,
        p=copy_to_cpu(density_grid.flatten() / np.sum(density_grid)),
    )
    indexes = np.array(
        indexes
    )  # finally convert to the correct backend. See above why.
    time_step = time_grid[0, 1] - time_grid[0, 0]
    assert time_step > 0, f"{time_step=}"
    deltaE_step = deltaE_grid[1, 0] - deltaE_grid[0, 0]
    assert deltaE_step > 0, f"{deltaE_step=}"
    # Randomize particles inside each grid cell (uniform distribution)
    # ``backend.random.triangular`` has rotational symmetry, but is
    # distributed within a square.
    dt_local = backend.array(
        time_grid.flatten()[indexes]
        + random_generator_cpu.triangular(
            left=-1, mode=0, right=1, size=n_macroparticles_local
        )
        * time_step
    )
    dE_local = backend.array(
        deltaE_grid.flatten()[indexes]
        + random_generator_cpu.triangular(
            left=-1, mode=0, right=1, size=n_macroparticles_local
        )
        * deltaE_step
    )
    return dt_local, dE_local


def populate_beam(
    beam: BeamBaseClass,
    time_grid: NumpyArray,
    deltaE_grid: NumpyArray,
    density_grid: NumpyArray,
    n_macroparticles: int,
    seed: int | None,
) -> None:
    """
    Fill bunch with macroparticles according to `density_distribution`.

    Parameters
    ----------
    beam
        Simulation beam object.
    time_grid
        2D grid of positions in time, in [s].
    deltaE_grid
        2D grid of energies, in [eV].
    density_grid
        2D grid of densities according to time vs. energy.
    n_macroparticles
        Number of macroparticles to distribute, according to the grid.
    seed
        Random seed, to make function with same seed
        always return the same value.

    Notes
    -----
    The beam coordinate `dt` and `dE` will be overwritten.
    """
    dt_local, dE_local = generate_particle_coordinates(
        time_grid=time_grid,
        deltaE_grid=deltaE_grid,
        density_grid=density_grid,
        n_macroparticles=n_macroparticles,
        seed=seed,
    )

    beam.setup_beam(dt=dt_local, dE=dE_local, mpi_mode="all-ranks")
