# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import backend
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.generals.distributed.helpers import (
    mpi_aware_random_generator_cpu,
    mpi_local_size,
)

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass


def generate_particle_coordinates(
    time_grid: NumpyArray,
    deltaE_grid: NumpyArray,
    density_grid: NumpyArray,
    n_macroparticles: int,
    seed: int | None,
) -> tuple[NumpyArray | CupyArray, NumpyArray | CupyArray]:
    """
    Fill bunch with macroparticles according to `density_distribution`

    Parameters
    ----------
    time_grid
        2D grid of positions in time, in [s]
    deltaE_grid
        2D grid of energies, in [eV]
    density_grid
        2D grid of densities according to time vs. energy
    n_macroparticles
        Number of macroparticles to distribute, according to the grid
    seed
        Random seed, to make function with same seed
        always return the same value
    """
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
    indexes = backend.array(
        indexes
    )  # finally convert to the correct backend. See above why.
    time_step = time_grid[0, 1] - time_grid[0, 0]
    assert time_step > 0, f"{time_step=}"
    deltaE_step = deltaE_grid[1, 0] - deltaE_grid[0, 0]
    assert deltaE_step > 0, f"{deltaE_step=}"
    # Randomize particles inside each grid cell (uniform distribution)
    # ``backend.random.triangular`` has rotational symmetry, but is
    # distributed within a square.
    dt_local = (
        time_grid.flatten()[indexes]
        + backend.array(
            random_generator_cpu.triangular(
                left=-1, mode=0, right=1, size=n_macroparticles_local
            )
        )
        * time_step
    )
    dE_local = (
        deltaE_grid.flatten()[indexes]
        + backend.array(
            random_generator_cpu.triangular(
                left=-1, mode=0, right=1, size=n_macroparticles_local
            )
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
    Fill bunch with macroparticles according to `density_distribution`

    Notes
    -----
    The beam coordinate `dt` and `dE` will be overwritten.

    Parameters
    ----------
    beam
        Simulation beam object
    time_grid
        2D grid of positions in time, in [s]
    deltaE_grid
        2D grid of energies, in [eV]
    density_grid
        2D grid of densities according to time vs. energy
    n_macroparticles
        Number of macroparticles to distribute, according to the grid
    seed
        Random seed, to make function with same seed
        always return the same value
    """
    dt_local, dE_local = generate_particle_coordinates(
        time_grid=time_grid,
        deltaE_grid=deltaE_grid,
        density_grid=density_grid,
        n_macroparticles=n_macroparticles,
        seed=seed,
    )

    beam.setup_beam(dt=dt_local, dE=dE_local, mpi_mode="all-ranks")


def repopulate_beam(
    beam: BeamBaseClass,
    time_grid: NumpyArray,
    deltaE_grid: NumpyArray,
    density_grid: NumpyArray,
    n_macroparticles_overwrite: int,
    seed: int,
) -> None:
    """
    Partially overwrite bunch with macroparticles according to `density_distribution`

    Notes
    -----
    The beam coordinate `dt` and `dE` will be overwritten.

    Parameters
    ----------
    beam
        Simulation beam object
    time_grid
        2D grid of positions in time, in [s]
    deltaE_grid
        2D grid of energies, in [eV]
    density_grid
        2D grid of densities according to time vs. energy
    n_macroparticles_overwrite
        Number of macroparticles to distribute, according to the grid
    seed
        Random seed, to make function with same seed
        always return the same value
    """
    assert n_macroparticles_overwrite <= (beam._dE), (
        "Number of particles to be overwritten is larger than number of macroparticles."
    )
    dt, dE = generate_particle_coordinates(
        time_grid=time_grid,
        deltaE_grid=deltaE_grid,
        density_grid=density_grid,
        n_macroparticles=n_macroparticles_overwrite,
        seed=seed,
    )
    indexes = np.random.choice(
        np.arange(0, beam.n_macroparticles_partial()),
        n_macroparticles_overwrite,
    )
    beam._dE[indexes] = dE

    beam._dt[indexes] = dt
