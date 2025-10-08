from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..._core.backends.backend import backend

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional, Tuple

    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray

    from blond._core.beam.base import BeamBaseClass


def generate_particle_coordinates(
    time_grid: NumpyArray,
    deltaE_grid: NumpyArray,
    density_grid: NumpyArray,
    n_macroparticles: int,
    seed: Optional[int],
) -> Tuple[NumpyArray | CupyArray, NumpyArray | CupyArray]:
    """
    Fill bunch with macroparticles according to `density_distribution`

    Notes
    -----
    The beam coordinate `dt and` `dE` will be overwritten.

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
    if seed is not None:
        np.random.seed(seed=seed)
    # Generating particles randomly inside the grid cells according to the
    # provided density_grid
    indexes = np.random.choice(
        np.arange(0, np.size(density_grid)),
        n_macroparticles,
        p=density_grid.flatten() / np.sum(density_grid),
    )
    indexes = backend.array(
        indexes
    )  # to always use the same random seed/choice
    time_step = time_grid[0, 1] - time_grid[0, 0]
    assert time_step > 0, f"{time_step=}"
    deltaE_step = deltaE_grid[1, 0] - deltaE_grid[0, 0]
    assert deltaE_step > 0, f"{deltaE_step=}"
    # Randomize particles inside each grid cell (uniform distribution)
    dt = (
        time_grid.flatten()[indexes]
        + backend.random.triangular(
            left=-1, mode=0, right=1, size=n_macroparticles
        )
        * time_step
    )
    dE = (
        deltaE_grid.flatten()[indexes]
        + backend.random.triangular(
            left=-1, mode=0, right=1, size=n_macroparticles
        )
        * deltaE_step
    )
    return dt, dE


def populate_beam(
    beam: BeamBaseClass,
    time_grid: NumpyArray,
    deltaE_grid: NumpyArray,
    density_grid: NumpyArray,
    n_macroparticles: int,
    seed: Optional[int],
) -> None:
    """
    Fill bunch with macroparticles according to `density_distribution`

    Notes
    -----
    The beam coordinate `dt and` `dE` will be overwritten.

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
    dt, dE = generate_particle_coordinates(
        time_grid=time_grid,
        deltaE_grid=deltaE_grid,
        density_grid=density_grid,
        n_macroparticles=n_macroparticles,
        seed=seed,
    )

    beam.setup_beam(dt=dt, dE=dE)


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
    The beam coordinate `dt and` `dE` will be overwritten.

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
    assert n_macroparticles_overwrite <= (beam._dE)
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
    if beam._dE is not None:
        beam._dE[indexes] = dE
    else:
        raise ValueError(f"{beam._dE=}")
    if beam._dt is not None:
        beam._dt[indexes] = dt
    else:
        raise ValueError(f"{beam._dt=}")
