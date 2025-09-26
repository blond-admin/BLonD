from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from blond import Simulation
from blond._core.helpers import int_from_float_with_warning
from blond.beam_preparation.base import MatchingRoutine
from blond.experimental.acc_math.empiric.hammiltonian import (
    calc_hamiltonian,
    separatrixes,
)

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

    from blond._core.beam.base import BeamBaseClass


def populate_beam(
    beam: BeamBaseClass,
    time_grid: NumpyArray,
    deltaE_grid: NumpyArray,
    density_grid: NumpyArray,
    n_macroparticles: int,
    seed: int,
    normalize_density: bool = False,
) -> None:
    """
    Fill bunch with macroparticles according to density_distribution

    Notes
    -----
    The beam coordinate dt and dE will be overwritten.

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
    if normalize_density:
        density_grid /= np.sum(density_grid)
    # Initialise the random number generator
    np.random.seed(seed=seed)
    # Generating particles randomly inside the grid cells according to the
    # provided density_grid
    indexes = np.random.choice(
        np.arange(0, np.size(density_grid)),
        n_macroparticles,
        p=density_grid.flatten(),
    )
    time_step = time_grid[0, 1] - time_grid[0, 0]
    assert time_step > 0, f"{time_step=}"
    deltaE_step = deltaE_grid[1, 0] - deltaE_grid[0, 0]
    assert deltaE_step > 0, f"{deltaE_step=}"
    # Randomize particles inside each grid cell (uniform distribution)
    dt = (
        time_grid.flatten()[indexes]
        + np.random.triangular(left=-1, mode=0, right=1, size=n_macroparticles)
        * time_step
    )
    dE = (
        deltaE_grid.flatten()[indexes]
        + np.random.triangular(left=-1, mode=0, right=1, size=n_macroparticles)
        * deltaE_step
    )
    beam.setup_beam(dt=dt, dE=dE)
