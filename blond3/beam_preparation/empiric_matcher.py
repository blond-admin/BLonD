from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from blond3 import Simulation
from blond3._core.helpers import int_from_float_with_warning
from blond3.acc_math.empiric.hammiltonian import calc_hamiltonian, separatrixes
from blond3.beam_preparation.base import MatchingRoutine

if TYPE_CHECKING:  # pragma: no cover
    from blond3._core.beam.base import BeamBaseClass
    from numpy.typing import NDArray as NumpyArray


def populate_beam(
    beam: BeamBaseClass,
    time_grid: NumpyArray,
    deltaE_grid: NumpyArray,
    density_grid: NumpyArray,
    n_macroparticles: int,
    seed: int,
):
    """Fill bunch with macroparticles according to density_distribution"""

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
    assert time_step > 0
    deltaE_step = deltaE_grid[1, 0] - deltaE_grid[0, 0]
    assert deltaE_step > 0
    # Randomize particles inside each grid cell (uniform distribution)
    dt = (
        time_grid.flatten()[indexes]
        + (np.random.rand(n_macroparticles) - 0.5) * time_step
    )
    dE = (
        deltaE_grid.flatten()[indexes]
        + (np.random.rand(n_macroparticles) - 0.5) * deltaE_step
    )
    beam.setup_beam(dt=dt, dE=dE)


def to_density_grid(hamilton_2D):
    h_levels = separatrixes(hamilton_2D=hamilton_2D)
    h_max = np.max(h_levels)
    hamilton_2D[hamilton_2D > h_max] = h_max
    hamilton_2D[hamilton_2D > h_max] -= h_max
    hamilton_2D = -(hamilton_2D**2)
    hamilton_2D -= np.max(hamilton_2D)
    hamilton_2D *= -1
    hamilton_2D /= np.sum(hamilton_2D)
    return hamilton_2D


class EmpiricMatcher(MatchingRoutine):
    def __init__(
        self,
        grid_base_dt: NumpyArray,
        grid_base_dE: NumpyArray,
        n_macroparticles: int | float,
        seed: int = 0,
        maxiter=10,
    ):
        self.grid_base_dt = grid_base_dt
        self.grid_base_dE = grid_base_dE

        self.n_macroparticles = int_from_float_with_warning(
            n_macroparticles,
            warning_stacklevel=2,
        )

        self.seed = int_from_float_with_warning(
            seed,
            warning_stacklevel=2,
        )
        self.maxiter = int_from_float_with_warning(
            maxiter,
            warning_stacklevel=2,
        )

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        super().prepare_beam(
            simulation=simulation,
            beam=beam,
        )

        time_grid, deltaE_grid = np.meshgrid(self.grid_base_dt, self.grid_base_dE)
        shape_2d = time_grid.shape
        dt_flat_init = time_grid.flatten()
        dE_flat_init = deltaE_grid.flatten()
        users_beam = beam
        beam_gridded = deepcopy(users_beam)
        beam_gridded.setup_beam(
            dt=dt_flat_init.copy(),
            dE=dE_flat_init.copy(),
            reference_time=users_beam.reference_time,
            # reference_total_energy=users_beam.reference_total_energy,
            # flags=None # TODO
        )
        simulation.intensity_effect_manager.set_wakefields(False)
        simulation.run_simulation(
            beams=(beam_gridded,),
            n_turns=1,
            turn_i_init=0,
            observe=tuple(),
            show_progressbar=False,
            callback=None,
        )
        hamilton_2D = calc_hamiltonian(
            deltaE_grid,
            beam_gridded._dE.reshape(shape_2d),
            time_grid,
            beam_gridded._dt.reshape(shape_2d),
            maxiter=10,
        )
        hamilton_2D = to_density_grid(hamilton_2D)
        plt.matshow(hamilton_2D)
        plt.colorbar()
        plt.show()
        populate_beam(
            beam=users_beam,
            time_grid=time_grid,
            deltaE_grid=deltaE_grid,
            density_grid=hamilton_2D,
            n_macroparticles=self.n_macroparticles,
            seed=self.seed,
        )
        simulation.intensity_effect_manager.set_wakefields(active=True)
        for i in range(self.maxiter):
            simulation.intensity_effect_manager.set_profiles(active=True)
            simulation.run_simulation(
                beams=(users_beam,),
                n_turns=1,
                turn_i_init=0,
                observe=tuple(),
                show_progressbar=False,
                callback=None,
            )
            # apply the same intensity effects of users_beam to beam_gridded
            simulation.intensity_effect_manager.set_profiles(active=False)
            simulation.run_simulation(
                beams=(beam_gridded,),
                n_turns=1,
                turn_i_init=0,
                observe=tuple(),
                show_progressbar=False,
                callback=None,
            )
            hamilton_2D = calc_hamiltonian(
                deltaE_grid,
                beam_gridded._dE.reshape(shape_2d),
                time_grid,
                beam_gridded._dt.reshape(shape_2d),
                maxiter=10,
            )
            hamilton_2D = to_density_grid(hamilton_2D)
            populate_beam(
                beam=users_beam,
                time_grid=time_grid,
                deltaE_grid=deltaE_grid,
                density_grid=hamilton_2D,
                n_macroparticles=self.n_macroparticles,
                seed=self.seed,
            )
        simulation.intensity_effect_manager.set_wakefields(active=True)
        simulation.intensity_effect_manager.set_profiles(active=True)
