from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from blond import Simulation
from blond._core.helpers import int_from_float_with_warning
from blond.beam_preparation.base import MatchingRoutine
from blond.experimental.acc_math.empiric.hamiltonian import (
    calc_hamiltonian,
    separatrixes,
)
from blond.experimental.beam_preparation.helpers import populate_beam

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from numpy.typing import NDArray as NumpyArray

    from blond._core.beam.base import BeamBaseClass


def populate_beam(
    beam: BeamBaseClass,
    time_grid: NumpyArray,
    deltaE_grid: NumpyArray,
    density_grid: NumpyArray,
    n_macroparticles: int,
    seed: int | None,
) -> None:
    """Fill bunch with macroparticles according to density_distribution.

    Notes
    -----
    The beam coordinate dt and dE will be overwritten.

    Parameters
    ----------
    beam
        Simulation `Beam` object
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
        p=density_grid.flatten(),
    )
    time_step = time_grid[0, 1] - time_grid[0, 0]
    assert time_step > 0
    deltaE_step = deltaE_grid[1, 0] - deltaE_grid[0, 0]
    assert deltaE_step > 0
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


def normalize_as_density(hamilton_2D: NumpyArray):
    """Convert 2D Hamiltonian to density.

    Parameters
    ----------
    hamilton_2D
        2D array containing the Hamiltonian

    Returns
    -------
    density

    """
    h_levels = separatrixes(hamilton_2D=hamilton_2D)
    h_max = np.max(h_levels)

    density = hamilton_2D.copy()  # TODO better inplace for memory?

    density[density > h_max] = h_max
    density -= h_max
    density = density**2
    # density *= -1
    density /= np.sum(density)

    return density


class EmpiricMatcher(MatchingRoutine):
    def __init__(
        self,
        grid_base_dt: NumpyArray,
        grid_base_dE: NumpyArray,
        n_macroparticles: int | float,
        seed: int | None = None,
        maxiter_intensity_effects=10,
        maxiter_hamiltonian=20,
        atol_hamiltonian=1e-4,
        animate=False,
        hamiltonian_to_density_function: Callable[
            [NumpyArray], NumpyArray
        ] = normalize_as_density,
    ):
        """Matching routine based on the particle movement within one turn.

        Step 1:
        This routine uses a 2D grid (dt vs dE) and executes one turn
        of the simulation with this grid as `Beam`. After one turn,
        the movement of the particles are used to derive the 2D Hamiltonian
        by the equations dH/dp = do/dt and dH/dq = -dq/dt,
        because dH/dt and dH/dq are observed within one turn.

        Step 2:
        The obtained 2D Hamilton is converted to a density distribution
        via `normalize_as_density`.

        Step 3:
        The 2D density distribution is converted to beam dt and dE coordinates.

        Step 4 - For Intensity effects
        Repeat 1-3 until the shape of the beam converges to a stable solution.
        Use `maxiter_hamiltonian` for to control the convergence.

        Notes
        -----
        Due to using an internal 2D grid, this method can be very demanding
        in terms of runtime and memory. It is better to use
        `SemiEmpiricMatcher`, as it works on a 1D data format.


        Notes
        -----
        This routine only works properly if the phase advance is low enough

        Parameters
        ----------
        grid_base_dt
            Base axis for a 2D grid of positions in time, in [s]
            This defines the boundaries of observation,
            i.e. where the bunch is going to be defined.
            This can also span several RF buckets.
        grid_base_dE
            Base axis for a 2D grid of energies, in [eV].
            This defines the boundaries of observation,
            i.e. where the bunch is going to be defined.
            This can also span several RF buckets.
        n_macroparticles
            Number of macroparticles to distribute, according to the grid
        seed
            Random seed, to make function with same seed
            always return the same value
        maxiter_intensity_effects
            Maximum number of iterations to refine the matched beam
            for intensity effects
        hamiltonian_to_density_function
            A function that converts from the 2D Hamiltonian to the 2D density.
            The 2D density is used to create the beam dt and dE coordinates.
            The default is `normalize_as_density`.
            It is intended to be replaced by user-defined functions.

        Examples
        --------
        >>> simulation.prepare_beam(
        >>>     beam=beam1,
        >>>     preparation_routine=EmpiricMatcher(
        >>>         grid_base_dt=np.linspace(0, 2.5e-9, 100),
        >>>         grid_base_dE=np.linspace(
        >>>             -(777538700.0 * 2), 777538700.0 * 2, 100
        >>>         ),
        >>>         n_macroparticles=1e6,
        >>>         seed=0, # For reproducible results
        >>>     ),
        >>> )
        """
        assert callable(hamiltonian_to_density_function)
        self.hamiltonian_to_density_function = hamiltonian_to_density_function
        warnings.warn(
            "This method is still in development and subject to changes. "
            "Expect bugs!",
            UserWarning,
            stacklevel=1,
        )
        self._grid_base_dt = grid_base_dt

        # Print cutoff if grid points are equidistant
        _time_diffs = np.diff(grid_base_dt)
        if np.all(_time_diffs == _time_diffs[0]):
            _freqs = np.fft.rfftfreq(3, _time_diffs[0])
            print(f"Cutoff frequency of the grid is {_freqs[-1]} Hz")

        self._grid_base_dE = grid_base_dE

        self._n_macroparticles = int_from_float_with_warning(
            n_macroparticles,
            warning_stacklevel=2,
        )
        if seed is not None:
            self._seed = int_from_float_with_warning(
                seed,
                warning_stacklevel=2,
            )
        else:
            self._seed = None
        self._maxiter_intensity_effects = int_from_float_with_warning(
            maxiter_intensity_effects,
            warning_stacklevel=2,
        )
        self._maxiter_hamiltonian = maxiter_hamiltonian
        self._atol_hamiltonian = atol_hamiltonian

        self.animate = animate

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """Carries out the empiric matching.

        Notes
        -----
        The beam coordinate dt and dE will be overwritten.

        Parameters
        ----------
        simulation
            Simulation context manager
        beam
            Beam class to interact with this element

        """
        super().prepare_beam(
            simulation=simulation,
            beam=beam,
        )
        reference_time = deepcopy(beam.reference_time)
        reference_total_energy = deepcopy(beam.reference_total_energy)

        time_grid, deltaE_grid = np.meshgrid(
            self._grid_base_dt, self._grid_base_dE
        )
        shape_2d = time_grid.shape
        dt_flat_init = time_grid.flatten()
        dE_flat_init = deltaE_grid.flatten()
        users_beam = beam
        beam_gridded = deepcopy(users_beam)
        beam_gridded.setup_beam(
            dt=dt_flat_init.copy(),
            dE=dE_flat_init.copy(),
            reference_time=reference_time,
            reference_total_energy=reference_total_energy,
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
            maxiter=self._maxiter_hamiltonian,
            atol=self._atol_hamiltonian,
        )
        hamilton_2D = self.hamiltonian_to_density_function(hamilton_2D)
        users_beam.reference_total_energy = reference_total_energy
        users_beam.reference_time = reference_time
        populate_beam(
            beam=users_beam,
            time_grid=time_grid,
            deltaE_grid=deltaE_grid,
            density_grid=hamilton_2D,
            n_macroparticles=self._n_macroparticles,
            seed=self._seed,
        )

        simulation.intensity_effect_manager.set_wakefields(active=True)
        if self.animate:
            plt.figure("EmpiricMatcher")
        for i in tqdm(
            range(self._maxiter_intensity_effects), desc="EmpiricMatcher:"
        ):
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
            beam_gridded.setup_beam(
                dt=dt_flat_init.copy(),
                dE=dE_flat_init.copy(),
                reference_time=users_beam.reference_time,
                # reference_total_energy=users_beam.reference_total_energy,
                # flags=None # TODO
            )
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
                maxiter=self._maxiter_hamiltonian,
                atol=self._atol_hamiltonian,
            )
            hamilton_2D = self.hamiltonian_to_density_function(hamilton_2D)
            users_beam.reference_total_energy = reference_total_energy
            users_beam.reference_time = reference_time
            populate_beam(
                beam=users_beam,
                time_grid=time_grid,
                deltaE_grid=deltaE_grid,
                density_grid=hamilton_2D,
                n_macroparticles=self._n_macroparticles,
                seed=self._seed,
            )
            if self.animate:
                plt.figure("EmpiricMatcher")
                plt.clf()
                plt.title(f"Iteration {i}")
                plt.hist2d(
                    users_beam._dt,
                    users_beam._dE,
                    bins=len(self._grid_base_dt),
                )
                plt.draw()
                plt.pause(0.1)
                plt.clf()

        simulation.intensity_effect_manager.set_wakefields(active=True)
        simulation.intensity_effect_manager.set_profiles(active=True)

        plt.close("EmpiricMatcher")
