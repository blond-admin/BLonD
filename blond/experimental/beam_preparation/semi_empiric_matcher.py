# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from blond import AllowPlotting, backend
from blond.beam_preparation.base import MatchingRoutine
from blond.core.helpers import int_from_float_with_warning
from blond.experimental.beam_preparation.helpers import populate_beam
from blond.generals.cupy.no_cupy_import import copy_to_cpu

# Oversampling factor for potential well calculation
_POTENTIAL_WELL_OVERSAMPLING = 10

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from typing import Any

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from blond import (
        Simulation,
    )
    from blond.core.beam.base import BeamBaseClass


def hamilton_to_density_by_max(
    time_grid: NumpyArray | CupyArray,
    deltaE_grid: NumpyArray | CupyArray,
    hamilton_2D: NumpyArray | CupyArray,
    density_modifier: float,
    hamilton_max: float,
) -> NumpyArray | CupyArray:
    """
    Converts a 2D Hamilton 2D array into a density distribution.

    This function normalizes the input Hamilton by a specified maximum value,
    inverts it to represent particle density (i.e., lower energy = higher density),
    and optionally adjusts the shape of the distribution using a power-law modifier.

    Notes
    -----
    - The density is highest in regions with the lowest Hamilton values.
    - Values in `hamilton_2D` greater than `hamilton_max` are clipped before processing.
    - The function preserves the array type (NumPy or CuPy) of the input.

    Parameters
    ----------
    deltaE_grid
        The time coordinates corresponding to `hamilton_2D`, in [eV].
    time_grid
        The time coordinates corresponding to `hamilton_2D`, in [s].
    hamilton_2D
        A 2D array representing the spatial Hamilton field.
    density_modifier
        Exponent applied to the normalized and inverted Hamilton values
        to shape the final density distribution.
        Higher values exaggerate differences in density.
    hamilton_max
        The maximum reference value for normalizing the Hamilton.
        Values above this threshold are truncated.

    Returns
    -------
    density : NumpyArray or CupyArray
        A 2D array of the same shape as `hamilton_2D`, representing the
        computed density distribution. Values are scaled between 0 and 1.

    Examples
    --------
    Defining a custom function to convert between hamilton and particle
    density.
    >>> import numpy as np
    >>>
    >>> def custom_density_function(
    ...         time_grid, deltaE_grid, hamilton_2D, # required arguments
    ...         custom_param, hamilton_max # your custom arguments
    ...    ):
    ...    '''Example custom density mapping with exponential falloff.'''
    ...    normalized_H = hamilton_2D / hamilton_max
    ...    normalized_H[normalized_H > 1] = 1
    ...    density = np.exp(-custom_param * normalized_H)
    ...    return density / density.max()  # Normalize to [0, 1]
    >>>
    >>> matcher = SemiEmpiricMatcher(
    ...    time_limit=(-2e-9, 2e-9),
    ...    n_macroparticles=100_000,
    ...    hamilton_to_density_function=custom_density_function,
    ...    hamilton_to_density_kwargs=dict(
    ...        custom_param=5.0,
    ...        hamilton_max=1.0
    ...    ),
    ...    internal_grid_shape=(1023, 1023),
    ...    tolerance=1e-6,
    ...    verbose=True,
    ... )
    >>> matcher.prepare_beam(...)

    """
    _density = hamilton_2D.copy()  # So the changes stay in this scope

    _density /= hamilton_max
    # Now 1 representing the limit between particles/no-particles.
    # Smaller 1 means there should be particles.

    _density[_density > 1] = 1  # Truncate
    # Flip max with min
    _density *= -1
    _density -= _density.min()

    # Modify of the density to be more/less dense in different regions.
    _density **= density_modifier
    return _density


def get_hamilton_semi_analytic(
    ts: NumpyArray | CupyArray,
    potential_well: NumpyArray | CupyArray,
    reference_total_energy: float,
    eta: float,
    beta: float,
    shape: tuple[int, int],
    energy_range: tuple[float, float] | None = None,
) -> (
    tuple[NumpyArray, NumpyArray, NumpyArray]
    | tuple[CupyArray, CupyArray, CupyArray]
):
    r"""
    Compute the 2D Hamiltonian :math:`H_{2D}(t, \Delta E)` based on an arbitrary potential well.

    This function computes a semi-analytic Hamiltonian over a 2D grid defined by
    time (:math:`t`) and energy difference (:math:`\Delta E`).

    Notes
    -----
    The Hamiltonian is computed using the relation:

    .. math::

        H_{2D}(t, \Delta E) = \frac{1}{2} \frac{\eta E_0}{\beta^2 c^2} (\Delta E)^2 + V(t)

    where

    - :math:`E_0` is the reference total energy [eV].
    - :math:`V(t)` is the potential well interpolated at times :math:`t` [V].
    - :math:`c` is the speed of light [m/s].

    Parameters
    ----------
    ts : array_like
        Time coordinates of the potential well [s].
    potential_well : array_like
        Potential energy values corresponding to ``ts`` [V].
    reference_total_energy : float
        Reference total energy :math:`E_0` [eV].
    eta : float
        General synchrotron parameter (zeroth-order slippage factor) [unitless].
    beta : float
        Relativistic velocity factor :math:`\beta = v/c` [unitless].
    shape : tuple of int
        Shape of the output Hamiltonian grid, as ``(num_time_points, num_energy_points)``.
    energy_range : tuple of float or None, optional
        Range of :math:`\Delta E` values to evaluate [eV].
        If ``None``, it will comprise the largest separatrix inside the given potential.

    Returns
    -------
    hamilton_2D : ndarray
        2D array representing the semi-analytic Hamiltonian evaluated on a grid of
        time vs. energy difference [eV]. Uses the same device (NumPy or CuPy) as the inputs.
    """
    assert len(ts) == len(potential_well), (
        f"{len(ts)=}, but {len(potential_well)=}"
    )

    E0 = reference_total_energy  # [eV]

    # Compute kinetic energy term constant
    drift_term = np.abs(eta) / (np.square(beta) * E0)  # [1/eV]

    # Auto-estimate ΔE range if not provided
    if energy_range is None:
        dE_max = backend.sqrt(
            (potential_well.max() - potential_well.min())
            / (0.5 * abs(drift_term))
        )
        _energy_range = (-dE_max, dE_max)
    else:
        _energy_range = energy_range

    assert _energy_range[1] > _energy_range[0], (
        f"``energy_range`` must be increasing, but got, {_energy_range=}"
    )

    # Uniformly sample energy differences ΔE
    _dE_base = backend.linspace(
        _energy_range[0], _energy_range[1], shape[1]
    )  # [eV]

    # Create 2D meshgrid: time_grid is time [s], deltaE_grid is ΔE [eV]
    time_grid, deltaE_grid = backend.meshgrid(ts, _dE_base, indexing="ij")
    # Expand potential V(t) to 2D grid
    V = potential_well[:, None]  # [V]

    # Compute the Hamiltonian hamilton_2D(t, ΔE) = 0.5 * const * ΔE² + V(t)
    hamilton_2D = 0.5 * drift_term * backend.square(deltaE_grid) + V  # [eV]

    return deltaE_grid, time_grid, hamilton_2D


class SemiEmpiricMatcher(MatchingRoutine):
    r"""Match a distribution to ``potential_well_empiric`` using an analytic drift term.

    This function matches a beam distribution to the empirically determined potential well,
    including an analytic drift term. The process iteratively adjusts the distribution until
    it converges within a specified tolerance, optionally accounting for intensity effects.

    Parameters
    ----------
    time_limit : tuple of float
        Start and stop of time, in seconds (s).
        The Hamiltonian will be calculated within this time range.
    n_macroparticles : int
        Number of macroparticles to inject into the beam.
    hamilton_to_density_kwargs : dict, optional
        Keyword arguments passed to ``hamilton_to_density_function``.
        For the default function, the following keys may be used:

        - ``density_modifier`` : float
          Exponent that shapes the density distribution according to :math:`H^{\text{density_modifier}}`.
        - ``hamilton_max`` : float
          Maximum value of the Hamiltonian, in arbitrary units.
          Tip: Set `animate=True` to see the internal state of the Hamilton values.
    hamilton_to_density_function : callable
        Function that converts the 2D Hamiltonian array into a density function.
        See also :func:`hamilton_to_density_by_max`.
    internal_grid_shape : tuple of int
        Shape of the internal grid as ``(n_time, n_energy)``, used to generate
        the beam particle coordinates.
    maxiter_intensity_effects : int, optional
        Maximum number of iterations allowed for convergence with intensity effects.
    increment_intensity_effects_until_iteration_i : int, optional
        Number of iterations during which intensity effects are gradually increased
        before matching at full beam intensity.
        Useful for convergence when intensity effects are strong.
    seed : int, optional
        Random seed ensuring reproducibility.
        Runs with the same seed will produce identical distributions.
    animate : bool, default=False
        If ``True``, draws a plot using ``matplotlib.pyplot.draw()`` at each iteration.
    tolerance : float, optional
        Convergence threshold. The matching process stops when the error
        falls below this tolerance.
    verbose : bool, default=False
        If ``True``, prints convergence and status messages to the console.

    Notes
    -----
    This routine is intended for iterative beam-matching workflows where
    the potential well and intensity effects are both considered analytically.
    """

    def __init__(
        self,
        time_limit: tuple[float, float],
        n_macroparticles: int | float,
        hamilton_to_density_kwargs: dict[str, Any],
        hamilton_to_density_function: Callable = hamilton_to_density_by_max,
        internal_grid_shape: tuple[int, int] = (1023, 1023),
        seed: int | None = 0,
        tolerance: float = 1e-6,
        maxiter_intensity_effects=1000,
        increment_intensity_effects_until_iteration_i: int = 0,
        animate: bool = False,
        verbose: bool = True,
    ) -> None:
        self.n_macroparticles = int_from_float_with_warning(
            n_macroparticles,
            warning_stacklevel=2,
        )
        self.maxiter_intensity_effects = int_from_float_with_warning(
            maxiter_intensity_effects,
            warning_stacklevel=2,
        )
        self.increment_intensity_effects_until_iteration_i = (
            int_from_float_with_warning(
                increment_intensity_effects_until_iteration_i,
                warning_stacklevel=2,
            )
        )

        self.internal_grid_shape: tuple[int, int] = internal_grid_shape
        self.seed = (
            int_from_float_with_warning(
                seed,
                warning_stacklevel=2,
            )
            if seed is not None
            else None
        )
        self.time_limit: tuple[float, float] = time_limit
        assert callable(hamilton_to_density_function)
        self.hamilton_to_density_function = hamilton_to_density_function
        self.hamilton_to_density_kwargs = hamilton_to_density_kwargs
        self.animate = animate
        self.tolerance = tolerance
        self.verbose = verbose

        # For error calculation and plotting during `prepare_beam`
        self._last_potential_well: NumpyArray | CupyArray | None = None
        self._prelast_potential_well: NumpyArray | CupyArray | None = None

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """Populates the `Beam` object with macro-particles

        Parameters
        ----------
        simulation
            `Simulation` context manager
        beam
            Simulation beam object
        """

        super().prepare_beam(
            simulation=simulation,
            beam=beam,
        )

        # make sure there is not a mixed state
        simulation.intensity_effect_manager.is_active_wakefields()
        simulation.intensity_effect_manager.is_active_profiles()

        ts = backend.linspace(
            self.time_limit[0], self.time_limit[1], self.internal_grid_shape[0]
        )
        # match beam without intensity effects
        simulation.intensity_effect_manager.set_wakefields(active=False)
        simulation.intensity_effect_manager.set_profiles(active=False)
        self._match_beam(beam, simulation, ts)

        # iterate solution with intensity effects
        intensity_org = beam.intensity

        # Get decimal places from the tolerance (e.g., 1e-6 → 6)
        tolerance_decimal_places = abs(
            int(math.floor(math.log10(self.tolerance)))
        )
        if self.animate:
            plt.figure("SemiEmpiricMatcher")
            plt.clf()
            self._plot_current_state(beam, 0, 0, ts)
            plt.draw()
            plt.pause(0.1)

        if simulation.intensity_effect_manager.has_wakefields():
            simulation.intensity_effect_manager.set_wakefields(active=True)
            for i_intensity in range(self.maxiter_intensity_effects):
                sim_tmp = deepcopy(simulation)

                # Change the strength of intensity effects to allow
                # convergence to a stable solution (if there is any?)
                if (
                    i_intensity
                    < self.increment_intensity_effects_until_iteration_i
                ):
                    scalar = (
                        i_intensity
                        / self.increment_intensity_effects_until_iteration_i
                    )  # t
                else:
                    scalar = 1.0
                beam.intensity = scalar * intensity_org

                # run simulation with beam to collect the actual profiles
                # that cause the wake-fields
                sim_tmp.intensity_effect_manager.set_profiles(active=True)

                # this might get changed by the simulation
                beam_reference_time = beam.reference.time
                beam_reference_total_energy = beam.reference.total_energy
                turn_i_org = int(simulation.turn_i.value)
                section_i_org = int(simulation.section_i.value)

                sim_tmp.run_simulation(
                    beams=(beam,),
                    n_turns=1,
                    show_progressbar=False,
                )

                # reset to original value before simulation
                beam.reference.time = beam_reference_time
                beam.reference.total_energy = beam_reference_total_energy
                sim_tmp.turn_i.value = turn_i_org
                sim_tmp.section_i.value = section_i_org

                # Prevent the profiles from updating.
                sim_tmp.intensity_effect_manager.set_profiles(active=False)
                # This is intended as override, so that the line density
                # inside `_match_beam` experiences the forces from the
                # previously run with the full beam
                self._match_beam(beam, sim_tmp, ts)

                if self.animate:
                    plt.figure("SemiEmpiricMatcher")
                    plt.clf()
                    self._plot_current_state(beam, i_intensity, scalar, ts)
                    plt.draw()
                    plt.pause(0.1)
                    assert sim_tmp.turn_i.value == 0

                error_calculable = (
                    self._last_potential_well is not None
                    and self._prelast_potential_well is not None
                )
                # calculate errors on wakes (not on beam profiles)
                # because this will more stable as noise is smoothed out
                if error_calculable and i_intensity > 1:
                    # Root mean square deviation
                    obs = self._last_potential_well
                    obs = obs / obs.max()
                    obs_previous = self._prelast_potential_well
                    obs_previous = obs_previous / obs_previous.max()
                    error = float(
                        np.sqrt(backend.mean((obs - obs_previous) ** 2))
                    )
                    if self.verbose:
                        print(
                            f"Iteration: {i_intensity:<5}"
                            f" | Intensity {(scalar * 100):3.1f}%"
                            f" | Error: {error:.{tolerance_decimal_places}f}"
                        )
                    if (
                        error < self.tolerance
                        and i_intensity
                        > self.increment_intensity_effects_until_iteration_i
                    ):
                        break

            beam.intensity = intensity_org

        simulation.intensity_effect_manager.set_wakefields(active=True)
        simulation.intensity_effect_manager.set_profiles(active=True)

    def _match_beam(
        self,
        beam: BeamBaseClass,
        simulation: Simulation,
        ts: NumpyArray | CupyArray,
    ) -> None:
        """Matches the beam coordinates to the current potential well.

        Notes
        -----
        The potential well is overwritten by the intensity effect manager
        outside this method, so that the line distribution experiences
        the same forces than the actual bunch.

        Parameters
        ----------
        beam
            Simulation beam object
        simulation
            `Simulation` context manager
        ts
            Time coordinate, in [s] for observation of the potential well.
        """
        assert simulation.turn_i.value == 0
        potential_well, factor, tilt_dt_per_dE = (
            simulation.get_potential_well_empiric(
                dt=np.linspace(
                    ts.min(),
                    ts.max(),
                    len(ts) * _POTENTIAL_WELL_OVERSAMPLING,
                ),
                particle_type=beam.particle_type,
                intensity=beam.intensity,
            )
        )
        potential_well = (
            potential_well[::_POTENTIAL_WELL_OVERSAMPLING] * factor
        )

        self._prelast_potential_well = self._last_potential_well
        self._last_potential_well = potential_well
        if self._prelast_potential_well is None:
            avg_pot_well = potential_well
        else:
            avg_pot_well = (potential_well + self._prelast_potential_well) / 2

        deltaE_grid, time_grid, hamilton_2D = get_hamilton_semi_analytic(
            ts=ts,
            potential_well=avg_pot_well,
            reference_total_energy=beam.reference.total_energy,
            beta=beam.reference.beta,
            eta=float(
                simulation.ring.calc_average_eta_0(beam.reference.gamma)
            ),
            shape=self.internal_grid_shape,
        )
        density = self.hamilton_to_density_function(
            time_grid=time_grid,
            deltaE_grid=deltaE_grid,
            hamilton_2D=hamilton_2D,
            **self.hamilton_to_density_kwargs,
        )  # type: ignore

        populate_beam(
            beam=beam,
            time_grid=time_grid.T,
            deltaE_grid=deltaE_grid.T,
            density_grid=density.T,
            n_macroparticles=self.n_macroparticles,
            seed=self.seed,
        )

    def _plot_current_state(
        self,
        beam: BeamBaseClass,
        i: int,
        scalar: float,
        ts: NumpyArray | CupyArray,
    ) -> None:
        """Make a plot of the current state of the matcher.

        Parameters
        ----------
        beam
            Simulation beam object
        i
            Current iteration
        scalar
            Current strength if the intensity effects
        ts
            Time coordinate, in [s] for observation of the potential well.


        """
        plt.figure("SemiEmpiricMatcher")
        with AllowPlotting():
            plt.subplot(2, 1, 1)
            plt.title(
                f"Iteration {i}, Intensity strength {(scalar * 100):3.1f}%"
            )
            plt.hist(
                copy_to_cpu(beam.read_partial_dt()),
                bins=self.internal_grid_shape[0],
                range=self.time_limit,
                density=True,
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Density (arb. unit)")

            plt.subplot(2, 1, 2)
            if self._last_potential_well is not None:
                plt.plot(ts, self._last_potential_well)
            if self._prelast_potential_well is not None:
                plt.plot(ts, self._prelast_potential_well)
            plt.xlabel("Time (s)")
            plt.ylabel("Potential (arb. unit)")
