from __future__ import annotations

import math
from typing import Callable  # NOQA
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from blond import AllowPlotting, WakeField, backend
from blond.beam_preparation.base import MatchingRoutine

from ..._core.helpers import int_from_float_with_warning
from ...physics.profiles import ProfileBaseClass
from .helpers import populate_beam

if TYPE_CHECKING:  # pragma: no cover
    from typing import Callable, Optional, Tuple

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from blond import (
        Simulation,
    )
    from blond._core.beam.base import BeamBaseClass


def _apply_density_function(
    hamilton_2D: NumpyArray | CupyArray,
    density_modifier: float,
) -> NumpyArray | CupyArray:
    """
    Converts Hamiltonian to a density distribution

    Parameters
    ----------
    hamilton_2D
        2D representation of the Hamiltonian
        with 1 representing the limit between particles/no-particles.
        Smaller 1 means there should be particles.
    density_modifier
        Exponent that modifies the denisity distribution.

    Returns
    -------
    density
        The density according to the Hamiltonian

    """
    _density = hamilton_2D.copy()
    _density[_density > 1] = 1
    _density *= -1
    _density -= _density.min()
    _density **= density_modifier
    return _density


def get_hamiltonian_semi_analytic(
    ts: NumpyArray | CupyArray,
    potential_well: NumpyArray | CupyArray,
    reference_total_energy: float,
    eta: float,
    beta: float,
    shape: Tuple[int, int],
    energy_range: Optional[Tuple[float, float]] = None,
) -> (
    Tuple[NumpyArray, NumpyArray, NumpyArray]
    | Tuple[CupyArray, CupyArray, CupyArray]
):
    """
    Computes hamilton_2D(Δt, ΔE) based on an arbitrary potential_well.

    Computes a semi-analytic Hamiltonian hamilton_2D(t, ΔE)
    over a 2D grid defined by time (t) and energy difference (ΔE).

    Notes
    -----
    The Hamiltonian is computed using the relation:
        hamilton_2D(t, ΔE) = 0.5 * (η * E0) / (β² * c²) * ΔE² + V(t)

    Where:
        - E0 is the reference total energy.
        - V(t) is the potential well interpolated at times t.
        - c is the speed of light.

    Parameters
    ----------
    ts
        Time coordinates of the potential well, in [s].
    potential_well
        Potential energy values corresponding to `ts`, in [V].
    reference_total_energy
        Reference total energy (E0), in [eV].
    eta
        General synchrotron parameter - Zeroth order slippage factor, unitless
    shape
        Shape of the output Hamiltonian grid
        as (num_time_points, num_energy_points).
    energy_range
        Range of ΔE values to evaluate, in [eV].
        If None, it will comprise the biggest separatrix inside the given
        potential.

    Returns
    -------
    hamilton_2D
        2D array representing the semi-analytic Hamiltonian evaluated on a grid of
        time vs. energy difference. Same device (NumPy or CuPy) as inputs.
        Units: [eV]
    """

    E0 = reference_total_energy  # [eV]

    # Compute kinetic energy term constant
    drift_term = eta / (beta * beta * E0)  # [1/eV]
    assert len(ts) == len(potential_well), (
        f"{len(ts)=}, but {len(potential_well)=}"
    )

    # Auto-estimate ΔE range if not provided
    if energy_range is None:
        dE_max = backend.sqrt(
            (potential_well.max() - potential_well.min())
            / (0.5 * abs(drift_term))
        )
        _energy_range = (-dE_max, dE_max)
    else:
        _energy_range = energy_range

    assert _energy_range[1] > _energy_range[0], f"{_energy_range=}"

    # Uniformly sample time and interpolate potential well to that grid
    _ts = backend.linspace(ts.min(), ts.max(), shape[0])  # [s]
    _potential_well = backend.interp(_ts, ts, potential_well)  # [V]

    # Uniformly sample energy differences ΔE
    _dE_base = backend.linspace(
        _energy_range[0], _energy_range[1], shape[1]
    )  # [eV]

    # Create 2D meshgrid: time_grid is time [s], deltaE_grid is ΔE [eV]
    time_grid, deltaE_grid = backend.meshgrid(_ts, _dE_base, indexing="ij")
    # Expand potential V(t) to 2D grid
    V = _potential_well[:, None]  # [V]

    # Compute the Hamiltonian hamilton_2D(t, ΔE) = 0.5 * const * ΔE² + V(t)
    hamilton_2D = 0.5 * drift_term * deltaE_grid * deltaE_grid + V  # [eV]

    return deltaE_grid, time_grid, hamilton_2D


class SemiEmpiricMatcher(MatchingRoutine):
    def __init__(
        self,
        time_limit: Tuple[float, float],
        hamilton_max: float,
        n_macroparticles: int | float,
        density_modifier: float
        | Callable[[NumpyArray | CupyArray], NumpyArray | CupyArray] = 1,
        internal_grid_shape: Tuple[int, int] = (1023, 1023),
        seed: int = 0,
        tolerance: float = 1e-6,
        maxiter_intensity_effects=1000,
        increment_intensity_effects_until_iteration_i: int = 0,
        animate: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Match distribution to ``potential_well_empiric`` with analytic drift term

        Parameters
        ----------

        time_limit
            Start and stop of time, in [s]
        hamilton_max
            Maximum value of the Hamilton, in [arb. unit]
        n_macroparticles
            Number of macroparticles to inject into the beam.
        density_modifier
            H**density_modifier shapes the density distribution.

            If Callable, should be a function that maps the 2D Hamiltonian to
            a density. The Hamiltonian is 1 at the user-given maximum
            contour (given by `h_max`). and 0 at the potential minimum.
        internal_grid_shape
            Shape (n_time, t_energy) of the internal grid, which will be
            used to generate the beam particle coordinates.
        maxiter_intensity_effects
            Maximum number of iterations to convergence with intensity effects.
        increment_intensity_effects_until_iteration_i
            Number of turns to increment intensity effects
            before matching with the full beam intensity.
            This is intended to help with convergence of the algorithm
            and might be used when intensity effects are strong.
        seed
            Random seed. Runs with the same seed will return the same
            distribution
        animate
            If True, pyplot will draw() a plot on each iteration.
        tolerance
            If the error is below the tolerance, the matching is stopped.
        verbose
            If True, allows printing of convergence message
        """
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

        self.internal_grid_shape = internal_grid_shape
        self.seed = int_from_float_with_warning(
            seed,
            warning_stacklevel=2,
        )
        self.time_limit = time_limit
        self.hamilton_max = hamilton_max
        self.density_modifier = density_modifier
        self.animate = animate
        self.tolerance = tolerance
        self.verbose = verbose

        self._previous_potential_well: NumpyArray | CupyArray | None = None
        self._preprevious_potential_well: NumpyArray | CupyArray | None = None

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        super().prepare_beam(
            simulation=simulation,
            beam=beam,
        )
        ts = backend.linspace(
            self.time_limit[0], self.time_limit[1], self.internal_grid_shape[0]
        )
        # match beam without intensity effects
        simulation.intensity_effect_manager.set_wakefields(active=False)
        simulation.intensity_effect_manager.set_profiles(active=False)
        self._match_beam(beam, simulation, ts)

        # iterate solution with intensity effects
        intensity_org = beam.intensity
        hist_y_previous = None

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
            for i in range(self.maxiter_intensity_effects):
                # Change the strength of intensity effects to allow
                # convergence to a stable solution (if there is any?)
                if i < self.increment_intensity_effects_until_iteration_i:
                    scalar = (
                        i / self.increment_intensity_effects_until_iteration_i
                    )  # t
                else:
                    scalar = 1.0
                beam.intensity = scalar * intensity_org

                # run simulation with beam to collect the actual profiles
                # that cause the wake-fields
                simulation.intensity_effect_manager.set_wakefields(active=True)
                simulation.intensity_effect_manager.set_profiles(active=True)

                simulation.run_simulation(
                    beams=(beam,),
                    n_turns=1,
                    turn_i_init=0,
                    show_progressbar=False,
                )
                # Prevent the profiles from updating.
                simulation.intensity_effect_manager.set_profiles(active=False)
                # This is intended as override, so that the line density
                # inside `_match_beam` experiences the forces from the
                # previously run with the full beam
                self._match_beam(beam, simulation, ts)

                hist_y, _ = np.histogram(
                    beam.read_partial_dt(),
                    bins=self.internal_grid_shape[0],
                    range=self.time_limit,
                )
                hist_y = hist_y / np.max(hist_y)

                if self.animate:
                    plt.figure("SemiEmpiricMatcher")
                    plt.clf()
                    self._plot_current_state(beam, i, scalar, ts)
                    plt.draw()
                    plt.pause(0.1)

                error_calculable = hist_y_previous is not None
                if error_calculable:
                    # Root mean square deviation
                    error = float(
                        np.sqrt(np.mean((hist_y - hist_y_previous) ** 2))
                    )
                    if self.verbose:
                        print(
                            f"Iteration: {i:<5} | Error: {error:.{tolerance_decimal_places}f}"
                        )
                    if (
                        error < self.tolerance
                        and i
                        > self.increment_intensity_effects_until_iteration_i
                    ):
                        break

                hist_y_previous = hist_y

            simulation.intensity_effect_manager.set_wakefields(active=True)
            simulation.intensity_effect_manager.set_profiles(active=True)
            beam.intensity = intensity_org

            simulation.run_simulation(
                beams=(beam,),
                n_turns=1,
                turn_i_init=0,
                show_progressbar=False,
            )
            plt.figure("mega_debug")
            plt.subplot(2, 1, 1)
            prof = simulation.ring.elements.get_element(WakeField).profile
            plt.plot(prof.hist_x, prof.hist_y)
            plt.subplot(2, 1, 2)
            simulation.intensity_effect_manager.set_profiles(active=False)

            potential_well, factor = simulation.get_potential_well_empiric(
                ts=ts,
                particle_type=beam.particle_type,
                intensity=beam.intensity,
            )
            plt.plot(ts, beam.intensity * potential_well * factor)

    def _match_beam(
        self,
        beam: BeamBaseClass,
        simulation: Simulation,
        ts: NumpyArray | CupyArray,
    ) -> None:
        """
        Matches the beam coordinates to the current potential well

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
            Simulation context manager
        ts
            Time coordinate, in [s] for observation of the potential well.
        """
        potential_well, factor = simulation.get_potential_well_empiric(
            ts=ts, particle_type=beam.particle_type, intensity=beam.intensity
        )
        potential_well = potential_well * factor
        self._preprevious_potential_well = self._previous_potential_well
        self._previous_potential_well = potential_well  # for debugging
        if self._preprevious_potential_well is None:
            avg_pot_well = potential_well
        else:
            avg_pot_well = (
                potential_well + self._preprevious_potential_well
            ) / 2
        deltaE_grid, time_grid, hamilton_2D = get_hamiltonian_semi_analytic(
            ts=ts,
            potential_well=avg_pot_well,
            reference_total_energy=beam.reference_total_energy,
            beta=beam.reference_beta,
            eta=float(
                simulation.ring.calc_average_eta_0(beam.reference_gamma)
            ),
            shape=self.internal_grid_shape,
        )
        hamilton_2D /= self.hamilton_max
        if callable(self.density_modifier):
            density = self.density_modifier(hamilton_2D=hamilton_2D)  # type: ignore
        else:
            density = _apply_density_function(
                hamilton_2D=hamilton_2D,
                density_modifier=float(self.density_modifier),
            )
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
        """
        Make a plot of the current state of the matcher

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
        plt.figure("mega_debug")
        plt.subplot(2, 1, 1)
        plt.cla()
        plt.hist(
            beam.read_partial_dt(),
            bins=self.internal_grid_shape[0],
            range=self.time_limit,
        )
        plt.figure("SemiEmpiricMatcher")
        with AllowPlotting():
            plt.subplot(2, 1, 1)
            plt.title(f"Iteration {i}, Intensity strength {scalar * 100} %")
            plt.hist(
                beam.read_partial_dt(),
                bins=self.internal_grid_shape[0],
                range=self.time_limit,
                density=True,
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Density (arb. unit)")

            plt.subplot(2, 1, 2)
            plt.axhline(self.hamilton_max, c="C1", linestyle="--")
            if self._previous_potential_well is not None:
                plt.plot(ts, self._previous_potential_well)
            if self._preprevious_potential_well is not None:
                plt.plot(ts, self._preprevious_potential_well)
            plt.xlabel("Time (s)")
            plt.ylabel("Potential (arb. unit)")
