from __future__ import annotations

from typing import Callable  # NOQA
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from blond import WakeField, backend
from blond.beam_preparation.base import MatchingRoutine

from ..._core.helpers import int_from_float_with_warning
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
    hamilton_2D: NumpyArray | CupyArray, density_modifier: float
):
    _hamilton_2D = hamilton_2D.copy()
    _hamilton_2D[_hamilton_2D > 1] = 1
    _hamilton_2D *= -1
    _hamilton_2D -= _hamilton_2D.min()
    _hamilton_2D **= density_modifier
    return _hamilton_2D


def get_hamiltonian_semi_analytic(
    ts: NumpyArray | CupyArray,
    potential_well: NumpyArray | CupyArray,
    reference_total_energy: float,
    eta: float,
    shape: Tuple[int, int],
    energy_range: Optional[Tuple[float, float]] = None,
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
    kinetic_energy_term = eta / E0  # [1/eV]
    assert len(ts) == len(potential_well), (
        f"{len(ts)=}, but {len(potential_well)=}"
    )

    # Auto-estimate ΔE range if not provided
    if energy_range is None:
        dE_max = backend.sqrt(
            (potential_well.max() - potential_well.min())
            / (0.5 * abs(kinetic_energy_term))
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
    hamilton_2D = (
        0.5 * kinetic_energy_term * deltaE_grid * deltaE_grid + V
    )  # [eV]

    return deltaE_grid, time_grid, hamilton_2D


class SemiEmpiricMatcher(MatchingRoutine):
    def __init__(
        self,
        t_lim: Tuple[float, float],
        h_max: float,
        n_macroparticles: int | float,
        density_modifier: float
        | Callable[[NumpyArray | CupyArray], NumpyArray | CupyArray] = 1,
        internal_grid_shape: Tuple[int, int] = (1023, 1023),
        maxiter_intensity_effects=1000,
        seed: int = 0,
    ):
        """
        Match distribution to ``potential_well_empiric`` with analytic drift term

        Parameters
        ----------
        t_lim
            Start and stop of time, in [s]
        h_max
            Maximum value of the Hamilton, in [arb. unit]
        n_macroparticles
            Number of macroparticles to inject into the beam
        density_modifier
            H**density_modifier shapes the density distribution.

            If Callable, should be a function that maps the 2D Hamiltonian to
            a density. The Hamiltonian is 1 at the user-given maximum
            contour (given by `h_max`)

        internal_grid_shape
        seed
        """
        self._n_macroparticles = int_from_float_with_warning(
            n_macroparticles,
            warning_stacklevel=2,
        )
        self._maxiter_intensity_effects = int_from_float_with_warning(
            maxiter_intensity_effects,
            warning_stacklevel=2,
        )

        self._internal_grid_shape = internal_grid_shape
        self._seed = int_from_float_with_warning(
            seed,
            warning_stacklevel=2,
        )
        self.t_lim = t_lim
        self.h_max = h_max
        self.density_modifier = density_modifier

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
            self.t_lim[0], self.t_lim[1], self._internal_grid_shape[0]
        )
        # match beam without intensity effects
        simulation.intensity_effect_manager.set_wakefields(active=False)
        simulation.intensity_effect_manager.set_profiles(active=False)
        self._match_beam(beam, simulation, ts)

        # iterate solution with intensity effects
        intensity_org = beam.intensity
        if simulation.intensity_effect_manager.has_wakefields():
            for i in range(self._maxiter_intensity_effects):
                if i < 100:
                    scalar = i / 100  # t
                else:
                    scalar = 1
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
                plt.figure(264542)
                plt.cla()
                plt.title(f"{i=}")

                a = beam.histogram(
                    bins=self._internal_grid_shape[0], range=self.t_lim
                )
                plt.plot(a)
                plt.draw()
                plt.pause(0.1)

            simulation.intensity_effect_manager.set_wakefields(active=True)
            simulation.intensity_effect_manager.set_profiles(active=True)

    def _match_beam(self, beam, simulation, ts):
        potential_well = simulation.get_potential_well_empiric(
            ts=ts, particle_type=beam.particle_type, intensity=beam.intensity
        )
        deltaE_grid, time_grid, hamilton_2D = get_hamiltonian_semi_analytic(
            ts=ts,
            potential_well=potential_well,
            reference_total_energy=beam.reference_total_energy,
            eta=float(
                simulation.ring.calc_average_eta_0(beam.reference_gamma)
            ),
            shape=self._internal_grid_shape,
        )
        hamilton_2D /= self.h_max
        if isinstance(self.density_modifier, Callable):
            density = self.density_modifier(hamilton_2D=hamilton_2D)
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
            n_macroparticles=self._n_macroparticles,
            seed=self._seed,
            normalize_density=True,
        )
        plt.figure(1234)
        plt.clf()
        plt.matshow(density.T, fignum=1234)
        plt.draw()
        plt.pause(0.1)
