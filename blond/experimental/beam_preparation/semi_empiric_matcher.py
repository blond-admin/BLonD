from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
from numba.cuda.libdevicedecl import retty

from blond import backend
from blond.beam_preparation.base import MatchingRoutine

from ..._core.helpers import int_from_float_with_warning
from .helpers import populate_beam

if TYPE_CHECKING:  # pragma: no cover
    from typing import Tuple

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from blond import (
        Simulation,
    )
    from blond._core.beam.base import BeamBaseClass


def apply_density_function(array: NumpyArray | CupyArray):
    array[array < (array.max() * 0.9)] = 0
    return  # TODO


def get_hamiltonian_semi_analytic(
    ts: NumpyArray | CupyArray,
    potential_well: NumpyArray | CupyArray,
    reference_total_energy: float,
    beta: float,
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
    beta
        Normalized particle velocity (v / c), unitless.
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
    from scipy.constants import speed_of_light as c

    E0 = reference_total_energy  # [eV]

    # Compute kinetic energy term constant
    kinetic_energy_term = (eta * E0) / (beta * beta * c * c)  # [eV⁻¹]

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


def _plot_ham(Delta, T, hamiltonian):
    cs = plt.contour(T, Delta, hamiltonian, levels=50)
    plt.xlabel("Time [s]")
    plt.ylabel("ΔE  [eV]")
    plt.title("Longitudinal Hamiltonian [eV]")
    plt.colorbar(label="hamiltonian [eV]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


class SemiEmpiricMatcher(MatchingRoutine):
    def __init__(
        self,
        n_macroparticles: int | float,
        internal_grid_shape: Tuple[int, int] = (1024, 1024),
        seed: int = 0,
    ):
        self._n_macroparticles = int_from_float_with_warning(
            n_macroparticles,
            warning_stacklevel=2,
        )
        self._internal_grid_shape = internal_grid_shape
        self._seed = int_from_float_with_warning(
            seed,
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
        t_max = simulation.magnetic_cycle.get_t_rev_init(
            simulation.ring.circumference,
            turn_i_init=0,
            t_init=0,
            particle_type=beam.particle_type,
        )
        ts = np.linspace(0, t_max / 35640, self._internal_grid_shape[0])
        potential_well = simulation.get_potential_well_empiric(
            ts=ts, particle_type=beam.particle_type
        )
        potential_well *= -1
        deltaE_grid, time_grid, hamilton_2D = get_hamiltonian_semi_analytic(
            ts=ts,
            potential_well=potential_well,
            reference_total_energy=beam.reference_total_energy,
            beta=beam.reference_beta,
            eta=float(
                simulation.ring.calc_average_eta_0(beam.reference_gamma)
            ),
            shape=self._internal_grid_shape,
        )
        hamilton_2D *= -1
        hamilton_2D -= hamilton_2D.min()
        apply_density_function(hamilton_2D)
        populate_beam(
            beam=beam,
            time_grid=time_grid.T,
            deltaE_grid=deltaE_grid.T,
            density_grid=hamilton_2D.T,
            n_macroparticles=self._n_macroparticles,
            seed=self._seed,
            normalize_density=True,
        )
