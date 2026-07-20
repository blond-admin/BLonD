# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Analytic 2D longitudinal Hamiltonian on a (time, energy) grid.

Full-analytical generation of the single-turn Hamiltonian, following the
BLonD 2 ``matched_from_distribution_function`` convention:

.. math::

    H(t, \\Delta E) = \\frac{|\\eta_0|}{2 \\beta^2 E}\\, \\Delta E^2 + V(t)

where :math:`V(t)` is the analytic RF potential well (see
:mod:`~blond.experimental.beam_preparation.analytic_potential_well`).

The overlap with the semi-empiric ``get_hamilton_semi_analytic`` is tracked in
``redundancy_notes.md`` in the project base folder (outside the blond repo).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


def eom_factor_dE(eta_0: float, beta: float, total_energy: float) -> float:
    r"""
    Kinetic coefficient of the longitudinal Hamiltonian.

    :math:`|\eta_0| / (2 \beta^2 E)`, in [1/eV]. This is the factor in front
    of :math:`\Delta E^2` in the Hamiltonian, matching the BLonD 2
    ``eom_factor_dE`` and solfege ``eom_factor_energy``.

    Parameters
    ----------
    eta_0
        Zeroth-order slippage factor.
    beta
        Relativistic beta of the reference particle.
    total_energy
        Total energy of the reference particle, in [eV].

    Returns
    -------
    factor
        Kinetic coefficient, in [1/eV].
    """
    return abs(eta_0) / (2.0 * beta**2 * total_energy)


def hamiltonian_grid(
    time_potential: NumpyArray,
    potential_well: NumpyArray,
    *,
    eom_factor_dE: float,
    n_points_deltaE: int | None = None,
    energy_range: tuple[float, float] | None = None,
    verbose: bool = False,
    plot: bool = False,
) -> tuple[NumpyArray, NumpyArray, NumpyArray]:
    r"""
    Build the analytic 2D Hamiltonian over a (time, energy) grid.

    :math:`H(t, \Delta E) = \mathrm{eom\_factor\_dE}\, \Delta E^2 + V(t)`.

    The returned arrays follow the BLonD 2 convention (``np.meshgrid`` with
    default ``"xy"`` indexing): shape ``(n_points_deltaE, len(time_potential))``,
    with :math:`\Delta E` varying along axis 0 and time along axis 1. Summing a
    density over axis 0 therefore yields the line density versus time.

    Parameters
    ----------
    time_potential
        Time coordinates of the potential well, in [s].
    potential_well
        Potential well values at ``time_potential``, in [eV]. Expected to be
        already restricted to a single bucket and shifted so its minimum is 0.
    eom_factor_dE
        Kinetic coefficient :math:`|\eta_0|/(2\beta^2 E)`, in [1/eV]
        (see :func:`eom_factor_dE`).
    n_points_deltaE
        Number of energy points. Defaults to ``len(time_potential)`` (square).
    energy_range
        ``(dE_min, dE_max)`` for the energy axis, in [eV]. If ``None``, the
        range is taken from the separatrix:
        :math:`\Delta E_\max = \sqrt{(V_\max - V_\min)/\mathrm{eom\_factor\_dE}}`.
    verbose
        If True, print diagnostic quantities.
    plot
        If True, draw a diagnostic contour of the Hamiltonian.

    Returns
    -------
    time_grid
        2D time grid, in [s], shape ``(n_points_deltaE, n_time)``.
    deltaE_grid
        2D energy grid, in [eV], same shape.
    hamilton_2D
        2D Hamiltonian, in [eV], same shape.
    """
    time_potential = np.asarray(time_potential, dtype=float)
    potential_well = np.asarray(potential_well, dtype=float)
    assert time_potential.shape == potential_well.shape, (
        f"{time_potential.shape=} must match {potential_well.shape=}"
    )

    if n_points_deltaE is None:
        n_points_deltaE = len(time_potential)

    if energy_range is None:
        barrier = float(potential_well.max() - potential_well.min())
        deltaE_max = np.sqrt(barrier / eom_factor_dE)
        energy_range = (-deltaE_max, deltaE_max)

    assert energy_range[1] > energy_range[0], (
        f"`energy_range` must be increasing, got {energy_range=}"
    )

    deltaE_array = np.linspace(
        energy_range[0], energy_range[1], n_points_deltaE
    )

    time_grid, deltaE_grid = np.meshgrid(time_potential, deltaE_array)
    hamilton_2D = (
        eom_factor_dE * deltaE_grid**2 + potential_well[np.newaxis, :]
    )

    if verbose:
        print(
            "[hamiltonian_grid] "
            f"shape={hamilton_2D.shape}, "
            f"dE_range=[{energy_range[0]:.3e}, {energy_range[1]:.3e}] eV, "
            f"H separatrix (barrier)={potential_well.max():.3e} eV, "
            f"H max={hamilton_2D.max():.3e} eV"
        )

    if plot:
        _plot_hamiltonian(time_grid, deltaE_grid, hamilton_2D, potential_well)

    return time_grid, deltaE_grid, hamilton_2D


def _plot_hamiltonian(
    time_grid: NumpyArray,
    deltaE_grid: NumpyArray,
    hamilton_2D: NumpyArray,
    potential_well: NumpyArray,
) -> None:
    """Quick diagnostic contour of the 2D Hamiltonian with the separatrix."""
    import matplotlib.pyplot as plt

    plt.figure("hamiltonian_grid")
    plt.contourf(
        time_grid * 1e9, deltaE_grid / 1e6, hamilton_2D, levels=40
    )
    plt.colorbar(label="Hamiltonian [eV]")
    plt.contour(
        time_grid * 1e9,
        deltaE_grid / 1e6,
        hamilton_2D,
        levels=[float(potential_well.max())],
        colors="w",
        linewidths=1.5,
    )
    plt.xlabel("Time [ns]")
    plt.ylabel("Energy offset [MeV]")
    plt.title("Analytic 2D Hamiltonian")
