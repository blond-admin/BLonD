# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

r"""Abel transform: distribution function from a line density.

Reconstructs the stationary phase-space distribution :math:`F(H)` from a
line density :math:`\lambda(t)` and the potential well :math:`V(t)` it
lives in, following the BLonD 2 ``matched_from_line_density`` route
(Abel inversion over one monotonic branch of the well):

.. math::

    F(H_i) = \pm \frac{\sqrt{\mathrm{eom\_factor\_dE}}}{\pi}
        \int \frac{\mathrm{d}\lambda/\mathrm{d}t}
                  {\sqrt{V(t) - V(t_i)}} \,\mathrm{d}t

evaluated on the ``"first"`` (left of the well minimum) or ``"second"``
(right) branch, or on ``"both"`` with the two results averaged — the
robust choice for asymmetric (e.g. intensity-distorted) profiles.

The line density is used as given: measured profiles are expected to be
clean (baseline-subtracted, low noise). The transform differentiates
:math:`\lambda(t)`, so noise is amplified; filtering is deliberately
left to the caller, as an appropriate filter (type, parameters) depends
on the bunch length and can bias the profile.

Deviations from BLonD 2 (accuracy fixes, not behavior changes):

* the singular integrand endpoint is linearly extrapolated on *both*
  branches — BLonD 2 built the extrapolation for the second branch but
  immediately overwrote it with a nearest-neighbour copy (a missing
  ``elif``);
* the trapezoid integration uses the actual (resampled) grid spacing —
  BLonD 2 passed the original profile resolution, a constant scale
  error absorbed by the final density normalization;
* the branches are split at the potential-well minimum rather than at
  the line-density maximum (identical for a centred bunch, robust for
  a flat-topped or noisy profile).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from typing import Literal

    from numpy.typing import NDArray as NumpyArray


def _abel_transform_branch(
    time_branch: NumpyArray,
    line_density_derivative_branch: NumpyArray,
    potential_branch: NumpyArray,
    *,
    eom_factor_dE: float,
    branch: Literal["first", "second"],
    n_points_abel: int | None,
) -> tuple[NumpyArray, NumpyArray]:
    r"""
    Abel-invert one monotonic branch of the potential well.

    Returns the Hamiltonian coordinates :math:`H_i = V(t_i)` (referenced
    to the branch minimum) and the distribution values :math:`F(H_i)`,
    in branch order (not sorted). NaNs from a non-monotonic branch and
    negative values are zeroed, as in BLonD 2.
    """
    if n_points_abel is not None:
        time_abel = np.linspace(
            float(time_branch[0]), float(time_branch[-1]), int(n_points_abel)
        )
        line_density_derivative_abel = np.interp(
            time_abel, time_branch, line_density_derivative_branch
        )
        potential_abel = np.interp(time_abel, time_branch, potential_branch)
    else:
        time_abel = np.asarray(time_branch, dtype=float)
        line_density_derivative_abel = line_density_derivative_branch
        potential_abel = potential_branch
    potential_abel = potential_abel - np.min(potential_abel)

    n_points = len(time_abel)
    distribution_values = np.zeros(n_points)
    prefactor = np.sqrt(eom_factor_dE) / np.pi

    with np.errstate(invalid="ignore", divide="ignore"):
        for i in range(n_points):
            if branch == "first":
                integrand = line_density_derivative_abel[: i + 1] / np.sqrt(
                    potential_abel[: i + 1] - potential_abel[i]
                )
                # The integrand diverges (integrably) at its last point:
                # extrapolate it linearly from the previous two samples.
                if len(integrand) > 2:
                    integrand[-1] = integrand[-2] + (
                        integrand[-2] - integrand[-3]
                    )
                elif len(integrand) > 1:
                    integrand[-1] = integrand[-2]
                else:
                    integrand = np.zeros(1)
                distribution_values[i] = prefactor * np.trapezoid(
                    integrand, x=time_abel[: i + 1]
                )
            else:
                integrand = line_density_derivative_abel[i:] / np.sqrt(
                    potential_abel[i:] - potential_abel[i]
                )
                if len(integrand) > 2:
                    integrand[0] = integrand[1] - (integrand[2] - integrand[1])
                elif len(integrand) > 1:
                    integrand[0] = integrand[1]
                else:
                    integrand = np.zeros(1)
                distribution_values[i] = -prefactor * np.trapezoid(
                    integrand, x=time_abel[i:]
                )

    # Unphysical results are zeroed, as in BLonD 2 (which cleaned NaN
    # and negatives): NaN/inf from non-monotonic or duplicated
    # potential samples, negative density from noise.
    distribution_values[~np.isfinite(distribution_values)] = 0.0
    distribution_values[distribution_values < 0.0] = 0.0

    return potential_abel, distribution_values


def distribution_from_line_density(
    time_array: NumpyArray,
    line_density_values: NumpyArray,
    potential_well: NumpyArray,
    *,
    eom_factor_dE: float,
    half_option: Literal["first", "second", "both"] = "first",
    n_points_abel: int | None = None,
    verbose: bool = False,
    plot: bool = False,
) -> tuple[NumpyArray, NumpyArray]:
    r"""
    Reconstruct :math:`F(H)` from a line density (Abel transform).

    The bunch is assumed centred in the well: the well minimum must sit
    inside the profile support (the caller — e.g. the line-density
    matcher — is responsible for centering measured profiles, which are
    arbitrarily positioned).

    Parameters
    ----------
    time_array
        Time coordinates, in [s]. Line density and potential well must
        be sampled on this same grid (interpolate beforehand).
    line_density_values
        Line density :math:`\lambda(t)` at ``time_array`` (arbitrary
        normalization; expected clean — see the module docstring).
    potential_well
        Single-bucket potential well :math:`V(t)` at ``time_array``,
        in [eV].
    eom_factor_dE
        Kinetic coefficient :math:`|\eta_0|/(2\beta^2 E)`, in [1/eV]
        (see
        :func:`~blond.experimental.beam_preparation.analytic_hamiltonian.calc_eom_factor_dE`).
    half_option
        Branch of the well used for the inversion: ``"first"`` (left of
        the minimum, the BLonD 2 default), ``"second"`` (right), or
        ``"both"`` (average of the two — robust for asymmetric
        profiles).
    n_points_abel
        Resample each branch to this many uniform points before the
        inversion (the BLonD 2 route used ``1e4``). ``None`` (default)
        keeps the input sampling. The 1/sqrt singularity converges
        slowly, so a finer grid improves accuracy.
    verbose
        If True, print diagnostic quantities.
    plot
        If True, draw the line density, the well and the reconstructed
        :math:`F(H)`.

    Returns
    -------
    hamiltonian_coord
        Hamiltonian coordinates, in [eV], sorted ascending, referenced
        to the well minimum (:math:`H=0` at the bunch centre).
    distribution_values
        :math:`F(H)` at ``hamiltonian_coord`` (normalization inherited
        from ``line_density_values``; negative/NaN values zeroed).
    """
    time_array = np.asarray(time_array, dtype=float)
    line_density_values = np.asarray(line_density_values, dtype=float)
    potential_well = np.asarray(potential_well, dtype=float)
    assert time_array.shape == line_density_values.shape, (
        f"{time_array.shape=} must match {line_density_values.shape=}"
    )
    assert time_array.shape == potential_well.shape, (
        f"{time_array.shape=} must match {potential_well.shape=}"
    )
    if half_option not in ("first", "second", "both"):
        raise ValueError(
            f"Unknown {half_option=}; use 'first', 'second' or 'both'."
        )

    # Central-difference derivative on the full profile, so the bunch
    # centre keeps a two-sided estimate on both branches.
    line_density_derivative = np.gradient(line_density_values, time_array)

    # A symmetric well sampled on an even grid carries two (or more)
    # equal minimum samples: end the first branch at the first
    # occurrence and start the second at the last, so neither branch
    # opens with a duplicated potential value (an exact division by
    # zero in the Abel integrand).
    minimum_index_first = int(np.argmin(potential_well))
    minimum_index_second = int(
        len(potential_well) - 1 - np.argmin(potential_well[::-1])
    )
    if minimum_index_first < 2 or minimum_index_second > len(time_array) - 3:
        raise ValueError(
            "The potential well minimum sits on the frame edge "
            f"(index {minimum_index_first} of {len(time_array)}): the "
            "well does not contain a centred bunch to invert."
        )

    branches: list[Literal["first", "second"]]
    if half_option == "both":
        branches = ["first", "second"]
    else:
        branches = [half_option]

    results = {}
    for branch in branches:
        if branch == "first":
            branch_slice = slice(None, minimum_index_first + 1)
        else:
            branch_slice = slice(minimum_index_second, None)
        results[branch] = _abel_transform_branch(
            time_array[branch_slice],
            line_density_derivative[branch_slice],
            potential_well[branch_slice],
            eom_factor_dE=eom_factor_dE,
            branch=branch,
            n_points_abel=n_points_abel,
        )

    if half_option == "both":
        hamiltonian_first, distribution_first = results["first"]
        hamiltonian_second, distribution_second = results["second"]
        # The second branch runs from the minimum outwards, so its
        # Hamiltonian coordinates are already ascending for np.interp.
        distribution_values = (
            distribution_first
            + np.interp(
                hamiltonian_first, hamiltonian_second, distribution_second
            )
        ) / 2.0
        hamiltonian_coord = hamiltonian_first
    else:
        hamiltonian_coord, distribution_values = results[half_option]

    ascending = np.argsort(hamiltonian_coord)
    hamiltonian_coord = hamiltonian_coord[ascending]
    distribution_values = distribution_values[ascending]

    if verbose:
        print(
            "[distribution_from_line_density] "
            f"{half_option=}, branches at minimum index "
            f"{minimum_index_first}, "
            f"H range [0, {hamiltonian_coord.max():.6e}] eV, "
            f"F(0)={distribution_values[0]:.6e}, "
            f"zeroed points="
            f"{int(np.sum(distribution_values == 0.0))}"
            f"/{len(distribution_values)}"
        )

    if plot:
        _plot_abel_transform(
            time_array,
            line_density_values,
            potential_well,
            hamiltonian_coord,
            distribution_values,
        )

    return hamiltonian_coord, distribution_values


def _plot_abel_transform(
    time_array: NumpyArray,
    line_density_values: NumpyArray,
    potential_well: NumpyArray,
    hamiltonian_coord: NumpyArray,
    distribution_values: NumpyArray,
) -> None:
    """Diagnostic figure: input line density and well, reconstructed F(H)."""
    import matplotlib.pyplot as plt

    fig, (ax_input, ax_result) = plt.subplots(
        1, 2, num="distribution_from_line_density", figsize=(9, 4)
    )
    ax_input.plot(
        time_array * 1e9,
        line_density_values / line_density_values.max(),
        label="Line density (norm.)",
    )
    ax_well = ax_input.twinx()
    ax_well.plot(
        time_array * 1e9, potential_well, color="C1", label="Potential well"
    )
    ax_input.set_xlabel("Time [ns]")
    ax_input.set_ylabel("Line density [norm.]")
    ax_well.set_ylabel("Potential well [eV]")
    ax_input.legend(loc="upper left")
    ax_well.legend(loc="upper right")
    ax_result.plot(hamiltonian_coord, distribution_values)
    ax_result.set_xlabel("Hamiltonian [eV]")
    ax_result.set_ylabel("Distribution function F(H)")
    fig.tight_layout()
