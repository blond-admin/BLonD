# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

r"""Analytic distribution families and bunch-length targeting.

Ports the BLonD 2 stationary distribution families (formulas from
Laclare) evaluated in the Hamiltonian or Action variable
:math:`X \in \{H, J\}`:

.. math::

    g(X) = (1 - X/X_0)^{\mu} \quad (X \leq X_0), \qquad
    g(X) = e^{-2 X/X_0} \ \text{(gaussian)}

and the corresponding analytic line densities
:math:`\lambda(t)`. The named types map to fixed exponents
(``DISTRIBUTION_EXPONENTS``): waterbag :math:`\mu=0`,
parabolic_amplitude :math:`\mu=1`, parabolic_line :math:`\mu=1/2`.

Exponent convention (+1/2)
--------------------------
Projecting a binomial phase-space density over :math:`\Delta E` with
:math:`H = \mathrm{eom\_factor\_dE}\,\Delta E^2 + V(t)` gives

.. math::

    \lambda(t) \propto (1 - V(t)/H_0)^{\mu + 1/2},

so a phase-space exponent :math:`\mu` corresponds to a line-density
exponent :math:`\mu + 1/2` — in the linear regime
(:math:`V \propto t^2`) a binomial in :math:`H` therefore projects to a
binomial in :math:`t` with exponent :math:`\mu + 1/2`. The line-density
families here bake that shift in, exactly as BLonD 2 did: the *same*
``distribution_type``/``exponent`` inputs describe matching phase-space
and line-density shapes.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from typing import Literal

    from numpy.typing import NDArray as NumpyArray

# Named distribution types and their phase-space exponent mu.
# The corresponding line-density exponent is mu + 1/2 (see module doc).
DISTRIBUTION_EXPONENTS = {
    "waterbag": 0.0,
    "parabolic_amplitude": 1.0,
    "parabolic_line": 0.5,
}


def _binomial_family(
    values: NumpyArray, scale: float, exponent: float
) -> NumpyArray:
    """
    Masked binomial kernel ``(1 - values/scale)**exponent``.

    The support mask (``values <= scale``) is applied *before* the
    power, so grids containing ``inf`` (e.g. ``action_grid`` output
    outside the bucket) or values beyond the support produce exact
    zeros without RuntimeWarnings — unlike the BLonD 2 formulation,
    which computed NaNs first and cleaned them after under suppressed
    warnings.
    """
    result = np.zeros_like(values, dtype=float)
    inside = values <= scale
    result[inside] = (1.0 - values[inside] / scale) ** exponent
    return result


def _resolve_exponent(distribution_type: str, exponent: float | None) -> float:
    """Map a named type to its exponent; validate the binomial input."""
    if distribution_type in DISTRIBUTION_EXPONENTS:
        if exponent is not None:
            warnings.warn(
                f"exponent is ignored for {distribution_type=}",
                UserWarning,
                stacklevel=3,
            )
        return DISTRIBUTION_EXPONENTS[distribution_type]
    if distribution_type == "binomial":
        if exponent is None:
            raise ValueError(
                "distribution_type='binomial' requires an exponent"
            )
        return float(exponent)
    raise ValueError(
        f"Unknown {distribution_type=}; expected one of "
        f"{list(DISTRIBUTION_EXPONENTS) + ['binomial', 'gaussian']}"
    )


def distribution_function(
    x_array: NumpyArray,
    distribution_type: str,
    x_0: float,
    exponent: float | None = None,
) -> NumpyArray:
    r"""
    Stationary phase-space distribution :math:`g(X)` (BLonD 2 families).

    Parameters
    ----------
    x_array
        Values of the distribution variable :math:`X` — the Hamiltonian
        in [eV] or the Action in [eV.s] — any shape (typically the 2D
        grid). May contain ``inf`` (outside-bucket marker of
        ``action_grid``): those points evaluate to 0.
    distribution_type
        ``"waterbag"``, ``"parabolic_amplitude"``, ``"parabolic_line"``,
        ``"binomial"`` or ``"gaussian"``.
    x_0
        Distribution size parameter :math:`X_0`, same unit as
        ``x_array``: support edge for the binomial families
        (:math:`g = (1 - X/X_0)^{\mu}`), decay constant for the
        gaussian (:math:`g = e^{-2X/X_0}`).
    exponent
        Binomial exponent :math:`\mu`; required for ``"binomial"``,
        ignored (with a warning) for the named types.

    Returns
    -------
    distribution
        :math:`g(X)` evaluated at ``x_array`` (not normalized).

    Notes
    -----
    The gaussian family has unbounded support: when evaluated in the
    Hamiltonian variable, points outside the separatrix keep non-zero
    density — zero them with an inside-bucket mask (as BLonD 2 did with
    ``density_grid[H_grid > H_max] = 0``).
    """
    x_array = np.asarray(x_array, dtype=float)
    if distribution_type == "gaussian":
        if exponent is not None:
            warnings.warn(
                f"exponent is ignored for {distribution_type=}",
                UserWarning,
                stacklevel=2,
            )
        return np.exp(-2.0 * x_array / x_0)
    resolved_exponent = _resolve_exponent(distribution_type, exponent)
    return _binomial_family(x_array, x_0, resolved_exponent)


def line_density(
    time_array: NumpyArray,
    distribution_type: str,
    bunch_length: float,
    bunch_position: float = 0.0,
    exponent: float | None = None,
) -> NumpyArray:
    r"""
    Analytic line density :math:`\lambda(t)` (BLonD 2 families).

    For the binomial families the exponent is the phase-space exponent
    :math:`\mu` **plus 1/2** (see the module docstring), so the same
    ``distribution_type``/``exponent`` inputs describe the line density
    matching :func:`distribution_function`.

    Parameters
    ----------
    time_array
        Time coordinates, in [s].
    distribution_type
        ``"waterbag"``, ``"parabolic_amplitude"``, ``"parabolic_line"``,
        ``"binomial"``, ``"gaussian"`` or ``"cosine_squared"``.
    bunch_length
        Full bunch length :math:`\tau`, in [s]: full support of the
        binomial/cosine families; :math:`4\sigma` for the gaussian.
    bunch_position
        Bunch centre, in [s].
    exponent
        Binomial phase-space exponent :math:`\mu`; required for
        ``"binomial"``, ignored (with a warning) otherwise.

    Returns
    -------
    line_density_values
        :math:`\lambda(t)` evaluated at ``time_array`` (not normalized).
    """
    time_array = np.asarray(time_array, dtype=float)
    normalized_offset = 2.0 * (time_array - bunch_position) / bunch_length
    if distribution_type == "gaussian":
        if exponent is not None:
            warnings.warn(
                f"exponent is ignored for {distribution_type=}",
                UserWarning,
                stacklevel=2,
            )
        sigma = bunch_length / 4.0
        return np.exp(-((time_array - bunch_position) ** 2) / (2.0 * sigma**2))
    if distribution_type == "cosine_squared":
        if exponent is not None:
            warnings.warn(
                f"exponent is ignored for {distribution_type=}",
                UserWarning,
                stacklevel=2,
            )
        result = np.zeros_like(time_array, dtype=float)
        inside = np.abs(normalized_offset) <= 1.0
        result[inside] = np.cos(0.5 * np.pi * normalized_offset[inside]) ** 2
        return result
    resolved_exponent = _resolve_exponent(distribution_type, exponent)
    # Phase-space exponent mu -> line-density exponent mu + 1/2.
    return _binomial_family(normalized_offset**2, 1.0, resolved_exponent + 0.5)


def _bunch_length_rms(
    time_array: NumpyArray, line_density_values: NumpyArray
) -> float:
    """4-sigma RMS bunch length of a line density (BLonD 2 default)."""
    total = float(np.sum(line_density_values))
    if total <= 0.0:
        return 0.0
    mean_time = float(np.sum(line_density_values * time_array) / total)
    variance = float(
        np.sum(line_density_values * (time_array - mean_time) ** 2) / total
    )
    return 4.0 * np.sqrt(variance)


def _bunch_length_fwhm(
    time_array: NumpyArray, line_density_values: NumpyArray
) -> float:
    """
    FWHM bunch length rescaled to gaussian-equivalent 4 sigma.

    Ports the BLonD 2 ``filters_and_fitting.fwhm`` (shift=0):
    interpolated half-maximum crossings, then
    ``4 * fwhm / (2 * sqrt(2 * ln 2))``.
    """
    if np.all(line_density_values <= 0.0):
        return 0.0
    half_maximum = 0.5 * float(line_density_values.max())
    above = np.flatnonzero(line_density_values >= half_maximum)
    first, last = int(above[0]), int(above[-1])
    bin_size = time_array[1] - time_array[0]
    if first > 0:
        time_left = time_array[first] - bin_size * (
            line_density_values[first] - half_maximum
        ) / (line_density_values[first] - line_density_values[first - 1])
    else:
        time_left = time_array[first]
    if last < len(time_array) - 1:
        time_right = time_array[last] + bin_size * (
            line_density_values[last] - half_maximum
        ) / (line_density_values[last] - line_density_values[last + 1])
    else:
        time_right = time_array[last]
    return float(
        4.0 * (time_right - time_left) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    )


def x0_from_bunch_length(
    time_array: NumpyArray,
    x_grid: NumpyArray,
    *,
    target_bunch_length: float,
    distribution_type: str,
    exponent: float | None = None,
    bunch_length_fit: Literal["rms", "fwhm", "full"] = "rms",
    inside_bucket_mask: NumpyArray | None = None,
    max_iterations: int = 100,
    verbose: bool = False,
) -> float:
    r"""
    Find :math:`X_0` giving a requested bunch length (BLonD 2 port).

    Bisects :math:`X_0` between the smallest and largest finite value
    of ``x_grid`` until the bunch length of the projected line density
    matches ``target_bunch_length`` within one time bin.

    Parameters
    ----------
    time_array
        1D time coordinates of the grid columns, in [s].
    x_grid
        2D distribution-variable grid (Hamiltonian in [eV] or Action in
        [eV.s]), shape ``(n_deltaE, n_time)`` — the ``"xy"`` convention
        of ``hamiltonian_grid``/``action_grid``. ``inf`` entries
        (outside the bucket) are allowed.
    target_bunch_length
        Requested bunch length, in [s] (meaning set by
        ``bunch_length_fit``).
    distribution_type
        Family passed to :func:`distribution_function`.
    exponent
        Binomial exponent :math:`\mu` (see
        :func:`distribution_function`).
    bunch_length_fit
        ``"rms"``: 4-sigma RMS of the line density (the BLonD 2
        default); ``"fwhm"``: interpolated FWHM rescaled to
        gaussian-equivalent 4 sigma; ``"full"``: full extent of the
        :math:`X \leq X_0` contour. The BLonD 2 ``"gauss"`` mode was
        broken (dead code) and is not ported — requesting it raises.
    inside_bucket_mask
        Optional boolean grid (True inside the separatrix): the
        candidate density is zeroed outside before measuring. Required
        for meaningful gaussian fits in the Hamiltonian variable.
        NB BLonD 2 did *not* apply this zeroing inside its fit loop
        (only to the final density) — a small inconsistency for
        unbounded families, corrected here.
    max_iterations
        Iteration cap. BLonD 2 iterated unboundedly and could hang when
        its interval-collapse guard evaluated to zero width.
    verbose
        If True, print per-iteration diagnostics.

    Returns
    -------
    x_0
        The fitted :math:`X_0`, in the unit of ``x_grid``.

    Warns
    -----
    UserWarning
        If the bucket is too small for the requested bunch length, the
        requested bunch length is too small to resolve, or the
        iteration cap is reached (mirroring the BLonD 2 warnings).
    """
    if bunch_length_fit == "gauss":
        raise ValueError(
            "bunch_length_fit='gauss' was broken dead code in BLonD 2 "
            "and is not ported; use 'rms', 'fwhm' or 'full'."
        )
    if bunch_length_fit not in ("rms", "fwhm", "full"):
        raise ValueError(
            f"Unknown {bunch_length_fit=}; use 'rms', 'fwhm' or 'full'."
        )

    time_array = np.asarray(time_array, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    finite = np.isfinite(x_grid)
    if inside_bucket_mask is not None:
        finite = finite & inside_bucket_mask
    x_low = float(x_grid[finite].min())
    x_high = float(x_grid[finite].max())
    x_span = x_high - x_low
    time_resolution = float(time_array[1] - time_array[0])

    def measure(x_0: float) -> float:
        if bunch_length_fit == "full":
            columns = np.any((x_grid <= x_0) & finite, axis=0)
            if not np.any(columns):
                return 0.0
            occupied = np.flatnonzero(columns)
            return float(time_array[occupied[-1]] - time_array[occupied[0]])
        density = distribution_function(
            x_grid, distribution_type, x_0, exponent
        )
        if inside_bucket_mask is not None:
            density = np.where(inside_bucket_mask, density, 0.0)
        line_density_values = density.sum(axis=0)
        if bunch_length_fit == "rms":
            return _bunch_length_rms(time_array, line_density_values)
        return _bunch_length_fwhm(time_array, line_density_values)

    x_0 = x_high
    for iteration in range(max_iterations):
        x_0 = 0.5 * (x_low + x_high)
        achieved = measure(x_0)
        if verbose:
            print(
                f"[x0_from_bunch_length] iter {iteration:3d}: "
                f"x_0={x_0:.6e}, bunch length {achieved:.6e} s "
                f"(target {target_bunch_length:.6e})"
            )
        if abs(achieved - target_bunch_length) <= time_resolution:
            return float(x_0)
        if achieved >= target_bunch_length:
            x_high = x_0
        else:
            x_low = x_0
        if (x_high - x_low) < 1e-12 * x_span:
            if achieved < target_bunch_length and np.isclose(
                x_high, x_grid[finite].max()
            ):
                warnings.warn(
                    "The bucket is too small for the requested bunch "
                    f"length: requested {target_bunch_length:.4e} s, "
                    f"obtained {achieved:.4e} s.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "The requested bunch length is too small to be "
                    "resolved on this grid: requested "
                    f"{target_bunch_length:.4e} s, obtained "
                    f"{achieved:.4e} s.",
                    UserWarning,
                    stacklevel=2,
                )
            return float(x_0)
    warnings.warn(
        f"x0_from_bunch_length did not converge in {max_iterations} "
        f"iterations (last bunch length {achieved:.4e} s for target "
        f"{target_bunch_length:.4e} s).",
        UserWarning,
        stacklevel=2,
    )
    return float(x_0)
