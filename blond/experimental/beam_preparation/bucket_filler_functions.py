# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Functions that turn the Hamiltonian of a bucket into a particle density.

Each function can be passed as ``hamilton_to_density_function`` to
:class:`~blond.experimental.beam_preparation.semi_empiric_matcher.SemiEmpiricMatcher`,
its keyword arguments besides the grids as ``hamilton_to_density_kwargs``.
:func:`hamilton_to_density_by_bunch_length` and
:func:`hamilton_to_density_by_emittance` fill the bucket to a target bunch
length or emittance, :func:`hamilton_to_density_by_metric` to a target of
any metric, and :func:`hamilton_to_density_by_max` up to a given
Hamiltonian.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from numpy.typing import NDArray as NumpyArray

# Smallest number of time steps that the fitted bunch should span; shorter
# bunches are not resolved by the grid of the matcher.
_MIN_RESOLVED_TIME_STEPS = 5
# Decay of the Gaussian distribution, in units of the Hamiltonian at the
# separatrix, beyond which it is a uniform filling of the bucket
_GAUSSIAN_DECAY_MAX = 1e6
_DISTRIBUTIONS = ("binomial", "gaussian")


def hamilton_to_density_by_bunch_length(
    time_grid: NumpyArray,
    deltaE_grid: NumpyArray,
    hamilton_2D: NumpyArray,
    bunch_length_4sigma: float,
    distribution: str = "binomial",
    exponent: float = 1.0,
) -> NumpyArray:
    r"""Density in :math:`H`, fitted to a 4 sigma bunch length.

    See :func:`hamilton_to_density_by_metric` for the distributions, the
    bucket that is filled, and targets that do not fit. The matcher calls it
    in every iteration, so the target holds for the matched beam, with its
    intensity effects.

    Parameters
    ----------
    time_grid
        Time coordinates of the grid, in [s].
    deltaE_grid
        Energy coordinates of the grid, in [eV].
    hamilton_2D
        Hamiltonian on the grid, in [eV].
    bunch_length_4sigma
        Target bunch length, four times the RMS length of the line density,
        in [s], i.e. four times
        :attr:`~blond.handle_results.observables.BeamStatisticsOncePerTurn.bunch_length`.
    distribution
        ``"binomial"`` or ``"gaussian"``, see
        :func:`hamilton_to_density_by_metric`.
    exponent
        Shape of the binomial distribution, see
        :func:`hamilton_to_density_by_metric`.

    Returns
    -------
    density
        Density on the grid, zero outside the bucket.

    Warns
    -----
    UserWarning
        If the target does not fit the bucket, or is too short for the grid.

    See Also
    --------
    hamilton_to_density_by_emittance : Fill to a target emittance.
    hamilton_to_density_by_metric : Fill to a target of any metric.
    hamilton_to_density_by_max : Fill up to a given Hamiltonian.
    blond.experimental.beam_preparation.semi_empiric_matcher_extensions.line_density.classes.ProfileMatcherAddon.hamilton_to_density_function :
        Fill to a measured line density.
    blond.experimental.beam_preparation.semi_empiric_matcher.SemiEmpiricMatcher :
        Calls these functions, see its ``hamilton_to_density_function``.
    """
    return hamilton_to_density_by_metric(
        time_grid=time_grid,
        deltaE_grid=deltaE_grid,
        hamilton_2D=hamilton_2D,
        metric=_bunch_length_4sigma,
        target=bunch_length_4sigma,
        distribution=distribution,
        exponent=exponent,
        description="4 sigma bunch length",
        unit="s",
    )


def hamilton_to_density_by_emittance(
    time_grid: NumpyArray,
    deltaE_grid: NumpyArray,
    hamilton_2D: NumpyArray,
    emittance_4pi_rms: float,
    distribution: str = "binomial",
    exponent: float = 1.0,
) -> NumpyArray:
    r"""Density in :math:`H`, fitted to a longitudinal emittance.

    The emittance is
    :math:`4 \pi \sqrt{\sigma_t^2 \sigma_E^2 - \sigma_{tE}^2}`, the area of
    the :math:`2 \sigma` ellipse of a Gaussian bunch. See
    :func:`hamilton_to_density_by_metric` for the distributions, the bucket
    that is filled, and targets that do not fit.

    Parameters
    ----------
    time_grid
        Time coordinates of the grid, in [s].
    deltaE_grid
        Energy coordinates of the grid, in [eV].
    hamilton_2D
        Hamiltonian on the grid, in [eV].
    emittance_4pi_rms
        Target emittance, in [eVs], i.e. :math:`4 \pi` times
        :attr:`~blond.core.beam.beams.Beam.rms_emittance`.
    distribution
        ``"binomial"`` or ``"gaussian"``, see
        :func:`hamilton_to_density_by_metric`.
    exponent
        Shape of the binomial distribution, see
        :func:`hamilton_to_density_by_metric`.

    Returns
    -------
    density
        Density on the grid, zero outside the bucket.

    Warns
    -----
    UserWarning
        If the target does not fit the bucket, or is too small for the grid.

    See Also
    --------
    hamilton_to_density_by_bunch_length : Fill to a target bunch length.
    hamilton_to_density_by_metric : Fill to a target of any metric.
    hamilton_to_density_by_max : Fill up to a given Hamiltonian.
    blond.experimental.beam_preparation.semi_empiric_matcher_extensions.line_density.classes.ProfileMatcherAddon.hamilton_to_density_function :
        Fill to a measured line density.
    blond.experimental.beam_preparation.semi_empiric_matcher.SemiEmpiricMatcher :
        Calls these functions, see its ``hamilton_to_density_function``.
    """
    return hamilton_to_density_by_metric(
        time_grid=time_grid,
        deltaE_grid=deltaE_grid,
        hamilton_2D=hamilton_2D,
        metric=_emittance_4pi_rms,
        target=emittance_4pi_rms,
        distribution=distribution,
        exponent=exponent,
        description="emittance",
        unit="eVs",
    )


def hamilton_to_density_by_metric(
    time_grid: NumpyArray,
    deltaE_grid: NumpyArray,
    hamilton_2D: NumpyArray,
    metric: Callable[[NumpyArray, NumpyArray, NumpyArray], float],
    target: float,
    distribution: str = "binomial",
    exponent: float = 1.0,
    description: str = "metric",
    unit: str = "",
) -> NumpyArray:
    r"""Density in :math:`H` with one parameter, fitted to a target metric.

    Inside the separatrix, with :math:`h = H - H_\mathrm{bottom}`, the
    density is

    - ``"binomial"``: :math:`(1 - h / h_\mathrm{cut})^\mathrm{exponent}`
      for :math:`h < h_\mathrm{cut}`, zero above. ``exponent=0`` fills the
      bucket uniformly up to :math:`h_\mathrm{cut}`; larger values give
      more peaked bunches.
    - ``"gaussian"``: :math:`\exp(-h / h_0)`, cut at the separatrix. In a
      linear bucket, :math:`h \propto t^2 + (\ldots) \Delta E^2`, so the
      bunch is Gaussian in time and energy.

    The parameter, :math:`h_\mathrm{cut}` or :math:`h_0`, is solved for
    with :func:`scipy.optimize.brentq` such that ``metric`` of the density
    equals ``target``.

    Parameters
    ----------
    time_grid
        Time coordinates of the grid, in [s].
    deltaE_grid
        Energy coordinates of the grid, in [eV].
    hamilton_2D
        Hamiltonian on the grid, in [eV]. The bucket around the minimum of
        the potential well is filled, see Notes.
    metric
        ``metric(weights, ts, deltaEs)``, evaluated on the grid cells inside
        the bucket: ``weights`` is the density there (not normalised), ``ts``
        and ``deltaEs`` their coordinates, in [s] and [eV]. It must grow
        with the parameter, as e.g. a bunch length or an emittance does.
    target
        Target value of ``metric``.
    distribution
        ``"binomial"`` or ``"gaussian"``, see above.
    exponent
        Exponent of the binomial distribution; not used for the Gaussian.
    description
        Name of the metric, for warnings.
    unit
        Unit of the metric, for warnings.

    Returns
    -------
    density
        Density on the grid, zero outside the bucket.

    Raises
    ------
    ValueError
        If ``distribution`` is unknown.

    Warns
    -----
    UserWarning
        If the target does not fit the bucket. The bucket is then filled as
        far as the distribution allows (binomial: up to the separatrix,
        Gaussian: uniformly), which gives the largest metric possible. Also
        if the target is too small for the grid.

    See Also
    --------
    hamilton_to_density_by_bunch_length : Fill to a target bunch length.
    hamilton_to_density_by_emittance : Fill to a target emittance.
    hamilton_to_density_by_max : Fill up to a given Hamiltonian.
    blond.experimental.beam_preparation.semi_empiric_matcher_extensions.line_density.classes.ProfileMatcherAddon.hamilton_to_density_function :
        Fill to a measured line density.
    blond.experimental.beam_preparation.semi_empiric_matcher.SemiEmpiricMatcher :
        Calls these functions, see its ``hamilton_to_density_function``.

    Notes
    -----
    The bucket is the one around the minimum of the potential well
    :math:`H(t, \Delta E \approx 0)`, up to the lower of the highest
    barriers on its two sides. A single ``time_limit`` should therefore hold
    one bucket; to fill several, give one window per bucket.
    """
    if distribution not in _DISTRIBUTIONS:
        raise ValueError(
            f"`distribution` must be one of {_DISTRIBUTIONS}, but got"
            f" {distribution!r}."
        )
    hamilton_2D = np.asarray(hamilton_2D, dtype=float)
    inside, hamilton_separatrix = _cells_of_bucket(hamilton_2D)
    # only the cells inside the bucket, H from the bottom of the bucket
    hamiltons = hamilton_2D[inside] - hamilton_2D.min()
    ts = np.asarray(time_grid)[inside]
    deltaEs = np.asarray(deltaE_grid)[inside]

    if distribution == "binomial":

        def weights_of(hamilton_cut: float) -> NumpyArray:
            below = hamiltons < hamilton_cut
            weights = np.zeros_like(hamiltons)
            weights[below] = (1 - hamiltons[below] / hamilton_cut) ** exponent
            return weights

        parameter_largest = hamilton_separatrix
        fullest = f"binomial distribution with exponent={exponent} (a smaller"
        fullest += " exponent gives more)"
    else:

        def weights_of(hamilton_decay: float) -> NumpyArray:
            return np.exp(-hamiltons / hamilton_decay)

        parameter_largest = _GAUSSIAN_DECAY_MAX * hamilton_separatrix
        fullest = "Gaussian distribution, which then fills the bucket"
        fullest += " uniformly"

    def achieved(parameter: float) -> float:
        return float(metric(weights_of(parameter), ts, deltaEs))

    # The smallest parameter that the grid resolves: the bunch spans a few
    # time steps around the bottom of the bucket.
    time_step = abs(float(time_grid[1, 0] - time_grid[0, 0]))
    distance = np.abs(ts - ts[np.argmin(hamiltons)])
    resolved = distance >= _MIN_RESOLVED_TIME_STEPS * time_step
    parameter_smallest = float(
        hamiltons[resolved].min() if resolved.any() else hamilton_separatrix
    )

    if achieved(parameter_largest) <= target:
        parameter = parameter_largest
        warnings.warn(
            f"The target {description} {target:.4g} {unit} does not fit the"
            f" bucket. It is filled up to the separatrix, giving"
            f" {achieved(parameter):.4g} {unit}, the largest possible with"
            f" the {fullest}.",
            stacklevel=3,
        )
    elif achieved(parameter_smallest) >= target:
        parameter = parameter_smallest
        warnings.warn(
            f"The target {description} {target:.4g} {unit} is not resolved"
            f" by the grid, giving {achieved(parameter):.4g} {unit}."
            " Increase `internal_grid_shape`, or narrow the time range.",
            stacklevel=3,
        )
    else:
        parameter = brentq(
            lambda parameter: achieved(parameter) - target,
            parameter_smallest,
            parameter_largest,
            rtol=1e-10,
        )
    density = np.zeros(hamilton_2D.shape)
    density[inside] = weights_of(parameter)
    return density


def _bunch_length_4sigma(
    weights: NumpyArray, ts: NumpyArray, deltaEs: NumpyArray
) -> float:
    """Four times the RMS length of weighted points.

    Parameters
    ----------
    weights
        Density at the points.
    ts
        Time of the points, in [s].
    deltaEs
        Energy of the points, in [eV]; not used.

    Returns
    -------
    bunch_length_4sigma
        In [s].
    """
    mean = np.average(ts, weights=weights)
    return float(4 * np.sqrt(np.average((ts - mean) ** 2, weights=weights)))


def _emittance_4pi_rms(
    weights: NumpyArray, ts: NumpyArray, deltaEs: NumpyArray
) -> float:
    r""":math:`4 \pi` times the RMS emittance of weighted points.

    Parameters
    ----------
    weights
        Density at the points.
    ts
        Time of the points, in [s].
    deltaEs
        Energy of the points, in [eV].

    Returns
    -------
    emittance_4pi_rms
        In [eVs].
    """
    t_centred = ts - np.average(ts, weights=weights)
    deltaE_centred = deltaEs - np.average(deltaEs, weights=weights)
    var_t = np.average(t_centred**2, weights=weights)
    var_deltaE = np.average(deltaE_centred**2, weights=weights)
    cov = np.average(t_centred * deltaE_centred, weights=weights)
    return float(4 * np.pi * np.sqrt(max(var_t * var_deltaE - cov**2, 0.0)))


def _cells_of_bucket(
    hamilton_2D: NumpyArray,
) -> tuple[NumpyArray, float]:
    """Grid cells inside the separatrix of the bucket around the minimum.

    Parameters
    ----------
    hamilton_2D
        Hamiltonian on the grid, in [eV], shape ``(n_time, n_energy)``.

    Returns
    -------
    inside
        Boolean mask of the cells inside the separatrix.
    hamilton_separatrix
        Hamiltonian at the separatrix, counted from the bottom of the
        bucket, in [eV].
    """
    potential_well = hamilton_2D.min(axis=1)  # H(t, dE = 0)
    i_min = int(np.argmin(potential_well))
    i_left = int(np.argmax(potential_well[: i_min + 1]))
    i_right = i_min + int(np.argmax(potential_well[i_min:]))
    hamilton_separatrix = min(potential_well[i_left], potential_well[i_right])
    in_time = np.zeros(len(potential_well), dtype=bool)
    in_time[i_left : i_right + 1] = True
    inside = in_time[:, None] & (hamilton_2D < hamilton_separatrix)
    return inside, float(hamilton_separatrix - potential_well[i_min])


def hamilton_to_density_by_max(
    time_grid: NumpyArray,
    deltaE_grid: NumpyArray,
    hamilton_2D: NumpyArray,
    density_modifier: float,
    hamilton_max: float,
) -> NumpyArray:
    r"""Binomial density in :math:`H`, up to a given Hamiltonian.

    The density is

    .. math::

        \rho = \left(1 - \frac{H}{H_\mathrm{max}}\right)^{n}

    for :math:`H < H_\mathrm{max}`, zero above, with :math:`n` =
    ``density_modifier`` and :math:`H_\mathrm{max}` = ``hamilton_max``. This
    is the default ``hamilton_to_density_function`` of
    :class:`~blond.experimental.beam_preparation.semi_empiric_matcher.SemiEmpiricMatcher`.

    Parameters
    ----------
    time_grid
        Time coordinates of the grid, in [s]; not used.
    deltaE_grid
        Energy coordinates of the grid, in [eV]; not used.
    hamilton_2D
        Hamiltonian on the grid, in [eV]. The matcher passes it about zero
        at the bottom of the potential well (with windows: exactly zero at
        the bottom of each bucket).
    density_modifier
        Exponent :math:`n`: 0 fills uniformly up to ``hamilton_max``, larger
        values give more peaked bunches.
    hamilton_max
        Hamiltonian up to which the bucket is filled, in [eV], counted from
        zero. Above the separatrix, the particles are not bound: with
        windows, they are removed with a warning; with a single
        ``time_limit``, they are not. Set ``animate=True`` in the matcher to
        see the potential well, or give a bunch length or emittance
        instead, see See Also.

    Returns
    -------
    density
        Density on the grid, from 0 to 1.

    See Also
    --------
    hamilton_to_density_by_bunch_length : Fill to a target bunch length.
    hamilton_to_density_by_emittance : Fill to a target emittance.
    hamilton_to_density_by_metric : Fill to a target of any metric.
    blond.experimental.beam_preparation.semi_empiric_matcher_extensions.line_density.classes.ProfileMatcherAddon.hamilton_to_density_function :
        Fill to a measured line density.
    blond.experimental.beam_preparation.semi_empiric_matcher.SemiEmpiricMatcher :
        Calls these functions, see its ``hamilton_to_density_function``.
    """
    _density = hamilton_2D.copy()  # So the changes stay in this scope

    _density /= hamilton_max
    # Now 1 representing the limit between particles/no-particles.
    # Smaller 1 means there should be particles.

    np.clip(_density, 0, 1, out=_density)  # Truncate
    # Flip max with min
    _density *= -1
    _density += 1

    # Modify of the density to be more/less dense in different regions.
    _density **= density_modifier
    return _density
