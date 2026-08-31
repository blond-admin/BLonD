# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Interpolation routines complementing the ones shipped with `scipy`.

Notes
-----
The following classes are currently available:
- :class:`~blond.generals.interpolators.DerivativeInterpolator`

Authors:
Simon Lauber
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

#: A derivative can only be estimated from two samples onwards.
_MIN_SAMPLES = 2


class DerivativeInterpolator:
    r"""
    Smooth a program by interpolating its first derivative.

    The derivative of the input data is estimated by finite differences,
    interpolated linearly, and integrated back analytically. The result
    therefore has a **continuous first derivative**, which for a magnetic
    cycle means a continuous :math:`\dot{B}` and hence a smooth demand on
    the accelerating voltage.

    Parameters
    ----------
    x
        Sample positions, e.g. time [s]. Must be strictly increasing, but
        need not be equidistant.
    y
        Sample values, e.g. momentum [eV/c].
    left
        Value returned below ``x[0]``. ``None`` (default) clamps to
        ``y[0]``, like `numpy.interp`. Set to ``numpy.nan`` to raise a
        `ValueError` instead of extrapolating.
    right
        Value returned above ``x[-1]``. ``None`` (default) clamps to
        ``y[-1]``, like `numpy.interp`. Set to ``numpy.nan`` to raise a
        `ValueError` instead of extrapolating.

    Raises
    ------
    ValueError
        If evaluated outside ``[x[0], x[-1]]`` while the corresponding
        `left` or `right` is ``numpy.nan``.

    See Also
    --------
    scipy.interpolate.interp1d : 1D interpolator similar to `numpy.interp`.
    scipy.interpolate.Akima1DInterpolator : Modified Akima Interpolation.
    scipy.interpolate.PchipInterpolator : Piecewise Cubic Hermite Interpolating Polynomial.
    blond.cycles.magnetic_cycle.MagneticCycleByTime : Main consumer of this class.

    Notes
    -----
    **This is a smoother, not an interpolant.** Unlike `interp1d`,
    `Akima1DInterpolator` or `PchipInterpolator`, the curve does *not* pass
    through the input samples: the integral of the finite-difference
    derivative does not reproduce the original increments. Only the two end
    points are matched exactly. Use one of the `scipy` interpolators when
    the samples must be honoured.

    The construction is, with :math:`d_k` the finite-difference derivative
    at :math:`x_k` and :math:`\tilde{d}` its piecewise-linear interpolant,

    .. math::

        Y(x) = y_0 + \int_{x_0}^{x} \tilde{d}(\xi)\,\mathrm{d}\xi

    Because :math:`\tilde{d}` is piecewise linear, :math:`Y` is a
    :math:`C^1` piecewise-quadratic spline and the integral is evaluated in
    closed form. On the segment starting at :math:`x_k`, with
    :math:`s = x - x_k`, :math:`\Delta_k = x_{k+1} - x_k` and
    :math:`m_k = (d_{k+1} - d_k) / \Delta_k`:

    .. math::

        Y(x) = y_0 + C_k + d_k s + \tfrac{1}{2} m_k s^2, \qquad
        C_k = \sum_{j<k} \tfrac{1}{2}\Delta_j (d_j + d_{j+1})

    Integrating a differentiated program does not land exactly on the final
    sample. The residual drift :math:`\varepsilon = y_{N} - Y(x_{N})` is
    absorbed by adding a straight line that vanishes at :math:`x_0`:

    .. math::

        \hat{y}(x) = Y(x)
                     + \varepsilon \frac{x - x_0}{x_{N} - x_0}

    This pins both ends, adds only a constant offset to the derivative, and
    -- unlike a multiplicative rescaling of the whole curve -- never
    divides by the total swing :math:`y_N - y_0`. Programs that return to
    their starting value (a bump) are therefore handled correctly.

    Non-equidistant samples are fully supported: every spacing enters per
    segment, so the deviation from the samples vanishes at second order in
    the sample spacing on stretched grids just as it does on uniform ones.

    Compared to the ``interpolation='derivative'`` option of BLonD 2, the
    derivative is estimated with `numpy.gradient` on the actual sample
    positions (second-order accurate on non-uniform grids) and the
    integration is analytic rather than a first-order rectangle sum
    accumulated turn by turn, so the result no longer depends on the
    revolution period.

    Examples
    --------
    >>> import numpy as np
    >>> from blond import DerivativeInterpolator
    >>> time = np.linspace(0.0, 1.0, 11)
    >>> momentum = 1e9 + 24e9 * (0.5 - 0.5 * np.cos(np.pi * time))
    >>> interpolator = DerivativeInterpolator(time, momentum)
    >>> float(interpolator(0.0))
    1000000000.0

    Feeding a magnetic cycle with it:

    >>> from blond import MagneticCycleByTime, proton
    >>> cycle = MagneticCycleByTime(
    ...     reference_particle=proton,
    ...     reference_time=time,
    ...     reference_values=momentum,
    ...     in_unit="momentum",
    ...     interpolator=DerivativeInterpolator,
    ... )
    """

    def __init__(
        self,
        x: NumpyArray,
        y: NumpyArray,
        left: float | None = None,
        right: float | None = None,
    ) -> None:
        positions = np.asarray(x, dtype=float)
        values = np.asarray(y, dtype=float)

        assert positions.ndim == 1, (
            f"Expected 1D array, but {positions.shape=}"
        )
        assert positions.shape == values.shape, (
            f"Shape mismatch: {positions.shape=} vs {values.shape=}"
        )
        assert positions.size >= _MIN_SAMPLES, (
            f"At least {_MIN_SAMPLES} samples required,"
            f" but got {positions.size}"
        )
        assert not np.any(np.isnan(positions)), "NaN occurred in `x`"
        assert not np.any(np.isnan(values)), "NaN occurred in `y`"
        assert np.all(np.diff(positions) > 0), (
            "`x` must be strictly increasing"
        )

        derivative = np.gradient(values, positions)
        segment_duration = np.diff(positions)

        # exact integral of the piecewise-linear derivative, knot by knot
        segment_increment = (
            0.5 * segment_duration * (derivative[:-1] + derivative[1:])
        )
        cumulative_increment = np.concatenate(
            ([0.0], np.cumsum(segment_increment))
        )

        residual_drift = (values[-1] - values[0]) - cumulative_increment[-1]

        self._positions = positions
        self._values = values
        self._derivative = derivative
        self._segment_duration = segment_duration
        self._cumulative_increment = cumulative_increment
        self._detilt_slope = residual_drift / (positions[-1] - positions[0])
        self._left = values[0] if left is None else float(left)
        self._right = values[-1] if right is None else float(right)

    def __call__(self, x: float | NumpyArray) -> float | NumpyArray:
        """
        Evaluate the smoothed program.

        Parameters
        ----------
        x
            Position(s) at which to evaluate, e.g. time [s].

        Returns
        -------
        y
            Smoothed value(s) at `x`, with the shape of `x`.

        Raises
        ------
        ValueError
            If `x` reaches outside the sampled range while the
            corresponding `left` or `right` is ``numpy.nan``.
        """
        position = np.asarray(x, dtype=float)

        below_range = position < self._positions[0]
        above_range = position > self._positions[-1]

        if np.isnan(self._left) and np.any(below_range):
            raise ValueError(
                f"Evaluated below the sampled range "
                f"(smallest `x` is {self._positions[0]})."
            )
        if np.isnan(self._right) and np.any(above_range):
            raise ValueError(
                f"Evaluated above the sampled range "
                f"(largest `x` is {self._positions[-1]})."
            )

        segment_i = np.clip(
            np.searchsorted(self._positions, position, side="right") - 1,
            0,
            self._segment_duration.size - 1,
        )
        offset = position - self._positions[segment_i]
        curvature = (
            self._derivative[segment_i + 1] - self._derivative[segment_i]
        ) / self._segment_duration[segment_i]

        result = (
            self._values[0]
            + self._cumulative_increment[segment_i]
            + self._derivative[segment_i] * offset
            + 0.5 * curvature * offset * offset
            + self._detilt_slope * (position - self._positions[0])
        )

        result = np.where(below_range, self._left, result)
        result = np.where(above_range, self._right, result)
        return result
