# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

r"""
Shared math for bin-averaging a pole-residue wake.

A BLonD profile is a histogram, so the induced voltage of a bin is the wake
averaged over the source bin and over the observation bin. Doing only that
-- weighting the wake with the box of one bin twice -- removes the
*amplitude* error of an above-Nyquist resonance but leaves a **half-bin
lag**: it models the beam as a staircase, whose derivative is a train of
deltas sitting exactly on the bin edges, and a causal wake assigns each of
those edges wholly to the following bin. For a reactive impedance that turns
the exact (lossless) answer into a resistive one.

Averaging over a third box -- equivalently, reconstructing the line density
as piecewise linear through the bin centres instead of as a staircase --
fixes it. The wake is weighted with the quadratic B-spline
:math:`B_2 = \mathrm{box} * \mathrm{box} * \mathrm{box}`, whose support
:math:`(-3\Delta t / 2,\, 3\Delta t / 2)` straddles the causal onset
symmetrically. The price is one **non-causal tap**: the kernel is non-zero
from :math:`-3\Delta t / 2`, so the voltage of a bin depends on the charge of
the *next* one.

Every caller of :func:`triple_box_average_poles` -- the direct convolution in
:class:`~blond.physics.impedances.sources.Resonators` and the near-diagonal
correction in
:class:`~blond.physics.impedances.solvers.MultiPoleSparseSolve` -- computes
exactly this kernel, so the closed form and its onset correction are written
once here instead of once per call site.

Notes
-----
Authors:
Simon Lauber
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import backend

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray

# Terms summed for the series form of `causal_third_antiderivative_factor`
# (|p t| < 1, so the omitted tail is below 1 / 23! ~ 4e-23).
_PHI3_SERIES_TERMS = 20
_FACTORIAL_3 = 6.0

# Knots of the quadratic B-spline `box * box * box`, in units of the bin
# width: it is a different quadratic on each of (-3/2, -1/2), (-1/2, 1/2)
# and (1/2, 3/2), and zero outside.
_BSPLINE_KNOTS = (-1.5, -0.5, 0.5, 1.5)


def causal_third_antiderivative_factor(
    t: NumpyArray | CupyArray, pole: complex
) -> NumpyArray | CupyArray:
    r"""
    Causal factor :math:`\varphi_3(t)` of a pole's third antiderivative.

    :math:`\varphi_3(t) = (e^{p t} - 1 - p t - (p t)^2 / 2) / p^3` for
    :math:`t > 0` and :math:`\varphi_3(t \le 0) = 0`, so that the third
    antiderivative of the wake :math:`2 \,\mathrm{Re}[\rho e^{p t}]` is
    :math:`A_3(t) = 2 \,\mathrm{Re}[\rho \varphi_3(t)]`. Three box averages of
    the wake are a third difference of :math:`A_3`; see
    :func:`triple_box_average_poles`.

    Written as above the numerator cancels to :math:`O((p t)^3)` -- three
    digits lost for every decade :math:`|p t|` falls below one -- so for
    :math:`|p t| < 1` the equivalent series
    :math:`t^3 \sum_k (p t)^k / (k + 3)!` is summed instead, which has no
    cancellation at all.

    Parameters
    ----------
    t
        Time array at which to evaluate :math:`\varphi_3`, in [s].
    pole
        Pole :math:`p = -\alpha + i \bar\omega`, in [rad/s].

    Returns
    -------
    phi_3
        :math:`\varphi_3(t)`, in [s^3].
    """
    causal = t > 0.0  # (and phi_3(0) = 0 anyway)
    t_causal = backend.where(causal, t, 0.0)
    pole_t = pole * t_causal
    direct = (backend.exp(pole_t) - 1.0 - pole_t - 0.5 * pole_t**2) / pole**3
    # t**3 * sum_k (p t)**k / (k + 3)!, by Horner on the ratio of successive
    # coefficients (a_(k+1) / a_k = 1 / (k + 4)).
    series_factor = backend.ones_like(pole_t)
    for k in range(_PHI3_SERIES_TERMS - 2, -1, -1):
        series_factor = 1.0 + pole_t * series_factor / (k + 4)
    series = t_causal**3 * series_factor / _FACTORIAL_3
    return backend.where(
        (backend.abs(pole_t) < 1.0) & causal,
        series,
        backend.where(causal, direct, 0.0),
    )


def _smoothed_pole(
    t: NumpyArray | CupyArray, pole: complex, dt: float
) -> NumpyArray | CupyArray:
    r"""
    B-spline-averaged :math:`e^{p t}`, for lags past the causal onset.

    The value is
    :math:`((e^{p \Delta t} - 1) / (p \Delta t))^3 e^{p (t - 3\Delta t/2)}`,
    i.e. the factor multiplying the residue in
    :func:`triple_box_average_poles` once the whole smoothing B-spline
    :math:`(t - 3\Delta t/2,\, t + 3\Delta t/2)` is causal.

    Which of the two algebraically identical forms is evaluated depends on
    how well the bin resolves the pole. For :math:`|p \Delta t| \ge 1` the
    cubed factor on its own overflows as soon as the wake decays by more
    than :math:`e^{709}` within a bin, while :math:`e^{p t}` underflows to
    zero -- ``inf * 0 = nan``. Expanding it into the third difference of
    :math:`e^{p t}` keeps every exponent non-positive; its cancellation is
    only :math:`O(|p \Delta t|^3)`, which is harmless there and exactly
    what rules that form out for a resolved pole.

    Parameters
    ----------
    t
        Time array, in [s]. Every entry must exceed ``1.5 * dt``.
    pole
        Pole :math:`p = -\alpha + i \bar\omega`, in [rad/s].
    dt
        Bin width, in [s].

    Returns
    -------
    smoothed_pole
        The B-spline-averaged pole, dimensionless.
    """
    pole_dt = pole * dt
    if abs(pole_dt) < 1.0:
        # Resolved pole: |Re(pole_dt)| < 1, so nothing can overflow.
        return (np.expm1(pole_dt) / pole_dt) ** 3 * backend.exp(
            pole * (t - 1.5 * dt)
        )
    return (
        backend.exp(pole * (t + 1.5 * dt))
        - 3.0 * backend.exp(pole * (t + 0.5 * dt))
        + 3.0 * backend.exp(pole * (t - 0.5 * dt))
        - backend.exp(pole * (t - 1.5 * dt))
    ) / pole_dt**3


def triple_box_average_pole(
    t: NumpyArray | CupyArray, pole: complex, residue: complex, dt: float
) -> NumpyArray | CupyArray:
    r"""
    Triple bin-average of a single pole's wake :math:`2\,\mathrm{Re}[\rho e^{p t}]`.

    The factor 2 stands in for the implicit, unstored complex-conjugate
    partner of a complex pole (vector-fitting convention). A **real** pole
    (``pole.imag == 0``) has no partner and contributes
    :math:`\mathrm{Re}[\rho e^{p t}]` undoubled -- the same rule the
    far-field recursion applies (``injection_factor`` in
    ``wake_from_pole_residue``); without it, a real pole's near-field taps
    would be counted twice relative to its recursion tail.

    Writing the pole :math:`p` and residue :math:`\rho`, the causal third
    antiderivative of the wake is
    :math:`A_3(t) = 2 \,\mathrm{Re}[\rho \varphi_3(t)]` with
    :math:`\varphi_3` from :func:`causal_third_antiderivative_factor`. Three
    box averages are the third difference

    .. math::
        \frac{A_3(t + \tfrac{3}{2}\Delta t)
              - 3 A_3(t + \tfrac{1}{2}\Delta t)
              + 3 A_3(t - \tfrac{1}{2}\Delta t)
              - A_3(t - \tfrac{3}{2}\Delta t)}{\Delta t^3} .

    Once the whole B-spline is causal (:math:`t \ge 3\Delta t / 2`) the
    polynomial parts of the four :math:`\varphi_3` cancel *analytically*
    (see ``_smoothed_pole``); differencing :math:`A_3` itself there would
    lose the digits by which the difference is smaller than :math:`A_3`.
    Only the samples straddling the onset, where no such cancellation
    occurs, come from :math:`\varphi_3` directly.

    Parameters
    ----------
    t
        Time array (bin centres) at which the wake is evaluated, in [s]. May
        be negative: the kernel reaches back to ``-1.5 * dt``.
    pole
        Pole :math:`p = -\alpha + i \bar\omega`, in [rad/s].
    residue
        Residue :math:`\rho`.
    dt
        Bin width, in [s].

    Returns
    -------
    wake
        Bin-averaged wake of this single pole, in the units of ``residue``.
    """
    out = backend.zeros(len(t), dtype=backend.float, order="C")
    pair_factor = 1.0 if pole.imag == 0 else 2.0
    fully_causal = t > 1.5 * dt
    onset = ~fully_causal
    out[fully_causal] = (
        pair_factor * (residue * _smoothed_pole(t[fully_causal], pole, dt)).real
    )
    onset_t = t[onset]
    out[onset] = (
        pair_factor
        / dt**3
        * (
            residue
            * (
                causal_third_antiderivative_factor(onset_t + 1.5 * dt, pole)
                - 3.0
                * causal_third_antiderivative_factor(onset_t + 0.5 * dt, pole)
                + 3.0
                * causal_third_antiderivative_factor(onset_t - 0.5 * dt, pole)
                - causal_third_antiderivative_factor(onset_t - 1.5 * dt, pole)
            )
        ).real
    )
    return out


def triple_box_average_poles(
    t: NumpyArray | CupyArray,
    poles: NumpyArray | CupyArray,
    residues: NumpyArray | CupyArray,
    dt: float,
) -> NumpyArray | CupyArray:
    """
    Sum of :func:`triple_box_average_pole` over several poles.

    Parameters
    ----------
    t
        Time array (bin centres) at which the wake is evaluated, in [s]. May
        be negative: the kernel reaches back to ``-1.5 * dt``.
    poles
        Complex poles of the model, in [rad/s].
    residues
        Complex residues of the model, matching ``poles`` one-to-one.
    dt
        Bin width, in [s].

    Returns
    -------
    wake
        Bin-averaged wake summed over all poles.
    """
    out = backend.zeros(len(t), dtype=backend.float, order="C")
    for pole_entry, residue_entry in zip(poles, residues, strict=True):
        out += triple_box_average_pole(
            t, complex(pole_entry), complex(residue_entry), dt
        )
    return out


def quadratic_bspline(x: float) -> float:
    r"""
    Quadratic B-spline :math:`box * box * box` of unit width, at ``x``.

    The kernel every time-domain source is bin-averaged with (see
    :meth:`~blond.physics.impedances.base.TimeDomain.get_wake_per_bin`),
    normalised to unit integral and expressed in units of the bin width, so
    that its support is :math:`(-3/2,\, 3/2)`.

    Parameters
    ----------
    x
        Position in units of the bin width.

    Returns
    -------
    weight
        :math:`B_2(x)`, in units of one over the bin width.
    """
    abs_x = abs(x)
    if abs_x >= 1.5:  # NOQA PLR2004
        return 0.0
    if abs_x <= 0.5:  # NOQA PLR2004
        return 0.75 - x * x
    return 0.5 * (1.5 - abs_x) ** 2


def bspline_window_moments(offset: float, width: float) -> tuple[float, float]:
    r"""
    The two B-spline moments over a window of ``width`` bins.

    The moments are

    .. math::
        I_0 = \int_0^{w} B_2(v + \eta) \,\mathrm{d}\eta , \qquad
        I_1 = \int_0^{w} (w - \eta) B_2(v + \eta) \,\mathrm{d}\eta

    with :math:`v` = ``offset`` and :math:`w` = ``width``, both in units of
    the bin width. :math:`I_0` is the B-spline average of the box that spans
    the window, and :math:`I_1 / w` is the B-spline average of the ramp that
    rises linearly from 0 to 1 across it, both evaluated at ``offset`` past
    the window's upper end.

    The integrands are a quadratic and a cubic, so splitting the window at
    the B-spline's knots and applying Simpson's rule -- exact up to cubics --
    on each piece evaluates them exactly. Written as an integral of a
    non-negative integrand there is no cancellation, which a divided
    difference of the B-spline's antiderivatives would suffer from as
    ``width`` goes to zero.

    Parameters
    ----------
    offset
        Lower end of the integration window, in units of the bin width.
    width
        Width of the integration window, in units of the bin width.

    Returns
    -------
    moments
        The pair :math:`(I_0,\, I_1)`, dimensionless.
    """
    if width <= 0.0:
        return 0.0, 0.0
    edges = [0.0]
    edges += [
        knot - offset for knot in _BSPLINE_KNOTS if 0.0 < knot - offset < width
    ]
    edges += [width]
    zeroth = first = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        middle = 0.5 * (lower + upper)
        span = upper - lower
        weight_lower = quadratic_bspline(offset + lower)
        weight_middle = quadratic_bspline(offset + middle)
        weight_upper = quadratic_bspline(offset + upper)
        zeroth += (
            span / 6.0 * (weight_lower + 4.0 * weight_middle + weight_upper)
        )
        first += (
            span
            / 6.0
            * (
                (width - lower) * weight_lower
                + 4.0 * (width - middle) * weight_middle
                + (width - upper) * weight_upper
            )
        )
    return zeroth, first
