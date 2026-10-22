# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

r"""
Bin-averaged wake of a pole-residue model, in closed form.

A profile is a histogram of bin width :math:`\Delta t` and the induced
voltage lives on the same grid, so a time-domain solver applies the wake at
whole-bin lags only. Point-sampling the wake there aliases every resonance
above the profile's Nyquist frequency back onto the bunch spectrum -- the
low-Q / broadband resonator bug. Averaging the wake with the quadratic
B-spline :math:`B_2 = \mathrm{box} * \mathrm{box} * \mathrm{box}` of one bin
width (piecewise-linear line density, box read-out) suppresses the alias
images as :math:`|j|^{-3}` and leaves the loss factor right at any binning.
The price is one non-causal tap: :math:`B_2` reaches to
:math:`-\tfrac32 \Delta t`, so the voltage of a bin depends on the charge of
the next one.

For a pole :math:`p` with residue :math:`\rho`, wake
:math:`W(t) = \kappa\,\mathrm{Re}[\rho e^{p t}]` for :math:`t > 0`, where
:math:`\kappa = 2` for a complex pole, which stands in for its unstored
conjugate partner, and :math:`\kappa = 1` for a real pole, which has none.
The average :math:`\overline W = W * B_2` is the third difference of a
third antiderivative (Steps 47-51 of
``docs/models_new/pole-residue/bin_averaged_wake_derivation.py``):

.. math::
    \overline W(t) = \kappa\,\mathrm{Re}\!\left[\rho\,
      \frac{\varphi_3(t + \tfrac32\Delta t) - 3\varphi_3(t + \tfrac12\Delta t)
          + 3\varphi_3(t - \tfrac12\Delta t) - \varphi_3(t - \tfrac32\Delta t)}
      {\Delta t^3}\right],
    \qquad
    \varphi_3(t) = \frac{e^{p t} - 1 - p t - \tfrac12 (p t)^2}{p^3}\,\Theta(t),

which past the onset, :math:`t > \tfrac32\Delta t`, collapses to the pure
exponential
:math:`\kappa\,\mathrm{Re}[\rho\,((e^{p\Delta t} - 1) / (p\Delta t))^3
e^{p (t - \frac32 \Delta t)}]`.

Notes
-----
Authors:
Simon Lauber
"""

from __future__ import annotations

import cmath
import math
from typing import TYPE_CHECKING

from blond.core.backends.backend import backend

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray

# Terms summed for the series form of `causal_third_antiderivative_factor`
# (|p t| < 1, so the omitted tail is below 1 / 23! ~ 4e-23).
_PHI3_SERIES_TERMS = 20
_FACTORIAL_3 = 6.0
# From this many bins on, `triple_box_average_poles` evaluates the near
# branch on the few bins around the onset only (one host sync per call on
# CuPy); below it, on the whole array (no sync).
_NEAR_WINDOW_MIN_BINS = 4096
# Below that, the poles are evaluated together, lags times poles at once:
# at most this many elements per batch, so the temporaries stay a few MB.
_MAX_BROADCAST_ELEMENTS = 2**18


def causal_third_antiderivative_factor(
    t: NumpyArray | CupyArray, pole: complex
) -> NumpyArray | CupyArray:
    r"""
    Causal factor :math:`\varphi_3(t)` of a pole's third antiderivative.

    :math:`\varphi_3(t) = (e^{p t} - 1 - p t - (p t)^2 / 2) / p^3` for
    :math:`t > 0` and zero otherwise. For :math:`|p t| < 1` that numerator
    cancels to :math:`O((p t)^3)`, so the equivalent series
    :math:`t^3 \sum_k (p t)^k / (k + 3)!` is summed instead.

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
    causal = t > 0.0
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
    B-spline-averaged :math:`e^{p t}` for lags past the causal onset.

    :math:`((e^{p \Delta t} - 1) / (p \Delta t))^3 e^{p (t - 3\Delta t/2)}`.
    For a damped pole and ``t > 1.5 * dt`` every factor is bounded by one, so
    nothing here can overflow at any binning.

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
    return (backend.expm1(pole_dt) / pole_dt) ** 3 * backend.exp(
        pole * (t - 1.5 * dt)
    )


def triple_box_average_pole(
    t: NumpyArray | CupyArray,
    pole: complex,
    residue: complex,
    dt: float,
    near_idx: NumpyArray | CupyArray | None = None,
) -> NumpyArray | CupyArray:
    r"""
    Bin-averaged wake of a single pole, :math:`\overline W = W * B_2`.

    The third difference of :func:`causal_third_antiderivative_factor` is
    used where the B-spline straddles the onset, and the exponential closed
    form (``_smoothed_pole``) past it, where differencing would cancel
    digits. A complex pole stands in for its unstored conjugate partner and
    is doubled; a real pole (``pole.imag == 0``) is not.

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
    near_idx
        Indices of ``t`` with ``-1.5 * dt < t <= 1.5 * dt``, the only lags
        where the near branch is nonzero (see ``_near_window``). If
        given, the near branch is evaluated there only; if None, on the
        whole array, which needs no host sync but is slow on long grids.

    Returns
    -------
    wake
        Bin-averaged wake of this single pole, in the units of ``residue``.
    """
    onset = 1.5 * dt
    fully_causal = t > onset
    # 2 - True == 1 for a real pole, 2 - False == 2 for a complex one; works
    # for a host scalar and a device slice alike, without a branch.
    pair_factor = 2.0 - (pole.imag == 0)
    # The far argument is clamped to its side of the onset so the discarded
    # values stay finite.
    t_far = backend.where(fully_causal, t, onset)
    far = (residue * _smoothed_pole(t_far, pole, dt)).real
    if near_idx is not None:
        # Zero, not the clamped far value, before the kernel's support.
        wake = backend.where(fully_causal, far, 0.0)
        wake[near_idx] = _near_branch(t[near_idx], pole, residue, dt)
        return pair_factor * wake
    # Both branches are evaluated everywhere and selected with `where`: a
    # boolean-mask gather would sync a CuPy array to the host, and this is
    # reached from the per-turn loop. The near argument is clamped to its
    # side of the onset so the discarded values stay finite.
    t_near = backend.where(fully_causal, onset, t)
    near = _near_branch(t_near, pole, residue, dt)
    return pair_factor * backend.where(fully_causal, far, near)


def _near_branch(
    t: NumpyArray | CupyArray, pole: complex, residue: complex, dt: float
) -> NumpyArray | CupyArray:
    r"""
    Single-pole wake from the third difference of :math:`\varphi_3`.

    Parameters
    ----------
    t
        Time array, in [s]. Meant for ``t <= 1.5 * dt``; zero for
        ``t <= -1.5 * dt``.
    pole
        Pole :math:`p = -\alpha + i \bar\omega`, in [rad/s].
    residue
        Residue :math:`\rho`.
    dt
        Bin width, in [s].

    Returns
    -------
    wake
        Bin-averaged wake, without the pair factor.
    """
    return (
        residue
        * (
            causal_third_antiderivative_factor(t + 1.5 * dt, pole)
            - 3.0 * causal_third_antiderivative_factor(t + 0.5 * dt, pole)
            + 3.0 * causal_third_antiderivative_factor(t - 0.5 * dt, pole)
            - causal_third_antiderivative_factor(t - 1.5 * dt, pole)
        )
    ).real / dt**3


def _near_window(
    t: NumpyArray | CupyArray, dt: float
) -> NumpyArray | CupyArray | None:
    """
    Find the indices where the near branch of a bin average is nonzero.

    Parameters
    ----------
    t
        Time array, in [s].
    dt
        Bin width, in [s].

    Returns
    -------
    near_idx
        Indices of ``-1.5 * dt < t <= 1.5 * dt``, or None for an array
        shorter than ``_NEAR_WINDOW_MIN_BINS``, which is evaluated whole.
    """
    # Finding the window syncs a CuPy array to the host once. Short arrays,
    # the per-turn lags of `MultiPoleSparseSolve` among them, skip it: the
    # full-array evaluation is cheap there and needs no sync.
    if len(t) < _NEAR_WINDOW_MIN_BINS:
        return None
    return backend.where((t > -1.5 * dt) & (t <= 1.5 * dt))[0]


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
        Time array (bin centres) at which the wake is evaluated, in [s].
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
    assert len(poles) == len(residues)
    out = backend.zeros(len(t), dtype=backend.float)
    # The window depends on `t` and `dt` only, so it is found once.
    near_idx = _near_window(t, dt)
    if near_idx is None:
        # A short grid: one call per batch of poles, lags along axis 0 and
        # poles along axis 1. Per pole, a call is a few hundred array
        # operations -- kernel launches on a GPU -- whatever the length.
        batch_size = max(1, _MAX_BROADCAST_ELEMENTS // max(len(t), 1))
        lags = t[:, None]
        for first in range(0, len(poles), batch_size):
            batch = slice(first, first + batch_size)
            out += triple_box_average_pole(
                lags, poles[None, batch], residues[None, batch], dt
            ).sum(axis=1)
        return out
    # Length-1 slices rather than `complex(poles[i])`: the latter is a
    # device-to-host transfer per pole inside the per-turn loop.
    for pole_i in range(len(poles)):
        out += triple_box_average_pole(
            t,
            poles[pole_i : pole_i + 1],
            residues[pole_i : pole_i + 1],
            dt,
            near_idx,
        )
    return out


def _causal_third_antiderivative_factor_ramp(
    t: NumpyArray | CupyArray, pole: complex
) -> NumpyArray | CupyArray:
    r"""
    Third antiderivative of :math:`t e^{p t}\,\Theta(t)`.

    It is :math:`\psi_3 = \partial \varphi_3 / \partial p`, see
    :func:`causal_third_antiderivative_factor`:
    :math:`(t e^{p t} - t - p t^2) / p^3 - 3 \varphi_3 / p` for
    :math:`t > 0` and zero otherwise, summed as the series
    :math:`t^4 \sum_k (k + 1) (p t)^k / (k + 4)!` for :math:`|p t| < 1`.

    Parameters
    ----------
    t
        Time array at which to evaluate :math:`\psi_3`, in [s].
    pole
        Pole :math:`p`, in [rad/s].

    Returns
    -------
    psi_3
        :math:`\psi_3(t)`, in [s^4].
    """
    causal = t > 0.0
    t_causal = backend.where(causal, t, 0.0)
    pole_t = pole * t_causal
    series_factor = backend.zeros_like(pole_t)
    for k in range(_PHI3_SERIES_TERMS - 1, -1, -1):
        series_factor = (k + 1) / math.factorial(
            k + 4
        ) + pole_t * series_factor
    series = t_causal**4 * series_factor
    if pole == 0:
        return backend.where(causal, series, 0.0)
    direct = (
        t_causal * backend.exp(pole_t) - t_causal - pole * t_causal**2
    ) / pole**3 - 3.0 * causal_third_antiderivative_factor(t, pole) / pole
    return backend.where(
        (backend.abs(pole_t) < 1.0) & causal,
        series,
        backend.where(causal, direct, 0.0),
    )


def _causal_third_antiderivative_factor_any(
    t: NumpyArray | CupyArray, pole: complex
) -> NumpyArray | CupyArray:
    r"""
    :func:`causal_third_antiderivative_factor`, also for a zero pole.

    For :math:`p = 0` it is :math:`t^3 / 6`, which the direct form would
    reach as 0 / 0.

    Parameters
    ----------
    t
        Time array at which to evaluate :math:`\varphi_3`, in [s].
    pole
        Pole :math:`p`, a host scalar, in [rad/s].

    Returns
    -------
    phi_3
        :math:`\varphi_3(t)`, in [s^3].
    """
    if pole == 0:
        return backend.where(t > 0.0, t**3 / _FACTORIAL_3, 0.0)
    return causal_third_antiderivative_factor(t, pole)


def _expm1_ratio_and_ramp_factor(pole_dt: complex) -> tuple[complex, complex]:
    r"""
    The two factors of the far branch of a linear-times-pole bin average.

    With :math:`q = p \Delta t`, these are
    :math:`E = (e^q - 1) / q` and
    :math:`G = E' - E / 2 = (e^q (q - 2) + q + 2) / (2 q^2)`. Both cancel
    for small :math:`|q|`, where their series
    :math:`\sum_k q^k / (k + 1)!` and
    :math:`\sum_{n \geq 3} (n - 2) q^{n - 2} / (2\, n!)` are summed instead.

    Parameters
    ----------
    pole_dt
        :math:`q = p \Delta t`, dimensionless.

    Returns
    -------
    factors
        The pair :math:`(E,\, G)`, dimensionless.
    """
    if abs(pole_dt) < 1.0:
        expm1_ratio = sum(
            pole_dt**k / math.factorial(k + 1)
            for k in range(_PHI3_SERIES_TERMS)
        )
        ramp_factor = sum(
            (n - 2) * pole_dt ** (n - 2) / (2.0 * math.factorial(n))
            for n in range(3, _PHI3_SERIES_TERMS + 3)
        )
        return expm1_ratio, ramp_factor
    exp_q = cmath.exp(pole_dt)
    expm1_ratio = (exp_q - 1.0) / pole_dt
    ramp_factor = (exp_q * (pole_dt - 2.0) + pole_dt + 2.0) / (
        2.0 * pole_dt**2
    )
    return expm1_ratio, ramp_factor


def triple_box_average_linear_pole(
    t: NumpyArray | CupyArray,
    pole: complex,
    constant: complex,
    slope: complex,
    dt: float,
) -> NumpyArray | CupyArray:
    r"""
    Bin-averaged :math:`\mathrm{Re}[(c_0 + c_1 t)\, e^{p t}]\,\Theta(t)`.

    The B-spline average of a pole times a first-order polynomial, the
    building block of a wake that is a damped or undamped oscillation
    with a linear envelope (e.g. the traveling-wave cavity). The real part
    is taken as is; nothing is doubled for a conjugate partner, unlike
    :func:`triple_box_average_pole`.

    Where the B-spline straddles the onset, :math:`|t| \leq \tfrac32
    \Delta t`, it is the third difference of the third antiderivatives
    :math:`\varphi_3` and :math:`\psi_3 = \partial_p \varphi_3`. Past it,
    the average of :math:`e^{p t}` is :math:`E^3 e^{p (t - 3\Delta t/2)}`
    and that of :math:`t e^{p t}` its derivative in :math:`p`,
    :math:`(t E^3 + 3 \Delta t E^2 G)\, e^{p (t - 3\Delta t/2)}`, with
    :math:`E` and :math:`G` from ``_expm1_ratio_and_ramp_factor``.

    Parameters
    ----------
    t
        Time array (bin centres) at which the wake is evaluated, in [s].
    pole
        Pole :math:`p`, in [rad/s].
    constant
        :math:`c_0`, in the units of the wake.
    slope
        :math:`c_1`, in the units of the wake per second.
    dt
        Bin width, in [s].

    Returns
    -------
    wake
        Bin-averaged wake, in the units of ``constant``.
    """
    onset = 1.5 * dt
    fully_causal = t > onset
    t_near = backend.where(fully_causal, onset, t)
    near = 0.0
    for weight, shift in ((1.0, 1.5), (-3.0, 0.5), (3.0, -0.5), (-1.0, -1.5)):
        lag = t_near + shift * dt
        near = near + weight * (
            constant * _causal_third_antiderivative_factor_any(lag, pole)
            + slope * _causal_third_antiderivative_factor_ramp(lag, pole)
        )
    near = (near / dt**3).real
    expm1_ratio, ramp_factor = _expm1_ratio_and_ramp_factor(pole * dt)
    t_far = backend.where(fully_causal, t, onset)
    far = (
        backend.exp(pole * (t_far - onset))
        * (
            (constant + slope * t_far) * expm1_ratio**3
            + slope * 3.0 * dt * expm1_ratio**2 * ramp_factor
        )
    ).real
    return backend.where(fully_causal, far, near)


def bspline_kink_kernel(u: NumpyArray | CupyArray) -> NumpyArray | CupyArray:
    r"""
    B-spline average of a unit kink, minus the kink itself.

    The quadratic B-spline average of the ramp :math:`(u)_+` is its second
    antiderivative :math:`R(u)`; what is returned is
    :math:`R(u) - (u)_+`, which vanishes for :math:`|u| \geq 3/2`:

    .. math::
        \frac{13}{64} - \frac{|u|}{2} + \frac{3 u^2}{8} - \frac{u^4}{12}
        \quad (|u| \leq \tfrac12), \qquad
        \frac{(3/2 - |u|)^4}{24} \quad (\tfrac12 < |u| < \tfrac32).

    Parameters
    ----------
    u
        Distance past the kink, in units of the bin width.

    Returns
    -------
    excess
        :math:`R(u) - (u)_+`, in units of the bin width.
    """
    distance = backend.abs(u)
    distance = backend.where(distance < 1.5, distance, 1.5)  # NOQA PLR2004
    inner = 13.0 / 64.0 - distance / 2 + 3 * distance**2 / 8 - distance**4 / 12
    outer = (1.5 - distance) ** 4 / 24
    return backend.where(distance <= 0.5, inner, outer)  # NOQA PLR2004


def bspline_step_kernel(u: NumpyArray | CupyArray) -> NumpyArray | CupyArray:
    r"""
    B-spline average of a unit step, minus the step itself.

    The quadratic B-spline average of :math:`\Theta(u)` is its cumulative
    integral :math:`C(u)`; what is returned is :math:`C(u) - \Theta(u)`
    with :math:`\Theta(0) = 1`, which is odd about the step (but for
    :math:`u = 0` itself) and vanishes for :math:`|u| \geq 3/2`.

    Parameters
    ----------
    u
        Distance past the step, in units of the bin width.

    Returns
    -------
    excess
        :math:`C(u) - \Theta(u)`, dimensionless.
    """
    distance = backend.abs(u)
    distance = backend.where(distance < 1.5, distance, 1.5)  # NOQA PLR2004
    # C(-distance), the B-spline mass beyond `distance` on one side
    inner = 0.5 - 0.75 * distance + distance**3 / 3
    outer = (1.5 - distance) ** 3 / 6
    tail = backend.where(distance < 0.5, inner, outer)  # NOQA PLR2004
    return backend.where(u < 0.0, tail, -tail)


def quadratic_bspline(x: float) -> float:
    r"""
    Quadratic B-spline :math:`box * box * box` of unit width, at ``x``.

    The kernel a time-domain wake is bin-averaged with, normalised to
    unit integral and expressed in units of the bin width, so that its
    support is :math:`(-3/2,\, 3/2)`.

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
