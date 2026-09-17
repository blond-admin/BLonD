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
:math:`W(t) = 2\,\mathrm{Re}[\rho e^{p t}]` for :math:`t > 0`, the average
:math:`\overline W = W * B_2` is the third difference of a third
antiderivative (recipe 2 of ``explain_nearfield_farfield_model_rechenbuch.ipynb``):

.. math::
    \overline W(t) = 2\,\mathrm{Re}\!\left[\rho\,
      \frac{\varphi_3(t + \tfrac32\Delta t) - 3\varphi_3(t + \tfrac12\Delta t)
          + 3\varphi_3(t - \tfrac12\Delta t) - \varphi_3(t - \tfrac32\Delta t)}
      {\Delta t^3}\right],
    \qquad
    \varphi_3(t) = \frac{e^{p t} - 1 - p t - \tfrac12 (p t)^2}{p^3}\,\Theta(t),

which past the onset, :math:`t > \tfrac32\Delta t`, collapses to the pure
exponential :math:`2\,\mathrm{Re}[\rho\,((e^{p\Delta t} - 1) / (p\Delta t))^3
e^{p (t - \frac32 \Delta t)}]`.

Notes
-----
Authors:
Simon Lauber
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from blond.core.backends.backend import backend

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray

# Terms summed for the series form of `causal_third_antiderivative_factor`
# (|p t| < 1, so the omitted tail is below 1 / 23! ~ 4e-23).
_PHI3_SERIES_TERMS = 20
_FACTORIAL_3 = 6.0


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
    t: NumpyArray | CupyArray, pole: complex, residue: complex, dt: float
) -> NumpyArray | CupyArray:
    r"""
    Bin-averaged wake of a single pole, :math:`\overline W = W * B_2`.

    The third difference of :func:`causal_third_antiderivative_factor` is
    used where the B-spline straddles the onset, and the exponential closed
    form (:func:`_smoothed_pole`) past it, where differencing would cancel
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

    Returns
    -------
    wake
        Bin-averaged wake of this single pole, in the units of ``residue``.
    """
    onset = 1.5 * dt
    fully_causal = t > onset
    # Both branches are evaluated everywhere and selected with `where`: a
    # boolean-mask gather would sync a CuPy array to the host, and this is
    # reached from the per-turn loop. Each branch's argument is clamped to
    # its own side of the onset so the discarded values stay finite.
    t_far = backend.where(fully_causal, t, onset)
    t_near = backend.where(fully_causal, onset, t)

    far = (residue * _smoothed_pole(t_far, pole, dt)).real
    near = (
        residue
        * (
            causal_third_antiderivative_factor(t_near + 1.5 * dt, pole)
            - 3.0 * causal_third_antiderivative_factor(t_near + 0.5 * dt, pole)
            + 3.0 * causal_third_antiderivative_factor(t_near - 0.5 * dt, pole)
            - causal_third_antiderivative_factor(t_near - 1.5 * dt, pole)
        )
    ).real / dt**3
    # 2 - True == 1 for a real pole, 2 - False == 2 for a complex one; works
    # for a host scalar and a device slice alike, without a branch.
    pair_factor = 2.0 - (pole.imag == 0)
    return pair_factor * backend.where(fully_causal, far, near)


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
    # Length-1 slices rather than `complex(poles[i])`: the latter is a
    # device-to-host transfer per pole inside the per-turn loop.
    for pole_i in range(len(poles)):
        out += triple_box_average_pole(
            t, poles[pole_i : pole_i + 1], residues[pole_i : pole_i + 1], dt
        )
    return out
