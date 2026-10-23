# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Tests for :mod:`blond.physics.impedances.wake_per_bin.bin_average`.

To debug a failure, rerun with ``DEV_DRAW=True`` in the environment, e.g.
``DEV_DRAW=True python -m pytest -p no:randomly <this file>::<test>``:
every function of the module is then drawn against its reference by the
test that checks it, before that test asserts. The dotted lines mark the
branch switches (``t = +-1.5 dt`` for near / far, ``|p t| = 1`` for
series / direct), where a slip shows up first, and the footer of each
figure says what a healthy plot looks like.

Notes
-----
Authors:
Simon Lauber
"""

import os
import textwrap
import traceback
import unittest
import warnings
from unittest import mock

import mpmath as mp
import numpy as np
import pytest

from blond.core.backends.backend import backend
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.impedances.wake_per_bin import bin_average
from blond.physics.impedances.wake_per_bin.bin_average import (
    bspline_kink_kernel,
    bspline_step_kernel,
    quadratic_bspline,
    triple_box_average_linear_pole,
    triple_box_average_pole,
    triple_box_average_poles,
)
from blond.testing.backend_testing import BLonDTestCase

# Quadrature points used by the reference convolution, per sub-interval
# between two breakpoints of the integrand.
_QUADRATURE_POINTS = 20001

# Draw each function against its reference, see the module docstring.
DEV_DRAW = os.getenv("DEV_DRAW", "False").lower() == "true"


def _dev_draw(
    title: str,
    hint: str,
    panels: list,
    x_label: str,
    vlines: tuple = (),
    log_x: bool = False,
) -> None:
    """
    Draw one debugging figure, for ``DEV_DRAW=True`` runs only.

    Parameters
    ----------
    title
        Figure title; names the function under test.
    hint
        What a healthy figure looks like, printed under the plot.
    panels
        One ``(y_label, curves, log_y)`` per panel, top to bottom.
        ``curves`` maps a legend label to ``(x, y)`` or
        ``(x, y, format)``. On a ``log_y`` panel ``|y|`` is drawn and
        exact zeros are left out.
    x_label
        Label of the shared x axis.
    vlines
        x positions of branch switches, drawn dotted on every panel.
    log_x
        Logarithmic x axis.
    """
    import matplotlib.pyplot as plt  # only imported when drawing

    fig, axes = plt.subplots(
        len(panels),
        1,
        sharex=True,
        squeeze=False,
        figsize=(8, 2 + 2.5 * len(panels)),
    )
    fig.suptitle(title)
    for axis, (y_label, curves, log_y) in zip(axes[:, 0], panels, strict=True):
        anything_positive = False
        for label, (x, y, *style) in curves.items():
            y = np.asarray(y, dtype=float)
            if log_y:
                y = np.abs(y)
                y = np.where(y > 0, y, np.nan)
                anything_positive |= bool(np.any(y > 0))
            axis.plot(x, y, *style, label=label or None)
        for position in vlines:
            axis.axvline(position, color="grey", linestyle=":")
        if log_y and anything_positive:
            axis.set_yscale("log")
            # Underflowing tails would otherwise stretch the axis to 1e-300.
            axis.set_ylim(bottom=max(axis.get_ylim()[0], 1e-20))
        elif log_y:  # nothing to draw on a log axis: say why
            axis.text(
                0.5,
                0.5,
                "exactly zero everywhere (bit-identical)",
                transform=axis.transAxes,
                ha="center",
            )
        if log_x:
            axis.set_xscale("log")
        axis.set_ylabel(y_label)
        if any(curves):
            axis.legend(fontsize="small")
    axes[-1, 0].set_xlabel(x_label)
    fig.text(0.01, 0.005, textwrap.fill(hint, 110), fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    plt.show()


class TestQuadraticBspline(BLonDTestCase):
    """The B-spline the references below are built on, checked on its own."""

    def test_is_zero_outside_its_support(self):
        for x in (-10.0, -1.5, 1.5, 2.0, 1e9):
            with self.subTest(x=x):
                self.assertEqual(quadratic_bspline(x), 0.0)

    def test_is_box_convolved_with_itself_three_times(self):
        """Against a fine-grid ``box * box * box``, independently built."""
        step = 1e-3
        box = np.ones(1000)  # unit width, unit area at this step
        triple_box = np.convolve(np.convolve(box, box), box) * step**2
        x = (np.arange(len(triple_box)) - (len(triple_box) - 1) / 2) * step
        expected = np.interp(
            [-1.4, -1.0, -0.5, -0.2, 0.0, 0.7, 1.3], x, triple_box
        )
        result = [
            quadratic_bspline(entry)
            for entry in (-1.4, -1.0, -0.5, -0.2, 0.0, 0.7, 1.3)
        ]
        if DEV_DRAW:
            dense = np.linspace(-2.0, 2.0, 801)
            spline = np.array([quadratic_bspline(entry) for entry in dense])
            reference = np.interp(dense, x, triple_box, left=0.0, right=0.0)
            _dev_draw(
                "quadratic_bspline",
                "A bell on (-1.5, 1.5), 0.75 at 0 and 0.125 at +-1, smooth "
                "but for its knots (dotted), exactly zero outside. The "
                "difference to the fine-grid box*box*box stays at that "
                "grid's ~1e-6 level; a spike at a knot is a wrong piece.",
                [
                    (
                        "B_2(x)",
                        {
                            "quadratic_bspline": (dense, spline),
                            "box*box*box": (dense, reference, "--"),
                        },
                        False,
                    ),
                    ("|difference|", {"": (dense, spline - reference)}, True),
                ],
                x_label="x [bins]",
                vlines=(-1.5, -0.5, 0.5, 1.5),
            )
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-5)

    def test_has_unit_integral(self):
        nodes, weights = np.polynomial.legendre.leggauss(8)
        total = 0.0
        for lower, upper in ((-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5)):
            x = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
            total += (
                0.5
                * (upper - lower)
                * np.sum(weights * [quadratic_bspline(e) for e in x])
            )
        self.assertAlmostEqual(total, 1.0, delta=1e-15)


def reference_triple_box_average(
    time: np.ndarray,
    pole: complex,
    residue: complex,
    dt: float,
    pair_factor: float,
) -> np.ndarray:
    r"""
    Brute-force triple box average of a single pole's wake.

    Convolves the causal wake
    ``pair_factor * Re[residue * exp(pole * t)] * (t > 0)`` with the
    quadratic B-spline of one bin width on a fine grid, independently of
    the closed form under test.

    Parameters
    ----------
    time
        Times at which to evaluate the bin-averaged wake, in [s].
    pole
        Pole :math:`p = -\alpha + i \bar\omega`, in [rad/s].
    residue
        Residue :math:`\rho`.
    dt
        Bin width, in [s].
    pair_factor
        1 for a real pole (no conjugate partner), 2 for a complex one.

    Returns
    -------
    wake
        Bin-averaged wake, in the units of ``residue``.
    """
    knots = {-1.5 * dt, -0.5 * dt, 0.5 * dt, 1.5 * dt}
    out = np.zeros(len(time))
    for index, time_entry in enumerate(time):
        # The integrand is a different smooth function between the B-spline
        # knots, and jumps to zero at the causal onset ``lag == time_entry``,
        # so the integral is split at every breakpoint.
        breakpoints = knots | {time_entry} if -1.5 * dt < time_entry else knots
        for lower, upper in zip(
            sorted(breakpoints)[:-1], sorted(breakpoints)[1:], strict=True
        ):
            if upper > time_entry:  # acausal piece: the wake is zero
                continue
            lag = np.linspace(lower, upper, _QUADRATURE_POINTS)
            weight = np.array(
                [quadratic_bspline(entry / dt) / dt for entry in lag]
            )
            wake = (
                pair_factor
                * (residue * np.exp(pole * (time_entry - lag))).real
            )
            out[index] += np.trapezoid(wake * weight, lag)
    return out


def gauss_legendre_bspline_average(wake, time, dt, breakpoints, nodes=48):
    """
    Exact B-spline average of a wake that is smooth between breakpoints.

    The integral is split at the B-spline's knots and at every breakpoint
    of the wake, so Gauss-Legendre quadrature converges to round-off on
    each piece.

    Parameters
    ----------
    wake
        Vectorised wake ``W(t)``.
    time
        Times at which to evaluate the average, in [s].
    dt
        Bin width, in [s].
    breakpoints
        Times where ``W`` or one of its derivatives jumps, in [s].
    nodes
        Gauss-Legendre nodes per piece.

    Returns
    -------
    wake_per_bin
        ``W * B_2`` at ``time``.
    """
    abscissae, weights = np.polynomial.legendre.leggauss(nodes)
    out = np.zeros(len(time))
    for index, time_entry in enumerate(time):
        cuts = {-1.5, -0.5, 0.5, 1.5}
        cuts |= {
            (point - time_entry) / dt
            for point in breakpoints
            if -1.5 < (point - time_entry) / dt < 1.5
        }
        cuts = sorted(cuts)
        for lower, upper in zip(cuts[:-1], cuts[1:], strict=False):
            offset = 0.5 * (upper - lower) * abscissae + 0.5 * (upper + lower)
            weight = np.array([quadratic_bspline(x) for x in offset])
            out[index] += (
                0.5
                * (upper - lower)
                * np.sum(weights * weight * wake(time_entry + offset * dt))
            )
    return out


class TestBsplineStepAndKinkKernels(BLonDTestCase):
    """The B-spline averages of a unit step and a unit kink, minus each."""

    def setUp(self):
        # Includes the knots, the step itself and both ends of the support.
        self.u = np.array(
            [-2.0, -1.5, -1.2, -0.5, -0.3, -1e-9, 0.0, 0.2, 0.5, 0.9, 1.5, 3.0]
        )

    @staticmethod
    def _average(function, u):
        """``int function(u + x) B_2(x) dx`` by split Gauss-Legendre."""
        nodes, weights = np.polynomial.legendre.leggauss(16)
        cuts = sorted(
            {-1.5, -0.5, 0.5, 1.5} | ({-u} if abs(u) < 1.5 else set())
        )
        total = 0.0
        for lower, upper in zip(cuts[:-1], cuts[1:], strict=False):
            x = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
            weight = np.array([quadratic_bspline(entry) for entry in x])
            total += (
                0.5
                * (upper - lower)
                * np.sum(weights * weight * function(u + x))
            )
        return total

    def _draw_kernel(self, kernel, function, subtracted, hint):
        dense = np.linspace(-2.0, 2.0, 401)
        expected = [
            self._average(function, entry) - subtracted(entry)
            for entry in dense
        ]
        result = copy_to_cpu(kernel(backend.array(dense)))
        _dev_draw(
            kernel.__name__,
            hint,
            [
                (
                    "excess",
                    {
                        "closed form": (dense, result),
                        "quadrature": (dense, expected, "--"),
                    },
                    False,
                ),
                ("|difference|", {"": (dense, result - expected)}, True),
            ],
            x_label="u [bins]",
            vlines=(-1.5, -0.5, 0.5, 1.5),
        )

    def test_step_kernel(self):
        expected = [
            self._average(lambda v: (v >= 0).astype(float), entry)
            - float(entry >= 0)
            for entry in self.u
        ]
        if DEV_DRAW:
            self._draw_kernel(
                bspline_step_kernel,
                lambda v: (v >= 0).astype(float),
                lambda entry: float(entry >= 0),
                "Odd about u = 0: +0.5 just before the step, -0.5 at and "
                "after it (the subtracted step), smooth elsewhere and "
                "exactly zero for |u| >= 1.5. The difference is round-off; "
                "a bump between two knots (dotted) is a wrong piece.",
            )
        np.testing.assert_allclose(
            copy_to_cpu(bspline_step_kernel(backend.array(self.u))),
            expected,
            rtol=0,
            atol=1e-14,
        )

    def test_kink_kernel(self):
        expected = [
            self._average(lambda v: np.maximum(v, 0.0), entry)
            - max(entry, 0.0)
            for entry in self.u
        ]
        if DEV_DRAW:
            self._draw_kernel(
                bspline_kink_kernel,
                lambda v: np.maximum(v, 0.0),
                lambda entry: max(entry, 0.0),
                "Even and smooth, 13/64 at u = 0, falling to exactly zero "
                "for |u| >= 1.5. The difference is round-off; a bump "
                "between two knots (dotted) is a wrong piece.",
            )
        np.testing.assert_allclose(
            copy_to_cpu(bspline_kink_kernel(backend.array(self.u))),
            expected,
            rtol=0,
            atol=1e-14,
        )


class TestTripleBoxAverageLinearPole(BLonDTestCase):
    """``(c0 + c1 t) exp(p t)`` from ``t = 0`` on, against quadrature."""

    def _check(self, pole, constant, slope, dt):
        time = np.arange(-4, 40) * 0.37 * dt

        def wake(t):
            return np.where(
                t > 0, ((constant + slope * t) * np.exp(pole * t)).real, 0.0
            )

        expected = gauss_legendre_bspline_average(wake, time, dt, (0.0,))
        result = copy_to_cpu(
            triple_box_average_linear_pole(
                backend.array(time), pole, constant, slope, dt
            )
        )
        scale = np.max(np.abs(expected))
        if DEV_DRAW:
            _dev_draw(
                f"triple_box_average_linear_pole, p dt = {pole * dt:.3g}",
                "Closed form and quadrature on top of each other, nonzero "
                "from t = -1.5 dt on (the non-causal tap). The error stays "
                "flat at round-off: a step at t = 1.5 dt means near and "
                "far branch disagree; one where |p (t +- k dt / 2)| = 1 "
                "points at a series / direct switch of phi_3, psi_3 or "
                "the (E, G) factors. A zero pole goes through the t^3 / 6 "
                "limit.",
                [
                    (
                        "wake",
                        {
                            "closed form": (time / dt, result),
                            "quadrature": (time / dt, expected, "o"),
                        },
                        False,
                    ),
                    (
                        "|error| / max",
                        {"": (time / dt, (result - expected) / scale)},
                        True,
                    ),
                ],
                x_label="t [bins]",
                vlines=(-1.5, 1.5),
            )
        np.testing.assert_allclose(
            result, expected, rtol=0, atol=1e-11 * scale
        )

    def test_undamped_pole_well_resolved(self):
        self._check(2j * np.pi * 2e8, 3.0 + 1.0j, -2e9, 1e-10)

    def test_undamped_pole_beyond_the_bin_nyquist(self):
        self._check(2j * np.pi * 8e9, 3.0, -2e9 + 1e9j, 1e-10)

    def test_damped_pole(self):
        self._check(-3e9 + 2j * np.pi * 1e9, 1.0 - 2.0j, 5e8, 1e-10)

    def test_real_pole_is_not_doubled(self):
        self._check(-2e9 + 0j, 1.5, 4e8, 1e-10)

    def test_zero_pole_is_a_polynomial(self):
        self._check(0j, 2.0, -1e10, 1e-10)


class TestTripleBoxAveragePole(BLonDTestCase):
    """The bin-averaged wake of one pole, against a fine-grid convolution."""

    def setUp(self):
        self.dt = 1e-10  # [s]
        self.real_pole = -2.1e9 + 0.0j  # |pole * dt| = 0.21 [rad/s]
        self.real_residue = 3.7e4 + 0.0j
        self.complex_pole = -2.1e9 + 8.3e9j  # [rad/s]
        self.complex_residue = 3.7e4 - 1.2e4j

    def _assert_matches_reference(self, time, pole, residue, pair_factor):
        expected = reference_triple_box_average(
            time, pole, residue, self.dt, pair_factor=pair_factor
        )
        result = copy_to_cpu(
            triple_box_average_pole(
                backend.array(time, dtype=backend.float),
                pole,
                residue,
                self.dt,
            )
        )
        if DEV_DRAW:
            dense = np.linspace(-2.0, 12.0, 561) * self.dt
            closed_form = copy_to_cpu(
                triple_box_average_pole(
                    backend.array(dense, dtype=backend.float),
                    pole,
                    residue,
                    self.dt,
                )
            )
            _dev_draw(
                f"triple_box_average_pole, pair factor {pair_factor:g}",
                "The fine-grid references (circles) sit on the closed-form "
                "curve, which starts at t = -1.5 dt. Off by 2 everywhere: "
                "the conjugate-partner doubling is wrong (real poles are "
                "not doubled). A kink at t = 1.5 dt: near and far branch "
                "disagree, see TestOnsetBranchContinuity's draws.",
                [
                    (
                        "wake",
                        {
                            "closed form": (dense / self.dt, closed_form),
                            "reference": (time / self.dt, expected, "o"),
                        },
                        False,
                    ),
                    (
                        "|error| / max",
                        {
                            "": (
                                time / self.dt,
                                (result - expected) / np.abs(expected).max(),
                            )
                        },
                        True,
                    ),
                ],
                x_label="t [bins]",
                vlines=(-1.5, 1.5),
            )
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_real_pole_far_field_is_not_doubled(self):
        """A real pole has no conjugate partner past the causal onset."""
        self._assert_matches_reference(
            np.arange(2, 12) * self.dt, self.real_pole, self.real_residue, 1.0
        )

    def test_real_pole_onset_is_not_doubled(self):
        """The same holds for the taps straddling the causal onset."""
        self._assert_matches_reference(
            np.arange(-1, 2) * self.dt, self.real_pole, self.real_residue, 1.0
        )

    def test_complex_pole_far_field_is_doubled(self):
        """A complex pole stands in for its unstored conjugate partner."""
        self._assert_matches_reference(
            np.arange(2, 12) * self.dt,
            self.complex_pole,
            self.complex_residue,
            2.0,
        )

    def test_complex_pole_onset_is_doubled(self):
        """The onset branch keeps the conjugate-partner factor as well."""
        self._assert_matches_reference(
            np.arange(-1, 2) * self.dt,
            self.complex_pole,
            self.complex_residue,
            2.0,
        )

    def test_is_zero_before_the_kernel_starts(self):
        """Nothing is induced more than one and a half bins ahead."""
        time = backend.array([-5.0, -2.0, -1.5], dtype=backend.float)
        result = copy_to_cpu(
            triple_box_average_pole(
                time * self.dt,
                self.complex_pole,
                self.complex_residue,
                self.dt,
            )
        )
        np.testing.assert_array_equal(result, 0.0)

    def test_sum_over_poles(self):
        """`triple_box_average_poles` is the sum of the single-pole kernels."""
        time = backend.array(np.arange(-1, 6) * self.dt, dtype=backend.float)
        poles = backend.array(
            [self.real_pole, self.complex_pole], dtype=backend.complex
        )
        residues = backend.array(
            [self.real_residue, self.complex_residue], dtype=backend.complex
        )
        expected = copy_to_cpu(
            triple_box_average_pole(
                time, self.real_pole, self.real_residue, self.dt
            )
            + triple_box_average_pole(
                time, self.complex_pole, self.complex_residue, self.dt
            )
        )
        result = copy_to_cpu(
            triple_box_average_poles(time, poles, residues, self.dt)
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestTripleBoxAverageZeroPole(BLonDTestCase):
    """A zero pole is the step wake ``Re[rho] * Theta(t)``, not NaN.

    ``(e^q - 1) / q`` and the direct ``phi_3`` reach it as 0 / 0. A pole
    fitted to (almost) DC, or a padding entry, must not poison the whole
    wake, and the device path cannot afford a host-side check for it.
    """

    def setUp(self):
        self.dt = 1e-10  # [s]
        self.time = np.arange(-6, 40) * 0.37 * self.dt  # [s]
        self.residue = 2.5 + 0.0j

    def test_is_the_averaged_step(self):
        expected = gauss_legendre_bspline_average(
            lambda t: np.where(t > 0, self.residue.real, 0.0),
            self.time,
            self.dt,
            (0.0,),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no 0 / 0 on the way
            result = copy_to_cpu(
                triple_box_average_pole(
                    backend.array(self.time, dtype=backend.float),
                    0j,
                    self.residue,
                    self.dt,
                )
            )
        if DEV_DRAW:
            _dev_draw(
                "triple_box_average_pole, zero pole",
                "The B-spline-smoothed step: 0 before t = -1.5 dt, half "
                "the residue at t = 0, the full residue from t = 1.5 dt "
                "on, with quadrature (circles) on the curve. NaN (no line) "
                "means a 0 / 0 got through.",
                [
                    (
                        "wake",
                        {
                            "closed form": (self.time / self.dt, result),
                            "quadrature": (
                                self.time / self.dt,
                                expected,
                                "o",
                            ),
                        },
                        False,
                    ),
                    (
                        "|error|",
                        {"": (self.time / self.dt, result - expected)},
                        True,
                    ),
                ],
                x_label="t [bins]",
                vlines=(-1.5, 1.5),
            )
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-14)

    def test_among_other_poles_on_short_and_long_grids(self):
        """Broadcast and windowed paths alike: no NaN leaks to the rest."""
        poles = backend.array([0j, -2e8 + 4e9j], dtype=backend.complex)
        residues = backend.array(
            [self.residue, 1e4 - 3e3j], dtype=backend.complex
        )
        for n_bins in (512, 2 * bin_average._NEAR_WINDOW_MIN_BINS):
            with self.subTest(n_bins=n_bins):
                time = backend.array(
                    (np.arange(n_bins) - 40) * self.dt, dtype=backend.float
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    result = copy_to_cpu(
                        triple_box_average_poles(
                            time, poles, residues, self.dt
                        )
                    )
                expected = copy_to_cpu(
                    triple_box_average_pole(time, 0j, self.residue, self.dt)
                    + triple_box_average_pole(
                        time, -2e8 + 4e9j, 1e4 - 3e3j, self.dt
                    )
                )
                self.assertTrue(np.all(np.isfinite(result)))
                np.testing.assert_allclose(
                    result, expected, rtol=1e-13, atol=0.0
                )


class TestTripleBoxAverageLinearPoleLongGrid(BLonDTestCase):
    """On a long grid only the bins next to the onset take the near branch.

    The near branch of the linear pole is eight third antiderivatives,
    each a 20-step Horner loop; the traveling-wave cavity evaluates it per
    mode over the whole profile.
    """

    def setUp(self):
        self.dt = 1e-10  # [s]
        self.time = backend.array(
            (np.arange(20_000) - 50) * self.dt, dtype=backend.float
        )
        self.args = (-2e8 + 4e9j, 3.0 - 1.0j, 2e9, self.dt)

    def test_near_branch_only_evaluated_next_to_the_onset(self):
        with mock.patch.object(
            bin_average,
            "_causal_third_antiderivative_factor_ramp",
            wraps=bin_average._causal_third_antiderivative_factor_ramp,
        ) as psi_3:
            triple_box_average_linear_pole(self.time, *self.args)
        largest_call = max(len(call.args[0]) for call in psi_3.call_args_list)
        # -1.5 * dt < t <= 1.5 * dt holds for three bins of this grid.
        self.assertLessEqual(largest_call, 3)

    def test_matches_full_array_evaluation(self):
        with mock.patch.object(bin_average, "_NEAR_WINDOW_MIN_BINS", 10**9):
            expected = copy_to_cpu(
                triple_box_average_linear_pole(self.time, *self.args)
            )
        result = copy_to_cpu(
            triple_box_average_linear_pole(self.time, *self.args)
        )
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=0.0)
        # Before the kernel's support the wake is exactly zero.
        np.testing.assert_array_equal(result[:49], 0.0)

    @unittest.skipUnless(__debug__, "asserts are stripped under python -O")
    def test_rejects_an_array_pole(self):
        """The pole is branched on host-side, so it must be a scalar."""
        with self.assertRaises(AssertionError):
            triple_box_average_linear_pole(
                self.time,
                backend.array([self.args[0]], dtype=backend.complex),
                *self.args[1:],
            )


class TestTripleBoxAverageContinuumLimit(BLonDTestCase):
    """As the bin shrinks, the bin-averaged wake tends to the point wake."""

    def setUp(self):
        self.time = backend.array([1e-9], dtype=backend.float)  # [s]
        self.residue = 3.7e4 - 1.2e4j
        # Bin far shorter than any time scale of the poles below, so the
        # B-spline average is the point wake to better than 1e-6.
        self.dt = 1e-14  # [s]

    def test_real_pole_tends_to_undoubled_point_wake(self):
        """A real pole's point wake is ``Re[rho * exp(pole * t)]``."""
        pole = -2.1e8 + 0.0j  # [rad/s]
        expected = (self.residue * np.exp(pole * 1e-9)).real
        result = copy_to_cpu(
            triple_box_average_pole(self.time, pole, self.residue, self.dt)
        )
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_complex_pole_tends_to_doubled_point_wake(self):
        """A complex pole's point wake is ``2 * Re[rho * exp(pole * t)]``."""
        pole = -2.1e8 + 8.3e8j  # [rad/s]
        expected = 2.0 * (self.residue * np.exp(pole * 1e-9)).real
        result = copy_to_cpu(
            triple_box_average_pole(self.time, pole, self.residue, self.dt)
        )
        np.testing.assert_allclose(result, expected, rtol=1e-6)


# Working precision of the arbitrary-precision reference below. The loss
# factor is a near-cancelling sum over an oscillating wake, so a float64
# reference cannot arbitrate between two closed forms that already agree to
# 1e-12 pointwise.
_REFERENCE_DPS = 50


def reference_loss_factor(
    time: np.ndarray,
    pole: complex,
    residue: complex,
    dt: float,
    sigma: float,
) -> float:
    r"""
    Loss factor of one pole's bin-averaged wake, at 50 decimal digits.

    Evaluates the triple box average as the third difference of the causal
    third antiderivative :math:`\varphi_3` in arbitrary precision, so that
    neither the cancellation of the third difference nor that of the
    weighted sum below costs any digits.

    Parameters
    ----------
    time
        Times at which the bin-averaged wake is evaluated, in [s].
    pole
        Pole :math:`p = -\alpha + i \bar\omega`, in [rad/s].
    residue
        Residue :math:`\rho`.
    dt
        Bin width, in [s].
    sigma
        R.m.s. width of the Gaussian line density the wake is weighted
        with, in [s].

    Returns
    -------
    loss_factor
        The wake averaged over the line density, in the units of
        ``residue``.
    """
    with mp.workdps(_REFERENCE_DPS):
        pole_mp = mp.mpc(pole.real, pole.imag)
        residue_mp = mp.mpc(residue.real, residue.imag)
        bin_dt = mp.mpf(dt)
        sigma_mp = mp.mpf(sigma)

        def phi_3(lag):
            if lag <= 0:
                return mp.mpc(0)
            pole_lag = pole_mp * lag
            return (
                mp.e**pole_lag - 1 - pole_lag - pole_lag**2 / 2
            ) / pole_mp**3

        wake = []
        for time_entry in time:
            centre = mp.mpf(float(time_entry))
            third_difference = (
                phi_3(centre + 3 * bin_dt / 2)
                - 3 * phi_3(centre + bin_dt / 2)
                + 3 * phi_3(centre - bin_dt / 2)
                - phi_3(centre - 3 * bin_dt / 2)
            )
            wake.append(2 * mp.re(residue_mp * third_difference) / bin_dt**3)

        weights = [
            mp.e ** (-(mp.mpf(float(entry)) ** 2) / (2 * sigma_mp**2))
            for entry in time
        ]
        normalisation = mp.fsum(weights)
        return float(
            mp.fsum(w * v for w, v in zip(weights, wake, strict=True))
            / normalisation
        )


class TestTripleBoxAverageLossFactor(BLonDTestCase):
    """The loss factor of a coarsely-binned, high-Q pole."""

    def setUp(self):
        # One bin per resonator period (|pole * dt| ~ 6.3): coarse enough
        # that the closed form past the onset is doing real work, and a
        # narrow resonance so the loss factor is a strongly cancelling sum.
        quality_factor = 1e4
        center_frequency = 1e9  # [Hz]
        shunt_impedance = 1e6  # [Ohm]
        omega = 2.0 * np.pi * center_frequency  # [rad/s]
        alpha = omega / (2.0 * quality_factor)  # [1/s]
        omega_bar = np.sqrt(omega**2 - alpha**2)  # [rad/s]

        self.pole = -alpha + 1j * omega_bar  # [rad/s]
        self.residue = shunt_impedance * alpha * (1.0 + 1j * alpha / omega_bar)
        self.dt = 1.0 / center_frequency  # [s], one resonator period
        self.time = np.arange(-1, 400) * self.dt  # [s]
        self.sigma = 30 * self.dt  # [s]

    def test_loss_factor_matches_high_precision_reference(self):
        """The bin-averaged wake integrates to the exact loss factor."""
        expected = reference_loss_factor(
            self.time, self.pole, self.residue, self.dt, self.sigma
        )
        wake = copy_to_cpu(
            triple_box_average_pole(
                backend.array(self.time, dtype=backend.float),
                self.pole,
                self.residue,
                self.dt,
            )
        )
        weights = np.exp(-0.5 * (self.time / self.sigma) ** 2)
        weights /= weights.sum()
        loss_factor = float(np.sum(wake * weights))

        self.assertAlmostEqual(
            loss_factor / expected,
            1.0,
            delta=2e-7,
            msg=(
                "the closed form past the onset loses digits the "
                "loss factor cannot afford"
            ),
        )


class TestTripleBoxAverageSumRule(BLonDTestCase):
    r"""The bin-averaged wake carries the wake's whole integral, at any dt.

    The B-spline has unit area, so summing the averaged wake over every
    whole-bin lag gives :math:`\int W\,dt = \kappa\,\mathrm{Re}(-\rho / p)`,
    with :math:`\kappa = 2` for a complex pole and 1 for a real one. That
    holds whether the wake decays within a fraction of a bin or over
    thousands, so it pins the near-field taps -- series and direct branch
    alike -- without relying on the closed form they are built from.
    """

    def test_sum_matches_wake_integral_across_bin_widths(self):
        dt = 1e-9  # [s]
        # Pole direction in the complex plane, by name; the magnitude is
        # set per case through |pole * dt|.
        pole_directions = {
            "real": -1.0 + 0.0j,
            "Q=0.6": -0.83 + 0.55j,
            "Q~2": -0.25 + 0.97j,
            "Q~100": -5e-3 + 1.0j,
        }
        # Relative sum-rule error per direction, for the dev draw.
        errors = {name: ([], []) for name in pole_directions}
        for name, direction in pole_directions.items():
            for pole_dt in (10.0 ** np.arange(-3.0, 3.5, 0.5)).tolist():
                pole = pole_dt * direction / dt
                if name == "real":
                    pole = complex(pole.real, 0.0)
                residue = 1.3 - 0.4j if pole.imag else 1.3 + 0.0j
                # Forty decay lengths leave a tail below 1e-17.
                decay_bins = 1.0 / (pole_dt * abs(direction.real))
                n_bins = int(40 * decay_bins) + 10
                if n_bins > 2_000_000:
                    continue
                with self.subTest(pole=name, pole_dt=pole_dt):
                    lags = np.arange(-2, n_bins) * dt
                    wake = copy_to_cpu(
                        triple_box_average_pole(
                            backend.array(lags, dtype=backend.float),
                            pole,
                            residue,
                            dt,
                        )
                    )
                    pair_factor = 1.0 if pole.imag == 0 else 2.0
                    expected = pair_factor * (-residue / pole).real
                    errors[name][0].append(pole_dt)
                    errors[name][1].append(
                        (float(np.sum(wake)) * dt - expected)
                        / (pair_factor * abs(residue / pole))
                    )
                    self.assertAlmostEqual(
                        float(np.sum(wake)) * dt,
                        expected,
                        delta=1e-10 * pair_factor * abs(residue / pole),
                    )
        if DEV_DRAW:
            _dev_draw(
                "triple_box_average_pole: sum rule",
                "dt * sum of the taps against the wake integral, per pole "
                "direction, below the 1e-10 tolerance (dashed) over six "
                "decades of |p dt|. A rise on the left points at the "
                "series branch of phi_3 (|p dt| << 1, many taps); on the "
                "right at the near taps, where the wake dies within a bin.",
                [
                    (
                        "|relative error|",
                        {
                            **{
                                name: (np.array(x), np.array(y), "o-")
                                for name, (x, y) in errors.items()
                            },
                            "tolerance": (
                                np.array([1e-3, 10**3]),
                                np.array([1e-10, 1e-10]),
                                "k--",
                            ),
                        },
                        True,
                    )
                ],
                x_label="|p dt|",
                vlines=(1.0,),
                log_x=True,
            )


class TestTripleBoxAverageLongGrid(BLonDTestCase):
    """On a long grid only the bins next to the onset need the near branch.

    The near branch (four third antiderivatives, each a 19-step Horner
    loop) is nonzero only for ``-1.5 * dt < t <= 1.5 * dt``. Evaluating it
    on every bin made the one-time wake setup of a full SPS turn (~6e5
    bins, ~250 poles) take minutes.
    """

    def setUp(self):
        self.dt = 1e-10  # [s]
        # Starts well before -1.5 * dt, so bins exist where the near
        # branch is zero but the clamped far branch is not.
        self.time = backend.array(
            (np.arange(20_000) - 50) * self.dt, dtype=backend.float
        )
        self.poles = backend.array(
            [-3e9 + 0.0j, -2e8 + 4e9j, -5e6 + 7e8j], dtype=backend.complex
        )
        self.residues = backend.array(
            [2.5e3 + 0.0j, 1.0e4 - 3.0e3j, -7.0e2 + 2.0e2j],
            dtype=backend.complex,
        )

    def test_near_branch_only_evaluated_next_to_the_onset(self):
        """The near branch sees only the few bins around ``t = 0``."""
        with mock.patch.object(
            bin_average,
            "causal_third_antiderivative_factor",
            wraps=bin_average.causal_third_antiderivative_factor,
        ) as phi_3:
            bin_average.triple_box_average_poles(
                self.time, self.poles, self.residues, self.dt
            )
        largest_call = max(len(call.args[0]) for call in phi_3.call_args_list)
        # -1.5 * dt < t <= 1.5 * dt holds for three bins of this grid.
        self.assertLessEqual(largest_call, 3)

    def test_matches_full_array_evaluation(self):
        """The windowed sum equals the per-pole full-array evaluation."""
        expected = np.zeros(len(self.time))
        for pole_i in range(len(self.poles)):
            expected += copy_to_cpu(
                triple_box_average_pole(
                    self.time,
                    self.poles[pole_i : pole_i + 1],
                    self.residues[pole_i : pole_i + 1],
                    self.dt,
                )
            )
        result = copy_to_cpu(
            triple_box_average_poles(
                self.time, self.poles, self.residues, self.dt
            )
        )
        if DEV_DRAW:
            near_idx = copy_to_cpu(
                bin_average._near_window(self.time, self.dt)
            )
            lags = copy_to_cpu(self.time) / self.dt
            zoom = slice(0, 120)
            _dev_draw(
                "_near_window + triple_box_average_poles, long grid",
                "Only the squares -- the _near_window indices, -1.5 < "
                "t / dt <= 1.5 -- take the near branch. Windowed and "
                "full-array sums overlap -- usually bit-identical -- and "
                "are exactly zero before t = -1.5 dt. A "
                "jump at the window's edge means it misses a bin or "
                "takes one too many.",
                [
                    (
                        "wake",
                        {
                            "windowed": (lags[zoom], result[zoom]),
                            "full array": (lags[zoom], expected[zoom], "--"),
                            "near window": (
                                lags[near_idx],
                                result[near_idx],
                                "s",
                            ),
                        },
                        False,
                    ),
                    (
                        "|difference| / max",
                        {
                            "": (
                                lags[zoom],
                                (result - expected)[zoom]
                                / np.abs(expected).max(),
                                "o-",
                            )
                        },
                        True,
                    ),
                ],
                x_label="t [bins]",
                vlines=(-1.5, 1.5),
            )
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=0.0)
        # Before the kernel's support the wake is exactly zero.
        np.testing.assert_array_equal(result[:49], 0.0)


class TestTripleBoxAverageShortGridAllPolesAtOnce(BLonDTestCase):
    """On a short grid all poles are evaluated in one broadcast call.

    Each call of `triple_box_average_pole` is a few hundred small array
    operations -- on a GPU, a few hundred kernel launches. Walking the
    poles one by one made the per-turn near taps of a few hundred poles
    cost a second per turn on the GPU.
    """

    def setUp(self):
        self.dt = 1e-10  # [s]
        self.time = backend.array(
            (np.arange(512) - 50) * self.dt, dtype=backend.float
        )
        rng = np.random.default_rng(11)
        n_poles = 50
        is_real = np.arange(n_poles) % 4 == 1
        self.poles = backend.array(
            -rng.uniform(1e6, 5e9, n_poles)
            + 1j * np.where(is_real, 0.0, rng.uniform(1e8, 9e9, n_poles)),
            dtype=backend.complex,
        )
        self.residues = backend.array(
            rng.normal(size=n_poles)
            + 1j * np.where(is_real, 0.0, rng.normal(size=n_poles)),
            dtype=backend.complex,
        )

    def _per_pole_sum(self) -> np.ndarray:
        expected = np.zeros(len(self.time))
        for pole_i in range(len(self.poles)):
            expected += copy_to_cpu(
                triple_box_average_pole(
                    self.time,
                    self.poles[pole_i : pole_i + 1],
                    self.residues[pole_i : pole_i + 1],
                    self.dt,
                )
            )
        return expected

    def test_poles_evaluated_in_one_call(self):
        with mock.patch.object(
            bin_average,
            "triple_box_average_pole",
            wraps=bin_average.triple_box_average_pole,
        ) as single_pole:
            bin_average.triple_box_average_poles(
                self.time, self.poles, self.residues, self.dt
            )
        self.assertEqual(single_pole.call_count, 1)

    def test_matches_per_pole_sum(self):
        expected = self._per_pole_sum()
        result = copy_to_cpu(
            triple_box_average_poles(
                self.time, self.poles, self.residues, self.dt
            )
        )
        if DEV_DRAW:
            lags = copy_to_cpu(self.time) / self.dt
            _dev_draw(
                "triple_box_average_poles, short grid (all poles at once)",
                "The broadcast sum over 50 poles overlaps the per-pole "
                "loop and differs by round-off. A difference growing "
                "with t: lags and poles broadcast along the wrong axes; "
                "constant offsets: a pole batch dropped or counted twice.",
                [
                    (
                        "wake",
                        {
                            "broadcast": (lags, result),
                            "per pole": (lags, expected, "--"),
                        },
                        False,
                    ),
                    (
                        "|difference| / max",
                        {
                            "": (
                                lags,
                                (result - expected) / np.abs(expected).max(),
                            )
                        },
                        True,
                    ),
                ],
                x_label="t [bins]",
                vlines=(-1.5, 1.5),
            )
        np.testing.assert_allclose(
            result, expected, rtol=1e-13, atol=1e-13 * np.abs(expected).max()
        )

    def test_poles_batched_to_bound_memory(self):
        """Batches of poles keep the broadcast array small."""
        expected = self._per_pole_sum()
        # room for 7 poles of 512 lags per batch: 8 batches for 50 poles
        with (
            mock.patch.object(bin_average, "_MAX_BROADCAST_ELEMENTS", 7 * 512),
            mock.patch.object(
                bin_average,
                "triple_box_average_pole",
                wraps=bin_average.triple_box_average_pole,
            ) as single_pole,
        ):
            result = copy_to_cpu(
                bin_average.triple_box_average_poles(
                    self.time, self.poles, self.residues, self.dt
                )
            )
        self.assertEqual(single_pole.call_count, 8)
        np.testing.assert_allclose(
            result, expected, rtol=1e-13, atol=1e-13 * np.abs(expected).max()
        )


class TestTripleBoxAverageStaysOnTheDevice(BLonDTestCase):
    """`triple_box_average_poles` must not copy back to the host.

    `MultiPoleSparseSolve` reaches it from the per-turn tracking loop, so
    a single device-to-host transfer per call costs a PCIe round trip
    every turn of every simulation. Two things would force one: gathering
    with a boolean mask (CuPy has to count the selected elements on the
    host) and walking the poles with ``complex()``. The cost is invisible
    on a CPU backend, so without this test the regression would come back
    unnoticed and only show up as a slow GPU run.
    """

    @pytest.mark.cupy
    def test_no_host_transfer_per_call(self) -> None:
        cupy = pytest.importorskip("cupy")
        if backend.specials_mode != "cuda":
            self.skipTest("only meaningful on the cuda backend")

        transfers = []
        watched = [
            name
            for name in (
                "get",
                "item",
                "tolist",
                "__float__",
                "__int__",
                "__complex__",
                "__array__",
                "__bool__",
            )
            if hasattr(cupy.ndarray, name)
        ]
        originals = {n: getattr(cupy.ndarray, n) for n in watched}

        def _watch(name, original):
            def wrapper(self_, *args, **kwargs):
                stack = "".join(traceback.format_stack())
                if "bin_average.py" in stack:
                    transfers.append(name)
                return original(self_, *args, **kwargs)

            return wrapper

        for name, original in originals.items():
            setattr(cupy.ndarray, name, _watch(name, original))
        try:
            times = backend.array(
                np.linspace(-2.0, 10.0, 512), dtype=backend.float
            )
            # A real pole and complex ones together: the real one takes
            # the pair-factor branch that must not need a host-side `if`.
            poles = backend.array(
                np.array([-0.4 + 0.0j, -0.3 + 2.0j, -0.1 + 5.0j]),
                dtype=backend.complex,
            )
            residues = backend.array(
                np.array([2.5 + 0.0j, 1.0 + 0.5j, -3.2 + 7.1j]),
                dtype=backend.complex,
            )
            for _ in range(5):
                triple_box_average_poles(times, poles, residues, 1.0)
        finally:
            for name, original in originals.items():
                setattr(cupy.ndarray, name, original)

        self.assertEqual(
            transfers,
            [],
            msg=(
                "`triple_box_average_poles` copied back to the host "
                f"{len(transfers)} time(s) via {sorted(set(transfers))}. "
                "It runs inside the per-turn tracking loop, so every one "
                "of those is a PCIe round trip per turn. Prefer "
                "`backend.where` over boolean-mask indexing, and index "
                "poles as length-1 slices rather than calling `complex()`."
            ),
        )


# Directions of p * t in the complex plane the series / direct switches are
# probed along, and magnitudes |p t| straddling the switch at 1.
_POLE_DIRECTIONS = {
    "real": -1.0 + 0.0j,
    "Q=0.6": -0.83 + 0.55j,
    "undamped": 1.0j,
    "Q~100": -5e-3 + 1.0j,
}
_SWITCH_MAGNITUDES = (1e-3, 0.5, 0.99, 1.0 - 1e-9, 1.0 + 1e-9, 1.01, 2.0, 1e2)


class TestSeriesDirectSwitch(BLonDTestCase):
    r"""The small-argument series hand over to the direct forms at |p t| = 1.

    Each is checked against a 50-digit evaluation of the direct form,
    which does not cancel at that precision, so a truncated series or a
    direct form used too close to zero would show up here first.
    """

    @staticmethod
    def _phi_3(pole_t):
        return (mp.e**pole_t - 1 - pole_t - pole_t**2 / 2) / pole_t**3

    def _cases(self):
        for name, direction in _POLE_DIRECTIONS.items():
            for magnitude in _SWITCH_MAGNITUDES:
                pole_t = complex(magnitude * direction)
                with self.subTest(direction=name, magnitude=magnitude):
                    yield pole_t, mp.mpc(pole_t.real, pole_t.imag)

    @staticmethod
    def _draw_switch(title, factors, tolerance):
        """Relative error against |p t| on a dense grid, per direction.

        ``factors`` is one ``(label, numeric, reference)`` per panel:
        ``numeric(z)`` evaluates the function under test at ``p t = z``,
        ``reference(z_mp)`` the same in 50-digit precision.
        """
        magnitudes = np.logspace(-3.0, 2.0, 121)
        panels = []
        with mp.workdps(_REFERENCE_DPS):
            for label, numeric, reference in factors:
                curves = {}
                for name, direction in _POLE_DIRECTIONS.items():
                    error = []
                    for magnitude in magnitudes:
                        z = complex(magnitude * direction)
                        expected = complex(reference(mp.mpc(z.real, z.imag)))
                        error.append(abs(numeric(z) / expected - 1.0))
                    curves[name] = (magnitudes, error)
                curves["tolerance"] = (
                    magnitudes[[0, -1]],
                    [tolerance, tolerance],
                    "k--",
                )
                panels.append((f"{label}: |relative error|", curves, True))
        _dev_draw(
            title,
            "Round-off (~1e-16) on both sides of the switch at |p t| = 1 "
            "(dotted), for every direction of p t. A rise towards 1 from "
            "the left: the series is truncated too early. One from the "
            "right: the direct form is used where it still cancels. "
            "Undamped poles (p t imaginary) cancel worst.",
            panels,
            x_label="|p t|",
            vlines=(1.0,),
            log_x=True,
        )

    def test_third_antiderivative_factor(self):
        if DEV_DRAW:
            self._draw_switch(
                "causal_third_antiderivative_factor (phi_3)",
                [
                    (
                        "phi_3",
                        lambda z: complex(
                            copy_to_cpu(
                                bin_average.causal_third_antiderivative_factor(
                                    backend.array([1.0], dtype=backend.float),
                                    z,
                                )
                            )[0]
                        ),
                        self._phi_3,
                    )
                ],
                1e-14,
            )
        with mp.workdps(_REFERENCE_DPS):
            for pole_t, pole_t_mp in self._cases():
                expected = complex(self._phi_3(pole_t_mp))
                result = complex(
                    copy_to_cpu(
                        bin_average.causal_third_antiderivative_factor(
                            backend.array([1.0], dtype=backend.float), pole_t
                        )
                    )[0]
                )
                self.assertLess(abs(result / expected - 1.0), 1e-14)

    def test_ramp_factor_is_derivative_in_the_pole(self):
        """:math:`\\psi_3 = \\partial_p \\varphi_3`, taken numerically."""
        if DEV_DRAW:
            self._draw_switch(
                "_causal_third_antiderivative_factor_ramp (psi_3)",
                [
                    (
                        "psi_3",
                        lambda z: complex(
                            copy_to_cpu(
                                bin_average._causal_third_antiderivative_factor_ramp(
                                    backend.array([1.0], dtype=backend.float),
                                    z,
                                )
                            )[0]
                        ),
                        lambda z_mp: mp.diff(self._phi_3, z_mp),
                    )
                ],
                1e-13,
            )
        with mp.workdps(_REFERENCE_DPS):
            for pole_t, pole_t_mp in self._cases():
                expected = complex(mp.diff(self._phi_3, pole_t_mp))
                result = complex(
                    copy_to_cpu(
                        bin_average._causal_third_antiderivative_factor_ramp(
                            backend.array([1.0], dtype=backend.float), pole_t
                        )
                    )[0]
                )
                self.assertLess(abs(result / expected - 1.0), 1e-13)

    def test_expm1_ratio_and_ramp_factor(self):
        if DEV_DRAW:
            self._draw_switch(
                "_expm1_ratio_and_ramp_factor (E, G), with q = p dt",
                [
                    (
                        "E",
                        lambda q: bin_average._expm1_ratio_and_ramp_factor(q)[
                            0
                        ],
                        lambda q: (mp.e**q - 1) / q,
                    ),
                    (
                        "G",
                        lambda q: bin_average._expm1_ratio_and_ramp_factor(q)[
                            1
                        ],
                        lambda q: (mp.e**q * (q - 2) + q + 2) / (2 * q**2),
                    ),
                ],
                1e-14,
            )
        with mp.workdps(_REFERENCE_DPS):
            for pole_dt, q in self._cases():
                expm1_ratio = complex((mp.e**q - 1) / q)
                ramp_factor = complex((mp.e**q * (q - 2) + q + 2) / (2 * q**2))
                result = bin_average._expm1_ratio_and_ramp_factor(pole_dt)
                self.assertLess(abs(result[0] / expm1_ratio - 1.0), 1e-14)
                self.assertLess(abs(result[1] / ramp_factor - 1.0), 1e-14)

    def test_factors_vanish_before_the_onset(self):
        """Exactly zero for ``t <= 0``: nothing is induced ahead."""
        time = backend.array([-3.0, -1e-300, 0.0], dtype=backend.float)
        for function in (
            bin_average.causal_third_antiderivative_factor,
            bin_average._causal_third_antiderivative_factor_ramp,
        ):
            with self.subTest(function=function.__name__):
                np.testing.assert_array_equal(
                    copy_to_cpu(function(time, -0.83 + 0.55j)), 0.0
                )

    def test_zero_pole_is_t_cubed_over_six(self):
        """With ``p = 0`` the direct form is 0 / 0; its limit is used."""
        time = np.linspace(-1.0, 2.0, 31)
        result = copy_to_cpu(
            bin_average._causal_third_antiderivative_factor_any(
                backend.array(time, dtype=backend.float), 0j
            )
        )
        expected = np.where(time > 0, time**3 / 6, 0.0)
        if DEV_DRAW:
            tiny_pole = copy_to_cpu(
                bin_average._causal_third_antiderivative_factor_any(
                    backend.array(time, dtype=backend.float), -1e-6 + 1e-6j
                )
            ).real
            _dev_draw(
                "_causal_third_antiderivative_factor_any",
                "p = 0 gives t^3 / 6 for t > 0 and exactly 0 before; a "
                "tiny nonzero pole (through the general function) lands "
                "on the same curve, so the zero-pole limit is continuous. "
                "NaN (no line) means the 0 / 0 guard was bypassed.",
                [
                    (
                        "phi_3",
                        {
                            "p = 0": (time, result, "o"),
                            "t^3 / 6": (time, expected),
                            "p = 1e-6 (1 + i)": (time, tiny_pole, "--"),
                        },
                        False,
                    )
                ],
                x_label="t [s]",
                vlines=(0.0,),
            )
        np.testing.assert_array_equal(np.isfinite(result), True)
        np.testing.assert_allclose(result, expected, rtol=1e-15, atol=0.0)


class TestOnsetBranchContinuity(BLonDTestCase):
    """The near branch meets the far closed form at ``t = 1.5 * dt``.

    The two are different formulas for the same function, switched on
    ``t > 1.5 * dt``; a slip in either one would show up as a jump there.
    The jump is measured against the largest tap: at the onset itself the
    wake is down to ``1 / |p dt|^3`` of it and the third difference keeps
    absolute, not relative, accuracy.
    """

    _JUMP_HINT = (
        "Jump between the two branches at t = 1.5 dt, over the largest "
        "tap, per pole direction: round-off, under the tolerance (dashed), "
        "at every |p dt|. A jump growing on one side points at the "
        "branch that loses digits there; one that is O(1) is a wrong "
        "formula, whatever |p dt|."
    )

    @staticmethod
    def _jump_panel(jumps, tolerance):
        curves = {
            name: (np.array(x), np.array(y), "o-")
            for name, (x, y) in jumps.items()
        }
        curves["tolerance"] = (
            np.array([1e-3, 1e3]),
            np.array([tolerance, tolerance]),
            "k--",
        )
        return ("|jump| / peak", curves, True)

    @staticmethod
    def _draw_single_pole_branches(pole_dt, direction, dt):
        """Both formulas of one pole over a range around the onset."""
        pole = pole_dt * direction / dt
        residue = 1.3 - 0.4j
        lags = np.linspace(-2.0, 8.0, 401)
        time = backend.array(lags * dt, dtype=backend.float)
        near = copy_to_cpu(bin_average._near_branch(time, pole, residue, dt))
        far = copy_to_cpu(
            (residue * bin_average._smoothed_pole(time, pole, dt)).real
        )
        used = copy_to_cpu(triple_box_average_pole(time, pole, residue, dt))
        _dev_draw(
            f"_near_branch vs _smoothed_pole, p dt = {pole * dt:.3g}",
            "Both are the same function past t = 1.5 dt; the solid line "
            "is what triple_box_average_pole uses (near before the onset, "
            "far after, both halved by the pair factor here). The near "
            "formula is valid everywhere but loses digits far out, which "
            "is why the far one takes over: their difference should be "
            "round-off near the onset and only grow beyond it. The far "
            "formula is drawn past the onset only: before it, it is "
            "meaningless and grows.",
            [
                (
                    "wake",
                    {
                        "used": (lags, used / 2.0),
                        "near (_near_branch)": (lags, near, "--"),
                        "far (_smoothed_pole * residue)": (
                            lags[lags > 1.5],
                            far[lags > 1.5],
                            ":",
                        ),
                    },
                    False,
                ),
                (
                    "|near - far| / max",
                    {
                        "": (
                            lags[lags > 1.5],
                            ((near - far) / np.abs(near).max())[lags > 1.5],
                        )
                    },
                    True,
                ),
            ],
            x_label="t [bins]",
            vlines=(-1.5, 1.5),
        )

    def test_single_pole(self):
        dt = 1e-9  # [s]
        onset = backend.array([1.5 * dt], dtype=backend.float)
        taps = backend.array([-1.0, 0.0, 1.0], dtype=backend.float) * dt
        jumps = {name: ([], []) for name in _POLE_DIRECTIONS}
        if DEV_DRAW:
            for pole_dt in (0.1, 3.0, 100.0):
                self._draw_single_pole_branches(
                    pole_dt, _POLE_DIRECTIONS["Q=0.6"], dt
                )
        for name, direction in _POLE_DIRECTIONS.items():
            for pole_dt in (10.0 ** np.arange(-3.0, 3.5, 0.5)).tolist():
                pole = pole_dt * direction / dt
                residue = 1.3 - 0.4j if name != "real" else 1.3 + 0.0j
                with self.subTest(pole=name, pole_dt=pole_dt):
                    near = copy_to_cpu(
                        bin_average._near_branch(onset, pole, residue, dt)
                    )
                    far = copy_to_cpu(
                        (
                            residue
                            * bin_average._smoothed_pole(onset, pole, dt)
                        ).real
                    )
                    peak = np.max(
                        np.abs(
                            copy_to_cpu(
                                bin_average._near_branch(
                                    taps, pole, residue, dt
                                )
                            )
                        )
                    )
                    jumps[name][0].append(pole_dt)
                    jumps[name][1].append(abs(near[0] - far[0]) / peak)
                    self.assertLess(abs(near[0] - far[0]), 1e-13 * peak)
        if DEV_DRAW:
            _dev_draw(
                "_near_branch vs _smoothed_pole at the onset",
                self._JUMP_HINT,
                [self._jump_panel(jumps, 1e-13)],
                x_label="|p dt|",
                vlines=(1.0,),
                log_x=True,
            )

    def test_linear_pole(self):
        """The far value just past the onset, against the near branch.

        Its two branches are not exposed separately, so the near branch is
        extrapolated linearly from two lags at and just before the onset;
        the wake is C^2 there, which leaves an O(hair^2) remainder.
        """
        dt = 1e-9  # [s]
        jumps = {name: ([], []) for name in _POLE_DIRECTIONS}
        for name, direction in _POLE_DIRECTIONS.items():
            for pole_dt in (10.0 ** np.arange(-3.0, 3.5, 0.5)).tolist():
                pole = pole_dt * direction / dt
                residue = 1.3 - 0.4j if name != "real" else 1.3 + 0.0j
                hair = 1e-6 * dt / max(1.0, pole_dt)
                onset = 1.5 * dt
                # The taps that set the scale, then the near branch at
                # onset - hair and onset, and the far one at onset + hair.
                time = backend.array(
                    [-dt, 0.0, dt, onset - hair, onset, onset + hair],
                    dtype=backend.float,
                )
                with self.subTest(pole=name, pole_dt=pole_dt):
                    wake = copy_to_cpu(
                        triple_box_average_linear_pole(
                            time, pole, residue, 0.3 * residue / dt, dt
                        )
                    )
                    extrapolated = 2.0 * wake[-2] - wake[-3]
                    jumps[name][0].append(pole_dt)
                    jumps[name][1].append(
                        abs(wake[-1] - extrapolated) / np.max(np.abs(wake))
                    )
                    self.assertLess(
                        abs(wake[-1] - extrapolated),
                        1e-11 * np.max(np.abs(wake)),
                    )
        if DEV_DRAW:
            _dev_draw(
                "triple_box_average_linear_pole at the onset",
                self._JUMP_HINT
                + " Here the near branch is extrapolated linearly across "
                "the onset, so an O(hair^2) remainder sits under the "
                "round-off.",
                [self._jump_panel(jumps, 1e-11)],
                x_label="|p dt|",
                vlines=(1.0,),
                log_x=True,
            )


class TestTripleBoxAverageSpectrum(BLonDTestCase):
    r"""The sampled bin-averaged wake has the aliased, sinc^3 spectrum.

    By Poisson summation, the transform of the wake taps,
    :math:`\Delta t \sum_k \overline W(k \Delta t) e^{-i\omega k \Delta t}`,
    is :math:`\sum_j Z(\omega_j) \mathrm{sinc}^3(\omega_j \Delta t / 2)`
    with :math:`\omega_j = \omega + 2\pi j / \Delta t`. That is the claim
    the module makes -- every alias image damped by the B-spline's
    :math:`\mathrm{sinc}^3` -- checked against the impedance
    :math:`Z(\omega) = \int W e^{-i\omega t} dt` of the point wake, at
    baseband frequencies up to the Nyquist.
    """

    # Images summed on each side; the omitted tail falls as 1 / j^4.
    _IMAGES = 20_000

    def _assert_spectrum(self, pole, residue, dt):
        n_bins = int(40 / (-pole.real * dt)) + 10
        lags = np.arange(-2, n_bins) * dt
        wake = copy_to_cpu(
            triple_box_average_poles(
                backend.array(lags, dtype=backend.float),
                backend.array([pole], dtype=backend.complex),
                backend.array([residue], dtype=backend.complex),
                dt,
            )
        )
        images = np.arange(-self._IMAGES, self._IMAGES + 1)
        if DEV_DRAW:
            self._draw_spectrum(pole, residue, dt, lags, wake, images)
        for fraction in (0.0, 0.1, 0.25, 0.45):
            omega = 2.0 * np.pi * fraction / dt
            with self.subTest(fraction_of_sampling_frequency=fraction):
                transform = np.sum(wake * np.exp(-1j * omega * lags)) * dt
                omega_j = omega + 2.0 * np.pi * images / dt
                expected = np.sum(
                    self._impedance(pole, residue, omega_j)
                    * np.sinc(omega_j * dt / (2.0 * np.pi)) ** 3
                )
                # Round-off scale of the sum over taps: its L1 norm.
                scale = np.sum(np.abs(wake)) * dt
                self.assertLess(abs(transform - expected), 1e-13 * scale)

    @staticmethod
    def _impedance(pole, residue, omega):
        """``Z(omega)`` of the point wake, with the conjugate partner."""
        impedance = -residue / (pole - 1j * omega)
        if pole.imag != 0:
            impedance += -np.conj(residue) / (np.conj(pole) - 1j * omega)
        return impedance

    def _draw_spectrum(self, pole, residue, dt, lags, wake, images):
        # From just above 0: a resonator has Z(0) = 0, which squashes the
        # log axis.
        fractions = np.linspace(0.005, 0.5, 100)
        omegas = 2.0 * np.pi * fractions / dt
        averaged, poisson, baseband, point = [], [], [], []
        pair_factor = 2.0 if pole.imag != 0 else 1.0
        # The point-sampled taps, half weight at t = 0 where W jumps.
        point_wake = np.where(
            lags > 0, pair_factor * (residue * np.exp(pole * lags)).real, 0.0
        )
        point_wake[lags == 0] = 0.5 * pair_factor * residue.real
        for omega in omegas:
            phase = np.exp(-1j * omega * lags)
            averaged.append(abs(np.sum(wake * phase) * dt))
            point.append(abs(np.sum(point_wake * phase) * dt))
            omega_j = omega + 2.0 * np.pi * images / dt
            poisson.append(
                abs(
                    np.sum(
                        self._impedance(pole, residue, omega_j)
                        * np.sinc(omega_j * dt / (2.0 * np.pi)) ** 3
                    )
                )
            )
            baseband.append(
                abs(
                    self._impedance(pole, residue, omega)
                    * np.sinc(omega * dt / (2.0 * np.pi)) ** 3
                )
            )
        error = np.abs(np.array(averaged) - np.array(poisson)) / (
            np.sum(np.abs(wake)) * dt
        )
        _dev_draw(
            f"triple_box_average_poles: spectrum, p dt = {pole * dt:.3g}",
            "The transform of the bin-averaged taps (solid) sits on the "
            "Poisson sum over every sinc^3-weighted alias image (circles), "
            "with an error at round-off over the whole baseband (a "
            "magnitude check: the test also compares phases). Its "
            "distance from Z sinc^3 without images (dashed) is the "
            "aliasing the B-spline leaves; the point-sampled taps "
            "(dotted) show the aliasing without it -- the low-Q bug.",
            [
                (
                    "|spectrum|",
                    {
                        "bin-averaged taps": (fractions, averaged),
                        "Poisson sum": (fractions[::5], poisson[::5], "o"),
                        "Z sinc^3, no images": (fractions, baseband, "--"),
                        "point-sampled taps": (fractions, point, ":"),
                    },
                    True,
                ),
                (
                    "|error| / (dt sum |taps|)",
                    {"": (fractions, error)},
                    True,
                ),
            ],
            x_label="f dt (0.5 = Nyquist)",
        )

    @staticmethod
    def _resonator(center_frequency, quality_factor):
        omega = 2.0 * np.pi * center_frequency  # [rad/s]
        alpha = omega / (2.0 * quality_factor)  # [1/s]
        omega_bar = np.sqrt(omega**2 - alpha**2)  # [rad/s]
        pole = complex(-alpha, omega_bar)
        return pole, 1e6 * alpha * (1.0 + 1j * alpha / omega_bar)

    def test_broadband_resonator_above_the_nyquist(self):
        """The low-Q case the bin average exists for (a short grid)."""
        dt = 1e-10  # [s]
        self._assert_spectrum(*self._resonator(3.0 / dt, 1.0), dt)

    def test_resolved_narrow_resonator(self):
        """Over ten thousand bins, so the windowed near branch is used."""
        dt = 1e-10  # [s]
        self._assert_spectrum(*self._resonator(0.05 / dt, 50.0), dt)

    def test_real_pole(self):
        self._assert_spectrum(complex(-3e9, 0.0), 2.0 + 0.0j, 1e-10)


class TestTripleBoxAverageHalfBinGrid(BLonDTestCase):
    """The window and the branch switch agree on lags of exactly 1.5 bins."""

    def test_windowed_matches_full_array_on_half_bin_lags(self):
        dt = 1e-10  # [s]
        # Lags step by half a bin, so +-1.5 * dt are grid points exactly.
        time = backend.array(
            (np.arange(2 * bin_average._NEAR_WINDOW_MIN_BINS) - 40) * 0.5 * dt,
            dtype=backend.float,
        )
        poles = backend.array(
            [-2e8 + 4e9j, -3e9 + 0.0j], dtype=backend.complex
        )
        residues = backend.array([1e4 - 3e3j, 2.5e3], dtype=backend.complex)
        expected = np.zeros(len(time))
        for pole_i in range(len(poles)):
            expected += copy_to_cpu(
                triple_box_average_pole(
                    time,
                    poles[pole_i : pole_i + 1],
                    residues[pole_i : pole_i + 1],
                    dt,
                )
            )
        result = copy_to_cpu(
            triple_box_average_poles(time, poles, residues, dt)
        )
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=0.0)
        # t <= -1.5 * dt is the first 38 lags: nothing induced there.
        np.testing.assert_array_equal(result[:38], 0.0)


if __name__ == "__main__":
    unittest.main()
