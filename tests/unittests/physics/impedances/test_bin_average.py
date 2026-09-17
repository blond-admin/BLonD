# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Tests for :mod:`blond.physics.impedances.bin_average`.

Notes
-----
Authors:
Simon Lauber
"""

import unittest

import mpmath as mp
import numpy as np

from blond.core.backends.backend import backend
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.impedances.bin_average import (
    triple_box_average_pole,
    triple_box_average_poles,
)

# Quadrature points used by the reference convolution, per sub-interval
# between two breakpoints of the integrand.
_QUADRATURE_POINTS = 20001


def quadratic_bspline(x: float) -> float:
    """
    Quadratic B-spline ``box * box * box`` of unit width, at ``x``.

    Parameters
    ----------
    x
        Position in units of the bin width.

    Returns
    -------
    weight
        The B-spline, unit integral, support ``(-3/2, 3/2)``.
    """
    abs_x = abs(x)
    if abs_x >= 1.5:
        return 0.0
    if abs_x <= 0.5:
        return 0.75 - x * x
    return 0.5 * (1.5 - abs_x) ** 2


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


class TestTripleBoxAveragePole(unittest.TestCase):
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


class TestTripleBoxAverageContinuumLimit(unittest.TestCase):
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


class TestTripleBoxAverageLossFactor(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
