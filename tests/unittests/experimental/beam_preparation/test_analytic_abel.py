"""Tests for the Abel-transform distribution reconstruction."""

import io
import unittest
from contextlib import redirect_stdout
from math import gamma, pi, sqrt

import numpy as np
import pytest

from blond.core.backends.backend import backend
from blond.experimental.beam_preparation.analytic_abel import (
    distribution_from_line_density,
)
from blond.experimental.beam_preparation.analytic_hamiltonian import (
    calc_eom_factor_dE,
    hamiltonian_grid,
)
from blond.experimental.beam_preparation.analytic_potential_well import (
    bucket_time_array,
    rf_potential_well,
)
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.testing.backend_testing import multi_backend_testcase

# LHC-like kinetic factor (450 GeV protons).
ETA_0 = 3.172867586042721e-04
BETA = 0.9999978262922387
TOTAL_ENERGY = 450.00104432e9
EOM_FACTOR_DE = calc_eom_factor_dE(ETA_0, BETA, TOTAL_ENERGY)

# Harmonic (linear-regime) well: V = b * t^2, 1 eV at the frame edge.
CURVATURE = 1.0e18  # eV/s^2
HALF_SPAN = 1.0e-9  # s


def _harmonic_well(n_time=2001):
    time_array = backend.linspace(
        -HALF_SPAN, HALF_SPAN, n_time, dtype=backend.float
    )
    return time_array, CURVATURE * time_array**2


def _binomial_projection(potential_well, h_0, mu):
    """Exact line density of a unit-amplitude binomial F(H).

    Parameters
    ----------
    potential_well : NumpyArray | CupyArray
        Potential well in eV.
    h_0 : float
        Hamiltonian at the edge of the distribution support, in eV.
    mu : float
        Binomial exponent.

    Returns
    -------
    NumpyArray | CupyArray
        The projected line density.
    """
    half_integral = sqrt(pi) * gamma(mu + 1.0) / gamma(mu + 1.5)
    support = backend.maximum(1.0 - potential_well / h_0, 0.0)
    return support ** (mu + 0.5) * sqrt(h_0 / EOM_FACTOR_DE) * half_integral


class TestBinomialRoundTrip(unittest.TestCase):
    def test_harmonic_well(self):
        # A binomial F(H) projects to a (mu + 1/2)-binomial line
        # density; the Abel inversion must recover F(H) in shape AND
        # absolute scale.
        time_array, well = _harmonic_well()
        h_0 = 0.4
        for mu in (0.0, 0.5, 1.0, 2.0):
            for half_option in ("first", "second", "both"):
                with self.subTest(mu=mu, half_option=half_option):
                    line_density_values = _binomial_projection(well, h_0, mu)
                    hamiltonian_coord, distribution_values = (
                        distribution_from_line_density(
                            time_array,
                            line_density_values,
                            well,
                            eom_factor_dE=EOM_FACTOR_DE,
                            half_option=half_option,
                        )
                    )
                    expected = (
                        backend.maximum(1.0 - hamiltonian_coord / h_0, 0.0)
                        ** mu
                    )
                    # Compare away from the support edge, where the
                    # finite grid smears the (1 - H/H_0)^mu cusp (or,
                    # for the waterbag, its discontinuity).
                    inside = hamiltonian_coord <= 0.9 * h_0
                    np.testing.assert_allclose(
                        copy_to_cpu(distribution_values[inside]),
                        copy_to_cpu(expected[inside]),
                        atol=0.02,
                        rtol=0.03,
                    )
                    # Outside the support it must vanish.
                    outside = hamiltonian_coord >= 1.1 * h_0
                    self.assertTrue(
                        backend.all(
                            backend.abs(distribution_values[outside]) < 0.02
                        )
                    )

    @multi_backend_testcase
    @pytest.mark.backend_mutation
    def test_rf_well(self):
        # Same anchor in a realistic sinusoidal RF well: the projection
        # identity holds for a general well, not just the linear regime.
        omega_rf = 2.0 * np.pi * 400.789e6
        t_rev = 88.9e-6
        time_array = bucket_time_array(omega_rf, n_points=2001)
        voltage = 6e6 * backend.sin(omega_rf * time_array)
        well = rf_potential_well(
            time_array, voltage, charge=1.0, t_rev=t_rev, eta_0=ETA_0
        )

        mu = 1.0
        h_0 = 0.5 * float(well.max())
        line_density_values = _binomial_projection(well, h_0, mu)

        hamiltonian_coord, distribution_values = (
            distribution_from_line_density(
                time_array,
                line_density_values,
                well,
                eom_factor_dE=EOM_FACTOR_DE,
                half_option="both",
            )
        )

        expected = backend.maximum(1.0 - hamiltonian_coord / h_0, 0.0) ** mu
        inside = hamiltonian_coord <= 0.9 * h_0
        np.testing.assert_allclose(
            copy_to_cpu(distribution_values[inside]),
            copy_to_cpu(expected[inside]),
            atol=0.02,
            rtol=0.03,
        )

    def test_gaussian(self):
        # lambda ~ exp(-V/H_bar) inverts to F ~ exp(-H/H_bar) in any
        # well.
        time_array, well = _harmonic_well()
        h_bar = 0.2
        line_density_values = backend.exp(-well / h_bar) * sqrt(
            pi * h_bar / EOM_FACTOR_DE
        )

        hamiltonian_coord, distribution_values = (
            distribution_from_line_density(
                time_array,
                line_density_values,
                well,
                eom_factor_dE=EOM_FACTOR_DE,
                half_option="first",
            )
        )

        expected = backend.exp(-hamiltonian_coord / h_bar)
        # The frame truncates the gaussian tails: compare where the
        # input line density is not dominated by the truncation.
        inside = hamiltonian_coord <= 3.0 * h_bar
        np.testing.assert_allclose(
            copy_to_cpu(distribution_values[inside]),
            copy_to_cpu(expected[inside]),
            atol=0.02,
            rtol=0.03,
        )

    def test_n_points_abel_resampling(self):
        # A coarse measured-like profile refined via n_points_abel must
        # stay close to the analytic distribution.
        time_array, well = _harmonic_well(n_time=201)
        h_0 = 0.4
        line_density_values = _binomial_projection(well, h_0, 1.0)

        hamiltonian_coord, distribution_values = (
            distribution_from_line_density(
                time_array,
                line_density_values,
                well,
                eom_factor_dE=EOM_FACTOR_DE,
                half_option="both",
                n_points_abel=5000,
            )
        )

        expected = backend.maximum(1.0 - hamiltonian_coord / h_0, 0.0)
        inside = hamiltonian_coord <= 0.9 * h_0
        np.testing.assert_allclose(
            copy_to_cpu(distribution_values[inside]),
            copy_to_cpu(expected[inside]),
            atol=0.03,
            rtol=0.05,
        )


class TestHalfOptions(unittest.TestCase):
    def test_half_options_agree_for_symmetric_input(self):
        time_array, well = _harmonic_well()
        line_density_values = _binomial_projection(well, 0.4, 1.0)

        results = {
            half_option: distribution_from_line_density(
                time_array,
                line_density_values,
                well,
                eom_factor_dE=EOM_FACTOR_DE,
                half_option=half_option,
            )
            for half_option in ("first", "second", "both")
        }

        h_first, f_first = results["first"]
        for half_option in ("second", "both"):
            with self.subTest(half_option=half_option):
                h_other, f_other = results[half_option]
                np.testing.assert_allclose(
                    copy_to_cpu(f_first),
                    copy_to_cpu(backend.interp(h_first, h_other, f_other)),
                    atol=1e-3,
                )

    def test_both_is_average_of_first_and_second(self):
        # Asymmetric input (gaussian bunch in a tilted well): "both"
        # must be the average of the two single-branch reconstructions.
        time_array, well = _harmonic_well()
        well = well * (1.0 + 0.3 * time_array / HALF_SPAN)
        well -= well.min()
        sigma = 0.25e-9
        minimum_time = time_array[backend.argmin(well)]
        line_density_values = backend.exp(
            -((time_array - minimum_time) ** 2) / (2.0 * sigma**2)
        )

        common_kwargs = dict(eom_factor_dE=EOM_FACTOR_DE)
        h_first, f_first = distribution_from_line_density(
            time_array,
            line_density_values,
            well,
            half_option="first",
            **common_kwargs,
        )
        h_second, f_second = distribution_from_line_density(
            time_array,
            line_density_values,
            well,
            half_option="second",
            **common_kwargs,
        )
        h_both, f_both = distribution_from_line_density(
            time_array,
            line_density_values,
            well,
            half_option="both",
            **common_kwargs,
        )

        # The two branches genuinely disagree for this input...
        scale = float(f_first.max())
        self.assertFalse(
            backend.all(
                backend.isclose(
                    f_first,
                    backend.interp(h_first, h_second, f_second),
                    atol=0.01 * scale,
                    rtol=0.0,
                )
            )
        )
        # ...and "both" is their pointwise average on the first grid.
        expected = (
            f_first + backend.interp(h_first, h_second, f_second)
        ) / 2.0
        expected[expected < 0.0] = 0.0
        np.testing.assert_allclose(
            copy_to_cpu(f_both),
            copy_to_cpu(backend.interp(h_both, h_first, expected)),
            atol=1e-6 * scale,
            rtol=0.0,
        )

    def test_duplicated_minimum_sample(self):
        # A symmetric well on an even grid has two equal minimum
        # samples: the branch split must not divide by the duplicated
        # value — a regression that returned inf (and, once sanitized,
        # F(0) = 0) on the second branch.
        n_time = 2000  # even: minimum falls between two equal samples

        # Construct one half grid then invert it to ensure bit-wise
        # identical values.
        half_grid = backend.linspace(
            HALF_SPAN / (n_time - 1),
            HALF_SPAN,
            n_time // 2,
            dtype=backend.float,
        )
        time_array = backend.concatenate((-half_grid[::-1], half_grid))
        well = CURVATURE * time_array**2
        self.assertEqual(
            float(well[n_time // 2 - 1]), float(well[n_time // 2])
        )

        h_0 = 0.4
        line_density_values = _binomial_projection(well, h_0, 1.0)

        for half_option in ("first", "second", "both"):
            with self.subTest(half_option=half_option):
                hamiltonian_coord, distribution_values = (
                    distribution_from_line_density(
                        time_array,
                        line_density_values,
                        well,
                        eom_factor_dE=EOM_FACTOR_DE,
                        half_option=half_option,
                    )
                )
                self.assertTrue(
                    backend.all(backend.isfinite(distribution_values))
                )
                expected = backend.maximum(1.0 - hamiltonian_coord / h_0, 0.0)
                inside = hamiltonian_coord <= 0.9 * h_0
                np.testing.assert_allclose(
                    copy_to_cpu(distribution_values[inside]),
                    copy_to_cpu(expected[inside]),
                    atol=0.02,
                    rtol=0.03,
                )


class TestLineDensityClosure(unittest.TestCase):
    def test_projection_recovers_input_line_density(self):
        # Full closure: reconstruct F(H), project it back on a 2D grid
        # and recover the input line density (the matcher's guarantee).
        time_array, well = _harmonic_well()
        line_density_values = _binomial_projection(well, 0.4, 1.0)

        hamiltonian_coord, distribution_values = (
            distribution_from_line_density(
                time_array,
                line_density_values,
                well,
                eom_factor_dE=EOM_FACTOR_DE,
                half_option="both",
            )
        )

        _, _, hamilton_2d = hamiltonian_grid(
            time_array,
            well,
            eom_factor_dE=EOM_FACTOR_DE,
            n_points_deltaE=1001,
            energy_range=(
                -sqrt(float(well.max()) / EOM_FACTOR_DE),
                sqrt(float(well.max()) / EOM_FACTOR_DE),
            ),
        )
        density_grid = backend.interp(
            hamilton_2d, hamiltonian_coord, distribution_values
        )
        reconstructed = density_grid.sum(axis=0)

        normalized_input = line_density_values / backend.sum(
            line_density_values
        )
        normalized_reconstructed = reconstructed / backend.sum(reconstructed)
        np.testing.assert_allclose(
            copy_to_cpu(normalized_reconstructed),
            copy_to_cpu(normalized_input),
            atol=0.02 * float(normalized_input.max()),
        )


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.time_array, self.well = _harmonic_well(n_time=101)
        self.line_density_values = _binomial_projection(self.well, 0.4, 1.0)

    def test_unknown_half_option_raises(self):
        with self.assertRaisesRegex(ValueError, "half_option"):
            distribution_from_line_density(
                self.time_array,
                self.line_density_values,
                self.well,
                eom_factor_dE=EOM_FACTOR_DE,
                half_option="not_a_half",
            )

    def test_shape_mismatch_raises(self):
        with self.assertRaisesRegex(AssertionError, "shape"):
            distribution_from_line_density(
                self.time_array,
                self.line_density_values[:-1],
                self.well,
                eom_factor_dE=EOM_FACTOR_DE,
            )

    def test_monotonic_well_raises(self):
        # A monotonic well has its minimum on the frame edge: no
        # centred bunch to invert.
        with self.assertRaisesRegex(ValueError, "frame edge"):
            distribution_from_line_density(
                self.time_array,
                self.line_density_values,
                backend.linspace(
                    0.0, 1.0, len(self.time_array), dtype=backend.float
                ),
                eom_factor_dE=EOM_FACTOR_DE,
            )


class TestVerboseAndPlot(unittest.TestCase):
    def test_verbose_and_plot_smoke(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        time_array, well = _harmonic_well(n_time=201)
        line_density_values = _binomial_projection(well, 0.4, 1.0)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            distribution_from_line_density(
                time_array,
                line_density_values,
                well,
                eom_factor_dE=EOM_FACTOR_DE,
                verbose=True,
                plot=True,
            )
        self.assertIn("distribution_from_line_density", stdout.getvalue())
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
