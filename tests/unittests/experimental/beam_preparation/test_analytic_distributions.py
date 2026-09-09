"""Tests for the analytic distribution families and targeting."""

import unittest
import warnings

import numpy as np
import pytest

from blond.core.backends.backend import backend
from blond.experimental.beam_preparation.analytic_distributions import (
    DISTRIBUTION_EXPONENTS,
    _bunch_length_fwhm,
    distribution_function,
    line_density,
    x0_from_bunch_length,
)
from blond.experimental.beam_preparation.analytic_hamiltonian import (
    calc_eom_factor_dE,
    hamiltonian_grid,
)
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.testing.backend_testing import multi_backend_testcase

# LHC-like kinetic factor (450 GeV protons).
ETA_0 = 3.172867586042721e-04
BETA = 0.9999978262922387
TOTAL_ENERGY = 450.00104432e9
EOM_FACTOR_DE = calc_eom_factor_dE(ETA_0, BETA, TOTAL_ENERGY)

# Harmonic (linear-regime) well: V = b * t^2.
CURVATURE = 1.0e18  # eV/s^2


def _harmonic_grid(n_time=1001, n_deltaE=801, half_span=1.0e-9):
    time_array = backend.linspace(
        -half_span, half_span, n_time, dtype=backend.float
    )
    well = CURVATURE * time_array**2
    _, _, hamilton = hamiltonian_grid(
        time_array,
        well,
        eom_factor_dE=EOM_FACTOR_DE,
        n_points_deltaE=n_deltaE,
    )
    return time_array, well, hamilton


class TestDistributionFunction(unittest.TestCase):
    def test_named_types_match_binomial_exponents(self):
        x_array = backend.linspace(0.0, 2.0, 100, dtype=backend.float)
        for name, exponent in DISTRIBUTION_EXPONENTS.items():
            with self.subTest(distribution_type=name):
                np.testing.assert_array_equal(
                    copy_to_cpu(distribution_function(x_array, name, 1.3)),
                    copy_to_cpu(
                        distribution_function(
                            x_array, "binomial", 1.3, exponent
                        )
                    ),
                )

    def test_gaussian_form(self):
        x_array = backend.array([0.0, 0.4, 1.0, 1.6], dtype=backend.float)
        np.testing.assert_allclose(
            copy_to_cpu(distribution_function(x_array, "gaussian", 0.8)),
            copy_to_cpu(backend.exp(-2.0 * x_array / 0.8)),
        )

    def test_binomial_form(self):
        x_array = backend.array([0.0, 0.4, 1.0, 1.6], dtype=backend.float)
        x_host = copy_to_cpu(x_array)
        np.testing.assert_allclose(
            copy_to_cpu(distribution_function(x_array, "binomial", 1.0, 2.0)),
            np.where(x_host <= 1.0, (1.0 - np.minimum(x_host, 1.0)) ** 2, 0.0),
        )

    def test_binomial_without_exponent_raises(self):
        x_array = backend.linspace(0.0, 1.0, 10, dtype=backend.float)
        with self.assertRaisesRegex(ValueError, "binomial"):
            distribution_function(x_array, "binomial", 1.0)

    def test_unknown_type_raises(self):
        x_array = backend.linspace(0.0, 1.0, 10, dtype=backend.float)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            distribution_function(x_array, "not_a_type", 1.0)

    def test_redundant_exponent_warns(self):
        x_array = backend.linspace(0.0, 1.0, 10, dtype=backend.float)
        with self.assertWarnsRegex(UserWarning, "ignored"):
            distribution_function(x_array, "waterbag", 1.0, exponent=2.0)
        with self.assertWarnsRegex(UserWarning, "ignored"):
            line_density(x_array, "gaussian", 1.0, exponent=2.0)

    @multi_backend_testcase
    @pytest.mark.backend_mutation
    def test_inf_grid_evaluates_to_zero_without_warnings(self):
        # action_grid marks outside-bucket points with inf; all families
        # must map them to 0 with no RuntimeWarning.
        x_array = backend.array([0.2, 0.9, np.inf], dtype=backend.float)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            binomial = distribution_function(x_array, "binomial", 1.0, 0.5)
            gaussian = distribution_function(x_array, "gaussian", 1.0)
        self.assertEqual(float(binomial[-1]), 0.0)
        self.assertEqual(float(gaussian[-1]), 0.0)
        self.assertGreater(float(binomial[0]), 0.0)

    def test_projection_gives_plus_half_exponent(self):
        # Project a binomial phase-space density over dE in a harmonic
        # well: lambda(t) must equal the line_density family with the
        # same inputs (which bakes in the mu + 1/2 exponent shift).
        # Tolerances measured at n_deltaE=2001: the smooth families
        # verify the convention to ~1e-6-1e-8; the waterbag (mu=0,
        # discontinuous support) is dominated by dE-grid staircase
        # quantization, converging as 1/n.
        time_array, _, hamilton = _harmonic_grid(n_deltaE=2001)
        hamiltonian_0 = CURVATURE * (0.5e-9) ** 2  # support: |t|<=0.5 ns
        for distribution_type, exponent, rtol in [
            ("parabolic_amplitude", None, 1e-4),
            ("waterbag", None, 2e-2),
            ("binomial", 1.7, 1e-6),
        ]:
            with self.subTest(distribution_type=distribution_type):
                density = distribution_function(
                    hamilton, distribution_type, hamiltonian_0, exponent
                )
                projected = density.sum(axis=0)
                projected /= projected.max()
                bunch_length = 2.0 * np.sqrt(hamiltonian_0 / CURVATURE)
                expected = line_density(
                    time_array,
                    distribution_type,
                    bunch_length,
                    0.0,
                    exponent,
                )
                expected /= expected.max()
                significant = expected > 0.05
                np.testing.assert_allclose(
                    copy_to_cpu(projected[significant]),
                    copy_to_cpu(expected[significant]),
                    rtol=rtol,
                )


class TestLineDensity(unittest.TestCase):
    def setUp(self):
        self.time_array = backend.linspace(
            -1.0, 1.0, 2001, dtype=backend.float
        )

    def test_parabolic_line_is_exact_parabola(self):
        # Total exponent 0.5 + 0.5 = 1 -> exact parabola.
        parabola = line_density(self.time_array, "parabolic_line", 2.0)
        np.testing.assert_allclose(
            copy_to_cpu(parabola),
            copy_to_cpu(1.0 - self.time_array**2),
            atol=1e-12,
        )

    def test_cosine_squared_with_support_tau(self):
        cosine = line_density(self.time_array, "cosine_squared", 2.0)
        np.testing.assert_allclose(
            copy_to_cpu(cosine),
            copy_to_cpu(backend.cos(0.5 * np.pi * self.time_array) ** 2),
            atol=1e-12,
        )

    def test_gaussian_sigma_is_tau_over_four(self):
        gaussian = line_density(self.time_array, "gaussian", 2.0)
        np.testing.assert_allclose(
            copy_to_cpu(gaussian),
            copy_to_cpu(backend.exp(-(self.time_array**2) / (2 * 0.5**2))),
            atol=1e-12,
        )


class TestBunchLengthFwhm(unittest.TestCase):
    def test_fwhm_helper_on_gaussian(self):
        time_array = backend.linspace(-1.0, 1.0, 4001, dtype=backend.float)
        sigma = 0.1
        gaussian = backend.exp(-(time_array**2) / (2 * sigma**2))
        np.testing.assert_allclose(
            _bunch_length_fwhm(time_array, gaussian),
            4.0 * sigma,
            rtol=1e-3,
        )


class TestX0FromBunchLength(unittest.TestCase):
    def setUp(self):
        self.time_array, _, self.hamilton = _harmonic_grid()

    def test_full_fit(self):
        target = 1.0e-9
        x_0 = x0_from_bunch_length(
            self.time_array,
            self.hamilton,
            target_bunch_length=target,
            distribution_type="waterbag",
            bunch_length_fit="full",
        )
        # Contour extent 2*sqrt(X0/b) = tau -> X0 = b*(tau/2)^2.
        np.testing.assert_allclose(
            x_0, CURVATURE * (target / 2.0) ** 2, rtol=2e-2
        )

    def test_rms_fit_for_waterbag_and_gaussian(self):
        target = 1.0e-9
        # Waterbag: semicircle line density, sigma = t_max/2, so
        # X0 = b*tau^2/4. Gaussian: sigma_t = sqrt(X0/(4b)) -> same X0.
        for distribution_type in ("waterbag", "gaussian"):
            with self.subTest(distribution_type=distribution_type):
                x_0 = x0_from_bunch_length(
                    self.time_array,
                    self.hamilton,
                    target_bunch_length=target,
                    distribution_type=distribution_type,
                    bunch_length_fit="rms",
                )
                np.testing.assert_allclose(
                    x_0, CURVATURE * target**2 / 4.0, rtol=2e-2
                )

    def test_fwhm_fit_for_gaussian(self):
        target = 1.0e-9
        x_0 = x0_from_bunch_length(
            self.time_array,
            self.hamilton,
            target_bunch_length=target,
            distribution_type="gaussian",
            bunch_length_fit="fwhm",
        )
        # For a true gaussian the fwhm measure returns exactly 4 sigma.
        np.testing.assert_allclose(x_0, CURVATURE * target**2 / 4.0, rtol=2e-2)

    def test_bucket_too_small_warns(self):
        with self.assertWarnsRegex(UserWarning, "too small for the requested"):
            x0_from_bunch_length(
                self.time_array,
                self.hamilton,
                target_bunch_length=50.0e-9,  # far beyond the frame
                distribution_type="waterbag",
                bunch_length_fit="full",
            )

    def test_target_below_resolution_converges_trivially(self):
        # A target below one time bin is within the convergence
        # tolerance of a zero-extent density: the fit returns quickly,
        # no warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            x_0 = x0_from_bunch_length(
                self.time_array,
                self.hamilton,
                target_bunch_length=1.0e-15,
                distribution_type="waterbag",
                bunch_length_fit="full",
            )
        self.assertGreaterEqual(x_0, 0.0)
        self.assertLessEqual(x_0, CURVATURE * (1.0e-9) ** 2)

    def test_invalid_fit_mode_raises(self):
        time_array, _, hamilton = _harmonic_grid(n_time=101, n_deltaE=51)
        with self.assertRaisesRegex(ValueError, "gauss"):
            x0_from_bunch_length(
                time_array,
                hamilton,
                target_bunch_length=1e-9,
                distribution_type="waterbag",
                bunch_length_fit="gauss",
            )
        with self.assertRaisesRegex(ValueError, "Unknown"):
            x0_from_bunch_length(
                time_array,
                hamilton,
                target_bunch_length=1e-9,
                distribution_type="waterbag",
                bunch_length_fit="nope",
            )

    def test_iteration_cap_warns(self):
        time_array, _, hamilton = _harmonic_grid(n_time=6001)
        with self.assertWarnsRegex(UserWarning, "did not converge"):
            x0_from_bunch_length(
                time_array,
                hamilton,
                target_bunch_length=1.0e-9,
                distribution_type="gaussian",
                bunch_length_fit="rms",
                max_iterations=2,
            )


if __name__ == "__main__":
    unittest.main()
