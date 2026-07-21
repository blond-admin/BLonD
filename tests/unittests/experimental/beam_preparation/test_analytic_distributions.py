"""Tests for the analytic distribution families and targeting."""

import warnings

import numpy as np
import pytest

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

# LHC-like kinetic factor (450 GeV protons).
ETA_0 = 3.172867586042721e-04
BETA = 0.9999978262922387
TOTAL_ENERGY = 450.00104432e9
EOM_FACTOR_DE = calc_eom_factor_dE(ETA_0, BETA, TOTAL_ENERGY)

# Harmonic (linear-regime) well: V = b * t^2.
CURVATURE = 1.0e18  # eV/s^2


def _harmonic_grid(n_time=1001, n_deltaE=801, half_span=1.0e-9):
    time_array = np.linspace(-half_span, half_span, n_time)
    well = CURVATURE * time_array**2
    time_grid, deltaE_grid, hamilton = hamiltonian_grid(
        time_array,
        well,
        eom_factor_dE=EOM_FACTOR_DE,
        n_points_deltaE=n_deltaE,
    )
    return time_array, well, hamilton


def test_named_types_match_binomial_exponents():
    x_array = np.linspace(0.0, 2.0, 100)
    for name, exponent in DISTRIBUTION_EXPONENTS.items():
        np.testing.assert_array_equal(
            distribution_function(x_array, name, 1.3),
            distribution_function(x_array, "binomial", 1.3, exponent),
        )


def test_gaussian_form_and_binomial_form():
    x_array = np.array([0.0, 0.4, 1.0, 1.6])
    np.testing.assert_allclose(
        distribution_function(x_array, "gaussian", 0.8),
        np.exp(-2.0 * x_array / 0.8),
    )
    np.testing.assert_allclose(
        distribution_function(x_array, "binomial", 1.0, 2.0),
        np.where(x_array <= 1.0, (1.0 - np.minimum(x_array, 1.0)) ** 2, 0.0),
    )


def test_input_validation():
    x_array = np.linspace(0.0, 1.0, 10)
    with pytest.raises(ValueError, match="binomial"):
        distribution_function(x_array, "binomial", 1.0)
    with pytest.raises(ValueError, match="Unknown"):
        distribution_function(x_array, "not_a_type", 1.0)
    with pytest.warns(UserWarning, match="ignored"):
        distribution_function(x_array, "waterbag", 1.0, exponent=2.0)
    with pytest.warns(UserWarning, match="ignored"):
        line_density(x_array, "gaussian", 1.0, exponent=2.0)


def test_inf_grid_evaluates_to_zero_without_warnings():
    # action_grid marks outside-bucket points with inf; all families
    # must map them to 0 with no RuntimeWarning.
    x_array = np.array([0.2, 0.9, np.inf])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        binomial = distribution_function(x_array, "binomial", 1.0, 0.5)
        gaussian = distribution_function(x_array, "gaussian", 1.0)
    assert binomial[-1] == 0.0
    assert gaussian[-1] == 0.0
    assert binomial[0] > 0.0


def test_projection_gives_plus_half_exponent():
    # Project a binomial phase-space density over dE in a harmonic well:
    # lambda(t) must equal the line_density family with the same inputs
    # (which bakes in the mu + 1/2 exponent shift). Tolerances measured
    # at n_deltaE=2001: the smooth families verify the convention to
    # ~1e-6-1e-8; the waterbag (mu=0, discontinuous support) is
    # dominated by dE-grid staircase quantization, converging as 1/n.
    time_array, well, hamilton = _harmonic_grid(n_deltaE=2001)
    hamiltonian_0 = CURVATURE * (0.5e-9) ** 2  # support: |t| <= 0.5 ns
    for distribution_type, exponent, rtol in [
        ("parabolic_amplitude", None, 1e-4),
        ("waterbag", None, 2e-2),
        ("binomial", 1.7, 1e-6),
    ]:
        density = distribution_function(
            hamilton, distribution_type, hamiltonian_0, exponent
        )
        projected = density.sum(axis=0)
        projected /= projected.max()
        bunch_length = 2.0 * np.sqrt(hamiltonian_0 / CURVATURE)
        expected = line_density(
            time_array, distribution_type, bunch_length, 0.0, exponent
        )
        expected /= expected.max()
        significant = expected > 0.05
        np.testing.assert_allclose(
            projected[significant],
            expected[significant],
            rtol=rtol,
            err_msg=distribution_type,
        )


def test_line_density_shapes():
    time_array = np.linspace(-1.0, 1.0, 2001)
    # parabolic_line: total exponent 0.5 + 0.5 = 1 -> exact parabola.
    parabola = line_density(time_array, "parabolic_line", 2.0)
    np.testing.assert_allclose(parabola, 1.0 - time_array**2, atol=1e-12)
    # cosine_squared with support tau
    cosine = line_density(time_array, "cosine_squared", 2.0)
    np.testing.assert_allclose(
        cosine, np.cos(0.5 * np.pi * time_array) ** 2, atol=1e-12
    )
    # gaussian: sigma = tau/4
    gaussian = line_density(time_array, "gaussian", 2.0)
    np.testing.assert_allclose(
        gaussian, np.exp(-(time_array**2) / (2 * 0.5**2)), atol=1e-12
    )


def test_fwhm_helper_on_gaussian():
    time_array = np.linspace(-1.0, 1.0, 4001)
    sigma = 0.1
    gaussian = np.exp(-(time_array**2) / (2 * sigma**2))
    assert np.isclose(
        _bunch_length_fwhm(time_array, gaussian), 4.0 * sigma, rtol=1e-3
    )


def test_x0_from_bunch_length_full():
    time_array, well, hamilton = _harmonic_grid()
    target = 1.0e-9
    x_0 = x0_from_bunch_length(
        time_array,
        hamilton,
        target_bunch_length=target,
        distribution_type="waterbag",
        bunch_length_fit="full",
    )
    # Contour extent 2*sqrt(X0/b) = tau -> X0 = b*(tau/2)^2.
    assert np.isclose(x_0, CURVATURE * (target / 2.0) ** 2, rtol=2e-2)


def test_x0_from_bunch_length_rms_waterbag_and_gaussian():
    time_array, well, hamilton = _harmonic_grid()
    target = 1.0e-9
    # Waterbag: semicircle line density, sigma = t_max/2 -> X0 = b*tau^2/4.
    # Gaussian: sigma_t = sqrt(X0/(4b)) -> the same X0 = b*tau^2/4.
    for distribution_type in ("waterbag", "gaussian"):
        x_0 = x0_from_bunch_length(
            time_array,
            hamilton,
            target_bunch_length=target,
            distribution_type=distribution_type,
            bunch_length_fit="rms",
        )
        assert np.isclose(x_0, CURVATURE * target**2 / 4.0, rtol=2e-2), (
            distribution_type
        )


def test_x0_from_bunch_length_fwhm_gaussian():
    time_array, well, hamilton = _harmonic_grid()
    target = 1.0e-9
    x_0 = x0_from_bunch_length(
        time_array,
        hamilton,
        target_bunch_length=target,
        distribution_type="gaussian",
        bunch_length_fit="fwhm",
    )
    # For a true gaussian the fwhm measure returns exactly 4 sigma.
    assert np.isclose(x_0, CURVATURE * target**2 / 4.0, rtol=2e-2)


def test_x0_bucket_too_small_warns():
    time_array, well, hamilton = _harmonic_grid()
    with pytest.warns(UserWarning, match="too small for the requested"):
        x0_from_bunch_length(
            time_array,
            hamilton,
            target_bunch_length=50.0e-9,  # far beyond the frame
            distribution_type="waterbag",
            bunch_length_fit="full",
        )


def test_x0_target_below_resolution_converges_trivially():
    # A target below one time bin is within the convergence tolerance
    # of a zero-extent density: the fit returns quickly, no warning.
    time_array, well, hamilton = _harmonic_grid()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x_0 = x0_from_bunch_length(
            time_array,
            hamilton,
            target_bunch_length=1.0e-15,
            distribution_type="waterbag",
            bunch_length_fit="full",
        )
    assert 0.0 <= x_0 <= CURVATURE * (1.0e-9) ** 2


def test_x0_invalid_fit_mode_raises():
    time_array, well, hamilton = _harmonic_grid(n_time=101, n_deltaE=51)
    with pytest.raises(ValueError, match="gauss"):
        x0_from_bunch_length(
            time_array,
            hamilton,
            target_bunch_length=1e-9,
            distribution_type="waterbag",
            bunch_length_fit="gauss",
        )
    with pytest.raises(ValueError, match="Unknown"):
        x0_from_bunch_length(
            time_array,
            hamilton,
            target_bunch_length=1e-9,
            distribution_type="waterbag",
            bunch_length_fit="nope",
        )


def test_x0_iteration_cap_warns():
    time_array, well, hamilton = _harmonic_grid(n_time=6001)
    with pytest.warns(UserWarning, match="did not converge"):
        x0_from_bunch_length(
            time_array,
            hamilton,
            target_bunch_length=1.0e-9,
            distribution_type="gaussian",
            bunch_length_fit="rms",
            max_iterations=2,
        )
