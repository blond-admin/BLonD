"""Tests for the Abel-transform distribution reconstruction."""

from math import gamma, pi, sqrt

import numpy as np
import pytest

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

# LHC-like kinetic factor (450 GeV protons).
ETA_0 = 3.172867586042721e-04
BETA = 0.9999978262922387
TOTAL_ENERGY = 450.00104432e9
EOM_FACTOR_DE = calc_eom_factor_dE(ETA_0, BETA, TOTAL_ENERGY)

# Harmonic (linear-regime) well: V = b * t^2, 1 eV at the frame edge.
CURVATURE = 1.0e18  # eV/s^2
HALF_SPAN = 1.0e-9  # s


def _harmonic_well(n_time=2001):
    time_array = np.linspace(-HALF_SPAN, HALF_SPAN, n_time)
    return time_array, CURVATURE * time_array**2


def _binomial_projection(potential_well, h_0, mu):
    """Exact line density of a unit-amplitude binomial F(H)."""
    half_integral = sqrt(pi) * gamma(mu + 1.0) / gamma(mu + 1.5)
    support = np.maximum(1.0 - potential_well / h_0, 0.0)
    return support ** (mu + 0.5) * sqrt(h_0 / EOM_FACTOR_DE) * half_integral


@pytest.mark.parametrize("mu", [0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("half_option", ["first", "second", "both"])
def test_binomial_round_trip_harmonic_well(mu, half_option):
    # A binomial F(H) projects to a (mu + 1/2)-binomial line density;
    # the Abel inversion must recover F(H) in shape AND absolute scale.
    time_array, well = _harmonic_well()
    h_0 = 0.4
    line_density_values = _binomial_projection(well, h_0, mu)

    hamiltonian_coord, distribution_values = distribution_from_line_density(
        time_array,
        line_density_values,
        well,
        eom_factor_dE=EOM_FACTOR_DE,
        half_option=half_option,
    )

    expected = np.maximum(1.0 - hamiltonian_coord / h_0, 0.0) ** mu
    # Compare away from the support edge, where the finite grid smears
    # the (1 - H/H_0)^mu cusp (or, for the waterbag, its discontinuity).
    inside = hamiltonian_coord <= 0.9 * h_0
    np.testing.assert_allclose(
        distribution_values[inside], expected[inside], atol=0.02, rtol=0.03
    )
    # Outside the support the reconstruction must vanish.
    outside = hamiltonian_coord >= 1.1 * h_0
    assert np.all(np.abs(distribution_values[outside]) < 0.02)


def test_binomial_round_trip_rf_well():
    # Same anchor in a realistic sinusoidal RF well: the projection
    # identity holds for a general well, not just the linear regime.
    omega_rf = 2.0 * np.pi * 400.789e6
    t_rev = 88.9e-6
    time_array = bucket_time_array(omega_rf, n_points=2001)
    voltage = 6e6 * np.sin(omega_rf * time_array)
    well = rf_potential_well(
        time_array, voltage, charge=1.0, t_rev=t_rev, eta_0=ETA_0
    )

    mu = 1.0
    h_0 = 0.5 * float(well.max())
    line_density_values = _binomial_projection(well, h_0, mu)

    hamiltonian_coord, distribution_values = distribution_from_line_density(
        time_array,
        line_density_values,
        well,
        eom_factor_dE=EOM_FACTOR_DE,
        half_option="both",
    )

    expected = np.maximum(1.0 - hamiltonian_coord / h_0, 0.0) ** mu
    inside = hamiltonian_coord <= 0.9 * h_0
    np.testing.assert_allclose(
        distribution_values[inside], expected[inside], atol=0.02, rtol=0.03
    )


def test_gaussian_round_trip():
    # lambda ∝ exp(-V/H_bar) inverts to F ∝ exp(-H/H_bar) in any well.
    time_array, well = _harmonic_well()
    h_bar = 0.2
    line_density_values = np.exp(-well / h_bar) * sqrt(
        pi * h_bar / EOM_FACTOR_DE
    )

    hamiltonian_coord, distribution_values = distribution_from_line_density(
        time_array,
        line_density_values,
        well,
        eom_factor_dE=EOM_FACTOR_DE,
        half_option="first",
    )

    expected = np.exp(-hamiltonian_coord / h_bar)
    # The frame truncates the gaussian tails: compare where the input
    # line density is not dominated by the truncation.
    inside = hamiltonian_coord <= 3.0 * h_bar
    np.testing.assert_allclose(
        distribution_values[inside], expected[inside], atol=0.02, rtol=0.03
    )


def test_half_options_agree_for_symmetric_input():
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
        h_other, f_other = results[half_option]
        np.testing.assert_allclose(
            f_first,
            np.interp(h_first, h_other, f_other),
            atol=1e-3,
        )


def test_both_is_average_of_first_and_second():
    # Asymmetric input (gaussian bunch in a tilted well): "both" must be
    # the average of the two single-branch reconstructions.
    time_array, well = _harmonic_well()
    well = well * (1.0 + 0.3 * time_array / HALF_SPAN)
    well -= well.min()
    sigma = 0.25e-9
    minimum_time = time_array[np.argmin(well)]
    line_density_values = np.exp(
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
    assert not np.allclose(
        f_first,
        np.interp(h_first, h_second, f_second),
        atol=0.01 * scale,
    )
    # ...and "both" is their pointwise average on the first-branch grid.
    expected = (f_first + np.interp(h_first, h_second, f_second)) / 2.0
    expected[expected < 0.0] = 0.0
    np.testing.assert_allclose(
        f_both, np.interp(h_both, h_first, expected), atol=1e-6 * scale
    )


def test_line_density_closure():
    # Full closure: reconstruct F(H), project it back on a 2D grid and
    # recover the input line density (the guarantee the matcher needs).
    time_array, well = _harmonic_well()
    line_density_values = _binomial_projection(well, 0.4, 1.0)

    hamiltonian_coord, distribution_values = distribution_from_line_density(
        time_array,
        line_density_values,
        well,
        eom_factor_dE=EOM_FACTOR_DE,
        half_option="both",
    )

    _, _, hamilton_2d = hamiltonian_grid(
        time_array,
        well,
        eom_factor_dE=EOM_FACTOR_DE,
        n_points_deltaE=1001,
        energy_range=(
            -sqrt(well.max() / EOM_FACTOR_DE),
            sqrt(well.max() / EOM_FACTOR_DE),
        ),
    )
    density_grid = np.interp(
        hamilton_2d, hamiltonian_coord, distribution_values
    )
    reconstructed = density_grid.sum(axis=0)

    normalized_input = line_density_values / np.sum(line_density_values)
    normalized_reconstructed = reconstructed / np.sum(reconstructed)
    np.testing.assert_allclose(
        normalized_reconstructed,
        normalized_input,
        atol=0.02 * normalized_input.max(),
    )


def test_n_points_abel_resampling():
    # A coarse measured-like profile refined via n_points_abel must stay
    # close to the analytic distribution.
    time_array, well = _harmonic_well(n_time=201)
    h_0 = 0.4
    line_density_values = _binomial_projection(well, h_0, 1.0)

    hamiltonian_coord, distribution_values = distribution_from_line_density(
        time_array,
        line_density_values,
        well,
        eom_factor_dE=EOM_FACTOR_DE,
        half_option="both",
        n_points_abel=5000,
    )

    expected = np.maximum(1.0 - hamiltonian_coord / h_0, 0.0)
    inside = hamiltonian_coord <= 0.9 * h_0
    np.testing.assert_allclose(
        distribution_values[inside], expected[inside], atol=0.03, rtol=0.05
    )


def test_duplicated_minimum_sample():
    # A symmetric well on an even grid has two equal minimum samples
    # (e.g. the LHC well of main.py step 8): the branch split must not
    # divide by the duplicated value — a regression that returned inf
    # (and, once sanitized, F(0) = 0) on the second branch.
    n_time = 2000  # even: minimum falls between two equal samples
    time_array = np.linspace(-HALF_SPAN, HALF_SPAN, n_time)
    well = CURVATURE * time_array**2
    assert well[n_time // 2 - 1] == well[n_time // 2]

    h_0 = 0.4
    line_density_values = _binomial_projection(well, h_0, 1.0)

    for half_option in ("first", "second", "both"):
        hamiltonian_coord, distribution_values = (
            distribution_from_line_density(
                time_array,
                line_density_values,
                well,
                eom_factor_dE=EOM_FACTOR_DE,
                half_option=half_option,
            )
        )
        assert np.all(np.isfinite(distribution_values))
        expected = np.maximum(1.0 - hamiltonian_coord / h_0, 0.0)
        inside = hamiltonian_coord <= 0.9 * h_0
        np.testing.assert_allclose(
            distribution_values[inside],
            expected[inside],
            atol=0.02,
            rtol=0.03,
        )


def test_input_validation():
    time_array, well = _harmonic_well(n_time=101)
    line_density_values = _binomial_projection(well, 0.4, 1.0)

    with pytest.raises(ValueError, match="half_option"):
        distribution_from_line_density(
            time_array,
            line_density_values,
            well,
            eom_factor_dE=EOM_FACTOR_DE,
            half_option="not_a_half",
        )
    with pytest.raises(AssertionError, match="shape"):
        distribution_from_line_density(
            time_array,
            line_density_values[:-1],
            well,
            eom_factor_dE=EOM_FACTOR_DE,
        )
    # A monotonic well has its minimum on the frame edge: no centred
    # bunch to invert.
    with pytest.raises(ValueError, match="frame edge"):
        distribution_from_line_density(
            time_array,
            line_density_values,
            np.linspace(0.0, 1.0, len(time_array)),
            eom_factor_dE=EOM_FACTOR_DE,
        )


def test_verbose_and_plot_smoke(capsys):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_array, well = _harmonic_well(n_time=201)
    line_density_values = _binomial_projection(well, 0.4, 1.0)
    distribution_from_line_density(
        time_array,
        line_density_values,
        well,
        eom_factor_dE=EOM_FACTOR_DE,
        verbose=True,
        plot=True,
    )
    assert "distribution_from_line_density" in capsys.readouterr().out
    plt.close("all")
