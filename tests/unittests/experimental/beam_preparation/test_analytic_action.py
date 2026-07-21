"""Tests for the analytic action J(H) building blocks."""

import numpy as np
import pytest

from blond.experimental.beam_preparation.analytic_action import (
    action_from_potential_well,
    action_grid,
    hamiltonian_from_emittance,
)
from blond.experimental.beam_preparation.analytic_hamiltonian import (
    calc_eom_factor_dE,
    hamiltonian_grid,
)
from blond.experimental.beam_preparation.analytic_potential_well import (
    bucket_time_array,
    rf_potential_well,
)

# LHC-like reference (450 GeV protons).
OMEGA_RF = 2518229887.224505
VOLTAGE = 6.0e6
T_REV = 8.892465516509709e-05
ETA_0 = 3.172867586042721e-04
BETA = 0.9999978262922387
TOTAL_ENERGY = 450.00104432e9

EOM_FACTOR_DE = calc_eom_factor_dE(ETA_0, BETA, TOTAL_ENERGY)


def _lhc_bucket(n_points=2000, phi_rf=0.0, eta_0=ETA_0):
    time_array = bucket_time_array(OMEGA_RF, n_points=n_points)
    total_voltage = VOLTAGE * np.sin(OMEGA_RF * time_array + phi_rf)
    well = rf_potential_well(
        time_array,
        total_voltage,
        charge=1.0,
        t_rev=T_REV,
        eta_0=eta_0,
    )
    return time_array, well


def test_action_matches_harmonic_oscillator():
    # For H = a*dE^2 + b*t^2 the orbit is an ellipse: J = H/(2*sqrt(a*b)).
    a = EOM_FACTOR_DE
    b = 1.0e18  # eV/s^2, arbitrary curvature
    time_array = np.linspace(-1e-9, 1e-9, 4001)
    well = b * time_array**2
    sorted_h, sorted_j = action_from_potential_well(
        time_array, well, eom_factor_dE=a
    )
    expected = sorted_h / (2.0 * np.sqrt(a * b))
    # Compare over the well-resolved central range (skip the edges).
    sel = (sorted_h > 0.05 * sorted_h.max()) & (
        sorted_h < 0.95 * sorted_h.max()
    )
    # Measured max rel. error at n=4001 is ~8.7e-5 (O(h^2) convergence).
    np.testing.assert_allclose(sorted_j[sel], expected[sel], rtol=5e-4)


def test_action_is_monotonic_in_hamiltonian():
    time_array, well = _lhc_bucket()
    sorted_h, sorted_j = action_from_potential_well(
        time_array, well, eom_factor_dE=EOM_FACTOR_DE
    )
    assert np.all(np.diff(sorted_j) >= -1e-12)
    assert sorted_j[0] >= 0.0


def test_separatrix_action_matches_bucket_area():
    # Stationary single-RF bucket: area = 8*dE_max/omega_rf = 2*pi*J_sep.
    time_array, well = _lhc_bucket(n_points=4000)
    sorted_h, sorted_j = action_from_potential_well(
        time_array, well, eom_factor_dE=EOM_FACTOR_DE
    )
    potential_well_amplitude = well.max() - well.min()
    deltaE_max = np.sqrt(potential_well_amplitude / EOM_FACTOR_DE)
    expected_action = 4.0 * deltaE_max / (np.pi * OMEGA_RF)
    # Measured rel. error at n=4000 is ~2.6e-8 (O(h^2) convergence).
    assert np.isclose(sorted_j[-1], expected_action, rtol=1e-6)


def test_below_transition_separatrix_action():
    # Below transition, with the BLonD 2 convention phi_rf = pi, the
    # whole chain works and the separatrix action matches the closed
    # form. Also pins |eta_0| in the kinetic factor (no NaN chain).
    factor = calc_eom_factor_dE(-ETA_0, BETA, TOTAL_ENERGY)
    assert factor > 0.0
    time_array, well = _lhc_bucket(n_points=4000, phi_rf=np.pi, eta_0=-ETA_0)
    sorted_h, sorted_j = action_from_potential_well(
        time_array, well, eom_factor_dE=factor
    )
    assert not np.any(np.isnan(sorted_j))
    deltaE_max = np.sqrt((well.max() - well.min()) / factor)
    expected_action = 4.0 * deltaE_max / (np.pi * OMEGA_RF)
    assert np.isclose(sorted_j[-1], expected_action, rtol=1e-6)


def test_action_on_nonuniform_grid():
    # Mildly non-uniform monotone grid over exactly one bucket: the
    # x=-based integration must still match the closed form.
    rf_period = 2.0 * np.pi / OMEGA_RF
    u = np.linspace(0.0, 1.0, 4001)
    time_array = rf_period * (u + 0.15 * np.sin(2.0 * np.pi * u) / (2 * np.pi))
    total_voltage = VOLTAGE * np.sin(OMEGA_RF * time_array)
    well = rf_potential_well(
        time_array,
        total_voltage,
        charge=1.0,
        t_rev=T_REV,
        eta_0=ETA_0,
    )
    sorted_h, sorted_j = action_from_potential_well(
        time_array, well, eom_factor_dE=EOM_FACTOR_DE
    )
    deltaE_max = np.sqrt((well.max() - well.min()) / EOM_FACTOR_DE)
    expected_action = 4.0 * deltaE_max / (np.pi * OMEGA_RF)
    assert np.isclose(sorted_j[-1], expected_action, rtol=1e-5)


def test_uncut_well_raises():
    time_array = bucket_time_array(
        OMEGA_RF, n_points=2000, dt_margin_fraction=0.4
    )
    total_voltage = VOLTAGE * np.sin(OMEGA_RF * time_array)
    well = rf_potential_well(
        time_array,
        total_voltage,
        charge=1.0,
        t_rev=T_REV,
        eta_0=ETA_0,
    )
    with pytest.raises(ValueError):
        action_from_potential_well(
            time_array, well, eom_factor_dE=EOM_FACTOR_DE
        )


def test_hamiltonian_from_emittance_round_trip():
    time_array, well = _lhc_bucket()
    sorted_h, sorted_j = action_from_potential_well(
        time_array, well, eom_factor_dE=EOM_FACTOR_DE
    )
    # Pick a level well inside the bucket, round-trip via emittance.
    index = len(sorted_h) // 2
    hamiltonian_0 = sorted_h[index]
    emittance = 2.0 * np.pi * sorted_j[index]
    recovered = hamiltonian_from_emittance(emittance, sorted_h, sorted_j)
    assert np.isclose(recovered, hamiltonian_0, rtol=1e-6)
    # Zero emittance maps to the bottom of the well.
    bottom = hamiltonian_from_emittance(0.0, sorted_h, sorted_j)
    assert np.isclose(bottom, sorted_h[0])


def test_shape_mismatch_raises():
    time_array, well = _lhc_bucket(n_points=300)
    with pytest.raises(AssertionError):
        action_from_potential_well(
            time_array, well[:-1], eom_factor_dE=EOM_FACTOR_DE
        )


def test_emittance_beyond_bucket_raises():
    time_array, well = _lhc_bucket()
    sorted_h, sorted_j = action_from_potential_well(
        time_array, well, eom_factor_dE=EOM_FACTOR_DE
    )
    bucket_area = 2.0 * np.pi * sorted_j[-1]
    with pytest.raises(ValueError):
        hamiltonian_from_emittance(1.01 * bucket_area, sorted_h, sorted_j)


def test_action_grid_shape_and_outside_bucket():
    time_array, well = _lhc_bucket(n_points=300)
    sorted_h, sorted_j = action_from_potential_well(
        time_array, well, eom_factor_dE=EOM_FACTOR_DE
    )
    _, _, hamilton = hamiltonian_grid(
        time_array,
        well,
        eom_factor_dE=EOM_FACTOR_DE,
        n_points_deltaE=200,
    )
    action_2D = action_grid(hamilton, sorted_h, sorted_j)
    assert action_2D.shape == hamilton.shape
    # Outside the bucket (H above the largest tabulated level) -> inf.
    assert np.isinf(
        action_grid(np.array([sorted_h[-1] * 10]), sorted_h, sorted_j)
    )[0]
    # Inside, the action stays bounded and non-negative.
    inside = hamilton <= sorted_h[-1]
    assert np.all(np.isfinite(action_2D[inside]))
    assert np.all(action_2D[inside] >= 0.0)


def test_split_well_via_loosened_tolerance_matches_legacy_route():
    # Legacy BLonD 2 tolerated wells with several minima (e.g. split by
    # an induced potential): it warned, took the deepest, and its
    # zero-padded J summed the islands below the inner separatrix. The
    # port's route for this is the single_bucket_tolerance knob; the
    # zero-padded integral keeps J well-behaved on the split well.
    time_array = bucket_time_array(
        OMEGA_RF, n_points=4000, dt_margin_fraction=0.2
    )
    total_voltage = VOLTAGE * (
        np.sin(OMEGA_RF * time_array)
        + 0.8 * np.sin(2.0 * OMEGA_RF * time_array)
    )
    from blond.experimental.beam_preparation.analytic_well_cut import (
        cut_potential_well,
    )

    well = rf_potential_well(
        time_array,
        total_voltage,
        charge=1.0,
        t_rev=T_REV,
        eta_0=ETA_0,
    )
    # Default: rejected (inner separatrix ~5 % of the amplitude)...
    with pytest.raises(ValueError):
        cut_potential_well(time_array, well)
    # ...loosened tolerance: outer bucket accepted, J finite and
    # monotone up to boundary-cell discretization noise.
    time_cut, well_cut = cut_potential_well(
        time_array, well, single_bucket_tolerance=0.12
    )
    sorted_h, sorted_j = action_from_potential_well(
        time_cut,
        well_cut,
        eom_factor_dE=EOM_FACTOR_DE,
        single_bucket_tolerance=0.12,
    )
    assert np.all(np.isfinite(sorted_j))
    max_backstep = float(np.max(-np.diff(sorted_j), initial=0.0))
    assert max_backstep < 1e-4 * sorted_j[-1]
    # The double-well outer bucket is larger than the single-harmonic
    # bucket.
    time_single, well_single = _lhc_bucket(n_points=4000)
    _, sorted_j_single = action_from_potential_well(
        time_single, well_single, eom_factor_dE=EOM_FACTOR_DE
    )
    assert sorted_j[-1] > sorted_j_single[-1]
    # The explicit allow_inner_buckets route (default tolerance) warns
    # and yields the same result as the loosened-tolerance route.
    with pytest.warns(UserWarning, match="inner"):
        time_cut_2, well_cut_2 = cut_potential_well(
            time_array, well, allow_inner_buckets=True
        )
    with pytest.warns(UserWarning, match="inner"):
        _, sorted_j_2 = action_from_potential_well(
            time_cut_2,
            well_cut_2,
            eom_factor_dE=EOM_FACTOR_DE,
            allow_inner_buckets=True,
        )
    assert np.isclose(sorted_j_2[-1], sorted_j[-1], rtol=1e-9)


def test_verbose_and_plot_smoke():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    time_array, well = _lhc_bucket(n_points=256)
    action_from_potential_well(
        time_array,
        well,
        eom_factor_dE=EOM_FACTOR_DE,
        verbose=True,
        plot=True,
    )
    plt.close("all")
