"""Tests for the analytic RF potential-well building blocks."""

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from blond.experimental.beam_preparation.analytic_potential_well import (
    bucket_time_array,
    check_single_bucket_well,
    rf_potential_well,
)

# LHC-like main-harmonic parameters (450 GeV protons, h=35640, V=6 MV).
OMEGA_RF = 2518229887.224505
VOLTAGE = 6.0e6
T_REV = 8.892465516509709e-05
ETA_0 = 3.172867586042721e-04  # above transition (eta_0 > 0)
HARMONIC = OMEGA_RF * T_REV / (2.0 * np.pi)


def _single_harmonic(time_array, phi_rf=0.0):
    return VOLTAGE * np.sin(OMEGA_RF * time_array + phi_rf)


def test_bucket_time_array_span():
    time_array = bucket_time_array(
        OMEGA_RF, n_points=128, dt_margin_fraction=0.4
    )
    rf_period = 2.0 * np.pi / OMEGA_RF
    assert len(time_array) == 128
    assert time_array[0] < 0.0
    assert time_array[-1] > rf_period


def test_shape_and_min_at_zero():
    time_array = bucket_time_array(OMEGA_RF, n_points=5000)
    well = rf_potential_well(
        time_array,
        _single_harmonic(time_array),
        charge=1.0,
        t_rev=T_REV,
        eta_0=ETA_0,
    )
    assert well.shape == time_array.shape
    assert np.isclose(well.min(), 0.0)


def test_matches_closed_form():
    # For V*sin(w t): Phi(t) = (eom*V/w)*(cos(w t) - 1), then min at 0.
    time_array = np.linspace(0.0, 2.0 * np.pi / OMEGA_RF, 20000)
    well = rf_potential_well(
        time_array,
        _single_harmonic(time_array),
        charge=1.0,
        t_rev=T_REV,
        eta_0=ETA_0,
    )
    eom = np.sign(ETA_0) * 1.0 / T_REV
    analytic = (eom * VOLTAGE / OMEGA_RF) * (
        np.cos(OMEGA_RF * time_array) - 1.0
    )
    analytic = analytic - analytic.min()
    # Measured max rel. error at n=20000 is ~8e-9 (O(h^2) convergence).
    np.testing.assert_allclose(
        well, analytic, rtol=1e-6, atol=1e-6 * well.max()
    )


def test_matches_legacy_cumtrapz_formula():
    # Exact parity with the BLonD 2 expression (identical integration).
    time_array = bucket_time_array(OMEGA_RF, n_points=4000)
    total_voltage = _single_harmonic(time_array)
    well = rf_potential_well(
        time_array,
        total_voltage,
        charge=1.0,
        t_rev=T_REV,
        eta_0=ETA_0,
        subtract_min=False,
    )
    eom = np.sign(ETA_0) * 1.0 / T_REV
    legacy = -cumulative_trapezoid(
        eom * total_voltage, x=time_array, initial=0.0
    )
    np.testing.assert_array_equal(well, legacy)


def test_eta_sign_flips_potential():
    time_array = bucket_time_array(OMEGA_RF, n_points=4000)
    total_voltage = _single_harmonic(time_array)
    common = dict(charge=1.0, t_rev=T_REV, subtract_min=False)
    above = rf_potential_well(
        time_array, total_voltage, eta_0=ETA_0, **common
    )
    below = rf_potential_well(
        time_array, total_voltage, eta_0=-ETA_0, **common
    )
    np.testing.assert_allclose(above, -below)


def test_energy_gain_adds_linear_tilt_with_pinned_slope():
    # Subtracting a constant (synchronous) voltage integrates to a
    # linear ramp of slope sign(eta_0)*sign(charge)*gain/t_rev.
    energy_gain = 1.0e5
    time_array = bucket_time_array(OMEGA_RF, n_points=8000)
    total_voltage = _single_harmonic(time_array)
    common = dict(charge=1.0, t_rev=T_REV, eta_0=ETA_0, subtract_min=False)
    base = rf_potential_well(time_array, total_voltage, **common)
    tilted = rf_potential_well(
        time_array, total_voltage, energy_gain_per_turn=energy_gain, **common
    )
    diff = tilted - base
    coeffs = np.polyfit(time_array, diff, 1)
    residual = diff - np.polyval(coeffs, time_array)
    assert np.max(np.abs(residual)) < 1e-6 * np.max(np.abs(diff))
    # Pin the sign and magnitude of the acceleration term (a sign flip
    # in the ported formula must fail here).
    expected_slope = np.sign(ETA_0) * np.sign(1.0) * energy_gain / T_REV
    assert np.isclose(coeffs[0], expected_slope, rtol=1e-6)


def test_energy_gain_slope_sign_with_negative_charge():
    energy_gain = 1.0e5
    time_array = bucket_time_array(OMEGA_RF, n_points=8000)
    total_voltage = _single_harmonic(time_array)
    common = dict(
        charge=-1.0, t_rev=T_REV, eta_0=ETA_0, subtract_min=False
    )
    base = rf_potential_well(time_array, total_voltage, **common)
    tilted = rf_potential_well(
        time_array, total_voltage, energy_gain_per_turn=energy_gain, **common
    )
    coeffs = np.polyfit(time_array, tilted - base, 1)
    expected_slope = np.sign(ETA_0) * np.sign(-1.0) * energy_gain / T_REV
    assert np.isclose(coeffs[0], expected_slope, rtol=1e-6)


def test_amplitude_scales_with_charge():
    # Stationary single-harmonic well amplitude is |q| * V / (pi * h).
    time_array = bucket_time_array(OMEGA_RF, n_points=8000)
    # q = +2, above transition, phi_rf = 0 (stable phase mid-frame)
    well_q2 = rf_potential_well(
        time_array,
        _single_harmonic(time_array),
        charge=2.0,
        t_rev=T_REV,
        eta_0=ETA_0,
    )
    assert np.isclose(
        well_q2.max(), 2.0 * VOLTAGE / (np.pi * HARMONIC), rtol=1e-6
    )
    # q = -1, above transition: sign(eta*q) < 0, so the convention is
    # phi_rf = pi to keep the stable phase mid-frame.
    well_qm1 = rf_potential_well(
        time_array,
        _single_harmonic(time_array, phi_rf=np.pi),
        charge=-1.0,
        t_rev=T_REV,
        eta_0=ETA_0,
    )
    assert np.isclose(
        well_qm1.max(), VOLTAGE / (np.pi * HARMONIC), rtol=1e-6
    )
    # The convention holds: minimum sits mid-frame, not on an edge.
    n = len(time_array)
    assert 0.25 * n < well_qm1.argmin() < 0.75 * n


def test_check_single_bucket_well():
    time_array = bucket_time_array(OMEGA_RF, n_points=2000)
    clean = rf_potential_well(
        time_array,
        _single_harmonic(time_array),
        charge=1.0,
        t_rev=T_REV,
        eta_0=ETA_0,
    )
    assert check_single_bucket_well(clean) is True

    # Margined frame: edges no longer reach the well maximum.
    time_margined = bucket_time_array(
        OMEGA_RF, n_points=2000, dt_margin_fraction=0.4
    )
    margined = rf_potential_well(
        time_margined,
        _single_harmonic(time_margined),
        charge=1.0,
        t_rev=T_REV,
        eta_0=ETA_0,
    )
    assert check_single_bucket_well(margined, raise_error=False) is False
    with pytest.raises(ValueError):
        check_single_bucket_well(margined)

    # Multi-bucket span: interior maxima.
    rf_period = 2.0 * np.pi / OMEGA_RF
    time_3 = np.linspace(0.0, 3.0 * rf_period, 6000)
    three_buckets = rf_potential_well(
        time_3,
        _single_harmonic(time_3),
        charge=1.0,
        t_rev=T_REV,
        eta_0=ETA_0,
    )
    with pytest.raises(ValueError):
        check_single_bucket_well(three_buckets)

    # Below transition with phi_rf=0 (convention violation): the well
    # minimum sits on a frame edge; the error message cites phi_rf.
    below = rf_potential_well(
        time_array,
        _single_harmonic(time_array),
        charge=1.0,
        t_rev=T_REV,
        eta_0=-ETA_0,
    )
    with pytest.raises(ValueError, match="phi_rf"):
        check_single_bucket_well(below)
    # With the convention (phi_rf = pi below transition), all is well.
    below_convention = rf_potential_well(
        time_array,
        _single_harmonic(time_array, phi_rf=np.pi),
        charge=1.0,
        t_rev=T_REV,
        eta_0=-ETA_0,
    )
    assert check_single_bucket_well(below_convention) is True

    # NaN wells must fail loudly (NaN compares False everywhere and
    # would otherwise silently pass the numeric checks).
    nan_well = clean.copy()
    nan_well[100] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        check_single_bucket_well(nan_well)

    # Degenerate input: too few samples.
    with pytest.raises(ValueError):
        check_single_bucket_well(np.array([0.0, 1.0]))


def test_check_accepts_sample_aligned_cut_of_tilted_well():
    # A separatrix cut done BLonD 2 style (sample-aligned, no endpoint
    # interpolation) of an accelerating well: the cut edges mismatch by
    # ~slope*dt (~4e-4 of the well amplitude at n=20000) — the default
    # tolerance must accept.
    time_array = bucket_time_array(OMEGA_RF, n_points=20000)
    well = rf_potential_well(
        time_array,
        _single_harmonic(time_array),
        charge=1.0,
        t_rev=T_REV,
        eta_0=ETA_0,
        energy_gain_per_turn=1.0e5,
    )
    # Uncut tilted well must be rejected...
    with pytest.raises(ValueError):
        check_single_bucket_well(well)
    # ...but its sample-aligned separatrix cut must pass: from the
    # interior (unstable) maximum to the first sample at the same
    # potential on the other side of the minimum.
    i_min = int(well.argmin())
    i_unstable = int(well[:i_min].argmax())
    level = well[i_unstable]
    i_right = i_min + int(np.argmax(well[i_min:] >= level))
    cut = well[i_unstable : i_right + 1]
    assert check_single_bucket_well(cut) is True


def test_shape_mismatch_raises():
    time_array = bucket_time_array(OMEGA_RF, n_points=100)
    with pytest.raises(AssertionError):
        rf_potential_well(
            time_array,
            _single_harmonic(time_array)[:-1],
            charge=1.0,
            t_rev=T_REV,
            eta_0=ETA_0,
        )


def test_verbose_and_plot_smoke():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    time_array = bucket_time_array(OMEGA_RF, n_points=256)
    rf_potential_well(
        time_array,
        _single_harmonic(time_array),
        charge=1.0,
        t_rev=T_REV,
        eta_0=ETA_0,
        verbose=True,
        plot=True,
    )
    plt.close("all")
