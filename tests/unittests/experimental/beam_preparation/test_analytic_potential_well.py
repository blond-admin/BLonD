"""Tests for the analytic RF potential-well building blocks."""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from blond.experimental.beam_preparation.analytic_potential_well import (
    bucket_time_array,
    rf_potential_well,
)

# LHC-like main-harmonic parameters (450 GeV protons, h=35640, V=6 MV).
OMEGA_RF = 2518229887.224505
VOLTAGE = 6.0e6
T_REV = 8.892465516509709e-05
ETA_0 = 3.172867586042721e-04  # above transition (eta_0 > 0)


def _single_harmonic(time_array, phi_rf=0.0):
    return VOLTAGE * np.sin(OMEGA_RF * time_array + phi_rf)


def test_bucket_time_array_span():
    time_array = bucket_time_array(OMEGA_RF, n_points=128, dt_margin_percent=0.4)
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
    # For V*sin(w t): Phi(t) = (eom*V/w)*(cos(w t) - 1), then shifted to min 0.
    time_array = np.linspace(0.0, 2.0 * np.pi / OMEGA_RF, 20000)
    well = rf_potential_well(
        time_array,
        _single_harmonic(time_array),
        charge=1.0,
        t_rev=T_REV,
        eta_0=ETA_0,
    )
    eom = np.sign(ETA_0) * 1.0 / T_REV
    analytic = (eom * VOLTAGE / OMEGA_RF) * (np.cos(OMEGA_RF * time_array) - 1.0)
    analytic = analytic - analytic.min()
    np.testing.assert_allclose(well, analytic, rtol=2e-3, atol=1e-3 * well.max())


def test_matches_legacy_cumtrapz_formula():
    # Exact parity with the BLonD 2 expression (identical integration path).
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
    above = rf_potential_well(time_array, total_voltage, eta_0=ETA_0, **common)
    below = rf_potential_well(time_array, total_voltage, eta_0=-ETA_0, **common)
    np.testing.assert_allclose(above, -below)


def test_energy_gain_adds_linear_tilt():
    # Subtracting a constant (synchronous) voltage integrates to a linear ramp.
    time_array = bucket_time_array(OMEGA_RF, n_points=8000)
    total_voltage = _single_harmonic(time_array)
    common = dict(charge=1.0, t_rev=T_REV, eta_0=ETA_0, subtract_min=False)
    base = rf_potential_well(time_array, total_voltage, **common)
    tilted = rf_potential_well(
        time_array, total_voltage, energy_gain_per_turn=1.0e5, **common
    )
    diff = tilted - base
    coeffs = np.polyfit(time_array, diff, 1)
    residual = diff - np.polyval(coeffs, time_array)
    assert np.max(np.abs(residual)) < 1e-6 * np.max(np.abs(diff))
