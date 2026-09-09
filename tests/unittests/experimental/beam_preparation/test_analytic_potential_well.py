"""Tests for the analytic RF potential-well building blocks."""

import unittest

import numpy as np
from scipy.integrate import cumulative_trapezoid

from blond.core.backends.backend import backend
from blond.experimental.beam_preparation.analytic_potential_well import (
    bucket_time_array,
    check_single_bucket_well,
    rf_potential_well,
)
from blond.generals.cupy.no_cupy_import import copy_to_cpu

# LHC-like main-harmonic parameters (450 GeV protons, h=35640, V=6 MV).
OMEGA_RF = 2518229887.224505
VOLTAGE = 6.0e6
T_REV = 8.892465516509709e-05
ETA_0 = 3.172867586042721e-04  # above transition (eta_0 > 0)
HARMONIC = OMEGA_RF * T_REV / (2.0 * np.pi)


def _single_harmonic(time_array, phi_rf=0.0):
    return VOLTAGE * backend.sin(OMEGA_RF * time_array + phi_rf)


class TestBucketTimeArray(unittest.TestCase):
    def test_span(self):
        time_array = bucket_time_array(
            OMEGA_RF, n_points=128, dt_margin_fraction=0.4
        )
        rf_period = 2.0 * np.pi / OMEGA_RF
        self.assertEqual(len(time_array), 128)
        self.assertLess(float(time_array[0]), 0.0)
        self.assertGreater(float(time_array[-1]), rf_period)


class TestRfPotentialWell(unittest.TestCase):
    def test_shape_and_min_at_zero(self):
        time_array = bucket_time_array(OMEGA_RF, n_points=5000)
        well = rf_potential_well(
            time_array,
            _single_harmonic(time_array),
            charge=1.0,
            t_rev=T_REV,
            eta_0=ETA_0,
        )
        self.assertEqual(well.shape, time_array.shape)
        np.testing.assert_allclose(float(well.min()), 0.0, atol=1e-12)

    def test_matches_closed_form(self):
        # For V*sin(w t): Phi(t) = (eom*V/w)*(cos(w t) - 1), min at 0.
        time_array = backend.linspace(
            0.0, 2.0 * np.pi / OMEGA_RF, 20000, dtype=backend.float
        )
        well = rf_potential_well(
            time_array,
            _single_harmonic(time_array),
            charge=1.0,
            t_rev=T_REV,
            eta_0=ETA_0,
        )
        eom = np.sign(ETA_0) * 1.0 / T_REV
        analytic = (eom * VOLTAGE / OMEGA_RF) * (
            backend.cos(OMEGA_RF * time_array) - 1.0
        )
        analytic = analytic - analytic.min()
        # Measured max rel. error at n=20000 is ~8e-9 (O(h^2)).
        np.testing.assert_allclose(
            copy_to_cpu(well),
            copy_to_cpu(analytic),
            rtol=1e-6,
            atol=1e-6 * float(well.max()),
        )

    def test_matches_legacy_cumtrapz_formula(self):
        # Exact parity with the BLonD 2 expression (same integration).
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
            eom * copy_to_cpu(total_voltage),
            x=copy_to_cpu(time_array),
            initial=0.0,
        )
        np.testing.assert_array_equal(copy_to_cpu(well), legacy)

    def test_eta_sign_flips_potential(self):
        time_array = bucket_time_array(OMEGA_RF, n_points=4000)
        total_voltage = _single_harmonic(time_array)
        common = dict(charge=1.0, t_rev=T_REV, subtract_min=False)
        above = rf_potential_well(
            time_array, total_voltage, eta_0=ETA_0, **common
        )
        below = rf_potential_well(
            time_array, total_voltage, eta_0=-ETA_0, **common
        )
        np.testing.assert_allclose(copy_to_cpu(above), copy_to_cpu(-below))

    def test_energy_gain_adds_linear_tilt_with_pinned_slope(self):
        # Subtracting a constant (synchronous) voltage integrates to a
        # linear ramp of slope sign(eta_0)*sign(charge)*gain/t_rev.
        energy_gain = 1.0e5
        time_array = bucket_time_array(OMEGA_RF, n_points=8000)
        total_voltage = _single_harmonic(time_array)
        common = dict(charge=1.0, t_rev=T_REV, eta_0=ETA_0, subtract_min=False)
        base = rf_potential_well(time_array, total_voltage, **common)
        tilted = rf_potential_well(
            time_array,
            total_voltage,
            energy_gain_per_turn=energy_gain,
            **common,
        )
        time_host = copy_to_cpu(time_array)
        diff = copy_to_cpu(tilted) - copy_to_cpu(base)
        coeffs = np.polyfit(time_host, diff, 1)
        residual = diff - np.polyval(coeffs, time_host)
        self.assertLess(np.max(np.abs(residual)), 1e-6 * np.max(np.abs(diff)))
        # Pin the sign and magnitude of the acceleration term (a sign
        # flip in the ported formula must fail here).
        expected_slope = np.sign(ETA_0) * np.sign(1.0) * energy_gain / T_REV
        np.testing.assert_allclose(coeffs[0], expected_slope, rtol=1e-6)

    def test_energy_gain_slope_sign_with_negative_charge(self):
        energy_gain = 1.0e5
        time_array = bucket_time_array(OMEGA_RF, n_points=8000)
        total_voltage = _single_harmonic(time_array)
        common = dict(
            charge=-1.0, t_rev=T_REV, eta_0=ETA_0, subtract_min=False
        )
        base = rf_potential_well(time_array, total_voltage, **common)
        tilted = rf_potential_well(
            time_array,
            total_voltage,
            energy_gain_per_turn=energy_gain,
            **common,
        )
        coeffs = np.polyfit(
            copy_to_cpu(time_array),
            copy_to_cpu(tilted) - copy_to_cpu(base),
            1,
        )
        expected_slope = np.sign(ETA_0) * np.sign(-1.0) * energy_gain / T_REV
        np.testing.assert_allclose(coeffs[0], expected_slope, rtol=1e-6)

    def test_amplitude_scales_with_charge(self):
        # Stationary single-harmonic well amplitude is |q|*V/(pi*h).
        time_array = bucket_time_array(OMEGA_RF, n_points=8000)
        # q = +2, above transition, phi_rf = 0 (stable phase mid-frame)
        well_q2 = rf_potential_well(
            time_array,
            _single_harmonic(time_array),
            charge=2.0,
            t_rev=T_REV,
            eta_0=ETA_0,
        )
        np.testing.assert_allclose(
            float(well_q2.max()),
            2.0 * VOLTAGE / (np.pi * HARMONIC),
            rtol=1e-6,
        )

    def test_negative_charge_uses_shifted_phase_convention(self):
        # q = -1, above transition: sign(eta*q) < 0, so the convention
        # is phi_rf = pi to keep the stable phase mid-frame.
        time_array = bucket_time_array(OMEGA_RF, n_points=8000)
        well_qm1 = rf_potential_well(
            time_array,
            _single_harmonic(time_array, phi_rf=np.pi),
            charge=-1.0,
            t_rev=T_REV,
            eta_0=ETA_0,
        )
        np.testing.assert_allclose(
            float(well_qm1.max()), VOLTAGE / (np.pi * HARMONIC), rtol=1e-6
        )
        # The convention holds: minimum sits mid-frame, not on an edge.
        n = len(time_array)
        self.assertGreater(int(well_qm1.argmin()), 0.25 * n)
        self.assertLess(int(well_qm1.argmin()), 0.75 * n)

    def test_shape_mismatch_raises(self):
        time_array = bucket_time_array(OMEGA_RF, n_points=100)
        with self.assertRaises(AssertionError):
            rf_potential_well(
                time_array,
                _single_harmonic(time_array)[:-1],
                charge=1.0,
                t_rev=T_REV,
                eta_0=ETA_0,
            )

    def test_verbose_and_plot_smoke(self):
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


class TestCheckSingleBucketWell(unittest.TestCase):
    def setUp(self):
        self.time_array = bucket_time_array(OMEGA_RF, n_points=2000)
        self.clean = rf_potential_well(
            self.time_array,
            _single_harmonic(self.time_array),
            charge=1.0,
            t_rev=T_REV,
            eta_0=ETA_0,
        )

    @staticmethod
    def _margined_well():
        # Margined frame: edges no longer reach the well maximum.
        time_margined = bucket_time_array(
            OMEGA_RF, n_points=2000, dt_margin_fraction=0.4
        )
        return rf_potential_well(
            time_margined,
            _single_harmonic(time_margined),
            charge=1.0,
            t_rev=T_REV,
            eta_0=ETA_0,
        )

    def test_clean_bucket_accepted(self):
        self.assertIs(check_single_bucket_well(self.clean), True)

    def test_margined_frame_rejected(self):
        margined = self._margined_well()
        self.assertIs(
            check_single_bucket_well(margined, raise_error=False), False
        )
        with self.assertRaises(ValueError):
            check_single_bucket_well(margined)

    def test_multi_bucket_span_rejected(self):
        # Multi-bucket span: interior maxima.
        rf_period = 2.0 * np.pi / OMEGA_RF
        time_3 = backend.linspace(
            0.0, 3.0 * rf_period, 6000, dtype=backend.float
        )
        three_buckets = rf_potential_well(
            time_3,
            _single_harmonic(time_3),
            charge=1.0,
            t_rev=T_REV,
            eta_0=ETA_0,
        )
        with self.assertRaises(ValueError):
            check_single_bucket_well(three_buckets)

    def test_below_transition_without_convention_cites_phi_rf(self):
        # Below transition with phi_rf=0 (convention violation): the
        # well minimum sits on a frame edge.
        below = rf_potential_well(
            self.time_array,
            _single_harmonic(self.time_array),
            charge=1.0,
            t_rev=T_REV,
            eta_0=-ETA_0,
        )
        with self.assertRaisesRegex(ValueError, "phi_rf"):
            check_single_bucket_well(below)

    def test_below_transition_with_convention_accepted(self):
        # With the convention (phi_rf = pi below transition), all is
        # well.
        below_convention = rf_potential_well(
            self.time_array,
            _single_harmonic(self.time_array, phi_rf=np.pi),
            charge=1.0,
            t_rev=T_REV,
            eta_0=-ETA_0,
        )
        self.assertIs(check_single_bucket_well(below_convention), True)

    def test_nan_well_rejected(self):
        # NaN compares False everywhere and would otherwise silently
        # pass the numeric checks.
        nan_well = backend.copy(self.clean)
        nan_well[100] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            check_single_bucket_well(nan_well)

    def test_too_few_samples_rejected(self):
        with self.assertRaises(ValueError):
            check_single_bucket_well(
                backend.array([0.0, 1.0], dtype=backend.float)
            )

    def test_accepts_sample_aligned_cut_of_tilted_well(self):
        # A separatrix cut done BLonD 2 style (sample-aligned, no
        # endpoint interpolation) of an accelerating well: the cut
        # edges mismatch by ~slope*dt (~4e-4 of the well amplitude at
        # n=20000) — the default tolerance must accept.
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
        with self.assertRaises(ValueError):
            check_single_bucket_well(well)
        # ...but its sample-aligned separatrix cut must pass: from the
        # interior (unstable) maximum to the first sample at the same
        # potential on the other side of the minimum.
        i_min = int(well.argmin())
        i_unstable = int(well[:i_min].argmax())
        level = well[i_unstable]
        i_right = i_min + int(backend.argmax(well[i_min:] >= level))
        cut = well[i_unstable : i_right + 1]
        self.assertIs(check_single_bucket_well(cut), True)

    def test_allow_inner_buckets_warns_instead_of_raising(self):
        # Double-harmonic well with two sub-wells (inner maximum ~5 %
        # of the amplitude) on a zero-margin frame: edges are the outer
        # barriers, so only the inner structure is at stake.
        time_array = bucket_time_array(OMEGA_RF, n_points=4000)
        split_voltage = VOLTAGE * (
            backend.sin(OMEGA_RF * time_array)
            + 0.8 * backend.sin(2.0 * OMEGA_RF * time_array)
        )
        split_well = rf_potential_well(
            time_array,
            split_voltage,
            charge=1.0,
            t_rev=T_REV,
            eta_0=ETA_0,
        )
        with self.assertRaises(ValueError):
            check_single_bucket_well(split_well)
        with self.assertWarnsRegex(UserWarning, "inner"):
            self.assertIs(
                check_single_bucket_well(split_well, allow_inner_buckets=True),
                True,
            )

    def test_allow_inner_buckets_does_not_relax_frame_edges(self):
        with self.assertRaises(ValueError):
            check_single_bucket_well(
                self._margined_well(), allow_inner_buckets=True
            )


if __name__ == "__main__":
    unittest.main()
