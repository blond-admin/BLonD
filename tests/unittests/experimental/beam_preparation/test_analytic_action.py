"""Tests for the analytic action J(H) building blocks."""

import unittest

import numpy as np

from blond.core.backends.backend import backend
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
from blond.experimental.beam_preparation.analytic_well_cut import (
    cut_potential_well,
)
from blond.generals.cupy.no_cupy_import import copy_to_cpu

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
    total_voltage = VOLTAGE * backend.sin(OMEGA_RF * time_array + phi_rf)
    well = rf_potential_well(
        time_array,
        total_voltage,
        charge=1.0,
        t_rev=T_REV,
        eta_0=eta_0,
    )
    return time_array, well


class TestActionFromPotentialWell(unittest.TestCase):
    def test_action_matches_harmonic_oscillator(self):
        # For H = a*dE^2 + b*t^2 the orbit is an ellipse:
        # J = H/(2*sqrt(a*b)).
        a = EOM_FACTOR_DE
        b = 1.0e18  # eV/s^2, arbitrary curvature
        time_array = backend.linspace(-1e-9, 1e-9, 4001, dtype=backend.float)
        well = b * time_array**2
        sorted_h, sorted_j = action_from_potential_well(
            time_array, well, eom_factor_dE=a
        )
        expected = sorted_h / (2.0 * float(np.sqrt(a * b)))
        # Compare over the well-resolved central range (skip the edges).
        sel = (sorted_h > 0.05 * sorted_h.max()) & (
            sorted_h < 0.95 * sorted_h.max()
        )
        # Measured max rel. error at n=4001 is ~8.7e-5 (O(h^2)).
        np.testing.assert_allclose(
            copy_to_cpu(sorted_j[sel]),
            copy_to_cpu(expected[sel]),
            rtol=5e-4,
        )

    def test_action_is_monotonic_in_hamiltonian(self):
        time_array, well = _lhc_bucket()
        _, sorted_j = action_from_potential_well(
            time_array, well, eom_factor_dE=EOM_FACTOR_DE
        )
        self.assertTrue(backend.all(backend.diff(sorted_j) >= -1e-12))
        self.assertGreaterEqual(float(sorted_j[0]), 0.0)

    def test_separatrix_action_matches_bucket_area(self):
        # Stationary single-RF bucket:
        # area = 8*dE_max/omega_rf = 2*pi*J_sep.
        time_array, well = _lhc_bucket(n_points=4000)
        _, sorted_j = action_from_potential_well(
            time_array, well, eom_factor_dE=EOM_FACTOR_DE
        )
        potential_well_amplitude = float(well.max() - well.min())
        deltaE_max = np.sqrt(potential_well_amplitude / EOM_FACTOR_DE)
        expected_action = 4.0 * deltaE_max / (np.pi * OMEGA_RF)
        # Measured rel. error at n=4000 is ~2.6e-8 (O(h^2)).
        np.testing.assert_allclose(
            float(sorted_j[-1]), expected_action, rtol=1e-6
        )

    def test_below_transition_separatrix_action(self):
        # Below transition, with the BLonD 2 convention phi_rf = pi, the
        # whole chain works and the separatrix action matches the closed
        # form. Also pins |eta_0| in the kinetic factor (no NaN chain).
        factor = calc_eom_factor_dE(-ETA_0, BETA, TOTAL_ENERGY)
        self.assertGreater(factor, 0.0)
        time_array, well = _lhc_bucket(
            n_points=4000, phi_rf=np.pi, eta_0=-ETA_0
        )
        _, sorted_j = action_from_potential_well(
            time_array, well, eom_factor_dE=factor
        )
        self.assertFalse(backend.any(backend.isnan(sorted_j)))
        deltaE_max = np.sqrt(float(well.max() - well.min()) / factor)
        expected_action = 4.0 * deltaE_max / (np.pi * OMEGA_RF)
        np.testing.assert_allclose(
            float(sorted_j[-1]), expected_action, rtol=1e-6
        )

    def test_action_on_nonuniform_grid(self):
        # Mildly non-uniform monotone grid over exactly one bucket: the
        # x-based integration must still match the closed form.
        rf_period = 2.0 * np.pi / OMEGA_RF
        u = backend.linspace(0.0, 1.0, 4001, dtype=backend.float)
        time_array = rf_period * (
            u + 0.15 * backend.sin(2.0 * np.pi * u) / (2 * np.pi)
        )
        total_voltage = VOLTAGE * backend.sin(OMEGA_RF * time_array)
        well = rf_potential_well(
            time_array,
            total_voltage,
            charge=1.0,
            t_rev=T_REV,
            eta_0=ETA_0,
        )
        _, sorted_j = action_from_potential_well(
            time_array, well, eom_factor_dE=EOM_FACTOR_DE
        )
        deltaE_max = np.sqrt(float(well.max() - well.min()) / EOM_FACTOR_DE)
        expected_action = 4.0 * deltaE_max / (np.pi * OMEGA_RF)
        np.testing.assert_allclose(
            float(sorted_j[-1]), expected_action, rtol=1e-5
        )

    def test_uncut_well_raises(self):
        time_array = bucket_time_array(
            OMEGA_RF, n_points=2000, dt_margin_fraction=0.4
        )
        total_voltage = VOLTAGE * backend.sin(OMEGA_RF * time_array)
        well = rf_potential_well(
            time_array,
            total_voltage,
            charge=1.0,
            t_rev=T_REV,
            eta_0=ETA_0,
        )
        with self.assertRaises(ValueError):
            action_from_potential_well(
                time_array, well, eom_factor_dE=EOM_FACTOR_DE
            )

    def test_shape_mismatch_raises(self):
        time_array, well = _lhc_bucket(n_points=300)
        with self.assertRaises(AssertionError):
            action_from_potential_well(
                time_array, well[:-1], eom_factor_dE=EOM_FACTOR_DE
            )

    def test_verbose_and_plot_smoke(self):
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


class TestHamiltonianFromEmittance(unittest.TestCase):
    def setUp(self):
        time_array, well = _lhc_bucket()
        self.sorted_h, self.sorted_j = action_from_potential_well(
            time_array, well, eom_factor_dE=EOM_FACTOR_DE
        )

    def test_round_trip(self):
        # Pick a level well inside the bucket, round-trip via emittance.
        index = len(self.sorted_h) // 2
        hamiltonian_0 = float(self.sorted_h[index])
        emittance = 2.0 * np.pi * float(self.sorted_j[index])
        recovered = hamiltonian_from_emittance(
            emittance, self.sorted_h, self.sorted_j
        )
        np.testing.assert_allclose(recovered, hamiltonian_0, rtol=1e-6)

    def test_zero_emittance_maps_to_bottom_of_well(self):
        bottom = hamiltonian_from_emittance(0.0, self.sorted_h, self.sorted_j)
        np.testing.assert_allclose(bottom, float(self.sorted_h[0]))

    def test_emittance_beyond_bucket_raises(self):
        bucket_area = 2.0 * np.pi * float(self.sorted_j[-1])
        with self.assertRaises(ValueError):
            hamiltonian_from_emittance(
                1.01 * bucket_area, self.sorted_h, self.sorted_j
            )


class TestActionGrid(unittest.TestCase):
    def setUp(self):
        time_array, well = _lhc_bucket(n_points=300)
        self.sorted_h, self.sorted_j = action_from_potential_well(
            time_array, well, eom_factor_dE=EOM_FACTOR_DE
        )
        _, _, self.hamilton = hamiltonian_grid(
            time_array,
            well,
            eom_factor_dE=EOM_FACTOR_DE,
            n_points_deltaE=200,
        )

    def test_shape_matches_hamiltonian(self):
        action_2D = action_grid(self.hamilton, self.sorted_h, self.sorted_j)
        self.assertEqual(action_2D.shape, self.hamilton.shape)

    def test_outside_bucket_is_infinite(self):
        # H above the largest tabulated level -> inf.
        outside = action_grid(
            backend.array([self.sorted_h[-1] * 10], dtype=backend.float),
            self.sorted_h,
            self.sorted_j,
        )
        self.assertEqual(float(outside[0]), np.inf)

    def test_inside_bucket_is_finite_and_non_negative(self):
        action_2D = action_grid(self.hamilton, self.sorted_h, self.sorted_j)
        inside = self.hamilton <= self.sorted_h[-1]
        self.assertTrue(backend.all(backend.isfinite(action_2D[inside])))
        self.assertTrue(backend.all(action_2D[inside] >= 0.0))


class TestSplitWell(unittest.TestCase):
    """The split-well route legacy BLonD 2 tolerated.

    Legacy tolerated wells with several minima (e.g. split by an induced
    potential): it warned, took the deepest, and its zero-padded J
    summed the islands below the inner separatrix. The port's route for
    this is the ``single_bucket_tolerance`` knob; the zero-padded
    integral keeps J well-behaved on the split well.
    """

    def setUp(self):
        self.time_array = bucket_time_array(
            OMEGA_RF, n_points=4000, dt_margin_fraction=0.2
        )
        total_voltage = VOLTAGE * (
            backend.sin(OMEGA_RF * self.time_array)
            + 0.8 * backend.sin(2.0 * OMEGA_RF * self.time_array)
        )
        self.well = rf_potential_well(
            self.time_array,
            total_voltage,
            charge=1.0,
            t_rev=T_REV,
            eta_0=ETA_0,
        )

    def _loosened_action(self):
        time_cut, well_cut = cut_potential_well(
            self.time_array, self.well, single_bucket_tolerance=0.12
        )
        return action_from_potential_well(
            time_cut,
            well_cut,
            eom_factor_dE=EOM_FACTOR_DE,
            single_bucket_tolerance=0.12,
        )

    def test_default_tolerance_rejects_split_well(self):
        # Inner separatrix is ~5 % of the amplitude.
        with self.assertRaises(ValueError):
            cut_potential_well(self.time_array, self.well)

    def test_loosened_tolerance_gives_finite_monotone_action(self):
        _, sorted_j = self._loosened_action()
        self.assertTrue(backend.all(backend.isfinite(sorted_j)))
        # Monotone up to boundary-cell discretization noise.
        max_backstep = max(float(backend.max(-backend.diff(sorted_j))), 0.0)
        self.assertLess(max_backstep, 1e-4 * float(sorted_j[-1]))

    def test_outer_bucket_is_larger_than_single_harmonic_bucket(self):
        _, sorted_j = self._loosened_action()
        time_single, well_single = _lhc_bucket(n_points=4000)
        _, sorted_j_single = action_from_potential_well(
            time_single, well_single, eom_factor_dE=EOM_FACTOR_DE
        )
        self.assertGreater(float(sorted_j[-1]), float(sorted_j_single[-1]))

    def test_allow_inner_buckets_matches_loosened_tolerance(self):
        # The explicit allow_inner_buckets route (default tolerance)
        # warns and yields the same result as the loosened route.
        _, sorted_j = self._loosened_action()
        with self.assertWarnsRegex(UserWarning, "inner"):
            time_cut_2, well_cut_2 = cut_potential_well(
                self.time_array, self.well, allow_inner_buckets=True
            )
        with self.assertWarnsRegex(UserWarning, "inner"):
            _, sorted_j_2 = action_from_potential_well(
                time_cut_2,
                well_cut_2,
                eom_factor_dE=EOM_FACTOR_DE,
                allow_inner_buckets=True,
            )
        np.testing.assert_allclose(
            float(sorted_j_2[-1]), float(sorted_j[-1]), rtol=1e-9
        )


if __name__ == "__main__":
    unittest.main()
