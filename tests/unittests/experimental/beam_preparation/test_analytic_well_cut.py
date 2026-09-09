"""Tests for the separatrix cut of analytic potential wells."""

import unittest

import numpy as np

from blond.core.backends.backend import backend
from blond.experimental.beam_preparation.analytic_action import (
    action_from_potential_well,
)
from blond.experimental.beam_preparation.analytic_hamiltonian import (
    calc_eom_factor_dE,
)
from blond.experimental.beam_preparation.analytic_potential_well import (
    bucket_time_array,
    check_single_bucket_well,
    rf_potential_well,
)
from blond.experimental.beam_preparation.analytic_well_cut import (
    cut_potential_well,
)

# LHC-like reference (450 GeV protons).
OMEGA_RF = 2518229887.224505
VOLTAGE = 6.0e6
T_REV = 8.892465516509709e-05
ETA_0 = 3.172867586042721e-04
BETA = 0.9999978262922387
TOTAL_ENERGY = 450.00104432e9
RF_PERIOD = 2.0 * np.pi / OMEGA_RF

EOM_FACTOR_DE = calc_eom_factor_dE(ETA_0, BETA, TOTAL_ENERGY)


def _well(time_array, phi_rf=0.0, eta_0=ETA_0, energy_gain_per_turn=0.0):
    total_voltage = VOLTAGE * backend.sin(OMEGA_RF * time_array + phi_rf)
    return rf_potential_well(
        time_array,
        total_voltage,
        charge=1.0,
        t_rev=T_REV,
        eta_0=eta_0,
        energy_gain_per_turn=energy_gain_per_turn,
    )


def _separatrix_action(well_cut):
    deltaE_max = backend.sqrt(
        (well_cut.max() - well_cut.min()) / EOM_FACTOR_DE
    )
    return 4.0 * deltaE_max / (np.pi * OMEGA_RF)


class TestCutPotentialWell(unittest.TestCase):
    def test_margined_frame_is_cut_to_one_bucket(self):
        time_array = bucket_time_array(
            OMEGA_RF, n_points=4000, dt_margin_fraction=0.4
        )
        well = _well(time_array)
        # The raw margined well violates the single-bucket contract...
        self.assertIs(check_single_bucket_well(well, raise_error=False), False)
        time_cut, well_cut = cut_potential_well(time_array, well)
        # ...the cut restores it and spans one RF period.
        np.testing.assert_allclose(float(well_cut.min()), 0.0, atol=1e-12)
        span = float(time_cut[-1] - time_cut[0])
        np.testing.assert_allclose(span, RF_PERIOD, rtol=2e-3)
        # J at the separatrix matches the closed form on the cut well.
        _, sorted_j = action_from_potential_well(
            time_cut, well_cut, eom_factor_dE=EOM_FACTOR_DE
        )
        np.testing.assert_allclose(
            float(sorted_j[-1]),
            float(_separatrix_action(well_cut)),
            rtol=1e-4,
        )

    def test_cut_is_idempotent_on_clean_bucket(self):
        time_array = bucket_time_array(OMEGA_RF, n_points=2000)
        well = _well(time_array)
        time_cut, well_cut = cut_potential_well(time_array, well)
        # An already-cut single bucket passes through (near-)unchanged.
        self.assertGreaterEqual(len(time_cut), len(time_array) - 2)
        np.testing.assert_allclose(
            float(well_cut.max()),
            float(well.max() - well.min()),
            rtol=1e-3,
        )

    def test_accelerating_well_cut_at_separatrix(self):
        time_array = bucket_time_array(OMEGA_RF, n_points=20000)
        well = _well(time_array, energy_gain_per_turn=1.0e5)
        # The raw tilted well violates the contract...
        self.assertIs(check_single_bucket_well(well, raise_error=False), False)
        time_cut, well_cut = cut_potential_well(time_array, well)
        # ...the cut runs between the unstable point and the
        # equal-potential crossing: both edges sit at the separatrix.
        amplitude = float(well_cut.max() - well_cut.min())
        self.assertGreaterEqual(
            float(well_cut[0]), float(well_cut.max()) - 1e-2 * amplitude
        )
        self.assertGreaterEqual(
            float(well_cut[-1]), float(well_cut.max()) - 1e-2 * amplitude
        )
        # The accelerating bucket is smaller than the stationary one.
        stationary_amplitude = VOLTAGE / (
            np.pi * OMEGA_RF * T_REV / (2.0 * np.pi)
        )
        self.assertLess(amplitude, stationary_amplitude)
        # The previously guarded downstream chain works end-to-end.
        _, sorted_j = action_from_potential_well(
            time_cut, well_cut, eom_factor_dE=EOM_FACTOR_DE
        )
        self.assertTrue(backend.all(backend.diff(sorted_j) >= -1e-12))
        self.assertTrue(backend.all(backend.isfinite(sorted_j)))

    def test_below_transition_with_convention_and_margin(self):
        time_array = bucket_time_array(
            OMEGA_RF, n_points=4000, dt_margin_fraction=0.2
        )
        well = _well(time_array, phi_rf=np.pi, eta_0=-ETA_0)
        time_cut, well_cut = cut_potential_well(time_array, well)
        # Minimum sits inside the cut, not on an edge.
        n = len(well_cut)
        self.assertGreater(int(well_cut.argmin()), 0.25 * n)
        self.assertLess(int(well_cut.argmin()), 0.75 * n)
        _, sorted_j = action_from_potential_well(
            time_cut, well_cut, eom_factor_dE=EOM_FACTOR_DE
        )
        np.testing.assert_allclose(
            float(sorted_j[-1]),
            float(_separatrix_action(well_cut)),
            rtol=1e-4,
        )

    def test_below_transition_without_convention_raises(self):
        time_array = bucket_time_array(OMEGA_RF, n_points=2000)
        well = _well(time_array, phi_rf=0.0, eta_0=-ETA_0)
        with self.assertRaisesRegex(ValueError, "phi_rf"):
            cut_potential_well(time_array, well)

    def test_deepest_bucket_spans_one_rf_period(self):
        time_array = backend.linspace(
            0.0, 3.0 * RF_PERIOD, 6000, dtype=backend.float
        )
        well = _well(time_array)
        time_cut, _ = cut_potential_well(time_array, well)
        span = float(time_cut[-1] - time_cut[0])
        np.testing.assert_allclose(span, RF_PERIOD, rtol=2e-2)

    def test_explicit_bucket_index_selects_the_right_bucket(self):
        time_array = backend.linspace(
            0.0, 3.0 * RF_PERIOD, 6000, dtype=backend.float
        )
        well = _well(time_array)
        # The first bucket's minimum sits at half an RF period, and the
        # cut does not leak into the second bucket.
        time_cut_0, well_cut_0 = cut_potential_well(
            time_array, well, bucket_index=0
        )
        time_of_minimum = float(time_cut_0[int(well_cut_0.argmin())])
        np.testing.assert_allclose(time_of_minimum, 0.5 * RF_PERIOD, rtol=5e-2)
        self.assertLessEqual(float(time_cut_0[-1]), 1.5 * RF_PERIOD)
        # The third physical bucket is addressable, centred at 2.5 T_rf.
        time_cut_2, well_cut_2 = cut_potential_well(
            time_array, well, bucket_index=2
        )
        time_of_minimum_2 = float(time_cut_2[int(well_cut_2.argmin())])
        np.testing.assert_allclose(
            time_of_minimum_2, 2.5 * RF_PERIOD, rtol=5e-2
        )

    def test_out_of_range_bucket_index_raises(self):
        time_array = backend.linspace(
            0.0, 3.0 * RF_PERIOD, 6000, dtype=backend.float
        )
        well = _well(time_array)
        with self.assertRaisesRegex(ValueError, "bucket_index"):
            cut_potential_well(time_array, well, bucket_index=99)

    def test_duplicate_bucket_detection_is_merged(self):
        # PotentialWellHelper reports the margined single bucket twice
        # (once per bounding maximum); after deduplication exactly one
        # physical bucket remains, so bucket_index=1 is out of range.
        time_array = bucket_time_array(
            OMEGA_RF, n_points=4000, dt_margin_fraction=0.4
        )
        well = _well(time_array)
        with self.assertRaisesRegex(ValueError, "bucket_index"):
            cut_potential_well(time_array, well, bucket_index=1)

    def test_subtract_min_false_keeps_offset(self):
        time_array = bucket_time_array(
            OMEGA_RF, n_points=2000, dt_margin_fraction=0.4
        )
        well = _well(time_array) + 5.0
        _, well_cut = cut_potential_well(time_array, well, subtract_min=False)
        self.assertGreater(float(well_cut.min()), 4.0)

    def test_verbose_and_plot_smoke(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        time_array = bucket_time_array(
            OMEGA_RF, n_points=512, dt_margin_fraction=0.4
        )
        well = _well(time_array)
        cut_potential_well(time_array, well, verbose=True, plot=True)
        plt.close("all")


class TestDoubleHarmonicSubWells(unittest.TestCase):
    """Characterizes the current multi-sub-well behaviour.

    A double harmonic with v2/v1 = 0.8 in phase gives one outer bucket
    enclosing two sub-wells.
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

    def _double_harmonic_well(self, second_harmonic_phase):
        return rf_potential_well(
            self.time_array,
            VOLTAGE
            * (
                backend.sin(OMEGA_RF * self.time_array)
                + 0.5
                * backend.sin(
                    2.0 * OMEGA_RF * self.time_array + second_harmonic_phase
                )
            ),
            charge=1.0,
            t_rev=T_REV,
            eta_0=ETA_0,
        )

    def test_inner_separatrix_is_rejected_by_default(self):
        # "deepest" resolves to the outer bucket, which contains the
        # inner separatrix -> loud rejection, no silent wrong physics.
        with self.assertRaisesRegex(ValueError, "local maximum"):
            cut_potential_well(self.time_array, self.well)

    def test_allow_inner_buckets_returns_outer_bucket_with_warning(self):
        with self.assertWarnsRegex(UserWarning, "inner"):
            _, outer_well = cut_potential_well(
                self.time_array, self.well, allow_inner_buckets=True
            )
        self.assertGreater(float(outer_well.max()), 55.0)  # ~56.6 eV

    def test_inner_sub_wells_are_addressable_and_equal(self):
        _, sub_well_1 = cut_potential_well(
            self.time_array, self.well, bucket_index=1
        )
        _, sub_well_2 = cut_potential_well(
            self.time_array, self.well, bucket_index=2
        )
        amplitude_1 = float(sub_well_1.max() - sub_well_1.min())
        amplitude_2 = float(sub_well_2.max() - sub_well_2.min())
        np.testing.assert_allclose(amplitude_1, amplitude_2, rtol=1e-3)
        self.assertLess(
            amplitude_1,
            0.1 * VOLTAGE / (np.pi * OMEGA_RF * T_REV / (2.0 * np.pi)),
        )

    def test_single_minimum_double_harmonic_wells_still_cut(self):
        for phase, name in ((np.pi, "steepened"), (0.0, "flattened")):
            with self.subTest(well=name):
                cut_potential_well(
                    self.time_array, self._double_harmonic_well(phase)
                )


if __name__ == "__main__":
    unittest.main()
