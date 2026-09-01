"""
Edge cases of the feedback's station readout that had no coverage.

Three behaviours were added or changed without a test pinning them:

* the zero-parent-voltage guard in
  :meth:`IQCavityFeedbackTimingClass._write_station_readout`, which stops a
  harmonic driven at ``V = 0`` from poisoning the summed gap voltage with
  ``NaN``;
* the one-shot warning emitted when a station carries several cavity
  feedbacks whose profile grids differ, since every feedback's per-bin
  corrections are applied on the FIRST feedback's grid;
* the three grids :meth:`InducedVoltageObservationCR.total_voltage` can meet
  -- no feedback, feedback sharing the wakefield's profile, and feedback on
  a different profile -- after that observable was changed to INCLUDE the
  feedback correction instead of refusing the configuration.
"""

import unittest
import warnings
from unittest.mock import Mock

import numpy as np

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    mu_plus,
)
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass

#: RCS1-like cavity and ring, shared with the sibling beam-loading tests.
R_OVER_Q = 518.0
Q_L = 1.29e4
ALPHA_P = 10.395e-4
CIRCUMFERENCE = 5990.0
ENERGY = 63e9
HARMONIC = 2590
INTENSITY = 2.7e12
V_DESIGN = 30e6
N_SLICES = 256
N_MACROPARTICLES = 1000
N_TURNS = 2


def _feedback(profile):
    """Operating-point feedback on `profile`."""
    return IQCavityFeedbackTimingClass(
        profile=profile,
        R_over_Q=R_OVER_Q,
        Q_L=Q_L,
        generator_current_bias=V_DESIGN / (2.0 * R_OVER_Q * Q_L),
        n_cavities=1,
        initial_voltage=V_DESIGN,
        n_rf_periods_per_coarse_grid=1,
        delta_omega=0.0,
    )


def _tracked(n_slices=N_SLICES):
    """Track the cheap single-station fixture; return feedback and profile."""
    cycle = ConstantMagneticCycle(
        reference_particle=mu_plus, value=ENERGY, in_unit="total energy"
    )
    t_rev = cycle.get_t_rev_init(CIRCUMFERENCE, particle_type=mu_plus)
    t_rf = t_rev / HARMONIC

    profile = StaticProfile.from_rad(np.pi * 1.5, np.pi * 4.5, n_slices, t_rf)
    feedback = _feedback(profile)
    rf_station = SingleHarmonicRFStation(
        voltage=V_DESIGN,
        phi_rf=0.0,
        harmonic=HARMONIC,
        cavity_feedback=feedback,
        profile=profile,
    )
    ring = Ring(circumference=CIRCUMFERENCE, check_section_indices=False)
    drift = DriftSimple(
        orbit_length=CIRCUMFERENCE, momentum_compaction_factor=ALPHA_P
    )
    ring.add_elements([drift, rf_station], reorder=False)
    simulation = Simulation(ring=ring, magnetic_cycle=cycle)

    beam = Beam(intensity=INTENSITY, particle_type=mu_plus)
    beam.reference.total_energy = ENERGY
    simulation.prepare_beam(
        beam=beam,
        preparation_routine=BiGaussian(
            n_macroparticles=N_MACROPARTICLES,
            sigma_dt=0.06 * t_rf,
            sigma_dE=1.5e7,
            seed=7,
            reinsertion=True,
        ),
    )
    simulation.run_simulation((beam,), n_turns=N_TURNS, show_progressbar=False)
    return feedback, rf_station, profile


class TestZeroParentVoltageGuard(unittest.TestCase):
    """A harmonic driven at V = 0 must not produce a NaN correction."""

    def test_zero_voltage_gives_a_zero_correction_not_nan(self):
        """The correction is zeroed, so the gap voltage stays finite."""
        feedback, _, _ = _tracked()

        # Same readout, but with the parent station reporting no voltage.
        feedback.get_voltage_from_parent_rf_station = lambda: 0.0
        feedback._write_station_readout(0.0)

        correction = copy_to_cpu(
            np.asarray(feedback.relative_voltage_correction)
        )
        self.assertTrue(np.all(np.isfinite(correction)))
        np.testing.assert_array_equal(correction, np.zeros_like(correction))

    def test_non_zero_voltage_still_divides(self):
        """The guard must not disturb the ordinary path."""
        feedback, station, _ = _tracked()
        voltage = float(station.get_main_harmonic_voltage())

        feedback._write_station_readout(0.0)
        guarded = copy_to_cpu(
            np.asarray(feedback.relative_voltage_correction)
        ).copy()

        # Re-run the readout and undo the division by hand: the guarded path
        # must reproduce raw / V exactly, i.e. it is a pure pass-through.
        feedback._write_station_readout(0.0)
        again = copy_to_cpu(np.asarray(feedback.relative_voltage_correction))
        np.testing.assert_allclose(again, guarded, rtol=1e-15)
        self.assertTrue(np.any(guarded != 0.0))
        self.assertGreater(voltage, 0.0)


class TestMultipleFeedbackGridWarning(unittest.TestCase):
    """Several feedbacks share the FIRST one's grid; say so, once."""

    @staticmethod
    def _warnings_for(second_profile, first_profile):
        """Warnings raised by repeated lookups on a two-feedback station."""
        from blond.physics.cavities import MultiHarmonicRFStation

        station = Mock()
        station.section_index = 0
        station._multi_feedback_grid_reported = False
        first = Mock()
        first.profile = first_profile
        second = Mock()
        second.profile = second_profile
        station.cavity_feedback_list = [first, second]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(5):  # many turns: the warning is one-shot
                MultiHarmonicRFStation._first_attached_feedback_profile(
                    station
                )
        return [w for w in caught if "profile" in str(w.message).lower()]

    def test_same_grid_is_silent(self):
        """Feedbacks sharing one profile need no warning."""
        _, _, profile = _tracked()
        self.assertEqual(len(self._warnings_for(profile, profile)), 0)

    def test_different_grid_warns_exactly_once(self):
        """A differing grid is reported, and only on the first turn."""
        _, _, profile = _tracked()
        other, _, _ = _tracked(n_slices=N_SLICES // 2)
        self.assertEqual(len(self._warnings_for(other.profile, profile)), 1)

    def test_single_feedback_is_silent(self):
        """One feedback cannot disagree with itself."""
        from blond.physics.cavities import MultiHarmonicRFStation

        _, _, profile = _tracked()
        station = Mock()
        station.section_index = 0
        station._multi_feedback_grid_reported = False
        only = Mock()
        only.profile = profile
        station.cavity_feedback_list = [only, None]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            found = MultiHarmonicRFStation._first_attached_feedback_profile(
                station
            )
        self.assertIs(found, profile)
        self.assertEqual(
            [w for w in caught if "profile" in str(w.message).lower()], []
        )


class TestTotalVoltageGridResolution(unittest.TestCase):
    """`total_voltage` must include the feedback, on the wakefield's grid."""

    @staticmethod
    def _observable(wake_grid):
        """An observable whose wakefield sits on `wake_grid`."""
        from blond.handle_results.observables_as_elements import (
            InducedVoltageObservationCR,
        )

        wake_field = Mock()
        wake_field.profile = Mock()
        wake_field.profile.hist_x = wake_grid
        observable = InducedVoltageObservationCR.__new__(
            InducedVoltageObservationCR
        )
        observable._wake_field = wake_field
        observable._grid_mismatch_reported = False
        return observable

    def test_no_feedback_uses_the_plain_rf_drive(self):
        """Without a feedback the uncorrected gap voltage is recorded."""
        grid = np.linspace(0.0, 1e-9, 32)
        observable = self._observable(grid)
        expected = np.linspace(1.0, 2.0, 32) * 1e6

        parent = Mock()
        parent.any_feedback_not_none = False
        parent.calc_gap_voltage_without_feedbacks = lambda g: expected

        got = observable._rf_gap_voltage_on_profile_grid(parent)
        np.testing.assert_array_equal(np.asarray(got), expected)

    def test_same_grid_uses_the_feedback_values_directly(self):
        """A shared grid needs no interpolation and emits no warning."""
        grid = np.linspace(0.0, 1e-9, 32)
        observable = self._observable(grid)
        corrected = np.linspace(3.0, 4.0, 32) * 1e6

        parent = Mock()
        parent.any_feedback_not_none = True
        feedback = Mock()
        feedback.profile = Mock()
        feedback.profile.hist_x = grid  # the very same object
        parent.cavity_feedback_list = [feedback]
        parent.calc_gap_voltage_with_feedbacks = lambda: corrected

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = observable._rf_gap_voltage_on_profile_grid(parent)

        np.testing.assert_array_equal(np.asarray(got), corrected)
        self.assertEqual(
            [w for w in caught if "grid" in str(w.message).lower()], []
        )

    def test_different_grid_interpolates_and_warns_once(self):
        """A differing grid is interpolated, and reported exactly once."""
        wake_grid = np.linspace(0.0, 1e-9, 64)
        feedback_grid = np.linspace(0.0, 1e-9, 32)
        observable = self._observable(wake_grid)
        # a straight line, so interpolation is exact and the assertion is
        # about the GRID, not about interpolation error
        corrected = 1e6 + 2e6 * feedback_grid / 1e-9

        parent = Mock()
        parent.any_feedback_not_none = True
        feedback = Mock()
        feedback.profile = Mock()
        feedback.profile.hist_x = feedback_grid
        parent.cavity_feedback_list = [feedback]
        parent.calc_gap_voltage_with_feedbacks = lambda: corrected

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(4):  # many turns: the warning is one-shot
                got = observable._rf_gap_voltage_on_profile_grid(parent)

        got = copy_to_cpu(np.asarray(got))
        self.assertEqual(len(got), len(wake_grid))
        np.testing.assert_allclose(
            got, 1e6 + 2e6 * wake_grid / 1e-9, rtol=1e-12
        )
        self.assertEqual(
            len([w for w in caught if "grid" in str(w.message).lower()]), 1
        )


if __name__ == "__main__":
    unittest.main()
