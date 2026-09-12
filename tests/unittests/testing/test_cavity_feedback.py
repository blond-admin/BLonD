# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Tests for the diagnostic IQ cavity feedback of :mod:`blond.testing`.

The three diagnostic switches -- ``debug``, ``validate_grid_each_turn`` and
``grid_only_no_correction`` -- are consumed only by tests. They live on
:class:`~blond.testing.cavity_feedback.DiagnosticIQCavityFeedbackTimingClass`,
not on the production
:class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass`.
"""

import unittest
from unittest import mock

import numpy as np

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Numpy64Bit,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    backend,
    mu_plus,
)
from blond.generals.distributed.distributed_array import DistributedArray
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass
from blond.testing.cavity_feedback import (
    DiagnosticIQCavityFeedbackTimingClass,
)

HARMONIC = 5
CIRCUMFERENCE = 5
STATION_VOLTAGE = 5e6
INITIAL_ANTENNA_VOLTAGE = 30.0e6

DIAGNOSTIC_SWITCHES = (
    "debug",
    "validate_grid_each_turn",
    "grid_only_no_correction",
)
GRID_SNAPSHOTS = (
    "current_slice_elements_forward",
    "reference_time_after_backfill",
    "reference_energy_after_backfill",
    "current_beam_reference_time",
    "current_beam_reference_energy",
)


def _track(
    feedback_class: type[IQCavityFeedbackTimingClass],
    n_turns: int = 1,
    **feedback_kwargs,
) -> IQCavityFeedbackTimingClass:
    """
    Track a single-section ring with one cavity feedback.

    The cavity is undriven and beam-loading free (``R_over_Q = 0``,
    ``generator_current_bias = 0``), so the antenna voltage simply decays
    from ``initial_voltage``. That is enough to tell a real readout from
    the neutral one: the relative voltage correction is
    ``|V_ant| / station voltage``, i.e. ~6 here, and only the neutral
    readout writes exactly 1.

    Parameters
    ----------
    feedback_class
        Feedback class to build.
    n_turns
        Number of turns to track, one passage each.
    **feedback_kwargs
        Extra keyword arguments for ``feedback_class``.

    Returns
    -------
    feedback
        The tracked feedback.
    """
    backend.change_backend(Numpy64Bit)
    profile = StaticProfile.from_cutoff(0, 1e-9, 5e9)
    rf_station = SingleHarmonicRFStation(
        phi_rf=0.0, harmonic=HARMONIC, voltage=STATION_VOLTAGE
    )
    ring = Ring(circumference=CIRCUMFERENCE, check_section_indices=False)
    ring.add_elements(
        [
            rf_station,
            DriftSimple(CIRCUMFERENCE, momentum_compaction_factor=0),
        ]
    )

    beam = Beam(intensity=1, particle_type=mu_plus, is_counter_rotating=False)
    beam._dt = DistributedArray(np.zeros(5))
    beam._dE = DistributedArray(np.zeros(5))
    beam._ids = DistributedArray(np.arange(5))
    beam._flags = DistributedArray(np.zeros(5))

    feedback = feedback_class(
        profile=profile,
        n_rf_periods_per_coarse_grid=1,
        R_over_Q=0,
        Q_L=100,
        generator_current_bias=0,
        n_cavities=1,
        initial_voltage=INITIAL_ANTENNA_VOLTAGE,
        **feedback_kwargs,
    )
    rf_station.attach_cavity_feedback(feedback)

    simulation = Simulation(
        ring,
        ConstantMagneticCycle(
            reference_particle=mu_plus, value=63.0e9, in_unit="momentum"
        ),
    )
    simulation.run_simulation(beam, n_turns=n_turns)
    return feedback


def _is_neutral_readout(feedback: IQCavityFeedbackTimingClass) -> bool:
    """
    Whether the feedback wrote the no-correction readout.

    Parameters
    ----------
    feedback
        Feedback whose station readout to inspect.

    Returns
    -------
    is_neutral
        True when the station is told unit gain and zero phase, i.e. it
        kicks exactly as if no feedback were attached.
    """
    return bool(
        np.all(feedback.relative_voltage_correction == 1.0)
        and np.all(feedback.phase_correction == 0.0)
    )


class TestProductionFeedbackHasNoDiagnosticSwitches(unittest.TestCase):
    """The production feedback neither takes nor runs the diagnostics."""

    def test_constructor_rejects_every_switch(self):
        """Each switch is an unexpected keyword of the production class."""
        for switch in DIAGNOSTIC_SWITCHES:
            with self.subTest(switch=switch), self.assertRaises(TypeError):
                IQCavityFeedbackTimingClass(
                    profile=mock.Mock(StaticProfile),
                    R_over_Q=0.0,
                    Q_L=100.0,
                    generator_current_bias=0.0,
                    n_cavities=1,
                    **{switch: True},
                )

    def test_tracking_never_validates_the_grid(self):
        """The per-turn grid check is not part of production tracking."""
        with mock.patch.object(
            IQCavityFeedbackTimingClass, "_validate_grid"
        ) as validate_grid:
            _track(IQCavityFeedbackTimingClass, n_turns=2)
        self.assertEqual(validate_grid.call_count, 0)

    def test_tracking_records_no_grid_snapshot(self):
        """None of the inspection-only snapshots appear on the instance."""
        feedback = _track(IQCavityFeedbackTimingClass, n_turns=2)
        for snapshot in GRID_SNAPSHOTS:
            with self.subTest(snapshot=snapshot):
                self.assertFalse(hasattr(feedback, snapshot))


class TestDiagnosticsDoNotDisableTheFeedback(unittest.TestCase):
    """Only ``grid_only_no_correction`` switches the physics off."""

    def test_default_applies_a_real_correction(self):
        """With no switch set the readout is a real correction."""
        feedback = _track(DiagnosticIQCavityFeedbackTimingClass)
        self.assertFalse(_is_neutral_readout(feedback))

    def test_snapshots_still_apply_a_real_correction(self):
        """``debug=True`` records without writing the neutral readout."""
        # A single ``debug`` flag used to short-circuit _track and write
        # the neutral readout, i.e. turning diagnostics on silently turned
        # the feedback off.
        feedback = _track(DiagnosticIQCavityFeedbackTimingClass, debug=True)
        self.assertFalse(_is_neutral_readout(feedback))

    def test_grid_validation_still_applies_a_real_correction(self):
        """``validate_grid_each_turn=True`` does not stop the correction."""
        feedback = _track(
            DiagnosticIQCavityFeedbackTimingClass,
            validate_grid_each_turn=True,
        )
        self.assertFalse(_is_neutral_readout(feedback))

    def test_observing_switches_leave_the_production_result_unchanged(
        self,
    ):
        """Snapshots and validation reproduce production bit-for-bit."""
        reference = _track(IQCavityFeedbackTimingClass, n_turns=2)
        diagnosed = _track(
            DiagnosticIQCavityFeedbackTimingClass,
            n_turns=2,
            debug=True,
            validate_grid_each_turn=True,
        )
        np.testing.assert_array_equal(
            diagnosed.relative_voltage_correction,
            reference.relative_voltage_correction,
        )
        np.testing.assert_array_equal(
            diagnosed.phase_correction, reference.phase_correction
        )
        np.testing.assert_array_equal(
            diagnosed.antenna_voltage_coarse_grid,
            reference.antenna_voltage_coarse_grid,
        )

    def test_grid_only_mode_applies_no_correction(self):
        """The one switch that stops the physics still builds the grid."""
        feedback = _track(
            DiagnosticIQCavityFeedbackTimingClass,
            grid_only_no_correction=True,
        )
        self.assertTrue(_is_neutral_readout(feedback))
        self.assertGreater(len(feedback.rf_centers), 0)


class TestGridValidationSwitch(unittest.TestCase):
    """``validate_grid_each_turn`` checks the grid once per passage."""

    def _count_grid_validations(self, **feedback_kwargs) -> int:
        """
        Track two turns and count the grid validations.

        Parameters
        ----------
        **feedback_kwargs
            Switches for the diagnostic feedback.

        Returns
        -------
        n_validations
            Number of ``_validate_grid`` calls during tracking.
        """
        with mock.patch.object(
            DiagnosticIQCavityFeedbackTimingClass,
            "_validate_grid",
            autospec=True,
            side_effect=IQCavityFeedbackTimingClass._validate_grid,
        ) as validate_grid:
            _track(
                DiagnosticIQCavityFeedbackTimingClass,
                n_turns=2,
                **feedback_kwargs,
            )
        return validate_grid.call_count

    def test_validates_every_passage(self):
        """One validation per passage; one station, so one per turn."""
        self.assertEqual(
            self._count_grid_validations(validate_grid_each_turn=True), 2
        )

    def test_off_by_default(self):
        """Without the switch the grid is never validated."""
        self.assertEqual(self._count_grid_validations(), 0)


class TestGridSnapshotSwitch(unittest.TestCase):
    """``debug`` records the forward-projection snapshot every passage."""

    def test_debug_records_the_forward_slice(self):
        """The walked element slice starts at this station."""
        feedback = _track(DiagnosticIQCavityFeedbackTimingClass, debug=True)
        self.assertIn(
            feedback._parent_rf_station,
            feedback.current_slice_elements_forward,
        )

    def test_off_by_default(self):
        """Without the switch no snapshot is recorded."""
        feedback = _track(DiagnosticIQCavityFeedbackTimingClass, n_turns=2)
        for snapshot in GRID_SNAPSHOTS:
            with self.subTest(snapshot=snapshot):
                self.assertFalse(hasattr(feedback, snapshot))


if __name__ == "__main__":
    unittest.main()
