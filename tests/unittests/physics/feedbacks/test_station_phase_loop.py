# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unit tests of the clocked, station-attached beam phase loop.

The tracked physics -- a phase step leaving the beam-induced field in
place, walking the design-locked drive off, and the loop damping a launch
error -- lives in
``tests/unittests/physics/feedbacks/accelerators/mucol/test_pi_feedback_full_tracking.py``.
"""

import unittest
from unittest.mock import Mock

import numpy as np

from blond import SingleHarmonicRFStation
from blond.physics.feedbacks.station_phase_loop import (
    StationPhaseLoop,
    StationPhaseLoopRecord,
    wrap_phase,
)


class TestPhiRfLoopEntersTheStationPhase(unittest.TestCase):
    """``phi_rf_loop`` is the third term of the station's actual RF phase."""

    def _station(self):
        return SingleHarmonicRFStation(voltage=1.0e6, phi_rf=0.3, harmonic=100)

    def test_default_is_zero_and_bit_neutral(self):
        """A station without a loop reads exactly design plus kick clock."""
        station = self._station()
        self.assertEqual(station.phi_rf_loop, 0.0)
        self.assertEqual(
            station.phi_rf, station.phi_rf_design + station.delta_phi_rf
        )

    def test_the_station_does_not_carry_the_loop(self):
        """The loop hangs off the cavity feedback that clocks it.

        The station holds only the offset the feedback writes; nothing
        reaches back from the station to the loop.
        """
        self.assertFalse(hasattr(self._station(), "phase_loop"))

    def test_offset_adds_to_the_actual_phase(self):
        """The loop's offset moves ``phi_rf``, not the design phase."""
        station = self._station()
        station.phi_rf_loop = 0.1
        self.assertAlmostEqual(station.phi_rf, 0.4, places=15)
        self.assertEqual(station.phi_rf_design, 0.3)


class TestStationPhaseLoop(unittest.TestCase):
    """The loop attaches to a station, records passages, samples in cells."""

    REFERENCE = 0.5
    INTERVAL = 4

    def _feedback(self, station=None):
        """A stand-in for the cavity feedback the loop attaches to.

        The loop is attached to, and clocked by, one cavity feedback, and
        never touches the feedback itself.  These tests exercise its own
        sample-and-hold arithmetic, so a stand-in carrying a free
        ``phase_loop`` slot and a parent station is enough.
        """
        return Mock(phase_loop=None, parent_rf_station=station)

    def _loop(self, gain=0.2, n_delay=1, record=None):
        feedback = self._feedback()
        loop = StationPhaseLoop(
            feedback=feedback,
            reference_phase=self.REFERENCE,
            gain=gain,
            n_delay=n_delay,
            record=record,
        )
        return loop, feedback

    def _offsets(self, loop, first_cell, n_cells, carried=0.0):
        return loop.offsets_for_cells(
            first_cell,
            n_cells,
            controller_update_interval=self.INTERVAL,
            carried=carried,
        )

    def test_attaches_to_its_feedback_once(self):
        loop, feedback = self._loop()
        self.assertIs(feedback.phase_loop, loop)
        self.assertIs(loop.feedback, feedback)
        with self.assertRaises(ValueError):
            StationPhaseLoop(feedback=feedback, reference_phase=0.0, gain=0.1)

    def test_the_station_is_the_feedbacks_own(self):
        """A loop regulates the station its own feedback drives.

        Attaching to the feedback makes a loop on a station that has none
        unrepresentable, rather than an error to raise: there is nothing
        to attach it to.
        """
        station = SingleHarmonicRFStation(
            voltage=1.0e6, phi_rf=0.0, harmonic=100
        )
        feedback = self._feedback(station=station)
        loop = StationPhaseLoop(
            feedback=feedback, reference_phase=self.REFERENCE, gain=0.2
        )
        self.assertIs(loop.station, station)

    def test_negative_latency_is_refused(self):
        with self.assertRaises(ValueError):
            self._loop(n_delay=-1)

    def test_measure_records_the_passage(self):
        loop, _ = self._loop()
        error = loop.measure(
            self.REFERENCE + 0.1, time=1.0e-6, cell=7, applied=-0.02
        )
        self.assertAlmostEqual(error, 0.1, places=12)
        self.assertEqual(loop.record.times, [1.0e-6])
        self.assertEqual(loop.record.cells, [7])
        self.assertAlmostEqual(loop.record.errors[0], 0.1, places=12)
        self.assertEqual(loop.record.corrections, [-0.02])

    def test_error_is_wrapped_into_the_principal_range(self):
        loop, _ = self._loop()
        loop.measure(
            self.REFERENCE + 2.0 * np.pi + 0.1, time=0.0, cell=0, applied=0.0
        )
        self.assertAlmostEqual(loop.record.errors[0], 0.1, places=9)
        self.assertAlmostEqual(
            wrap_phase(np.pi + 0.1), -np.pi + 0.1, places=12
        )

    def test_output_steps_on_samples_and_holds_between(self):
        """Cells 0..11 at interval 4: samples at 0, 4, 8; held elsewhere."""
        loop, _ = self._loop(gain=0.5, n_delay=1)
        loop.measure(self.REFERENCE + 0.2, time=0.0, cell=1, applied=0.0)
        offsets = self._offsets(loop, 0, 12, carried=0.3)
        # Sample 0: nothing at least 4 cells old -> 0.  Cells 1-3 hold it.
        # Sample 4: the measurement at cell 1 is 3 cells old -> not yet.
        # Sample 8: 7 cells old -> -0.5 * 0.2.
        np.testing.assert_allclose(offsets, [0.0] * 8 + [-0.1] * 4, atol=1e-15)
        # The carried value holds only until the first sample.
        offsets = self._offsets(loop, 1, 3, carried=0.3)
        np.testing.assert_allclose(offsets, [0.3, 0.3, 0.3])

    def test_latency_counts_controller_samples(self):
        loop, _ = self._loop(gain=1.0, n_delay=2)
        loop.measure(self.REFERENCE + 0.1, time=0.0, cell=0, applied=0.0)
        loop.measure(self.REFERENCE - 0.3, time=1.0, cell=6, applied=0.0)
        # Delay 2 samples = 8 cells: at cell 8 the first measurement (age
        # 8) qualifies, the second (age 2) does not; at cell 16 the second.
        offsets = self._offsets(loop, 0, 20)
        np.testing.assert_allclose(offsets[:8], 0.0)
        np.testing.assert_allclose(offsets[8:16], -0.1)
        np.testing.assert_allclose(offsets[16:], 0.3)

    def test_zero_latency_acts_from_the_next_sample(self):
        loop, _ = self._loop(gain=0.5, n_delay=0)
        loop.measure(self.REFERENCE + 0.2, time=0.0, cell=2, applied=0.0)
        offsets = self._offsets(loop, 0, 8)
        np.testing.assert_allclose(offsets, [0.0] * 4 + [-0.1] * 4)

    def test_any_bunch_is_a_passage(self):
        """Beam-blind: the record is the station's, whoever passed."""
        loop, _ = self._loop(gain=1.0, n_delay=0)
        loop.measure(self.REFERENCE + 0.1, time=0.0, cell=0, applied=0.0)
        loop.measure(self.REFERENCE + 0.2, time=0.5, cell=2, applied=0.0)
        self.assertEqual(len(loop.record.errors), 2)
        offsets = self._offsets(loop, 0, 8)
        # At sample 4 the newest entry is the second passage.
        np.testing.assert_allclose(offsets[4:], -0.2)

    def test_record_arrays_are_time_ordered(self):
        record = StationPhaseLoopRecord()
        loop, _ = self._loop(record=record)
        self.assertIs(loop.record, record)
        for error, time, cell in ((0.1, 0.0, 0), (0.3, 2.0, 8), (0.2, 1.0, 4)):
            loop.measure(
                self.REFERENCE + error, time=time, cell=cell, applied=-error
            )
        times, errors, corrections = record.as_arrays()
        np.testing.assert_allclose(times, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(errors, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(corrections, [-0.1, -0.2, -0.3])
        self.assertEqual(record.newest_at_or_before(5), 2)
        self.assertEqual(record.newest_at_or_before(8), 1)
        self.assertIsNone(record.newest_at_or_before(-1))


if __name__ == "__main__":
    unittest.main()
