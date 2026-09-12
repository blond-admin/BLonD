# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unit tests of the per-station beam phase loop and its RF-phase channel.

The tracked physics -- a phase step leaving the beam-induced field in
place, walking the design-locked drive off, and the loop damping a launch
error -- lives in
``tests/unittests/physics/feedbacks/accelerators/mucol/test_pi_feedback_full_tracking.py``.
"""

import unittest
from unittest.mock import Mock

import numpy as np

from blond import SingleHarmonicRFStation, StaticProfile
from blond.core.beam.beams import ProbeBeam
from blond.physics.feedbacks.station_phase_loop import (
    StationPhaseLoop,
    StationPhaseLoopRecord,
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

    def test_offset_adds_to_the_actual_phase(self):
        """The loop's offset moves ``phi_rf``, not the design phase."""
        station = self._station()
        station.phi_rf_loop = 0.1
        self.assertAlmostEqual(station.phi_rf, 0.4, places=15)
        self.assertEqual(station.phi_rf_design, 0.3)


class _BeamStub:
    """The two things the loop reads off a beam: its identity and its dt."""

    def __init__(self, dt):
        self._dt = Mock()
        self._dt.mean = Mock(return_value=float(np.mean(dt)))
        self.n_macroparticles_partial = len(dt)


class TestStationPhaseLoopElement(unittest.TestCase):
    """The element measures, remembers, delays and writes the offset."""

    OMEGA = 2.0 * np.pi * 1.0e9
    REFERENCE = 0.5

    def _loop(self, gain=0.2, delay=1, record=None):
        station = SingleHarmonicRFStation(
            voltage=1.0e6, phi_rf=0.0, harmonic=100
        )
        station.omega_rf_design = self.OMEGA
        beam = _BeamStub(dt=[0.0])
        record = StationPhaseLoopRecord() if record is None else record
        loop = StationPhaseLoop(
            station=station,
            beam=beam,
            reference_phase=self.REFERENCE,
            gain=gain,
            delay_stations=delay,
            turn_fraction=0.25,
            record=record,
        )
        return loop, station, beam, record

    def _pass(self, loop, beam, error):
        """Pass ``beam`` with a centroid ``error`` [rad] off the reference."""
        dt = (self.REFERENCE + error) / self.OMEGA
        beam._dt.mean = Mock(return_value=dt)
        loop.track(beam)

    def test_first_passage_records_but_has_nothing_to_act_on(self):
        loop, station, beam, record = self._loop(gain=0.2, delay=1)
        self._pass(loop, beam, 0.1)
        self.assertEqual(len(record.errors), 1)
        self.assertAlmostEqual(record.errors[0], 0.1, places=12)
        self.assertEqual(record.corrections[0], 0.0)
        self.assertEqual(station.phi_rf_loop, 0.0)

    def test_correction_uses_the_delayed_measurement(self):
        loop, station, beam, record = self._loop(gain=0.2, delay=1)
        self._pass(loop, beam, 0.1)
        self._pass(loop, beam, -0.3)
        # Acts on the PREVIOUS passage's error, with the documented sign.
        self.assertAlmostEqual(station.phi_rf_loop, -0.2 * 0.1, places=12)
        self.assertAlmostEqual(record.corrections[1], -0.02, places=12)
        self._pass(loop, beam, 0.0)
        self.assertAlmostEqual(station.phi_rf_loop, -0.2 * -0.3, places=12)

    def test_zero_delay_acts_on_its_own_measurement(self):
        loop, station, beam, _ = self._loop(gain=0.5, delay=0)
        self._pass(loop, beam, 0.1)
        self.assertAlmostEqual(station.phi_rf_loop, -0.05, places=12)

    def test_error_is_wrapped_into_the_principal_range(self):
        loop, _, beam, record = self._loop(gain=0.0, delay=0)
        self._pass(loop, beam, 2.0 * np.pi + 0.1)
        self.assertAlmostEqual(record.errors[0], 0.1, places=9)

    def test_turns_are_stamped_by_passage_count(self):
        loop, _, beam, record = self._loop()
        for _ in range(3):
            self._pass(loop, beam, 0.0)
        np.testing.assert_allclose(record.turns, [0.25, 1.25, 2.25])

    def test_other_beams_and_probes_are_ignored(self):
        loop, station, beam, record = self._loop(gain=0.5, delay=0)
        other = _BeamStub(dt=[1.0e-9])
        loop.track(other)
        loop.track(Mock(spec=ProbeBeam))
        self.assertEqual(len(record.errors), 0)
        self.assertEqual(station.phi_rf_loop, 0.0)
        self._pass(loop, beam, 0.1)
        self.assertEqual(len(record.errors), 1)

    def test_an_empty_beam_is_skipped(self):
        loop, _, beam, record = self._loop()
        beam.n_macroparticles_partial = 0
        loop.track(beam)
        self.assertEqual(len(record.errors), 0)

    def test_negative_delay_is_refused(self):
        with self.assertRaises(ValueError):
            self._loop(delay=-1)

    def test_record_arrays_are_time_ordered(self):
        loop, _, beam, record = self._loop(gain=0.1, delay=1)
        for error in (0.1, 0.2, 0.3):
            self._pass(loop, beam, error)
        turns, errors, corrections = record.as_arrays()
        self.assertTrue(np.all(np.diff(turns) > 0.0))
        np.testing.assert_allclose(errors, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(corrections, [0.0, -0.01, -0.02])


if __name__ == "__main__":
    unittest.main()
