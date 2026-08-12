"""
Tests for the shared beam-feedback base class.

The two methods under test both ask "does this rf station carry a cavity
feedback?".  Neither may answer that question with the main-harmonic-only
accessor ``get_main_harmonic_cavity_feedback``: a cavity feedback may be
attached to a NON-main harmonic of a `MultiHarmonicRFStation`, and the
main-harmonic slot is then ``None`` while a feedback is regulating.
"""

import unittest
import warnings

import numpy as np

from blond import (
    MultiHarmonicRFStation,
    Numpy64Bit,
    SingleHarmonicRFStation,
    StaticProfile,
    backend,
)
from blond.physics.feedbacks.beam_feedback import BeamFeedbackBase
from blond.physics.feedbacks.cavity_feedback import (
    IQCavityFeedbackTimingClass,
)

HARMONIC = 5
STATION_VOLTAGE = 5e6
INITIAL_ANTENNA_VOLTAGE = 30.0e6
CURRENT_THRES = 0.5


class _MinimalBeamControl(BeamFeedbackBase):
    """
    Smallest concrete beam feedback, for the guards only.

    `BeamFeedbackBase` is abstract on the two per-turn measurement
    hooks; neither is exercised here, because the tests call
    ``cavity_sum_phase`` and
    ``check_main_rf_stations_with_cavity_feedback`` directly instead of
    tracking a simulation.
    """

    def update_beam_attributes(self, beam):
        """
        Do nothing.

        Parameters
        ----------
        beam
            Unused.
        """

    def update_frequency_correction(self, beam):
        """
        Do nothing.

        Parameters
        ----------
        beam
            Unused.
        """


def _make_profile() -> StaticProfile:
    """
    Build the profile the feedbacks and beam controls share.

    Returns
    -------
    profile
        A static profile; never tracked, so its window is arbitrary.
    """
    return StaticProfile.from_cutoff(0, 1e-9, 5e9)


def _make_cavity_feedback(harmonic_index: int = 0):
    """
    Build a real IQ cavity feedback for a given harmonic slot.

    Parameters
    ----------
    harmonic_index
        Harmonic slot the feedback regulates.

    Returns
    -------
    feedback
        An `IQCavityFeedbackTimingClass`, not attached to any station.
    """
    return IQCavityFeedbackTimingClass(
        profile=_make_profile(),
        R_over_Q=0.0,
        Q_L=100.0,
        generator_current_bias=0.0,
        n_cavities=1,
        initial_voltage=INITIAL_ANTENNA_VOLTAGE,
        n_rf_periods_per_coarse_grid=1,
        harmonic_index=harmonic_index,
    )


def _make_two_harmonic_station(
    feedback_slot: int | None = None,
) -> MultiHarmonicRFStation:
    """
    Build a two-harmonic station, optionally with one cavity feedback.

    Parameters
    ----------
    feedback_slot
        Harmonic slot to attach a cavity feedback to; ``None`` attaches
        none.  Slot 0 is the main harmonic, slot 1 is not.

    Returns
    -------
    rf_station
        A station at harmonics ``[HARMONIC, 2 * HARMONIC]``.
    """
    rf_station = MultiHarmonicRFStation(
        n_harmonics=2,
        main_harmonic_idx=0,
        voltage=np.array([STATION_VOLTAGE, 0.5 * STATION_VOLTAGE]),
        phi_rf=np.array([0.0, 0.0]),
        harmonic=np.array([HARMONIC, 2 * HARMONIC], dtype=float),
    )
    if feedback_slot is not None:
        rf_station.attach_cavity_feedback(
            _make_cavity_feedback(harmonic_index=feedback_slot),
            harmonic_index=feedback_slot,
        )
    return rf_station


def _make_single_harmonic_station(
    with_feedback: bool = False,
) -> SingleHarmonicRFStation:
    """
    Build a single-harmonic station, optionally with a cavity feedback.

    Parameters
    ----------
    with_feedback
        Whether to attach a cavity feedback to the only slot.

    Returns
    -------
    rf_station
        A station at harmonic ``HARMONIC``.
    """
    return SingleHarmonicRFStation(
        voltage=STATION_VOLTAGE,
        phi_rf=0.0,
        harmonic=HARMONIC,
        cavity_feedback=_make_cavity_feedback() if with_feedback else None,
    )


def _make_beam_control(rf_stations) -> _MinimalBeamControl:
    """
    Build a beam control bound to ``rf_stations``.

    ``cavities`` is the list `GlobalFeedback.configure` installs at
    simulation setup; it is set directly so the guards can be reached
    without building and tracking a full simulation.

    Parameters
    ----------
    rf_stations
        RF stations the beam control acts on.

    Returns
    -------
    beam_control
        A beam control whose ``_main_cavities`` is already resolved.
    """
    backend.change_backend(Numpy64Bit)
    beam_control = _MinimalBeamControl(profile=_make_profile())
    beam_control.cavities = list(rf_stations)
    beam_control.update_main_rf_stations()
    return beam_control


class TestCavitySumPhaseGuard(unittest.TestCase):
    """``cavity_sum_phase`` must not skip its own NotImplementedError."""

    def assert_message_names_both_apis(self, message: str) -> None:
        """
        Assert the raise still explains which two APIs are involved.

        Parameters
        ----------
        message
            The `NotImplementedError` message.
        """
        self.assertIn("I_BEAM_COARSE", message)
        self.assertIn("antenna_voltage_coarse_grid", message)
        self.assertIn("open design task", message)

    def test_raises_for_feedback_on_a_non_main_harmonic(self):
        # The silent skip: the feedback sits on slot 1, so the
        # main-harmonic accessor returns None and the phase loop used to
        # run as if no cavity feedback existed at all.
        beam_control = _make_beam_control(
            [_make_two_harmonic_station(feedback_slot=1)]
        )

        with self.assertRaises(NotImplementedError) as ctx:
            beam_control.cavity_sum_phase(CURRENT_THRES)

        self.assert_message_names_both_apis(str(ctx.exception))

    def test_raises_for_feedback_on_the_main_harmonic(self):
        beam_control = _make_beam_control(
            [_make_two_harmonic_station(feedback_slot=0)]
        )

        with self.assertRaises(NotImplementedError) as ctx:
            beam_control.cavity_sum_phase(CURRENT_THRES)

        self.assert_message_names_both_apis(str(ctx.exception))

    def test_raises_for_feedback_on_a_single_harmonic_station(self):
        beam_control = _make_beam_control(
            [_make_single_harmonic_station(with_feedback=True)]
        )

        with self.assertRaises(NotImplementedError) as ctx:
            beam_control.cavity_sum_phase(CURRENT_THRES)

        self.assert_message_names_both_apis(str(ctx.exception))

    def test_raises_when_only_a_later_station_carries_a_feedback(self):
        # The loop must not stop at the first feedback-free station.
        beam_control = _make_beam_control(
            [
                _make_two_harmonic_station(),
                _make_two_harmonic_station(feedback_slot=1),
            ]
        )

        with self.assertRaises(NotImplementedError):
            beam_control.cavity_sum_phase(CURRENT_THRES)

    def test_silent_without_any_cavity_feedback(self):
        # Anti-regression: the LHC and SPS beam controls call
        # cavity_sum_phase unconditionally every turn, and a station
        # without a cavity feedback is a normal, supported setup.
        for label, rf_station in (
            ("multi-harmonic", _make_two_harmonic_station()),
            ("single-harmonic", _make_single_harmonic_station()),
        ):
            with self.subTest(station=label):
                beam_control = _make_beam_control([rf_station])

                self.assertIsNone(beam_control.cavity_sum_phase(CURRENT_THRES))


class TestMixedCavityFeedbackWarning(unittest.TestCase):
    """``check_main_rf_stations_with_cavity_feedback`` warns on mixtures."""

    WARNING_FRAGMENT = "do not have a cavity feedback model"

    def caught_warnings(self, rf_stations) -> list[str]:
        """
        Run the check and collect the warnings it emits.

        Parameters
        ----------
        rf_stations
            RF stations the beam control acts on.

        Returns
        -------
        messages
            The warning messages the check emitted.
        """
        beam_control = _make_beam_control(rf_stations)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            beam_control.check_main_rf_stations_with_cavity_feedback()
        return [
            str(warning.message)
            for warning in caught
            if self.WARNING_FRAGMENT in str(warning.message)
        ]

    def test_no_warning_when_no_station_has_a_feedback(self):
        messages = self.caught_warnings(
            [_make_two_harmonic_station(), _make_two_harmonic_station()]
        )

        self.assertEqual(messages, [])

    def test_no_warning_when_every_station_has_a_feedback(self):
        # The second station's feedback is on a NON-main harmonic, which
        # the main-harmonic-only accessor reported as "no feedback".
        messages = self.caught_warnings(
            [
                _make_two_harmonic_station(feedback_slot=0),
                _make_two_harmonic_station(feedback_slot=1),
            ]
        )

        self.assertEqual(messages, [])

    def test_no_warning_for_a_single_station_with_a_feedback(self):
        messages = self.caught_warnings(
            [_make_two_harmonic_station(feedback_slot=1)]
        )

        self.assertEqual(messages, [])

    def test_warns_when_only_some_stations_have_a_feedback(self):
        messages = self.caught_warnings(
            [
                _make_two_harmonic_station(feedback_slot=1),
                _make_two_harmonic_station(),
            ]
        )

        self.assertEqual(len(messages), 1)


if __name__ == "__main__":
    unittest.main()
