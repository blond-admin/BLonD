"""Cavity-feedback observation-class and diagnostic-flag tests.

The IQ-cavity-feedback timing-class grid tests and the RFCenterSegment
tests moved to test_rf_center_grid.py and test_rf_center_segment.py when
the grid builder and value class were split into their own modules.
"""

import numpy as np
from _pytest import unittest

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

HARMONIC = 5
CIRCUMFERENCE = 5
STATION_VOLTAGE = 5e6
INITIAL_ANTENNA_VOLTAGE = 30.0e6


class TestIQCavityFeedbackObservationClass(unittest.TestCase):
    pass


def _run_one_turn(**feedback_kwargs) -> IQCavityFeedbackTimingClass:
    """
    Track one turn of a single-section ring with a cavity feedback.

    The cavity is undriven and beam-loading free (``R_over_Q = 0``,
    ``generator_current_bias = 0``), so the antenna voltage simply decays
    from ``initial_voltage``. That is enough to tell a real readout from
    the neutral one: the relative voltage correction is
    ``|V_ant| / station voltage``, i.e. ~6 here, and only the neutral
    readout writes exactly 1.

    Parameters
    ----------
    **feedback_kwargs
        Extra keyword arguments for the
        :class:`IQCavityFeedbackTimingClass` under test.

    Returns
    -------
    feedback
        The tracked feedback, after one turn.
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

    feedback = IQCavityFeedbackTimingClass(
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
    simulation.run_simulation(beam, n_turns=1)
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


class TestDiagnosticsDoNotDisableTheFeedback:
    """The diagnostic flags must not switch the physics off."""

    def test_default_applies_a_real_correction(self) -> None:
        feedback = _run_one_turn()

        assert not _is_neutral_readout(feedback)

    def test_diagnostics_still_apply_a_real_correction(self) -> None:
        # ``debug=True`` used to short-circuit _track and write the
        # neutral readout, i.e. turning diagnostics on silently turned
        # the feedback off.
        feedback = _run_one_turn(debug=True)

        assert not _is_neutral_readout(feedback)

    def test_grid_validation_still_applies_a_real_correction(self) -> None:
        feedback = _run_one_turn(validate_grid_each_turn=True)

        assert not _is_neutral_readout(feedback)

    def test_diagnostic_flags_leave_the_readout_bit_identical(self) -> None:
        # Both flags are observation-only, so the tracked result must be
        # bit-for-bit the default one.
        reference = _run_one_turn()
        diagnosed = _run_one_turn(debug=True, validate_grid_each_turn=True)

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

    def test_grid_only_mode_applies_no_correction(self) -> None:
        # The one mode that does switch the physics off; it now says so
        # in its name.
        feedback = _run_one_turn(grid_only_no_correction=True)

        assert _is_neutral_readout(feedback)
        # The grid is still built -- that is the point of the mode.
        assert len(feedback._rf_centers) > 0
