"""Cavity-feedback observation-class and diagnostic-flag tests.

The IQ-cavity-feedback timing-class grid tests and the RFCenterSegment
tests moved to test_rf_center_grid.py and test_rf_center_segment.py when
the grid builder and value class were split into their own modules.
"""

from unittest.mock import Mock

import numpy as np
import pytest
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


def _make_bare_feedback(**feedback_kwargs) -> IQCavityFeedbackTimingClass:
    """
    Build a feedback without a simulation, on a mocked profile.

    Parameters
    ----------
    **feedback_kwargs
        Overrides for the :class:`IQCavityFeedbackTimingClass`
        constructor defaults.

    Returns
    -------
    feedback
        A feedback instance not attached to any RF station.
    """
    params = {
        "profile": Mock(StaticProfile),
        "R_over_Q": 0.0,
        "Q_L": 100.0,
        "generator_current_bias": 0.0,
        "n_cavities": 1,
        "initial_voltage": INITIAL_ANTENNA_VOLTAGE,
        "n_rf_periods_per_coarse_grid": 1,
    }
    params.update(feedback_kwargs)
    return IQCavityFeedbackTimingClass(**params)


class _MultiHarmonicStationStub:
    """
    Parent-station stand-in that is not a SingleHarmonicRFStation.

    Carries per-harmonic arrays, the way a multi-harmonic station does,
    so the RF-parameter properties must index them by ``harmonic_index``.
    """

    def __init__(self):
        self.harmonic = np.array([3.0, 7.0])
        self.omega_rf = np.array([1.1e9, 2.2e9])
        self.delta_phi_rf = None


class TestMultiHarmonicParentResolution:
    """RF-parameter accessors on a multi-harmonic parent station."""

    def _feedback_with_stub_parent(self) -> IQCavityFeedbackTimingClass:
        feedback = _make_bare_feedback()
        # Assigned directly: set_parent_rf_station() rejects anything but
        # the real station classes, and only the isinstance dispatch of
        # _resolve_main_harmonic is under test here.
        feedback._parent_rf_station = _MultiHarmonicStationStub()
        feedback.harmonic_index = 1
        return feedback

    def test_harmonic_indexes_the_per_harmonic_array(self) -> None:
        feedback = self._feedback_with_stub_parent()

        assert feedback.harmonic == 7.0

    def test_resolve_main_harmonic_indexes_the_value(self) -> None:
        # omega_rf goes through _resolve_main_harmonic, which must pick
        # the tracked harmonic out of the per-harmonic array.
        feedback = self._feedback_with_stub_parent()

        assert feedback.omega_rf == 2.2e9

    def test_delta_phi_rf_is_zero_before_any_slip(self) -> None:
        # The parent's kick clock is None before the first passage; the
        # accessor must report "no accumulated slip", not crash.
        feedback = self._feedback_with_stub_parent()

        assert feedback.delta_phi_rf == 0.0

    def test_delta_phi_rf_indexes_the_per_harmonic_array(self) -> None:
        feedback = self._feedback_with_stub_parent()
        feedback._parent_rf_station.delta_phi_rf = np.array([0.5, 1.5])

        assert feedback.delta_phi_rf == 1.5


def _prepare_hand_built_grid(
    feedback: IQCavityFeedbackTimingClass, rf_centers
) -> None:
    """
    Install a hand-built coarse grid and size the IQ arrays for it.

    Parameters
    ----------
    feedback
        Feedback to prepare.
    rf_centers
        Coarse-grid centre times [s] to install.
    """
    feedback._rf_centers = np.asarray(rf_centers, dtype=float)
    feedback._rf_centers_lengths = np.array([len(feedback._rf_centers)])
    feedback.reset_arrays()


class TestCoarseCellStepSizing:
    """Per-cell step sizing of the coarse-grid recursion."""

    # Arbitrary segment frequency; rf_period = 2 pi / omega = 1 s makes
    # the centre times below easy to read.
    omega_input = 2 * np.pi

    def test_single_cell_first_turn_uses_own_coarse_step(self) -> None:
        # First centre ever tracked AND the only centre of the segment:
        # there is no next centre to diff against, so the step proxy must
        # fall back to this segment's own coarse step (n * t_rf).
        single = _make_bare_feedback()
        _prepare_hand_built_grid(single, [0.5])
        single._circuit_track_cells_python(
            self.omega_input, no_beam=True, start_index=0, end_index=1
        )

        # Reference: a two-centre grid whose spacing IS one coarse step;
        # its first cell diffs the two centres and must give the same
        # step, hence the same first-cell voltage.
        step = 2 * np.pi / self.omega_input
        reference = _make_bare_feedback()
        _prepare_hand_built_grid(reference, [0.5, 0.5 + step])
        reference._circuit_track_cells_python(
            self.omega_input, no_beam=True, start_index=0, end_index=2
        )

        assert (
            single.antenna_voltage_coarse_grid[0]
            == reference.antenna_voltage_coarse_grid[0]
        )
        # Non-vacuous: the cell decayed away from the initial voltage.
        assert single.antenna_voltage_coarse_grid[0] != 0
        assert single.antenna_voltage_coarse_grid[0] != INITIAL_ANTENNA_VOLTAGE

    def test_ulp_negative_step_is_clamped_to_a_coincident_skip(self) -> None:
        # A first-cell step a few ULPs below zero (a centre landing almost
        # exactly on the segment boundary) is floating-point noise, not an
        # ordering violation: it must be clamped to zero and handled as a
        # coincident point (warn + skip), not trip the hard assertion.
        feedback = _make_bare_feedback(
            R_over_Q=518.0, generator_current_bias=0.01
        )
        _prepare_hand_built_grid(feedback, [-1e-12, 1.0])
        feedback._last_rf_centers_entry = 123.0  # not the first turn

        with pytest.warns(
            UserWarning, match="double taking of rf_centers value"
        ):
            feedback._circuit_track_cells_python(
                self.omega_input, no_beam=True, start_index=0, end_index=2
            )

        # The clamped cell was skipped (kept the zeros prefill)...
        assert feedback.antenna_voltage_coarse_grid[0] == 0
        # ...and tracking completed: the next cell was still advanced.
        assert feedback.antenna_voltage_coarse_grid[1] != 0

    def test_coincident_centers_warn_and_skip(self) -> None:
        # KNOWN DEFERRED DEFECT (characterization, not endorsement): a
        # duplicated rf_centers value is skipped with a warning, but the
        # skipped cell keeps the zeros prefill, so the next cell
        # propagates from V = 0 instead of the carried voltage. This test
        # pins the current warn-and-skip behaviour; it must be updated
        # when the defect is fixed.
        r_over_q = 518.0
        bias = 0.01
        feedback = _make_bare_feedback(
            R_over_Q=r_over_q, generator_current_bias=bias
        )
        _prepare_hand_built_grid(feedback, [0.25, 0.25, 0.5])
        feedback._last_rf_centers_entry = 123.0  # not the first turn

        with pytest.warns(
            UserWarning, match="double taking of rf_centers value"
        ):
            feedback._circuit_track_cells_python(
                self.omega_input, no_beam=True, start_index=0, end_index=3
            )

        assert feedback.antenna_voltage_coarse_grid[0] != 0
        # The duplicated cell kept the zeros prefill...
        assert feedback.antenna_voltage_coarse_grid[1] == 0
        # ...so the following cell is the pure drive term on top of
        # V = 0 (the deferred defect described above), not a decayed
        # carried voltage: V = R/Q * omega*dt * I_gen with v_prev = 0.
        omega_times_dt = self.omega_input * 0.25
        assert feedback.antenna_voltage_coarse_grid[2] == (
            r_over_q * omega_times_dt * bias
        )

    def test_vectorised_first_turn_step_matches_reference_path(self) -> None:
        # _coarse_step_sizes is the vectorised twin of the reference loop
        # and must reproduce the first-turn single-cell fallback step.
        feedback = _make_bare_feedback()
        _prepare_hand_built_grid(feedback, [0.5])

        delta_t = feedback._coarse_step_sizes(self.omega_input, 0, 1)

        np.testing.assert_array_equal(delta_t, [2 * np.pi / self.omega_input])

    def test_degenerate_segment_defers_to_the_reference_path(self) -> None:
        # A coincident (zero) step makes the vectorised sizing return
        # None, and the kernel path must then fall back to the reference
        # loop -- reproducing its warning and its result exactly.
        def _degenerate_feedback() -> IQCavityFeedbackTimingClass:
            feedback = _make_bare_feedback(
                R_over_Q=518.0, generator_current_bias=0.01
            )
            _prepare_hand_built_grid(feedback, [0.25, 0.25, 0.5])
            feedback._last_rf_centers_entry = 123.0
            return feedback

        kernel = _degenerate_feedback()
        assert kernel._coarse_step_sizes(self.omega_input, 0, 3) is None

        with pytest.warns(
            UserWarning, match="double taking of rf_centers value"
        ):
            kernel._circuit_track_cells_kernel(
                self.omega_input, no_beam=True, start_index=0, end_index=3
            )

        reference = _degenerate_feedback()
        with pytest.warns(
            UserWarning, match="double taking of rf_centers value"
        ):
            reference._circuit_track_cells_python(
                self.omega_input, no_beam=True, start_index=0, end_index=3
            )

        np.testing.assert_array_equal(
            kernel.antenna_voltage_coarse_grid,
            reference.antenna_voltage_coarse_grid,
        )
        # Non-vacuous: something was tracked.
        assert np.any(kernel.antenna_voltage_coarse_grid != 0)

    def test_kernel_empty_span_is_a_no_op(self) -> None:
        # start_index == end_index: nothing to track. The kernel must
        # return before sizing the (empty) span -- proceeding would index
        # into a zero-length step array -- and leave the grids untouched.
        feedback = _make_bare_feedback()
        _prepare_hand_built_grid(feedback, [0.5, 1.5, 2.5])

        feedback._circuit_track_cells_kernel(
            self.omega_input, no_beam=True, start_index=2, end_index=2
        )

        np.testing.assert_array_equal(
            feedback.antenna_voltage_coarse_grid, np.zeros(3)
        )
