"""Value-class + derived-array tests for RFCenterSegment -- moved from
test_cavity_feedback.py alongside the module extraction
(blond/physics/feedbacks/rf_center_segment.py)."""

import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytest
from _pytest import unittest

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    MagneticCyclePerTurn,
    MagneticCyclePerTurnAllRFStations,
    Numpy64Bit,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    backend,
    mu_minus,
    mu_plus,
)
from blond.generals.distributed.distributed_array import DistributedArray
from blond.physics.feedbacks.cavity_feedback import (
    IQCavityFeedbackTimingClass,
)
from blond.physics.feedbacks.rf_center_segment import RFCenterSegment
from blond.physics.impedances.solvers import (
    SingleTurnResonatorConvolutionSolver,
)

DEBUG_PLOTTING = False


class TestRFCenterSegment:
    """
    Unit tests for the RFCenterSegment value class and the derived flat arrays.

    These replace the length-bookkeeping checks that used to be reconstructed
    inside the timing-class integration tests from ``fdbk.rf_centers_lengths``.
    The flat ``rf_centers`` / ``rf_centers_lengths`` are now *derived* from the
    segment list, so length consistency is guaranteed by construction and only
    needs a focused check here.
    """

    # ---- field guards (__post_init__) -----------------------------------
    def test_rejects_non_positive_omega(self):
        with pytest.raises(ValueError, match="omega must be > 0"):
            RFCenterSegment(
                omega=0.0, duration=1.0, residual=0.1, centers=np.zeros(0)
            )

    def test_rejects_negative_duration(self):
        with pytest.raises(ValueError, match="duration must be >= 0"):
            RFCenterSegment(
                omega=1.0, duration=-1.0, residual=0.0, centers=np.zeros(0)
            )

    def test_rejects_multidimensional_centers(self):
        with pytest.raises(ValueError, match="centers must be 1-D"):
            RFCenterSegment(
                omega=1.0,
                duration=1.0,
                residual=0.1,
                centers=np.zeros((2, 2)),
            )

    def test_rejects_residual_outside_duration_for_nonempty(self):
        # A populated segment's residual is the leftover after its last centre,
        # so it must lie within [0, duration].
        with pytest.raises(ValueError, match="residual"):
            RFCenterSegment(
                omega=1.0,
                duration=1.0,
                residual=2.0,
                centers=np.array([0.25, 0.5, 0.75]),
            )

    def test_allows_carried_residual_for_empty_segment(self):
        # An empty segment legitimately carries the *previous* segment's
        # residual, which may exceed its own (near-zero) duration; that must be
        # accepted (no bound check when there are no centres).
        seg = RFCenterSegment(
            omega=1.0, duration=1e-6, residual=5.0, centers=np.zeros(0)
        )
        assert len(seg) == 0

    def test_len_matches_centers(self):
        seg = RFCenterSegment(
            omega=1.0,
            duration=1.0,
            residual=0.1,
            centers=np.array([0.1, 0.4, 0.7]),
        )
        assert len(seg) == 3

    # ---- derived flat arrays (_append_segment / _rebuild / _validate) ----
    @staticmethod
    def _bare_feedback():
        profile = StaticProfile.from_cutoff(0, 1e-9, 5e9)
        return IQCavityFeedbackTimingClass(
            profile=profile,
            n_rf_periods_per_coarse_grid=1,
            R_over_Q=0,
            Q_L=1e6,
            generator_current_bias=0,
            n_cavities=1,
        )

    def test_flat_arrays_derived_from_segments(self):
        fdbk = self._bare_feedback()
        seg_a = RFCenterSegment(
            omega=2.0, duration=1.0, residual=0.2, centers=np.array([0.3, 0.8])
        )
        seg_b = RFCenterSegment(
            omega=3.0,
            duration=0.6,
            residual=0.1,
            centers=np.array([0.1, 0.3, 0.5]),
        )
        fdbk._clear_segments()
        fdbk._append_segment(seg_a)
        fdbk._append_segment(seg_b)

        # rf_centers is the concatenation; rf_centers_lengths the per-segment
        # lengths -- both purely derived, so they cannot desync.
        np.testing.assert_allclose(
            fdbk._rf_centers, np.array([0.3, 0.8, 0.1, 0.3, 0.5])
        )
        assert list(fdbk._rf_centers_lengths) == [2, 3]
        # The per-turn consistency guard must pass.
        fdbk._validate_grid()

    def test_clear_segments_empties_derived_arrays(self):
        fdbk = self._bare_feedback()
        fdbk._append_segment(
            RFCenterSegment(
                omega=2.0,
                duration=1.0,
                residual=0.2,
                centers=np.array([0.3, 0.8]),
            )
        )
        fdbk._clear_segments()
        assert len(fdbk._rf_centers) == 0
        assert len(fdbk._rf_centers_lengths) == 0
        fdbk._validate_grid()

    def test_empty_segment_contributes_zero_length(self):
        # A segment shorter than one coarse step holds no centre; it must still
        # count as a (zero-length) entry in rf_centers_lengths so the segment
        # count matches the reverse+forward bookkeeping.
        fdbk = self._bare_feedback()
        fdbk._clear_segments()
        fdbk._append_segment(
            RFCenterSegment(
                omega=2.0, duration=1e-9, residual=1e-9, centers=np.zeros(0)
            )
        )
        fdbk._append_segment(
            RFCenterSegment(
                omega=2.0,
                duration=1.0,
                residual=0.1,
                centers=np.array([0.4, 0.9]),
            )
        )
        assert list(fdbk._rf_centers_lengths) == [0, 2]
        assert len(fdbk._rf_centers) == 2
        fdbk._validate_grid()

    def test_validate_grid_detects_direct_mutation(self):
        # If a code path mutates the derived arrays directly instead of going
        # through _append_segment, the guard must catch the desync.
        fdbk = self._bare_feedback()
        fdbk._append_segment(
            RFCenterSegment(
                omega=2.0,
                duration=1.0,
                residual=0.1,
                centers=np.array([0.4, 0.9]),
            )
        )
        fdbk._rf_centers_lengths = np.array(
            [5], dtype=int
        )  # corrupt on purpose
        with pytest.raises(AssertionError, match="out of sync"):
            fdbk._validate_grid()

    def test_segments_no_overlap_in_absolute_time(self):
        # The invariant that used to be reconstructed in the accelerating
        # integration test: offsetting each segment's (segment-relative)
        # centres by the cumulative durations of the segments before it, the
        # absolute centre times must be strictly increasing (no overlap or
        # duplicated span between the reverse and forward segments).
        fdbk = self._bare_feedback()
        fdbk._clear_segments()
        fdbk._append_segment(
            RFCenterSegment(
                omega=2.0,
                duration=1.0,
                residual=0.2,
                centers=np.array([0.2, 0.6]),
            )
        )
        fdbk._append_segment(
            RFCenterSegment(
                omega=2.0,
                duration=1.0,
                residual=0.2,
                centers=np.array([0.2, 0.6]),  # relative grid repeats
            )
        )
        durations = [seg.duration for seg in fdbk._segments]
        offsets = np.concatenate(([0.0], np.cumsum(durations[:-1])))
        absolute_centers = np.concatenate(
            [seg.centers + offsets[i] for i, seg in enumerate(fdbk._segments)]
        )
        # Raw values coincide across segments, but absolute times must not.
        assert np.all(np.diff(absolute_centers) > 0)


T_RF_BOUNDARY = 1e-9
OMEGA_BOUNDARY = 2 * np.pi / T_RF_BOUNDARY
# Three deliberately different residuals: what the PREVIOUS turn ended on,
# what the reverse segment of this turn ended on, and what the forward
# segment (generated last, hence the value left on the live host scalar)
# ended on. On an accelerating multi-section ring they differ by
# delta(t_rf) / 2; here they are pulled apart so the difference is visible.
RESIDUAL_PREVIOUS_TURN = 0.30 * T_RF_BOUNDARY
RESIDUAL_REVERSE_SEGMENT = 0.40 * T_RF_BOUNDARY
RESIDUAL_FORWARD_SEGMENT = 0.50 * T_RF_BOUNDARY
CENTERS_PER_SEGMENT = 4


class TestSegmentBoundaryStep:
    """
    The coarse step across a segment boundary is a PER-SEGMENT quantity.

    ``rf_centers`` are segment-LOCAL times, so the step into the first cell
    of segment ``j`` is that cell's local time plus the *preceding* segment's
    residual (the unfilled tail between the preceding segment's last centre
    and its end). ``_track`` generates the whole per-turn grid before walking
    any of it, so the live
    ``_residual_time_last_rf_centers_calculation`` scalar holds the
    LAST-GENERATED (forward) segment's residual by the time the loop reads
    it -- a value from the *future* of the walk. These tests pin the correct
    source for both the scalar reference loop and its vectorised twin.
    """

    @staticmethod
    def _two_segment_feedback():
        """
        Reverse+forward grid whose residuals deliberately all differ.

        Returns
        -------
        fdbk
            A feedback whose ``_segments`` hold two 4-centre segments with
            different residuals, with the host state set exactly as
            ``_track`` leaves it before the grid is walked.
        """
        fdbk = TestRFCenterSegment._bare_feedback()
        centers = (
            np.arange(CENTERS_PER_SEGMENT) * T_RF_BOUNDARY
            + 0.5 * T_RF_BOUNDARY
        )
        fdbk._clear_segments()
        fdbk._append_segment(
            RFCenterSegment(
                omega=OMEGA_BOUNDARY,
                duration=CENTERS_PER_SEGMENT * T_RF_BOUNDARY,
                residual=RESIDUAL_REVERSE_SEGMENT,
                centers=centers.copy(),
            )
        )
        fdbk._append_segment(
            RFCenterSegment(
                omega=OMEGA_BOUNDARY,
                duration=CENTERS_PER_SEGMENT * T_RF_BOUNDARY,
                residual=RESIDUAL_FORWARD_SEGMENT,
                centers=centers.copy(),
            )
        )
        # What _track leaves behind: the LAST generated (forward) residual on
        # the live scalar, and the previous turn's tail as the carry.
        fdbk._residual_time_last_rf_centers_calculation = (
            RESIDUAL_FORWARD_SEGMENT
        )
        fdbk._residual_time_carried_into_turn = RESIDUAL_PREVIOUS_TURN
        # Not the first centre ever tracked, so the residual branch is taken.
        fdbk._last_rf_centers_entry = 0.0
        return fdbk

    @staticmethod
    def _tracked_phases(fdbk, start_index, end_index):
        """
        Per-cell ``omega * delta_t`` the reference loop actually stepped with.

        Compared in the loop's own units (the product it passes to
        ``cavity_response``) rather than dividing ``omega`` back out, which
        would cost a ULP and blunt the bit-exact comparison.

        Parameters
        ----------
        fdbk
            Feedback whose grid should be walked.
        start_index
            First ``rf_centers`` index of the segment.
        end_index
            One past the last ``rf_centers`` index of the segment.

        Returns
        -------
        list
            ``omega_input * delta_t`` [rad] of every cell the loop advanced.
        """
        seen = []
        fdbk.cavity_response = lambda omega_times_T_s, **kwargs: seen.append(
            omega_times_T_s
        )
        fdbk._circuit_track_cells_python(
            OMEGA_BOUNDARY, True, start_index, end_index
        )
        return seen

    def test_segment_boundary_step_vectorised(self):
        # Into the forward segment: its local first centre plus the REVERSE
        # segment's tail -- not the forward segment's own (live-scalar) tail.
        fdbk = self._two_segment_feedback()
        delta_t = fdbk._coarse_step_sizes(
            OMEGA_BOUNDARY, CENTERS_PER_SEGMENT, 2 * CENTERS_PER_SEGMENT
        )
        assert delta_t[0] == (
            fdbk._rf_centers[CENTERS_PER_SEGMENT] + RESIDUAL_REVERSE_SEGMENT
        )

    def test_segment_boundary_step_reference_loop(self):
        fdbk = self._two_segment_feedback()
        phases = self._tracked_phases(
            fdbk, CENTERS_PER_SEGMENT, 2 * CENTERS_PER_SEGMENT
        )
        assert phases[0] == OMEGA_BOUNDARY * (
            fdbk._rf_centers[CENTERS_PER_SEGMENT] + RESIDUAL_REVERSE_SEGMENT
        )

    def test_turn_boundary_step_vectorised(self):
        # Into the FIRST segment of the turn: the step crosses the turn
        # boundary, so it must use the residual the PREVIOUS turn ended on.
        fdbk = self._two_segment_feedback()
        delta_t = fdbk._coarse_step_sizes(
            OMEGA_BOUNDARY, 0, CENTERS_PER_SEGMENT
        )
        assert delta_t[0] == (fdbk._rf_centers[0] + RESIDUAL_PREVIOUS_TURN)

    def test_turn_boundary_step_reference_loop(self):
        fdbk = self._two_segment_feedback()
        phases = self._tracked_phases(fdbk, 0, CENTERS_PER_SEGMENT)
        assert phases[0] == OMEGA_BOUNDARY * (
            fdbk._rf_centers[0] + RESIDUAL_PREVIOUS_TURN
        )

    def test_scalar_and_vectorised_paths_agree(self):
        # The kernel-vs-python byte-identity pin depends on both paths
        # deriving the boundary step the same way.
        fdbk = self._two_segment_feedback()
        for start_index in (0, CENTERS_PER_SEGMENT):
            end_index = start_index + CENTERS_PER_SEGMENT
            np.testing.assert_array_equal(
                OMEGA_BOUNDARY
                * fdbk._coarse_step_sizes(
                    OMEGA_BOUNDARY, start_index, end_index
                ),
                np.array(self._tracked_phases(fdbk, start_index, end_index)),
            )

    def test_hand_built_grid_without_segments_keeps_live_scalar(self):
        # Byte-compat guard for the direct-call tests: a grid built by hand
        # (no segments at all) must keep reading the live host scalar, which
        # is what those tests set up and what they were pinned against.
        fdbk = self._two_segment_feedback()
        fdbk._clear_segments()
        fdbk._rf_centers = (
            np.arange(1, 2 * CENTERS_PER_SEGMENT + 1) * T_RF_BOUNDARY
        )
        fdbk._rf_centers_lengths = np.array(
            [CENTERS_PER_SEGMENT, CENTERS_PER_SEGMENT], dtype=int
        )
        delta_t = fdbk._coarse_step_sizes(
            OMEGA_BOUNDARY, CENTERS_PER_SEGMENT, 2 * CENTERS_PER_SEGMENT
        )
        assert delta_t[0] == (
            fdbk._rf_centers[CENTERS_PER_SEGMENT] + RESIDUAL_FORWARD_SEGMENT
        )
