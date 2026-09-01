"""Value-class + derived-array tests for RFCenterSegment -- moved from
test_cavity_feedback.py alongside the module extraction
(blond/physics/feedbacks/rf_center_segment.py)."""

import unittest
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytest

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
from blond.physics.feedbacks.rf_center_segment import (
    PerTurnGridSpan,
    RFCenterSegment,
)
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

    def test_rejects_residual_outside_duration(self):
        # A segment's residual is the leftover after its last centre,
        # so it must lie within [0, duration].
        with pytest.raises(ValueError, match="residual"):
            RFCenterSegment(
                omega=1.0,
                duration=1.0,
                residual=2.0,
                centers=np.array([0.25, 0.5, 0.75]),
            )

    def test_rejects_empty_centers(self):
        # D2 retirement: an empty segment used to carry the previous
        # segment's residual through without adding its own duration to
        # the bridging coarse step. Such a segment is now unconstructible:
        # every segment must hold at least two coarse centres.
        with pytest.raises(ValueError, match="at least two") as excinfo:
            RFCenterSegment(
                omega=1.0, duration=1e-6, residual=5.0, centers=np.zeros(0)
            )
        message = str(excinfo.value)
        assert "1e-06" in message  # the segment duration is named
        assert "reduce n_rf_periods_per_coarse_grid" in message
        assert "fewer/longer sections" in message

    def test_rejects_single_center(self):
        # A single-centre segment is as degenerate as an empty one: the
        # coincidence-guard tolerance rf_centers[-1] - rf_centers[-2]
        # would cross the segment boundary (D1), so it must raise too.
        with pytest.raises(ValueError, match="at least two") as excinfo:
            RFCenterSegment(
                omega=1.0,
                duration=1.2,
                residual=0.7,
                centers=np.array([0.5]),
            )
        message = str(excinfo.value)
        assert "reduce n_rf_periods_per_coarse_grid" in message
        assert "fewer/longer sections" in message

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
        # duplicated span between the backfill and forward segments).
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
# what the backfill segment of this turn ended on, and what the forward
# segment (generated last, hence the value left on the live host scalar)
# ended on. On an accelerating multi-section ring they differ by
# delta(t_rf) / 2; here they are pulled apart so the difference is visible.
RESIDUAL_PREVIOUS_TURN = 0.30 * T_RF_BOUNDARY
RESIDUAL_BACKFILL_SEGMENT = 0.40 * T_RF_BOUNDARY
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
        Backfill+forward grid whose residuals deliberately all differ.

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
                residual=RESIDUAL_BACKFILL_SEGMENT,
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
        fdbk.cavity_response = lambda omega_times_dt, **kwargs: seen.append(
            omega_times_dt
        )
        fdbk._circuit_track_cells_python(
            OMEGA_BOUNDARY, True, start_index, end_index
        )
        return seen

    def test_segment_boundary_step_vectorised(self):
        # Into the forward segment: its local first centre plus the BACKFILL
        # segment's tail -- not the forward segment's own (live-scalar) tail.
        fdbk = self._two_segment_feedback()
        delta_t = fdbk._coarse_step_sizes(
            OMEGA_BOUNDARY, CENTERS_PER_SEGMENT, 2 * CENTERS_PER_SEGMENT
        )
        assert delta_t[0] == (
            fdbk._rf_centers[CENTERS_PER_SEGMENT] + RESIDUAL_BACKFILL_SEGMENT
        )

    def test_segment_boundary_step_reference_loop(self):
        fdbk = self._two_segment_feedback()
        phases = self._tracked_phases(
            fdbk, CENTERS_PER_SEGMENT, 2 * CENTERS_PER_SEGMENT
        )
        assert phases[0] == OMEGA_BOUNDARY * (
            fdbk._rf_centers[CENTERS_PER_SEGMENT] + RESIDUAL_BACKFILL_SEGMENT
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

    def test_per_turn_grid_span_lives_with_the_value_types(self):
        # PerTurnGridSpan is the per-turn coarse-grid value type, so it
        # belongs beside RFCenterSegment rather than in the (much larger)
        # feedback module, which only imports it.
        from blond.physics.feedbacks import cavity_feedback

        assert (
            PerTurnGridSpan.__module__
            == "blond.physics.feedbacks.rf_center_segment"
        )
        assert cavity_feedback.PerTurnGridSpan is PerTurnGridSpan

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


class TestGuardCellWidthInvariant:
    """
    D1: the coincidence-guard tolerance ``rf_centers[-1] - rf_centers[-2]``.

    The simultaneous-passage guard of the timing class compares arrival
    times against half a forward coarse-cell width, computed from the last
    two entries of the flat ``rf_centers``. Because centre times are
    segment-LOCAL, both entries must lie inside the forward (last) segment
    for the difference to be a genuine cell width -- which the >=2-centres
    invariant of :class:`RFCenterSegment` guarantees. Before the invariant,
    a single-centre forward segment made the difference cross the segment
    boundary and go negative, silently disarming the guard.
    """

    @staticmethod
    def _feedback_with_backfill_segment():
        """
        A feedback holding one 3-centre backfill segment.

        Returns
        -------
        fdbk
            Feedback whose ``_segments`` hold a single backfill segment with
            centres at ``(0.5, 1.5, 2.5) * T_RF_BOUNDARY``.
        """
        fdbk = TestRFCenterSegment._bare_feedback()
        fdbk._clear_segments()
        fdbk._append_segment(
            RFCenterSegment(
                omega=OMEGA_BOUNDARY,
                duration=3 * T_RF_BOUNDARY,
                residual=0.5 * T_RF_BOUNDARY,
                centers=np.array([0.5, 1.5, 2.5]) * T_RF_BOUNDARY,
            )
        )
        return fdbk

    def test_single_center_forward_segment_is_unconstructible(self):
        # Pre-invariant, appending this single-centre forward segment gave
        # rf_centers = [..., 2.5 T, 0.2 T], so the guard tolerance
        # rf_centers[-1] - rf_centers[-2] = -2.3 T was negative and
        # abs(arrival-time difference) < 0.5 * width could never fire.
        fdbk = self._feedback_with_backfill_segment()
        with pytest.raises(ValueError, match="at least two"):
            fdbk._append_segment(
                RFCenterSegment(
                    omega=OMEGA_BOUNDARY,
                    duration=0.4 * T_RF_BOUNDARY,
                    residual=0.2 * T_RF_BOUNDARY,
                    centers=np.array([0.2]) * T_RF_BOUNDARY,
                )
            )

    def test_forward_cell_width_is_forward_segment_spacing(self):
        # With the invariant holding, the last two flat entries both lie
        # inside the forward segment: the guard tolerance is strictly
        # positive and equals the forward segment's own cell spacing --
        # even when that spacing differs from the backfill segment's.
        forward_spacing = 0.8 * T_RF_BOUNDARY
        fdbk = self._feedback_with_backfill_segment()
        fdbk._append_segment(
            RFCenterSegment(
                omega=2 * np.pi / forward_spacing,
                duration=2 * forward_spacing,
                residual=0.5 * forward_spacing,
                centers=np.array([0.5, 1.5]) * forward_spacing,
            )
        )
        width = float(fdbk._rf_centers[-1] - fdbk._rf_centers[-2])
        assert width > 0
        np.testing.assert_allclose(width, forward_spacing, rtol=1e-12)


# Three backfill segments at deliberately different frequencies, the middle
# one at the two-centre minimum every segment must hold, plus the forward
# segment every passage ends with.
BACKFILL_OMEGAS = (0.9 * OMEGA_BOUNDARY, 1.1 * OMEGA_BOUNDARY, OMEGA_BOUNDARY)
BACKFILL_LENGTHS = (3, 2, 2)
FORWARD_OMEGA = 1.2 * OMEGA_BOUNDARY


SECOND_BACKFILL_OMEGAS = (
    1.05 * OMEGA_BOUNDARY,
    0.95 * OMEGA_BOUNDARY,
    1.25 * OMEGA_BOUNDARY,
)
SECOND_FORWARD_OMEGA = 1.35 * OMEGA_BOUNDARY


class TestBackfillSpanWalksSegments(unittest.TestCase):
    """
    The per-passage walks read the segment records, not parallel arrays.

    ``RFCenterSegment`` carries the frequency and the time span its centres
    were generated over, so the backfill-span replay and the multi-section
    registration phase take omega_k and T_seg,k from the segments
    themselves. The backfill segments of a passage are ``_segments[:-1]``:
    the grid is cleared at the start of every passage, the backfill
    generation appends exactly one segment per elapsed frequency span, and
    the forward generation then appends exactly one more.

    The registration phase additionally spans TWO passages: the carried
    envelope a backfill segment corrects was demodulated against the
    carrier of the passage that STARTED the interval, so the increment is
    ``sum_k (omega_prev - omega_k) T_seg,k`` with ``omega_prev`` the
    forward-segment design frequency of the PREVIOUS passage of this
    station -- not the one of the passage that ends the interval. Those
    tests therefore load two successive passages into the same feedback.
    """

    @staticmethod
    def _load_passage(
        fdbk,
        backfill_omegas=BACKFILL_OMEGAS,
        forward_omega=FORWARD_OMEGA,
    ):
        """
        Rebuild ``fdbk``'s grid as one passage's backfill + forward segments.

        Parameters
        ----------
        fdbk
            Feedback whose ``_segments`` are cleared and refilled, exactly
            as the start of every passage does.
        backfill_omegas
            The three backfill design frequencies [rad/s], one per elapsed
            past-station span (lengths from ``BACKFILL_LENGTHS``, the middle
            one at the two-centre minimum every segment must hold).
        forward_omega
            Design frequency [rad/s] of the forward segment this passage
            ends with.

        Returns
        -------
        fdbk
            The same feedback, with the coarse state resized to the derived
            flat grid.
        """
        fdbk._clear_segments()
        for omega, n_centers in zip(
            backfill_omegas, BACKFILL_LENGTHS, strict=True
        ):
            duration = (n_centers + 1) * T_RF_BOUNDARY
            fdbk._append_segment(
                RFCenterSegment(
                    omega=omega,
                    duration=duration,
                    residual=0.5 * T_RF_BOUNDARY,
                    centers=np.arange(n_centers) * T_RF_BOUNDARY
                    + 0.5 * T_RF_BOUNDARY,
                )
            )
        fdbk._append_segment(
            RFCenterSegment(
                omega=forward_omega,
                duration=4 * T_RF_BOUNDARY,
                residual=0.5 * T_RF_BOUNDARY,
                centers=np.arange(3) * T_RF_BOUNDARY + 0.5 * T_RF_BOUNDARY,
            )
        )
        n_cells = len(fdbk._rf_centers)
        fdbk.antenna_voltage_coarse_grid = np.zeros(n_cells, dtype=complex)
        fdbk.generator_current_coarse_grid = np.zeros(n_cells, dtype=complex)
        return fdbk

    @staticmethod
    def _passage_feedback():
        """
        A feedback holding one passage's backfill + forward segments.

        Returns
        -------
        fdbk
            Feedback whose ``_segments`` are three backfill segments (the
            middle one at the two-centre minimum) followed by the forward
            segment, with the coarse state sized to the derived flat grid.
        """
        return TestBackfillSpanWalksSegments._load_passage(
            TestRFCenterSegment._bare_feedback()
        )

    @staticmethod
    def _expected_increment(fdbk, omega_previous):
        """
        ``sum_k (omega_prev - omega_k) T_seg,k`` over the loaded backfill.

        Parameters
        ----------
        fdbk
            Feedback holding the passage whose backfill segments
            (``_segments[:-1]``) supply omega_k and T_seg,k.
        omega_previous
            The previous passage's forward-segment design frequency
            [rad/s] -- the carrier the carried envelope was demodulated
            against.

        Returns
        -------
        float
            The registration-phase increment [rad] this passage must add.
        """
        backfill_segments = fdbk._segments[:-1]
        segment_omegas = np.array(
            [segment.omega for segment in backfill_segments]
        )
        segment_durations = np.array(
            [segment.duration for segment in backfill_segments]
        )
        return float(
            np.sum((omega_previous - segment_omegas) * segment_durations)
        )

    @staticmethod
    def _recorded_replay(fdbk, n_backfill_centers):
        """
        Run the replay with ``circuit_track`` recorded instead of executed.

        Parameters
        ----------
        fdbk
            Feedback whose backfill span should be replayed.
        n_backfill_centers
            The passage's backfill centre count (the replay's gate).

        Returns
        -------
        list
            One ``(omega_input, start_index, end_index, no_beam)`` tuple per
            ``circuit_track`` call the replay made.
        """
        calls = []
        fdbk.circuit_track = (
            lambda omega_input, start_index, end_index, no_beam: calls.append(
                (omega_input, start_index, end_index, no_beam)
            )
        )
        fdbk._replay_backfill_span(n_backfill_centers=n_backfill_centers)
        return calls

    def test_replay_walks_backfill_segments_only(self):
        # One no-beam pass per backfill segment, at that segment's own omega
        # and over exactly its own centres -- and the forward segment (the
        # last one) is left to the real forward pass.
        fdbk = self._passage_feedback()
        calls = self._recorded_replay(fdbk, sum(BACKFILL_LENGTHS))

        self.assertEqual(
            calls,
            [
                (BACKFILL_OMEGAS[0], 0, 3, True),
                (BACKFILL_OMEGAS[1], 3, 5, True),
                (BACKFILL_OMEGAS[2], 5, 7, True),
            ],
        )

    def test_replay_is_a_no_op_without_backfill_centers(self):
        # The gate: a passage that generated no backfill centre must not walk
        # anything (a stale frequency list used to re-run the whole grid).
        fdbk = self._passage_feedback()
        self.assertEqual(self._recorded_replay(fdbk, 0), [])

    def test_registration_phase_uses_previous_passage_carrier(self):
        """
        ``dPsi = sum_k (omega_prev - omega_k) T_seg,k`` across two passages.

        The backfill segments of a passage reconstruct the interval since
        this station's PREVIOUS passage, and the carried envelope they
        correct was demodulated against the carrier of that previous
        passage. The increment is therefore referenced to ``omega_prev``,
        the forward-segment design frequency as it stood one passage ago --
        and it enters with that sign, ``omega_prev`` minus omega_k.

        Passage 1 has no predecessor, so it contributes exactly ``+0.0``
        and only records its carrier. Passage 2 (different backfill
        frequencies AND a different forward frequency, so the two candidate
        references are cleanly separated) must add the increment built from
        passage 1's carrier.
        """
        fdbk = self._passage_feedback()
        fdbk._n_rf_stations_in_ring = 2
        fdbk._forward_segment_omega_design = FORWARD_OMEGA
        n_backfill_centers = sum(BACKFILL_LENGTHS)

        first_passage_phase = fdbk._accumulate_registration_phase(
            n_backfill_centers=n_backfill_centers
        )

        # Nothing to correct yet: there is no previous carrier to refer to.
        self.assertEqual(first_passage_phase, 0.0)

        self._load_passage(
            fdbk,
            backfill_omegas=SECOND_BACKFILL_OMEGAS,
            forward_omega=SECOND_FORWARD_OMEGA,
        )
        fdbk._forward_segment_omega_design = SECOND_FORWARD_OMEGA

        second_passage_phase = fdbk._accumulate_registration_phase(
            n_backfill_centers=n_backfill_centers
        )

        expected = self._expected_increment(fdbk, FORWARD_OMEGA)
        self.assertEqual(second_passage_phase, expected)
        # The segments really do differ from the previous carrier, so the
        # pin above is not trivially satisfied by zero.
        self.assertNotEqual(expected, 0.0)
        # ... and it is not the former expression either (which referenced
        # THIS passage's carrier, with the opposite sign). The two agree
        # only for a linear frequency programme -- that second difference
        # is exactly what used to hide the defect -- so the fixture pulls
        # them apart by more than 10 % of the increment.
        former_expression = -self._expected_increment(
            fdbk, SECOND_FORWARD_OMEGA
        )
        self.assertGreater(
            abs(second_passage_phase - former_expression),
            0.1 * abs(second_passage_phase),
        )

    def test_registration_phase_accumulates_a_running_total(self):
        # The method returns the RUNNING TOTAL, not the increment: a third
        # passage adds its own increment (built from passage 2's carrier)
        # on top of what passage 2 left.
        fdbk = self._passage_feedback()
        fdbk._n_rf_stations_in_ring = 2
        fdbk._forward_segment_omega_design = FORWARD_OMEGA
        n_backfill_centers = sum(BACKFILL_LENGTHS)
        fdbk._accumulate_registration_phase(
            n_backfill_centers=n_backfill_centers
        )

        self._load_passage(
            fdbk,
            backfill_omegas=SECOND_BACKFILL_OMEGAS,
            forward_omega=SECOND_FORWARD_OMEGA,
        )
        fdbk._forward_segment_omega_design = SECOND_FORWARD_OMEGA
        after_second = fdbk._accumulate_registration_phase(
            n_backfill_centers=n_backfill_centers
        )

        self._load_passage(fdbk)
        fdbk._forward_segment_omega_design = FORWARD_OMEGA
        after_third = fdbk._accumulate_registration_phase(
            n_backfill_centers=n_backfill_centers
        )

        increment = self._expected_increment(fdbk, SECOND_FORWARD_OMEGA)
        self.assertEqual(after_third, after_second + increment)

    def test_registration_phase_snapshot_is_the_live_design_carrier(self):
        # The snapshot must take the instance scalar
        # ``_forward_segment_omega_design`` -- the live design clock the
        # demodulation and readout reference -- and NOT ``_segments[-1]
        # .omega``. Here the two are deliberately pulled apart so a
        # segment-sourced snapshot gives a different second passage.
        fdbk = self._passage_feedback()
        fdbk._n_rf_stations_in_ring = 2
        held_carrier = 1.15 * OMEGA_BOUNDARY
        self.assertNotEqual(held_carrier, fdbk._segments[-1].omega)
        fdbk._forward_segment_omega_design = held_carrier
        n_backfill_centers = sum(BACKFILL_LENGTHS)
        fdbk._accumulate_registration_phase(
            n_backfill_centers=n_backfill_centers
        )

        self._load_passage(
            fdbk,
            backfill_omegas=SECOND_BACKFILL_OMEGAS,
            forward_omega=SECOND_FORWARD_OMEGA,
        )
        fdbk._forward_segment_omega_design = SECOND_FORWARD_OMEGA

        self.assertEqual(
            fdbk._accumulate_registration_phase(
                n_backfill_centers=n_backfill_centers
            ),
            self._expected_increment(fdbk, held_carrier),
        )

    def test_registration_phase_snapshot_is_taken_outside_the_gate(self):
        # A passage that the gate skips (no backfill centres, or a
        # single-station ring) still records its carrier: the snapshot sits
        # OUTSIDE the gate, so the NEXT passage has a carrier to refer to.
        # Without that, a skipped passage would silently reset the
        # reference and the following increment would be dropped.
        fdbk = self._passage_feedback()
        fdbk._n_rf_stations_in_ring = 2
        fdbk._forward_segment_omega_design = FORWARD_OMEGA

        self.assertEqual(
            fdbk._accumulate_registration_phase(n_backfill_centers=0), 0.0
        )

        self._load_passage(
            fdbk,
            backfill_omegas=SECOND_BACKFILL_OMEGAS,
            forward_omega=SECOND_FORWARD_OMEGA,
        )
        fdbk._forward_segment_omega_design = SECOND_FORWARD_OMEGA

        self.assertEqual(
            fdbk._accumulate_registration_phase(
                n_backfill_centers=sum(BACKFILL_LENGTHS)
            ),
            self._expected_increment(fdbk, FORWARD_OMEGA),
        )

    def test_registration_phase_is_zero_for_a_single_station_ring(self):
        # A single section builds the whole passage from one segment at
        # omega_0: Psi is identically zero and must stay EXACTLY +0.0 over
        # repeated passages, so single-section runs stay bit-identical.
        fdbk = self._passage_feedback()
        fdbk._n_rf_stations_in_ring = 1
        fdbk._forward_segment_omega_design = FORWARD_OMEGA
        n_backfill_centers = sum(BACKFILL_LENGTHS)

        for _ in range(3):
            phase = fdbk._accumulate_registration_phase(
                n_backfill_centers=n_backfill_centers
            )
            self.assertEqual(phase, 0.0)
            self._load_passage(
                fdbk,
                backfill_omegas=SECOND_BACKFILL_OMEGAS,
                forward_omega=SECOND_FORWARD_OMEGA,
            )
            fdbk._forward_segment_omega_design = SECOND_FORWARD_OMEGA
