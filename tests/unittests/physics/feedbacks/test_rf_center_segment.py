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
    IQCavityFeedbackBase,
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
            fdbk.rf_centers, np.array([0.3, 0.8, 0.1, 0.3, 0.5])
        )
        assert list(fdbk.rf_centers_lengths) == [2, 3]
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
        assert len(fdbk.rf_centers) == 0
        assert len(fdbk.rf_centers_lengths) == 0
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
        assert list(fdbk.rf_centers_lengths) == [0, 2]
        assert len(fdbk.rf_centers) == 2
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
        fdbk.rf_centers_lengths = np.array(
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
