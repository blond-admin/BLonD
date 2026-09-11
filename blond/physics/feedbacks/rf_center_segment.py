# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


"""
The coarse-grid value classes of the cavity feedback.

:class:`RFCenterSegment` is one segment of the per-turn ``rf_centers`` grid
the cavity-feedback timing class builds; :class:`PerTurnGridSpan` is the
per-turn span one grid rebuild produces out of those segments. Both value
types are kept in this module so they and their validation stay independent
of the (much larger) feedback and grid-construction code, together with
:func:`accumulated_phases`, the phase each backfill segment stores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray


# Tolerance for the residual bound check in RFCenterSegment, as a fraction of
# the segment's RF period. The residual is a float difference of times, so
# rounding contributes a few ULPs; a millionth of an RF period is far above
# that, yet far below any genuine error, which is of order a coarse step.
# The former absolute 1e-9 s exceeded a whole RF period at 1.3 GHz and could
# not catch such an error.
_RF_CENTER_SEGMENT_RESIDUAL_RTOL = 1e-6


@dataclass(frozen=True, eq=False)
class RFCenterSegment:
    """
    One coarse-grid segment of the per-turn ``rf_centers`` grid.

    The timing-class grid is built per turn as an ordered list of these
    records -- one per backfill frequency segment plus one forward
    segment (see
    :meth:`~blond.physics.feedbacks.rf_center_grid.RFCenterGridMixin.calculate_rf_centers_for_backfill`
    and ``..._for_forward_direction``). Bundling the four pieces that used to
    live in loose parallel arrays / a mutable scalar keeps them coherent and
    self-validating: the flat ``rf_centers`` / ``rf_centers_lengths`` arrays the
    tracking loop indexes are *derived* from the segment list
    (``_rebuild_grid_arrays``), so they can no longer desync from it.
    """

    omega: float
    """RF angular frequency [rad/s] this segment was generated at."""
    duration: float
    """Time span [s] the segment covers (``until_time`` in
    ``_generate_rf_centers``)."""
    residual: float
    """Accumulator value after this segment -- the leftover time [s] between
    the last centre and the end of the segment. Feeds the sub-stepped
    cross-segment continuity and the demodulation frame, and is READ back by
    ``_preceding_segment_residual`` on
    :class:`~blond.physics.feedbacks.rf_center_grid.RFCenterGridMixin`
    to form the first coarse step of the FOLLOWING
    segment: ``rf_centers`` are segment-local, so that step is the following
    segment's first local centre time plus this unfilled tail. The live host
    scalar cannot serve there -- the whole per-turn grid is generated before
    any of it is walked, so by consumption time the scalar holds the
    last-generated (forward) segment's value."""
    centers: NumpyArray
    """The coarse-grid centre times [s] of this segment. Always holds at
    least two centres -- enforced in ``__post_init__``, and relied on by the
    coincidence-guard cell width of
    :class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass`."""
    accumulated_phase: float = 0.0
    """Grid-vs-carrier phase [rad] accumulated up to the end of this segment.

    A multi-section passage builds its grid piecewise, each backfill segment
    at the design frequency of the station it reconstructs, while the
    envelope carried across that interval was demodulated against a single
    carrier: the forward-segment frequency of this station's previous
    passage. This is the running phase difference between the two,
    continued from passage to passage (see :func:`accumulated_phases`). The
    forward segment holds the value of its passage, which the demodulation
    subtracts and the readout adds back. Exactly ``0.0`` for a single
    section, for an unaccelerated ring and on a station's first passage;
    the default, so hand-built segments need not state it."""

    def __post_init__(self) -> None:
        """Validate the segment fields (frequency, duration, residual, shape)."""
        if self.omega <= 0:
            raise ValueError(
                f"RFCenterSegment.omega must be > 0, got {self.omega}"
            )
        if self.duration < 0:
            raise ValueError(
                f"RFCenterSegment.duration must be >= 0, got {self.duration}"
            )
        if np.ndim(self.centers) != 1:
            raise ValueError(
                "RFCenterSegment.centers must be 1-D, got ndim "
                f"{np.ndim(self.centers)}"
            )
        # Every segment must hold at least two coarse centres. Two
        # correctness properties rest on this invariant (do not relax it
        # without revisiting them): the counter-rotating coincidence-guard
        # tolerance rf_centers[-1] - rf_centers[-2] only measures a genuine
        # forward cell width when both entries lie inside the forward
        # segment; and an empty segment would carry the preceding residual
        # through without adding its own duration to the bridging coarse
        # step.
        min_centers_per_segment = 2
        if len(self.centers) < min_centers_per_segment:
            t_rf = 2 * np.pi / self.omega
            raise ValueError(
                f"RFCenterSegment holds {len(self.centers)} coarse-grid "
                "centre(s), but every segment must hold at least two. "
                f"The segment spans {self.duration} s = "
                f"{self.duration / t_rf:.6g} RF periods "
                f"(t_rf = {t_rf} s): the walked interval -- a ring "
                "section, or the partial first-turn stretch before a "
                "station -- is shorter than two coarse steps "
                "(duration < 2 * n_rf_periods_per_coarse_grid * t_rf). "
                "Remedies: reduce n_rf_periods_per_coarse_grid, or use "
                "fewer/longer sections so every RF-station section spans "
                "at least two coarse cells."
            )
        # residual is the time left after the last centre, so it must fall
        # within [0, duration] (up to float noise, scaled to the RF period).
        tolerance = _RF_CENTER_SEGMENT_RESIDUAL_RTOL * 2 * np.pi / self.omega
        if not (-tolerance <= self.residual <= self.duration + tolerance):
            raise ValueError(
                f"RFCenterSegment.residual {self.residual} outside "
                f"[0, duration={self.duration}]"
            )
        if not np.isfinite(self.accumulated_phase):
            raise ValueError(
                "RFCenterSegment.accumulated_phase must be finite, got "
                f"{self.accumulated_phase}"
            )

    def __len__(self) -> int:
        """
        Number of coarse-grid centres in this segment.

        Returns
        -------
        int
            The number of centres held by the segment.
        """
        return len(self.centers)


@dataclass(frozen=True, eq=False)
class PerTurnGridSpan:
    """
    The per-turn coarse-grid span produced by one grid rebuild.

    Carries the three values the later phases of
    ``IQCavityFeedbackTimingClass._track`` need from the grid rebuild.
    They are *returned* rather than left on the feedback so that the
    ordering is enforced by the data flow: in particular
    ``residual_from_backfill_span`` can only be read from a span object, and
    a span is only produced by a rebuild that snapshotted it before the
    forward generation overwrote the host scalar.
    """

    n_backfill_centers: int
    """Number of coarse centres generated by the backfill segments of
    this passage (``0`` when this passage generated none)."""
    n_forward_centers: int
    """Number of coarse centres in the forward segment of this passage."""
    residual_from_backfill_span: float
    """``_residual_time_last_rf_centers_calculation`` [s] as it stood after the
    backfill segments and BEFORE the forward generation overwrote it. This is
    the demodulation frame
    :meth:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass.calculate_rf_beam_current_partial`
    needs; re-reading the host attribute later yields the forward value and
    silently shifts the frame."""


def accumulated_phases(
    carried_phase: float,
    carrier_omega: float | None,
    segment_omegas: NumpyArray,
    segment_durations: NumpyArray,
) -> NumpyArray:
    """
    Accumulated phase at the end of each backfill segment of a passage.

    The backfill segments rebuild the interval since this station's
    previous passage, segment ``k`` spanning ``T_k`` at the design frequency
    ``omega_k`` of the station it reconstructs. The envelope carried across
    that interval was demodulated against the carrier in force when the
    interval STARTED -- the previous passage's forward-segment frequency --
    so over segment ``k`` the grid and that carrier part by
    ``(omega_carrier - omega_k) T_k``.

    Parameters
    ----------
    carried_phase
        Accumulated phase [rad] the previous passage's forward segment
        ended on.
    carrier_omega
        Design frequency [rad/s] of that forward segment, or ``None`` when
        nothing accumulates: a station's first passage and a single-station
        ring.
    segment_omegas
        Design frequency [rad/s] of each backfill segment, in order.
    segment_durations
        Duration [s] of each backfill segment, in order.

    Returns
    -------
    phases
        ``carried_phase + sum_{j <= k} (carrier_omega - omega_j) T_j`` [rad]
        for each segment ``k``; every entry is ``carried_phase`` when
        ``carrier_omega`` is ``None``.

    Notes
    -----
    The reference is the PREVIOUS passage's carrier. Referring the increment
    to the carrier of the passage that ends the interval, with the opposite
    sign, differs by a second difference of the frequency programme; that
    vanishes for a linear ramp, which hid a secular drift against the
    multi-pass convolution.

    Each prefix is summed as one array slice, so the last entry equals
    ``carried_phase + float(np.sum(increments))`` bit-for-bit -- the value a
    passage reads off its forward segment.
    """
    omegas = np.asarray(segment_omegas, dtype=float)
    durations = np.asarray(segment_durations, dtype=float)
    if carrier_omega is None:
        return np.full(len(omegas), float(carried_phase))
    increments = (carrier_omega - omegas) * durations
    return np.array(
        [
            carried_phase + float(np.sum(increments[: index + 1]))
            for index in range(len(increments))
        ],
        dtype=float,
    )
