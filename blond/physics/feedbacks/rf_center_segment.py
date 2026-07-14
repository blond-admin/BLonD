# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


"""
The :class:`RFCenterSegment` coarse-grid value class.

One segment of the per-turn ``rf_centers`` grid the cavity-feedback timing
class builds. Kept in its own module so the value type and its validation are
independent of the (much larger) feedback and grid-construction code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray


# Absolute tolerance [s] for the residual bound check in RFCenterSegment: the
# residual is a float difference of times, so allow a few ULPs of slack.
_RF_CENTER_SEGMENT_RESIDUAL_TOL = 1e-9


@dataclass(frozen=True, eq=False)
class RFCenterSegment:
    """
    One coarse-grid segment of the per-turn ``rf_centers`` grid.

    The timing-class grid is built per turn as an ordered list of these
    records -- one per reverse-tracking frequency segment plus one forward
    segment (see :meth:`IQCavityFeedbackTimingClass.calculate_rf_centers_for_reverse_direction`
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
    the last centre and the end of the segment (carried unchanged for an empty
    segment). Feeds the sub-stepped cross-segment continuity and the
    demodulation frame."""
    centers: NumpyArray
    """The coarse-grid centre times [s] of this segment (may be empty when the
    segment is shorter than one coarse step)."""

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
        # residual is the time left after the last centre; for a non-empty
        # segment it must fall within [0, duration] (up to float noise). Empty
        # segments legitimately carry the *previous* segment's residual, which
        # can exceed their own (near-zero) duration, so skip the bound there.
        if len(self.centers) and not (
            -_RF_CENTER_SEGMENT_RESIDUAL_TOL
            <= self.residual
            <= self.duration + _RF_CENTER_SEGMENT_RESIDUAL_TOL
        ):
            raise ValueError(
                f"RFCenterSegment.residual {self.residual} outside "
                f"[0, duration={self.duration}]"
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
