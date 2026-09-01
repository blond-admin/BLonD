# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


"""
Coarse-grid (``rf_centers``) construction for the cavity-feedback timing class.

:class:`RFCenterGridMixin` bundles the per-turn coarse-grid construction of
:class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass`:
the forward/backfill reference walks that decide which RF frequency each
segment is generated at, the segment generation itself, and the derived flat
``rf_centers`` / ``rf_centers_lengths`` arrays the tracking loop indexes.

It is a *mixin*: its methods read and write state that the host feedback class
owns and -- with the one exception flagged under "Reference bookkeeping"
below -- initialises in ``__init__`` / ``on_run_simulation``.

- Grid state: ``_segments`` (the single source of truth) and the derived
  ``_rf_centers`` / ``_rf_centers_lengths``.
- Tiling carry-over from one segment to the next:
  ``_residual_time_last_rf_centers_calculation``,
  ``_residual_taps_last_rf_centers_calculation`` and
  ``_last_segment_omega_design``.
- Turn-boundary carry-over, captured by
  ``RFCenterGridMixin._close_previous_turn_grid`` before it clears the
  previous turn's segments: ``_residual_time_carried_into_turn``, which
  ``RFCenterGridMixin._preceding_segment_residual`` reads back at a turn
  boundary, and ``_last_rf_centers_entry``, the previous turn's last
  centre -- the only piece the host reads back (as a first-turn
  ``None`` / not-``None`` flag in its per-cell step sizing).
- Walk results: ``_backfill_time_array`` /
  ``_backfill_segment_omega_design_list``, which stay inside this module (the
  backfill generation turns them into segments, and the tracking loop then
  reads the segments), plus ``_forward_tracking_time`` /
  ``_forward_segment_omega_design`` and ``_tracked_forward_until_element``,
  which the tracking loop consumes.
- Reference bookkeeping: ``_reference_state_until_tracked``,
  ``_reference_altering_elements`` / ``..._reverse``,
  ``_own_index_in_reference_list`` / ``..._reverse``,
  ``_last_tracked_turn_frwrd`` and ``_last_tracked_beam_state_frwrd``.
  The two until-tracked indices ``_reference_index_until_tracked`` and
  ``_reference_index_until_tracked_reverse`` are the exception: nothing
  initialises them up front -- they come into existence on the first
  ``RFCenterGridMixin.get_passed_time_forward_direction`` projection, and
  every read is gated on ``_last_tracked_beam_state_frwrd is not None``.
- Plain host configuration: ``n_rf_periods_per_coarse_grid``,
  ``section_index``, ``_parent_rf_station``, ``_ring_circumference`` and
  ``_debug`` -- the last of which additionally gates the inspection-only
  ``current_slice_elements_forward`` / ``reference_time_after_backfill`` /
  ``reference_energy_after_backfill`` / ``current_beam_reference_time`` /
  ``current_beam_reference_energy`` diagnostics written here (nothing in
  BLonD reads them back; they are the debug feature's assertion interface,
  consumed by ``tests/unittests/physics/feedbacks/test_rf_center_grid.py``).

Two independent notions share the word "backwards" here, and the names keep
them apart. *Backfill* is a TIME direction: the feedback only runs at its own
station, so each passage must reconstruct the stretch of grid that already
elapsed since the previous passage -- every multi-section ring needs it,
whichever way its beam goes. *Reverse* is a SPACE direction: a
counter-rotating beam meets the ring's reference-altering elements in the
reversed order, hence ``_reference_altering_elements_reverse`` and its index
companions. A single-direction multi-section ring uses the backfill walk and
never the reversed element list.

Note which frequency the grid runs on: every segment frequency comes from the
parent station's ``calc_omega_rf_design``, reduced to the host's tracked
harmonic through ``_resolve_main_harmonic`` (a multi-harmonic station returns
the whole per-harmonic array; a single-harmonic one is untouched). Beyond
that dispatch the mixin reads none of the host's RF properties -- not
``omega_rf``, ``omega_rf_design``, ``harmonic``, ``delta_omega_rf`` or
``sampling_time_coarse``, and the host has no ``t_rf`` accessor at all. The
grid geometry is design-clock only; the RF-frequency offset enters solely as
a demodulation/readout phase. The traversals and the segment generation were
extracted verbatim from ``cavity_feedback.py`` for readability.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from blond.core.base import AltersReference
from blond.physics.cavities import RFStationBaseClass
from blond.physics.feedbacks.rf_center_segment import RFCenterSegment

if TYPE_CHECKING:
    from blond.core.beam.base import BeamBaseClass
    from blond.physics.feedbacks.cavity_feedback import (
        IQCavityFeedbackTimingClass,
    )


class RFCenterGridMixin:
    """Coarse-grid (``rf_centers``) construction mixin (see module docstring)."""

    def _reference_list_for_direction(
        self: IQCavityFeedbackTimingClass, is_counter_rotating: bool
    ) -> tuple[AltersReference, ...] | None:
        """
        Reference-altering element list for one beam direction.

        Both the forward-projection and the backfill traversals walk the
        ring's reference-altering elements; a counter-rotating beam sees them
        in the reversed order. This is the *space* sense of "reverse" (see
        the module docstring), independent of the backfill time direction.
        Selecting the list through this one helper keeps the two traversals
        free of duplicated direction ``if`` ladders while still supporting
        either direction (needed for counter-rotating beams).

        Parameters
        ----------
        is_counter_rotating : bool
            Whether the (current or last-tracked) beam is counter-rotating.

        Returns
        -------
        tuple
            ``reference_altering_elements_reverse`` when counter-rotating,
            otherwise ``reference_altering_elements``.
        """
        return (
            self._reference_altering_elements_reverse
            if is_counter_rotating
            else self._reference_altering_elements
        )

    def _own_index_for_direction(
        self: IQCavityFeedbackTimingClass, is_counter_rotating: bool
    ) -> int | None:
        """
        Return this feedback's index in the direction's reference list.

        Parameters
        ----------
        is_counter_rotating : bool
            Whether the (current or last-tracked) beam is counter-rotating.

        Returns
        -------
        int
            ``own_index_in_reference_list_reverse`` when counter-rotating,
            otherwise ``own_index_in_reference_list``.
        """
        return (
            self._own_index_in_reference_list_reverse
            if is_counter_rotating
            else self._own_index_in_reference_list
        )

    def get_passed_time_forward_direction(  # noqa: PLR0912
        self: IQCavityFeedbackTimingClass, beam: BeamBaseClass
    ):
        """
        Determine the slice of elements, which should be tracked in the forward direction.

        Parameters
        ----------
        beam
            Beam object to receive the reference frame.
        """
        next_reference_altering_element_index = -1

        dummy_reference = deepcopy(beam.reference)
        start_time = dummy_reference.time

        found = False

        own_index_tracking = self._own_index_for_direction(
            beam.is_counter_rotating
        )
        forward_list = self._reference_list_for_direction(
            beam.is_counter_rotating
        )

        # beam is tracked after the feedback, therefore we have to track the current element
        # the schedules are applied correctly though as this is done in the RFCavityBaseClass._track, which was already called
        for el_ind, element in enumerate(
            forward_list[own_index_tracking:]
        ):  # iterate through remaining current turn
            if isinstance(element, RFStationBaseClass) and el_ind != 0:
                found = True
                next_reference_altering_element_index = (
                    el_ind + own_index_tracking
                    # This will be the next element
                )
                self._last_tracked_turn_frwrd = deepcopy(
                    self._parent_rf_station._turn_counter.value
                )
                break
            element: AltersReference
            if isinstance(element, RFStationBaseClass):
                element.track_reference(
                    dummy_reference, beam.is_counter_rotating
                )
            else:
                element.track_reference(dummy_reference)

        if not found:
            if own_index_tracking != 0:
                for el_ind, element in enumerate(
                    forward_list[:own_index_tracking]
                ):  # iterate through initial next turn
                    element: AltersReference

                    if not isinstance(element, RFStationBaseClass):
                        element.track_reference(dummy_reference)
                    else:
                        next_reference_altering_element_index = (
                            el_ind
                            + len(
                                self._reference_altering_elements
                            )  # This will be the next element
                        )
                        self._last_tracked_turn_frwrd = deepcopy(
                            self._parent_rf_station._turn_counter.value + 1
                        )
                        break
            else:
                next_reference_altering_element_index = -1

        self._forward_tracking_time = dummy_reference.time - start_time
        # The coarse-grid GEOMETRY (spacing, tiling, residuals) AND the
        # beam-current demodulation carrier both stay on the *design* RF
        # clock even under an RF-frequency offset, so every time-valued
        # bookkeeping quantity (residuals, dT demodulation shifts) is
        # independent of delta_omega_rf. The offset enters only as the
        # accumulated kick-clock slip ``int delta_omega_rf dt`` (the parent
        # station's ``delta_phi_rf`` plus its live end-of-track tail),
        # applied as an explicit constant phase on the demodulation and
        # readout sides (see IQCavityFeedbackTimingClass docstring). Feeding
        # the offset into the grid times instead (detuned spacing or a
        # slipping seed) drags the whole envelope timeline against the
        # readout -- a frame drift no per-deposit bookkeeping can compensate
        # (measured against the retuning convolution). The only residual is
        # the intra-window mismatch delta_omega_rf * hist_x between the
        # design demodulation carrier and the actual RF; that is bounded by
        # the bunch-local time (order t_rf, reset each turn) to ~1e-6 rad and
        # never accumulates, so demodulating at the design carrier is
        # exact to well within the discretization floor.
        # _resolve_main_harmonic: calc_omega_rf_design returns the whole
        # per-harmonic array for a MultiHarmonicRFStation; the grid runs
        # on THIS feedback's harmonic (a no-op for a single-harmonic
        # station, whose value is a scalar already).
        self._forward_segment_omega_design = self._resolve_main_harmonic(
            self._parent_rf_station.calc_omega_rf_design(
                dummy_reference.beta, self._ring_circumference
            )
        )
        self._tracked_forward_until_element = (
            forward_list[
                next_reference_altering_element_index % len(forward_list)
            ]
            if next_reference_altering_element_index != -1
            else self._parent_rf_station
        )
        self._reference_index_until_tracked = (
            self._reference_altering_elements.index(
                self._tracked_forward_until_element
            )
        )
        self._reference_index_until_tracked_reverse = (
            self._reference_altering_elements_reverse.index(
                self._tracked_forward_until_element
            )
        )
        self._last_tracked_beam_state_frwrd = beam.is_counter_rotating
        self._reference_state_until_tracked = dummy_reference

        if self._debug:
            if (
                next_reference_altering_element_index == -1
                or next_reference_altering_element_index
                >= len(self._reference_altering_elements)
            ):
                # either none were found or it is around two turns
                self.current_slice_elements_forward = (
                    self._reference_altering_elements[
                        self._own_index_in_reference_list :
                    ]
                )
                self.current_slice_elements_forward += (
                    self._reference_altering_elements[
                        0 : next_reference_altering_element_index
                        - len(self._reference_altering_elements)
                    ]
                )
            else:  # element is in the same turn
                self.current_slice_elements_forward = self._reference_altering_elements[
                    self._own_index_in_reference_list : next_reference_altering_element_index
                ]

    def get_time_omega_array_backfill(  # noqa: PLR0912, PLR0915
        self: IQCavityFeedbackTimingClass, beam: BeamBaseClass
    ):
        """
        Determine the slice of elements to walk for the backfill.

        The backfill covers the stretch of the ring that has already elapsed
        since this feedback's previous passage. Only gets called after the
        first turn.

        Parameters
        ----------
        beam
            Beam object to receive the reference frame.
        """
        time_list = []
        omega_list = []
        start_time = self._reference_state_until_tracked.time

        found = False

        if (
            self._parent_rf_station._turn_counter.value
            > self._last_tracked_turn_frwrd
        ):
            reference_turn_offset = -1
        elif (
            self._parent_rf_station._turn_counter.value
            == self._last_tracked_turn_frwrd
        ):
            reference_turn_offset = 0
        else:
            raise RuntimeError("Turn value not possible, was a turn skipped?")

        # ONE direction for the whole walk. The backfill re-derives an
        # interval that has already elapsed, so the traversal order, the
        # indices into that order and the per-station section remap must all
        # describe the beam whose forward projection laid down the carried
        # state -- not whichever beam happens to be tracked now. Deriving
        # them from a single flag is what keeps them from disagreeing: with
        # two beams the current beam's direction can differ from the carried
        # one, and mixing the two would order the elements for one beam while
        # applying the other's energy program.
        if self._last_tracked_beam_state_frwrd is not None:
            # Continue from where the last forward projection stopped, in that
            # beam's direction.
            backfill_is_counter_rotating = self._last_tracked_beam_state_frwrd
            start_index = (
                self._reference_index_until_tracked_reverse
                if self._last_tracked_beam_state_frwrd
                else self._reference_index_until_tracked
            )
        else:
            # first turn, nothing has been tracked yet.
            backfill_is_counter_rotating = beam.is_counter_rotating
            start_index = 0
        backfill_element_list = self._reference_list_for_direction(
            backfill_is_counter_rotating
        )

        for element in backfill_element_list[
            start_index:
        ]:  # iterate through remaining last turn
            element: AltersReference  # TODO: are duplicate elements allowed in pipeline?
            if isinstance(
                element, RFStationBaseClass
            ):  # and element == self.tracked_forward_until_element:
                # Since we are in the previous turn, we need to decrease this manually
                # and increase it afterwards (only for cavities in case of scheduled acceleration).
                # this is not strictly true for all cases, but only cases, where the reference crosses the turn border on the forward tracking
                #
                # FOREIGN STATE: the turn counter belongs to an RF station
                # this feedback does not own; it is borrowed for the
                # duration of the call and must be handed back untouched.
                # The restore therefore runs in a ``finally``: both this
                # walk and ``track_reference`` itself can raise, and an
                # unrestored counter would silently mis-track every element
                # of the ring for the rest of the run.
                element._turn_counter._value += reference_turn_offset
                try:
                    element.track_reference(
                        self._reference_state_until_tracked,
                        backfill_is_counter_rotating,
                    )
                finally:
                    element._turn_counter._value -= reference_turn_offset
            else:
                element.track_reference(
                    self._reference_state_until_tracked
                )  # no need for CR flag

            # Per-harmonic array on a multi-harmonic parent; the grid
            # runs on THIS feedback's harmonic (see
            # forward_segment_omega_design).
            omega_list.append(
                self._resolve_main_harmonic(
                    self._parent_rf_station.calc_omega_rf_design(
                        self._reference_state_until_tracked.beta,
                        self._ring_circumference,
                    )
                )
            )
            time_list.append(self._reference_state_until_tracked.time)
            if self._backfill_walk_reached_beam(beam.reference.time):
                # counterrotation should break earlier
                found = True
                break

        until_index = self._own_index_for_direction(
            backfill_is_counter_rotating
        )

        if not found:
            for element in backfill_element_list[
                :until_index
            ]:  # iterate through initial current turn
                element: AltersReference
                if isinstance(element, RFStationBaseClass):
                    element.track_reference(
                        self._reference_state_until_tracked,
                        backfill_is_counter_rotating,
                    )
                else:
                    element.track_reference(
                        self._reference_state_until_tracked
                    )
                # Per-harmonic array on a multi-harmonic parent; the
                # grid runs on THIS feedback's harmonic (see
                # forward_segment_omega_design).
                omega_list.append(
                    self._resolve_main_harmonic(
                        self._parent_rf_station.calc_omega_rf_design(
                            self._reference_state_until_tracked.beta,
                            self._ring_circumference,
                        )
                    )
                )
                time_list.append(self._reference_state_until_tracked.time)
                if self._backfill_walk_reached_beam(beam.reference.time):
                    # counterrotation should break earlier
                    break

        if len(time_list) > 1:
            self._backfill_time_array = np.append(
                np.array(time_list[0] - start_time), np.diff(time_list)
            )
            # Grid geometry stays on the *design* RF clock (see
            # forward_segment_omega_design); the RF-frequency offset
            # enters only as the constant demodulation/readout phase,
            # not the grid.
            self._backfill_segment_omega_design_list = np.array(omega_list)
        else:
            # Durations since the walk's start, exactly as the branch above:
            # a single recorded reference time is still a SEGMENT LENGTH
            # (time_list[0] - start_time), never the absolute clock reading.
            # Storing the absolute time here made the segment grow with the
            # simulation clock, so the grid spanned the whole elapsed run.
            self._backfill_time_array = (
                np.asarray(time_list, dtype=float) - start_time
            )
            # Grid geometry stays on the *design* RF clock (see
            # forward_segment_omega_design); the RF-frequency offset
            # enters only as the constant demodulation/readout phase,
            # not the grid.
            self._backfill_segment_omega_design_list = np.array(omega_list)

        self._unify_same_frequency_time_points_backfill()

        if self._debug:
            self.reference_time_after_backfill = (
                self._reference_state_until_tracked.time
            )
            self.current_beam_reference_time = beam.reference.time
            self.reference_energy_after_backfill = (
                self._reference_state_until_tracked.total_energy
            )
            self.current_beam_reference_energy = beam.reference.total_energy

    def _rebuild_grid_arrays(
        self: IQCavityFeedbackTimingClass,
    ) -> None:
        """
        Rebuild the flat ``rf_centers`` / ``rf_centers_lengths`` from segments.

        :attr:`_segments` is the single source of truth for the per-turn coarse
        grid. The flat arrays are derived views kept because the tracking hot
        path indexes ``rf_centers`` directly; rebuilding them from the segment
        list on every mutation makes the two impossible to desync.
        """
        if self._segments:
            self._rf_centers = np.concatenate(
                [segment.centers for segment in self._segments]
            )
            self._rf_centers_lengths = np.array(
                [len(segment) for segment in self._segments], dtype=int
            )
        else:
            self._rf_centers = np.zeros(0)
            self._rf_centers_lengths = np.zeros(0, dtype=int)

    def _append_segment(
        self: IQCavityFeedbackTimingClass, segment: RFCenterSegment
    ) -> None:
        """
        Append a coarse-grid segment and refresh the derived flat arrays.

        Parameters
        ----------
        segment : RFCenterSegment
            The generated segment to add to the per-turn grid.
        """
        self._segments.append(segment)
        self._rebuild_grid_arrays()

    def _clear_segments(self: IQCavityFeedbackTimingClass) -> None:
        """Drop all segments (start-of-turn) and clear the derived arrays."""
        self._segments = []
        self._rebuild_grid_arrays()

    def _close_previous_turn_grid(
        self: IQCavityFeedbackTimingClass,
    ) -> None:
        """
        Carry the previous turn's tail over, then clear the segments.

        Notes
        -----
        ORDERING: both carries must be captured while the previous turn's
        grid is still standing. The centre sentinel needs ``_rf_centers``,
        which :meth:`_clear_segments` empties; the residual needs
        ``_residual_time_last_rf_centers_calculation``, which this turn's
        first :meth:`_generate_rf_centers` overwrites. The sentinel is
        consumed only as a ``None`` / not-``None`` first-turn flag by the
        host's ``_circuit_track_cells_python`` and ``_coarse_step_sizes``;
        losing it silently reverts every turn to the first-turn step proxy.
        """
        if len(self._rf_centers) != 0:
            self._last_rf_centers_entry = self._rf_centers[-1]

        # The first coarse cell of the new turn steps across the turn
        # boundary, so it needs the tail the PREVIOUS turn ended on -- which
        # is what the live scalar still holds at this exact point (see
        # _preceding_segment_residual).
        self._residual_time_carried_into_turn = (
            self._residual_time_last_rf_centers_calculation
        )

        self._clear_segments()

    def _generate_backfill_segments_if_due(
        self: IQCavityFeedbackTimingClass, beam: BeamBaseClass
    ) -> None:
        """
        Back-fill the segments elapsed since the last passage, if any.

        The backfill walk reconstructs the interval the previous forward
        projection did not already cover. It is therefore skipped when that
        projection stopped at this very station (the whole turn was tracked
        forward), and -- when no projection has been made yet -- runs only
        on the very first turn, whose interval reaches back to the start of
        the simulation.

        Parameters
        ----------
        beam
            Beam object to receive the reference frame.
        """
        if self._tracked_forward_until_element is not None:  # noqa: SIM102
            if (
                self._tracked_forward_until_element
                is not self._parent_rf_station
            ):  # otherwise, the full turn was already tracked
                self.calculate_rf_centers_for_backfill(beam=beam)
        elif self._parent_rf_station._turn_counter.value == 0:
            # at first call, this always needs to be tracked, since the values from the start of the simulation until now are not retrieved yet.
            self.calculate_rf_centers_for_backfill(beam=beam)

    def _validate_grid(self: IQCavityFeedbackTimingClass) -> None:
        """
        Assert the derived flat arrays are consistent with the segment list.

        Cheap invariant check run once per turn after grid generation. With the
        arrays derived from :attr:`_segments` this can only fail if a code path
        mutates ``rf_centers`` / ``rf_centers_lengths`` directly instead of
        going through :meth:`_append_segment` / :meth:`_clear_segments`.
        """
        segment_lengths = [len(segment) for segment in self._segments]
        assert list(self._rf_centers_lengths) == segment_lengths, (
            f"rf_centers_lengths {list(self._rf_centers_lengths)} out of sync "
            f"with segment lengths {segment_lengths}"
        )
        assert len(self._rf_centers) == sum(segment_lengths), (
            f"rf_centers length {len(self._rf_centers)} != sum of segment "
            f"lengths {sum(segment_lengths)}"
        )

    def _preceding_segment_residual(
        self: IQCavityFeedbackTimingClass, start_index: int
    ) -> float:
        """
        Residual [s] of the segment preceding the one at ``start_index``.

        ``rf_centers`` are segment-LOCAL times, so the coarse step into the
        first cell of a segment is that cell's local time plus the
        *preceding* segment's residual -- the unfilled tail between the
        preceding segment's last centre and its end. The whole per-turn grid
        is generated before any of it is walked, so the live
        ``_residual_time_last_rf_centers_calculation`` holds the
        last-generated (forward) segment's value by consumption time, not
        the preceding one's; read :attr:`_segments` instead.

        The first segment of a turn steps across the turn boundary and takes
        the residual the previous turn ended on
        (``_residual_time_carried_into_turn``, snapshotted before this
        turn's generation). Hand-built grids (tests, direct
        ``circuit_track`` callers) carry no segment list and fall back to the
        live scalar, reproducing the historical value bit-for-bit.

        Parameters
        ----------
        start_index : int
            First ``rf_centers`` index of the segment about to be walked.

        Returns
        -------
        float
            Unfilled tail [s] of the preceding segment.
        """
        offset = 0
        for segment_index, segment in enumerate(self._segments):
            # Segment offsets are strictly increasing (every segment holds
            # >= 2 centres, enforced in RFCenterSegment), so at most one
            # segment can match start_index.
            if offset == start_index:
                if segment_index == 0:
                    return (
                        self._residual_time_last_rf_centers_calculation
                        if self._residual_time_carried_into_turn is None
                        else self._residual_time_carried_into_turn
                    )
                return self._segments[segment_index - 1].residual
            offset += len(segment)
        # Falling through is legal for exactly ONE case: a hand-built grid
        # with no segments at all (tests, direct ``circuit_track`` callers),
        # which reproduces the historical live-scalar value. On a real
        # per-turn grid an unmatched start_index means the caller sliced
        # mid-segment, and the live scalar holds THIS turn's forward tail --
        # a plausible-but-wrong step. Trip immediately instead.
        assert not self._segments, (
            f"start_index {start_index} is not a segment boundary of the "
            "per-turn grid; segment start offsets are "
            f"{np.cumsum([0, *(len(seg) for seg in self._segments)])[:-1]}"
        )
        return self._residual_time_last_rf_centers_calculation

    def _backfill_walk_reached_beam(
        self: IQCavityFeedbackTimingClass, beam_reference_time: float
    ) -> bool:
        """
        Whether the backfill walk has reached the current beam's arrival.

        The walk reconstructs the ring interval between the end of the last
        forward projection and this passage's arrival, so it must stop at
        the FIRST element whose accumulated reference time reaches that
        arrival. Every further element adds a segment of grid for ring
        elements the beam has not traversed since the previous passage --
        segments later replayed into the cavity envelope as elapsed time,
        and leaving ``_reference_state_until_tracked`` (the seed of the
        NEXT forward projection) past the beam.

        Both stages of the walk -- the tail of the previous turn and the
        wrap-around into the current one -- stop through this one test so
        they cannot drift apart. Only the TARGET is passed in: both stages
        advance the same ``_reference_state_until_tracked`` object, so
        taking the walked time off the instance is what keeps the two call
        sites comparing the same pair of clocks. A stage testing for
        equality alone runs past the arrival whenever the two clocks
        disagree by more than the tolerance -- exactly the situation the
        warning below reports.

        Parameters
        ----------
        beam_reference_time : float
            Reference time [s] of the beam being tracked now: the end of
            the interval the walk has to reconstruct.

        Returns
        -------
        bool
            True once the walked reference has reached (to within 1e-12
            relative) or passed ``beam_reference_time``.

        Warns
        -----
        UserWarning
            If the walked reference lands strictly ABOVE the beam's, i.e.
            the two clocks have diverged rather than merely met.
        """
        walked_time = self._reference_state_until_tracked.time
        is_close = np.isclose(
            walked_time, beam_reference_time, rtol=1e-12, atol=0
        )
        is_above = walked_time > beam_reference_time
        if is_above and not is_close:
            warnings.warn(
                "Inconsistency with references, is a "
                "delta_omega_rf applied to the rf_stations?",
                stacklevel=2,
            )
        return is_close or is_above

    def _generate_rf_centers(
        self: IQCavityFeedbackTimingClass,
        t_rf,
        omega_design,
        until_time: float,
    ):
        """
        Generate the coarse-grid centre times of one segment.

        The returned times are segment-local (they start near 0, not at an
        absolute simulation time) and are laid out on the **design** RF
        clock: both ``t_rf`` and ``omega_design`` below carry the design
        frequency (``calc_omega_rf_design``), never the actual/offset one.
        The RF-frequency offset reaches the feedback only as a demodulation
        and readout phase, which keeps the grid geometry a pure function of
        the design frequency and the reference times.

        The first centre normally sits half an RF period into the segment
        (the falling-edge zero of ``sin(omega t)``). When the coarse grid
        sub-divides the RF period (``n_rf_periods_per_coarse_grid < 1``) the
        centres instead continue the previous segment's tiling -- one full
        *previous* step after its last centre -- so the tiling stays
        continuous across segment and turn boundaries.

        Parameters
        ----------
        t_rf
            RF period of this segment on the design clock [s].
        omega_design
            Design RF frequency of this segment [rad/s]; it is stored as
            ``_last_segment_omega_design`` for the next segment's
            sub-stepping continuation.
        until_time
            Duration of the segment [s]. Centres are generated up to, and
            excluding, this local time.

        Returns
        -------
        rf_centers
            Segment-local coarse-grid centre times [s]. Empty (with a
            warning) when the segment is shorter than one coarse step; the
            ``RFCenterSegment`` the caller then constructs rejects such a
            degenerate (fewer than two centres) segment with an actionable
            ``ValueError``.

        Notes
        -----
        Side effects on the host, all carrying the tiling to the next
        segment: ``_residual_time_last_rf_centers_calculation`` (the unfilled
        tail between the last centre and ``until_time``),
        ``_residual_taps_last_rf_centers_calculation`` (that tail in RF
        periods) and ``_last_segment_omega_design``.
        """
        # Segments always seed at the design bucket phase: the first centre
        # sits on the falling-edge zero half an RF period into the segment
        # (sin(omega * t) crosses zero falling at t = t_rf / 2). Station
        # phases (phi_rf_design, the accumulated delta_omega_rf kick-clock
        # slip) never shift the seed -- they are applied as explicit
        # demodulation/readout phases, keeping the grid geometry a pure
        # function of the design frequency and the reference times.
        time_to_next_falling_edge_zero = t_rf / 2

        step_width_rf_centers = t_rf * self.n_rf_periods_per_coarse_grid
        if (
            self._residual_taps_last_rf_centers_calculation != 0
            and self.n_rf_periods_per_coarse_grid < 1
        ):
            # Sub-stepping (n < 1): the coarse grid sub-divides the RF period,
            # so the centres tile continuously across the turn boundary rather
            # than re-aligning to an RF bucket. The first centre of this turn
            # lies one full step after the previous turn's last centre, i.e.
            # (step - residual) into the new turn. The residual was measured
            # against the *previous* segment's step, so use that step
            # (last_segment_omega_design), not the current one -- under
            # acceleration/detuning they differ and mixing them places the
            # first centre at the wrong (possibly negative) offset. The
            # phase-based falling-edge start is only used to seed the very
            # first turn (when there is no residual yet).
            step_width_previous = (
                self.n_rf_periods_per_coarse_grid
                * 2
                * np.pi
                / self._last_segment_omega_design
            )
            time_to_next_falling_edge_zero = (
                step_width_previous
                - self._residual_time_last_rf_centers_calculation
            )
        elif (
            self._residual_taps_last_rf_centers_calculation != 0
            and self.n_rf_periods_per_coarse_grid != 1
        ):
            # while time_to_next_falling_edge_zero + self._residual_taps_last_rf_centers_calculation < step_width_rf_centers:
            time_to_next_falling_edge_zero += t_rf * (
                self.n_rf_periods_per_coarse_grid
                - int(self._residual_taps_last_rf_centers_calculation)
                - 1
            )
        rf_centers = np.arange(
            start=time_to_next_falling_edge_zero,
            stop=until_time,  # ensure that the last value is taken even with float precision
            step=step_width_rf_centers,
        )

        if len(rf_centers) == 0:
            warnings.warn(
                f"no rf centers in turn {self._parent_rf_station._turn_counter.value} at {self.section_index}",
                stacklevel=2,
            )
            # A segment shorter than one coarse step contains no centre
            # (fine sectioning: a backfill segment shorter than step_width).
            # Return the empty array rather than indexing rf_centers[-1]
            # below; the RFCenterSegment the caller constructs right after
            # rejects the degenerate segment with the actionable >=2-centres
            # ValueError, and this warning adds the turn/section context
            # that error cannot know.
            return rf_centers

        # reset with current turn
        self._residual_time_last_rf_centers_calculation = (
            until_time - rf_centers[-1]
        )
        self._residual_taps_last_rf_centers_calculation = (
            self._residual_time_last_rf_centers_calculation / t_rf
        )
        self._last_segment_omega_design = omega_design
        return rf_centers

    def calculate_rf_centers_for_forward_direction(
        self: IQCavityFeedbackTimingClass, beam: BeamBaseClass
    ) -> None:
        """
        Calculate the centers of the rf buckets in the current turn.

        Parameters
        ----------
        beam
            Beam object to receive the reference frame.
        """
        self.get_passed_time_forward_direction(beam=beam)
        # The grid stays fully on the *design* RF clock (design spacing,
        # phase 0 at the bucket edges) even under an RF-frequency offset:
        # the accumulated phase slip ``int delta_omega_rf dt`` is carried as
        # a pure phase by the parent station's kick clock (``delta_phi_rf``)
        # and applied to the beam-current demodulation (see
        # ``calculate_rf_beam_current_partial``) -- never as a time shift or
        # spacing change of the grid (see forward_segment_omega_design).

        new_rf_centers = self._generate_rf_centers(
            t_rf=(2 * np.pi / self._forward_segment_omega_design),
            omega_design=self._forward_segment_omega_design,
            until_time=self._forward_tracking_time,
        )

        self._append_segment(
            RFCenterSegment(
                omega=self._forward_segment_omega_design,
                duration=self._forward_tracking_time,
                residual=self._residual_time_last_rf_centers_calculation,
                centers=new_rf_centers,
            )
        )

    def _unify_same_frequency_time_points_backfill(
        self: IQCavityFeedbackTimingClass,
    ):
        if len(self._backfill_time_array) > 1:
            time_arr_to_use = np.copy(self._backfill_time_array)
            omega_array_to_use = np.copy(
                self._backfill_segment_omega_design_list
            )

            for omega_ind in range(1, len(omega_array_to_use)):
                if (
                    omega_array_to_use[omega_ind - 1]
                    == omega_array_to_use[omega_ind]
                ):
                    time_arr_to_use[omega_ind] += time_arr_to_use[
                        omega_ind - 1
                    ]
                    time_arr_to_use[omega_ind - 1] = 0

            mask = time_arr_to_use != 0
            self._backfill_time_array = time_arr_to_use[mask]
            self._backfill_segment_omega_design_list = omega_array_to_use[mask]

    def calculate_rf_centers_for_backfill(
        self: IQCavityFeedbackTimingClass, beam: BeamBaseClass
    ) -> None:
        """
        Compute the coarse-grid rf_centers of the backfill span.

        This function determines the design RF frequencies, which were present
        between the last call of the module and now, and then computes the
        rf-centers from based on these values.

        Parameters
        ----------
        beam
            Beam object to receive the reference frame.
        """
        if (
            self._own_index_in_reference_list == 0
            and self._tracked_forward_until_element is None
        ):
            return
        if beam.reference.time == self._reference_state_until_tracked.time:
            return

        self.get_time_omega_array_backfill(beam=beam)

        for time_ind, time in enumerate(self._backfill_time_array):
            # if time == 0:  # cavities may cause this in debug mode
            #     continue
            segment_omega_design = self._backfill_segment_omega_design_list[
                time_ind
            ]
            new_rf_centers = self._generate_rf_centers(
                t_rf=(2 * np.pi / segment_omega_design),
                omega_design=segment_omega_design,
                until_time=time,
            )
            self._append_segment(
                RFCenterSegment(
                    omega=segment_omega_design,
                    duration=time,
                    residual=self._residual_time_last_rf_centers_calculation,
                    centers=new_rf_centers,
                )
            )
