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
the forward/reverse reference walks that decide which RF frequency each grid
segment is generated at, the segment generation itself, and the derived flat
``rf_centers`` / ``rf_centers_lengths`` arrays the tracking loop indexes.

It is a *mixin*: the methods read and write grid state
(``_segments``, ``rf_centers``, ``reference_state_until_tracked``,
``residual_time_last_rf_centers_calculation``, ``phase_offset_frwrd*``, the
``last_tracked_*`` / ``reference_*`` bookkeeping, ...) that the host feedback
class initialises in ``__init__`` / ``on_run_simulation``, and the RF-parameter
accessors (``omega_rf``, ``harmonic``, ``t_rf``, ``sampling_time_coarse``, ...)
it exposes. Extracted verbatim from ``cavity_feedback.py`` for readability; the
behaviour is unchanged.
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


class RFCenterGridMixin:
    """Coarse-grid (``rf_centers``) construction mixin (see module docstring)."""

    def _reference_list_for_direction(
        self, is_counter_rotating: bool
    ) -> tuple[AltersReference, ...]:
        """
        Reference-altering element list for one beam direction.

        Both the forward-projection and the reverse back-fill traversals walk
        the ring's reference-altering elements; a counter-rotating beam sees
        them in the reversed order. Selecting the list through this one helper
        keeps the two traversals free of duplicated direction ``if`` ladders
        while still supporting either direction (needed for counter-rotating
        beams).

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
            self.reference_altering_elements_reverse
            if is_counter_rotating
            else self.reference_altering_elements
        )

    def _own_index_for_direction(self, is_counter_rotating: bool) -> int:
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
            self.own_index_in_reference_list_reverse
            if is_counter_rotating
            else self.own_index_in_reference_list
        )

    def get_passed_time_forward_direction(self, beam: BeamBaseClass):  # noqa: PLR0912
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
                self.last_tracked_turn_frwrd = deepcopy(self.turn_i.value)
                self.reference_turn_offset = -1
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
                                self.reference_altering_elements
                            )  # This will be the next element
                        )
                        self.last_tracked_turn_frwrd = deepcopy(
                            self.turn_i.value + 1
                        )
                        self.reference_turn_offset = 0
                        break
            else:
                next_reference_altering_element_index = -1

        self.forward_tracking_time = dummy_reference.time - start_time
        # The coarse grid must follow the *actual* RF frequency: the design
        # frequency at the tracked reference plus the station's RF-frequency
        # offset delta_omega_rf, so the rf_center spacing tracks the detuned
        # period. delta_omega_rf == 0 leaves this unchanged. Applying the
        # parent's offset across the whole forward-tracked segment (which
        # spans other stations' sections) is safe because the RF station
        # forbids changing delta_omega_rf during the run when the ring has
        # more than one station (see RFStationBaseClass.delta_omega_rf).
        self.forward_tracking_omega_rf = (
            self._parent_rf_station.calc_omega_rf_design(
                dummy_reference.beta, self.ring.circumference
            )
            + self.delta_omega_rf
        )
        self.tracked_forward_until_element = (
            forward_list[
                next_reference_altering_element_index % len(forward_list)
            ]
            if next_reference_altering_element_index != -1
            else self._parent_rf_station
        )
        self.reference_index_until_tracked = (
            self.reference_altering_elements.index(
                self.tracked_forward_until_element
            )
        )
        self.reference_index_until_tracked_reverse = (
            self.reference_altering_elements_reverse.index(
                self.tracked_forward_until_element
            )
        )
        self.last_tracked_beam_state_frwrd = beam.is_counter_rotating
        self.reference_state_until_tracked = dummy_reference

        if self.debug:
            if (
                next_reference_altering_element_index == -1
                or next_reference_altering_element_index
                >= len(self.reference_altering_elements)
            ):
                # either none were found or it is around two turns
                self.current_slice_elements_forward = (
                    self.reference_altering_elements[
                        self.own_index_in_reference_list :
                    ]
                )
                self.current_slice_elements_forward += (
                    self.reference_altering_elements[
                        0 : next_reference_altering_element_index
                        - len(self.reference_altering_elements)
                    ]
                )
            else:  # element is in the same turn
                self.current_slice_elements_forward = self.reference_altering_elements[
                    self.own_index_in_reference_list : next_reference_altering_element_index
                ]

    def get_time_omega_array_reverse_direction(self, beam: BeamBaseClass):  # noqa: PLR0912, PLR0915
        """
        Determine the slice of elements, which should be tracked in the reverse direction.

        Only gets called after the first turn.

        Parameters
        ----------
        beam
            Beam object to receive the reference frame.
        """
        time_list = []
        omega_list = []
        start_time = self.reference_state_until_tracked.time

        found = False

        if self.turn_i.value > self.last_tracked_turn_frwrd:
            reference_turn_offset = -1
        elif self.turn_i.value == self.last_tracked_turn_frwrd:
            reference_turn_offset = 0
        else:
            raise RuntimeError("Turn value not possible, was a turn skipped?")

        if self.last_tracked_beam_state_frwrd is not None:
            # Continue from where the last forward projection stopped, in that
            # beam's direction.
            reverse_tracking_list = self._reference_list_for_direction(
                self.last_tracked_beam_state_frwrd
            )
            start_index = (
                self.reference_index_until_tracked_reverse
                if self.last_tracked_beam_state_frwrd
                else self.reference_index_until_tracked
            )
        else:
            # first turn, nothing has been tracked yet.
            reverse_tracking_list = self._reference_list_for_direction(
                beam.is_counter_rotating
            )
            start_index = 0

        for element in reverse_tracking_list[
            start_index:
        ]:  # iterate through remaining last turn
            element: AltersReference  # TODO: are duplicate elements allowed in pipeline?
            if isinstance(
                element, RFStationBaseClass
            ):  # and element == self.tracked_forward_until_element:
                # Since we are in the previous turn, we need to decrease this manually
                # and increase it afterwards (only for cavities in case of scheduled acceleration).
                # this is not strictly true for all cases, but only cases, where the reference crosses the turn border on the forward tracking
                element._turn_counter._value += reference_turn_offset
                element.track_reference(
                    self.reference_state_until_tracked,
                    beam.is_counter_rotating,
                )
            else:
                element.track_reference(
                    self.reference_state_until_tracked
                )  # no need for CR flag
            if isinstance(element, RFStationBaseClass):
                element._turn_counter._value -= reference_turn_offset

            omega_list.append(
                self._parent_rf_station.calc_omega_rf_design(
                    self.reference_state_until_tracked.beta,
                    self.ring.circumference,
                )
            )
            time_list.append(self.reference_state_until_tracked.time)
            isclose = np.isclose(
                self.reference_state_until_tracked.time,
                beam.reference.time,
                rtol=1e-12,
                atol=0,
            )
            is_above = (
                self.reference_state_until_tracked.time > beam.reference.time
            )
            if isclose or is_above:  # counterrotation should break earlier
                if is_above:
                    warnings.warn(
                        "Inconsistency with references, is a "
                        "delta_omega_rf applied to the rf_stations?",
                        stacklevel=2,
                    )
                found = True
                break

        until_index = self._own_index_for_direction(
            reverse_tracking_list is self.reference_altering_elements_reverse
        )

        if not found:
            for element in reverse_tracking_list[
                :until_index
            ]:  # iterate through initial current turn
                element: AltersReference
                if isinstance(element, RFStationBaseClass):
                    element.track_reference(
                        self.reference_state_until_tracked,
                        beam.is_counter_rotating,
                    )
                else:
                    element.track_reference(self.reference_state_until_tracked)
                omega_list.append(
                    self._parent_rf_station.calc_omega_rf_design(
                        self.reference_state_until_tracked.beta,
                        self.ring.circumference,
                    )
                )
                time_list.append(self.reference_state_until_tracked.time)
                if np.isclose(
                    self.reference_state_until_tracked.time,
                    beam.reference.time,
                    rtol=1e-12,
                    atol=0,
                ):  # counterrotation should break earlier
                    break

        if len(time_list) > 1:
            self.reverse_tracking_time_array = np.append(
                np.array(time_list[0] - start_time), np.diff(time_list)
            )
            # Track the actual RF frequency (design + delta_omega_rf), see
            # forward_tracking_omega_rf. delta_omega_rf == 0 leaves it unchanged.
            self.reverse_tracking_omega_list = (
                np.array(omega_list) + self.delta_omega_rf
            )
        else:
            self.reverse_tracking_time_array = np.array(time_list)
            # Track the actual RF frequency (design + delta_omega_rf), see
            # forward_tracking_omega_rf. delta_omega_rf == 0 leaves it unchanged.
            self.reverse_tracking_omega_list = (
                np.array(omega_list) + self.delta_omega_rf
            )

        self._unify_same_frequency_time_points_reverse()

        if self.debug:
            self.reference_time_after_reverse = (
                self.reference_state_until_tracked.time
            )
            self.current_beam_reference_time = beam.reference.time
            self.reference_energy_after_reverse = (
                self.reference_state_until_tracked.total_energy
            )
            self.current_beam_reference_energy = beam.reference.total_energy

    @staticmethod
    def _get_time_to_next_rising_edge_zero(
        phi: float, frequency: float
    ) -> float:
        phi_modulated = np.mod(phi, 2 * np.pi)
        return np.mod(np.pi - phi_modulated, 2 * np.pi) / frequency

    def _rebuild_grid_arrays(self) -> None:
        """
        Rebuild the flat ``rf_centers`` / ``rf_centers_lengths`` from segments.

        :attr:`_segments` is the single source of truth for the per-turn coarse
        grid. The flat arrays are derived views kept because the tracking hot
        path indexes ``rf_centers`` directly; rebuilding them from the segment
        list on every mutation makes the two impossible to desync.
        """
        if self._segments:
            self.rf_centers = np.concatenate(
                [segment.centers for segment in self._segments]
            )
            self.rf_centers_lengths = np.array(
                [len(segment) for segment in self._segments], dtype=int
            )
        else:
            self.rf_centers = np.zeros(0)
            self.rf_centers_lengths = np.zeros(0, dtype=int)

    def _append_segment(self, segment: RFCenterSegment) -> None:
        """
        Append a coarse-grid segment and refresh the derived flat arrays.

        Parameters
        ----------
        segment : RFCenterSegment
            The generated segment to add to the per-turn grid.
        """
        self._segments.append(segment)
        self._rebuild_grid_arrays()

    def _clear_segments(self) -> None:
        """Drop all segments (start-of-turn) and clear the derived arrays."""
        self._segments = []
        self._rebuild_grid_arrays()

    def _validate_grid(self) -> None:
        """
        Assert the derived flat arrays are consistent with the segment list.

        Cheap invariant check run once per turn after grid generation. With the
        arrays derived from :attr:`_segments` this can only fail if a code path
        mutates ``rf_centers`` / ``rf_centers_lengths`` directly instead of
        going through :meth:`_append_segment` / :meth:`_clear_segments`.
        """
        segment_lengths = [len(segment) for segment in self._segments]
        assert list(self.rf_centers_lengths) == segment_lengths, (
            f"rf_centers_lengths {list(self.rf_centers_lengths)} out of sync "
            f"with segment lengths {segment_lengths}"
        )
        assert len(self.rf_centers) == sum(segment_lengths), (
            f"rf_centers length {len(self.rf_centers)} != sum of segment "
            f"lengths {sum(segment_lengths)}"
        )

    def _generate_rf_centers(self, t_rf, omega_rf, phi_rf, until_time: float):
        time_to_next_falling_edge_zero = (
            self._get_time_to_next_rising_edge_zero(
                phi_rf,
                omega_rf,
            )
        )

        # 2nd part of if: floating precision would miss this in the last turn, hence has to be done this turn
        if time_to_next_falling_edge_zero <= 0 and not np.isclose(
            self.residual_taps_last_rf_centers_calculation, 1
        ):
            time_to_next_falling_edge_zero += t_rf

        step_width_rf_centers = t_rf * self.n_rf_periods_per_coarse_grid
        if (
            self.residual_taps_last_rf_centers_calculation != 0
            and self.n_rf_periods_per_coarse_grid < 1
        ):
            # Sub-stepping (n < 1): the coarse grid sub-divides the RF period,
            # so the centres tile continuously across the turn boundary rather
            # than re-aligning to an RF bucket. The first centre of this turn
            # lies one full step after the previous turn's last centre, i.e.
            # (step - residual) into the new turn. The residual was measured
            # against the *previous* segment's step, so use that step
            # (last_forward_tracking_freq), not the current one -- under
            # acceleration/detuning they differ and mixing them places the
            # first centre at the wrong (possibly negative) offset. The
            # phase-based falling-edge start is only used to seed the very
            # first turn (when there is no residual yet).
            step_width_previous = (
                self.n_rf_periods_per_coarse_grid
                * 2
                * np.pi
                / self.last_forward_tracking_freq
            )
            time_to_next_falling_edge_zero = (
                step_width_previous
                - self.residual_time_last_rf_centers_calculation
            )
        elif (
            self.residual_taps_last_rf_centers_calculation != 0
            and self.n_rf_periods_per_coarse_grid != 1
        ):
            # while time_to_next_falling_edge_zero + self.residual_time_last_rf_centers_calculation < step_width_rf_centers:
            time_to_next_falling_edge_zero += t_rf * (
                self.n_rf_periods_per_coarse_grid
                - int(self.residual_taps_last_rf_centers_calculation)
                - 1
            )
        rf_centers = np.arange(
            start=time_to_next_falling_edge_zero,
            stop=until_time,  # ensure that the last value is taken even with float precision
            step=step_width_rf_centers,
        )

        if len(rf_centers) == 0:
            warnings.warn(
                f"no rf centers in turn {self.turn_i.value} at {self.section_index}",
                stacklevel=2,
            )
            # A segment shorter than one coarse step legitimately contains no
            # centre (common with fine sectioning, where a reverse segment can
            # be shorter than step_width). Return the empty array -- callers
            # append a zero-length segment and circuit_track() no-ops over it
            # -- rather than None, which would crash len(new_rf_centers).
            return rf_centers

        # reset with current turn
        self.residual_time_last_rf_centers_calculation = (
            until_time - rf_centers[-1]
        )
        self.residual_taps_last_rf_centers_calculation = (
            self.residual_time_last_rf_centers_calculation / t_rf
        )
        self.last_forward_tracking_freq = omega_rf
        return rf_centers

    def calculate_rf_centers_for_forward_direction(
        self, beam: BeamBaseClass
    ) -> None:
        """
        Calculate the centers of the rf buckets in the current turn.

        Parameters
        ----------
        beam
            Beam object to receive the reference frame.
        """
        self.get_passed_time_forward_direction(beam=beam)
        self.phase_offset_frwrd += self.phase_offset_frwrd_next
        # Per-turn phase slip of the RF clock from the RF-frequency offset:
        # over one turn (2*pi*harmonic / omega_rf_design seconds) a frequency
        # offset delta_omega_rf advances the RF phase by
        # delta_omega_rf * turn_time. Accumulating it into phase_offset_frwrd
        # keeps the baseband/demodulated representation continuous across the
        # turn boundary when the RF is detuned. delta_omega_rf == 0 leaves
        # phase_offset_frwrd at 0 (unchanged behaviour).
        self.phase_offset_frwrd_next = (
            2.0
            * np.pi
            * self.harmonic
            * self.delta_omega_rf
            / self._parent_rf_station.calc_omega_rf_design(
                beam_beta=self.reference_state_until_tracked.beta,
                ring_circumference=self.ring.circumference,
            )
        )

        new_rf_centers = self._generate_rf_centers(
            t_rf=(2 * np.pi / self.forward_tracking_omega_rf),
            # TODO: this is indeed necessary for the multi-section acceleration tracking, delta_omega hast to be applied somewhere else if applicable
            omega_rf=self.forward_tracking_omega_rf,
            phi_rf=self.phase_offset_frwrd,  # phase_offset_frwrd,
            until_time=self.forward_tracking_time,
        )

        self._append_segment(
            RFCenterSegment(
                omega=self.forward_tracking_omega_rf,
                duration=self.forward_tracking_time,
                residual=self.residual_time_last_rf_centers_calculation,
                centers=new_rf_centers,
            )
        )

    def _unify_same_frequency_time_points_reverse(self):
        if len(self.reverse_tracking_time_array) > 1:
            time_arr_to_use = np.copy(self.reverse_tracking_time_array)
            omega_array_to_use = np.copy(self.reverse_tracking_omega_list)

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
            self.reverse_tracking_time_array = time_arr_to_use[mask]
            self.reverse_tracking_omega_list = omega_array_to_use[mask]

    def calculate_rf_centers_for_reverse_direction(
        self, beam: BeamBaseClass
    ) -> None:
        """
        Compute the coarse-grid rf_centers for the reverse-tracking direction.

        This function determines the omega_rf values, which were present
        between the last call of the module and now, and then computes the
        rf-centers from based on these values.

        Parameters
        ----------
        beam
            Beam object to receive the reference frame.
        """
        if (
            self.own_index_in_reference_list == 0
            and self.tracked_forward_until_element is None
        ):
            return
        if beam.reference.time == self.reference_state_until_tracked.time:
            return

        self.get_time_omega_array_reverse_direction(beam=beam)

        for time_ind, time in enumerate(self.reverse_tracking_time_array):
            # if time == 0:  # cavities may cause this in debug mode
            #     continue
            new_rf_centers = self._generate_rf_centers(
                t_rf=(2 * np.pi / self.reverse_tracking_omega_list[time_ind]),
                omega_rf=self.reverse_tracking_omega_list[time_ind],
                phi_rf=self.phi_rf,
                # The parent station accumulates the delta_omega_rf phase slip
                # exactly, from the elapsed reference time (see
                # RFStationBaseClass._update_delta_phi_rf_from_beam_feedback).
                # TODO: phi_rf is the phase at the *current* passage; with
                #  delta_omega_rf != 0 each reverse segment would need the
                #  phase at its own start, phi_rf - delta_omega_rf *
                #  (t_now - t_segment_start), reconstructable from the
                #  segment times gathered above.
                until_time=time,
            )
            self._append_segment(
                RFCenterSegment(
                    omega=self.reverse_tracking_omega_list[time_ind],
                    duration=time,
                    residual=self.residual_time_last_rf_centers_calculation,
                    centers=new_rf_centers,
                )
            )
