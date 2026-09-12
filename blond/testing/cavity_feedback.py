# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Diagnostic variant of the IQ cavity feedback, for tests.

:class:`DiagnosticIQCavityFeedbackTimingClass` is
:class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass`
plus three switches that only tests consume, which is why the production
class does not carry them:

``debug``
    Record the inspection-only grid snapshots. Pure observation.
``validate_grid_each_turn``
    Re-check the coarse grid on every passage. Pure observation.
``grid_only_no_correction``
    Build the grid and replay the backfill span, then write the neutral
    readout instead of tracking the forward span. The feedback applies
    **no correction**; this is not a physical mode.

With all three at their ``False`` default the class tracks bit-for-bit like
the production class. They were once a single ``debug`` flag that did all
three at once, so asking for diagnostics silently switched the physics off;
only ``grid_only_no_correction`` stops the physics now.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass
    from blond.physics.feedbacks.rf_center_segment import PerTurnGridSpan


class DiagnosticIQCavityFeedbackTimingClass(IQCavityFeedbackTimingClass):
    """
    IQ cavity feedback with the test-only diagnostic switches.

    Parameters
    ----------
    *args
        Positional arguments of
        :class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass`.
    debug
        Record the inspection-only grid snapshots: after every forward
        projection the walked element slice
        ``current_slice_elements_forward``, and after every backfill walk
        ``reference_time_after_backfill`` and
        ``reference_energy_after_backfill`` (where the walk ended) beside
        ``current_beam_reference_time`` and
        ``current_beam_reference_energy`` (where the beam is). Nothing in
        BLonD reads them back. Default is False.
    validate_grid_each_turn
        After every passage's grid generation, check that the flat
        ``rf_centers`` arrays still agree with the ``_segments`` they are
        derived from, and that the forward segment's boundary residual
        equals its demodulation frame. It walks the whole grid every
        passage. Default is False.
    grid_only_no_correction
        Build the coarse grid and replay the elapsed backfill span, then
        end the passage: the beam current is never demodulated, the
        forward span is never tracked, and the parent RF station is handed
        the neutral readout -- unit relative voltage, zero phase -- so it
        kicks as if no feedback were attached. Only for inspecting the grid
        geometry in isolation. Default is False.
    **kwargs
        Keyword arguments of
        :class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass`.
    """

    def __init__(
        self,
        *args: Any,
        debug: bool = False,
        validate_grid_each_turn: bool = False,
        grid_only_no_correction: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._debug = debug
        self._validate_grid_each_turn = validate_grid_each_turn
        self._grid_only_no_correction = grid_only_no_correction

    def _record_forward_projection(
        self, next_reference_altering_element_index: int
    ) -> None:
        """
        Record the element slice the forward projection walked.

        Parameters
        ----------
        next_reference_altering_element_index
            Index of the element the projection stopped at, counted through
            ``_reference_altering_elements`` and on into the next turn;
            ``-1`` when it found none.
        """
        if not self._debug:
            return
        elements = self._reference_altering_elements
        own_index = self._own_index_in_reference_list
        if next_reference_altering_element_index == -1 or (
            next_reference_altering_element_index >= len(elements)
        ):
            # Either none was found, or the stop lies in the next turn.
            self.current_slice_elements_forward = elements[own_index:]
            self.current_slice_elements_forward += elements[
                0 : next_reference_altering_element_index - len(elements)
            ]
        else:  # the stop lies in the same turn
            self.current_slice_elements_forward = elements[
                own_index:next_reference_altering_element_index
            ]

    def get_time_omega_array_backfill(self, beam: BeamBaseClass) -> None:
        """
        Walk the backfill, then record where the walk and the beam ended.

        Parameters
        ----------
        beam
            Beam object to receive the reference frame.
        """
        super().get_time_omega_array_backfill(beam=beam)
        if not self._debug:
            return
        walked_reference = self._reference_state_until_tracked
        self.reference_time_after_backfill = walked_reference.time
        self.current_beam_reference_time = beam.reference.time
        self.reference_energy_after_backfill = walked_reference.total_energy
        self.current_beam_reference_energy = beam.reference.total_energy

    def _rebuild_per_turn_grid(self, beam: BeamBaseClass) -> PerTurnGridSpan:
        """
        Rebuild the passage's grid, then validate it if switched on.

        Parameters
        ----------
        beam
            Beam passing this station now.

        Returns
        -------
        span
            The per-turn span of the production method.
        """
        span = super()._rebuild_per_turn_grid(beam=beam)
        if self._validate_grid_each_turn:
            self._validate_grid()
            # The demodulation frame of the forward segment and the coarse
            # step into its first cell are the same physical quantity -- the
            # tail of the segment preceding the forward one. They used to be
            # derived independently (snapshot vs live scalar) and silently
            # disagreed; this ties them together.
            boundary_residual = self._preceding_segment_residual(
                span.n_backfill_centers
            )
            assert boundary_residual == span.residual_from_backfill_span, (
                f"forward-segment boundary residual {boundary_residual} != "
                f"demodulation frame {span.residual_from_backfill_span}"
            )
        return span

    def _track_forward_span(
        self, beam: BeamBaseClass, span: PerTurnGridSpan
    ) -> None:
        """
        Track the forward span, unless only the grid is wanted.

        Parameters
        ----------
        beam
            Beam passing this station now.
        span
            The span of this passage.
        """
        if self._grid_only_no_correction:
            return
        super()._track_forward_span(beam=beam, span=span)

    def _write_station_readout(self, carrier_slip_gap: float) -> None:
        """
        Write the station readout, or the neutral one in grid-only mode.

        Parameters
        ----------
        carrier_slip_gap
            The accumulated actual-RF phase [rad] of this passage.
        """
        if self._grid_only_no_correction:
            self._write_no_correction_readout()
            return
        super()._write_station_readout(carrier_slip_gap=carrier_slip_gap)

    def _write_no_correction_readout(self) -> None:
        """
        Write the neutral readout: unit gain, zero phase.

        Notes
        -----
        Unit relative voltage and zero phase make the parent station's
        ``calc_gap_voltage_with_feedbacks`` reduce to
        ``voltage * sin(omega_rf * ts + phi_rf)``, the unperturbed RF wave.
        """
        self.relative_voltage_correction = np.ones_like(self.profile.hist_x)
        self.phase_correction = np.zeros_like(self.profile.hist_x)
