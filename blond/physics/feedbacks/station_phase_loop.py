# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Per-station beam phase loop.

A ring element placed directly in front of an RF station, in one beam's
traversal order. At every passage it measures that beam's centroid RF
phase against a reference, remembers the error, and writes the station's
:attr:`~blond.physics.cavities.RFStationBaseClass.phi_rf_loop` for the kick
the bunch is about to receive::

    phi_rf_loop = -gain * error(delay_stations passages ago)

``delay_stations = 0`` acts on the measurement just taken (an unphysical
zero-delay loop, useful as a bound), ``delay_stations = 1`` on the one taken
at the previous station the bunch passed.

Why per station and not once per turn: on a ring whose synchrotron tune is
of order one per turn (the muon-collider RCS advance 1.25 synchrotron
periods per turn at injection) a once-per-turn record of the bunch phase is
aliased and a once-per-turn correction is more than a synchrotron period
late; the stations sample the oscillation many times per period instead.
The global loops of :mod:`blond.physics.feedbacks.beam_feedback` stay what
they are -- their coupling to a cavity feedback is a deliberate non-goal --
while this element couples to the station's own RF phase, and through it
to the station's cavity feedback, which treats the offset as a step of the
RF reference (see
``IQCavityFeedbackTimingClass._absorb_phase_loop_step``).

The loop is a phase actuator: an RF phase offset is a thin lens on the
bunch's energy coordinate, so it damps only when the error it is built
from is about a quarter synchrotron period old. Which sign damps, and
the gain for a wanted damping time, follow from the linear model of the
lumped lattice (the muon-collider example's ``phase_loop_analysis``);
tracking confirmed that a positive gain damps with a one-station-old
measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from blond.core.base import BeamPhysicsRelevant
from blond.core.beam.beams import ProbeBeam

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass
    from blond.physics.cavities import RFStationBaseClass


def wrap_phase(phase: float) -> float:
    """
    Fold an RF phase into ``(-pi, pi]``.

    Parameters
    ----------
    phase
        Phase [rad].

    Returns
    -------
    wrapped
        The same angle in ``(-pi, pi]``.
    """
    return float((phase + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class StationPhaseLoopRecord:
    """
    What one beam's loop elements measured and did, newest last.

    One record is shared by all the elements acting on one beam, so the
    delayed measurement an element acts on may come from another station.
    """

    turns: list[float] = field(default_factory=list)
    """Fractional turn of each measurement, i.e. the element's passage
    count plus its ``turn_fraction``."""
    errors: list[float] = field(default_factory=list)
    """Centroid phase error at each measurement [rad]."""
    corrections: list[float] = field(default_factory=list)
    """RF phase offset written at each passage [rad]."""

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The three records as arrays sorted by turn.

        Returns
        -------
        turns, errors, corrections
            Time-ordered copies.
        """
        order = np.argsort(self.turns, kind="stable")
        return (
            np.asarray(self.turns, dtype=float)[order],
            np.asarray(self.errors, dtype=float)[order],
            np.asarray(self.corrections, dtype=float)[order],
        )


class StationPhaseLoop(BeamPhysicsRelevant):
    """
    One beam's phase-loop element in front of one RF station.

    Parameters
    ----------
    station
        The RF station the element precedes in ``beam``'s traversal order;
        its ``phi_rf_loop`` is written.
    beam
        The beam this element acts on; every other beam, and every
        :class:`~blond.core.beam.beams.ProbeBeam`, is ignored.
    reference_phase
        Centroid RF phase the loop regulates to [rad], e.g. the launch
        phase of the matched bunch; may be set later through the attribute
        of the same name.
    gain
        RF phase offset per radian of measured error [1]; ``0`` records
        without acting.
    delay_stations
        Age of the measurement the correction is built from, in passages
        of this beam: ``0`` its own, ``1`` the previous station's.
    turn_fraction
        Fraction of the beam's own turn elapsed where the element sits,
        used to time-stamp the record.
    record
        Shared record of this beam's measurements; a fresh one if
        ``None``.
    section_index
        Ring section the element belongs to.
    name
        Element name.

    Raises
    ------
    ValueError
        If ``delay_stations`` is negative.
    """

    def __init__(  # noqa: PLR0913 - one argument per physical quantity
        self,
        *,
        station: RFStationBaseClass,
        beam: BeamBaseClass,
        reference_phase: float,
        gain: float,
        delay_stations: int,
        turn_fraction: float = 0.0,
        record: StationPhaseLoopRecord | None = None,
        section_index: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(section_index=section_index, name=name)
        if delay_stations < 0:
            raise ValueError(f"delay_stations={delay_stations} must be >= 0")
        self._station = station
        self._beam_id = id(beam)
        #: Centroid RF phase the loop regulates to [rad].
        self.reference_phase = float(reference_phase)
        self._gain = float(gain)
        self._delay = int(delay_stations)
        self._turn_fraction = float(turn_fraction)
        self._record = StationPhaseLoopRecord() if record is None else record
        self._passages = 0

    @property
    def gain(self) -> float:
        """
        RF phase offset per radian of measured error [1].

        Returns
        -------
        gain
            The loop gain.
        """
        return self._gain

    @property
    def delay_stations(self) -> int:
        """
        Age of the measurement the correction is built from.

        Returns
        -------
        delay_stations
            In passages of the beam this element acts on.
        """
        return self._delay

    @property
    def record(self) -> StationPhaseLoopRecord:
        """
        The shared record of this beam's measurements and corrections.

        Returns
        -------
        record
            The record this element writes to.
        """
        return self._record

    def _track(self, beam: BeamBaseClass) -> None:
        if isinstance(beam, ProbeBeam) or id(beam) != self._beam_id:
            return
        if beam.n_macroparticles_partial == 0:
            return
        omega = float(self._station.omega_rf_design)
        phase = omega * float(beam._dt.mean())
        error = wrap_phase(phase - self.reference_phase)
        record = self._record
        record.turns.append(self._passages + self._turn_fraction)
        record.errors.append(error)
        self._passages += 1
        index = len(record.errors) - 1 - self._delay
        used = record.errors[index] if index >= 0 else 0.0
        correction = -self._gain * used
        record.corrections.append(correction)
        self._station.phi_rf_loop = correction
