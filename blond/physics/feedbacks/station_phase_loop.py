# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Station-attached beam phase loop, clocked by the station's cavity feedback.

A loop attached to one RF station, the way its cavity feedback is, and run
on that feedback's controller clock. At every passage of a bunch -- either
bunch of a counter-rotating pair, the loop does not know which -- the
feedback hands the loop the bunch's centroid RF phase, which the loop
measures against a reference and appends to the station's record, stamped
with the coarse-grid cell of the passage and the bunch's reference time.
On every controller sample the loop's output is::

    phi_rf_loop = -gain * error(newest entry >= n_delay samples old)

held between samples, and the feedback carries it cell by cell over its
whole grid, the empty tail of a profile window included: the RF reference
steps at the sample where a measurement becomes old enough, wherever that
falls between passages, and the cavity feedback keeps the field in place
across each step (its per-cell beam step rotations). The station's
:attr:`~blond.physics.cavities.RFStationBaseClass.phi_rf_loop` is the
offset in force at the bunch's own cell, which its kick, readout and
demodulation run in. A measurement cannot reach its own kick: the field
has no time to move. Nothing old enough writes ``0``.

Why per station and not once per turn: on a ring whose synchrotron tune is
of order one per turn (the muon-collider RCS advance 1.25 synchrotron
periods per turn at injection) a once-per-turn record of the bunch phase is
aliased and a once-per-turn correction is more than a synchrotron period
late; the stations sample the oscillation many times per period instead.
The global loops of :mod:`blond.physics.feedbacks.beam_feedback` stay what
they are -- their coupling to a cavity feedback is a deliberate non-goal --
while this loop writes the station's own RF phase.

What the bunch is kicked with is the station's *field*, which the LLRF
settles on the written reference after the reference moved: a bunch's own
command reaches it when it returns, a turn later, and a counter-rotating
bunch's command in between. On a ring symmetric for the two bunches both
carry the same dipole, so a beam-blind loop acts on one oscillation sampled
at every passage. The linear model of that (the muon-collider example's
``phase_loop_analysis``) says which gain and latency damp on which ring.

A station without a cavity feedback never clocks its loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NumpyArray

    from blond.physics.cavities import RFStationBaseClass
    from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackBase


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
    What one station's loop measured and applied, in the order it happened.

    One record per station by default; several loops may share one, in
    which case ``as_arrays`` interleaves them by time.
    """

    times: list[float] = field(default_factory=list)
    """Reference time of each passage [s], as the passing bunch carries it."""
    cells: list[int] = field(default_factory=list)
    """Coarse-grid cell of each passage, on the feedback's cell clock."""
    errors: list[float] = field(default_factory=list)
    """Centroid phase error at each passage [rad]."""
    corrections: list[float] = field(default_factory=list)
    """RF phase offset in force at each passage's kick [rad]."""

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Times, errors and corrections as arrays sorted by time.

        Returns
        -------
        times, errors, corrections
            Time-ordered copies.
        """
        order = np.argsort(self.times, kind="stable")
        return (
            np.asarray(self.times, dtype=float)[order],
            np.asarray(self.errors, dtype=float)[order],
            np.asarray(self.corrections, dtype=float)[order],
        )

    def newest_at_or_before(self, cell: int) -> int | None:
        """
        Index of the newest entry stamped at or before ``cell``.

        Parameters
        ----------
        cell
            Cell on the feedback's cell clock.

        Returns
        -------
        index
            Position in the lists, or ``None`` if no entry is that old.
            Entries are scanned from the newest, so a record appended out
            of order still yields the newest old-enough one.
        """
        best: int | None = None
        for index in range(len(self.cells) - 1, -1, -1):
            stamp = self.cells[index]
            if stamp <= cell and (best is None or stamp > self.cells[best]):
                best = index
        return best


class StationPhaseLoop:
    """
    A beam phase loop attached to one cavity feedback, which clocks it.

    Parameters
    ----------
    feedback
        The cavity feedback that clocks this loop; its ``phase_loop``
        attribute is set to this loop, and it writes its parent station's
        ``phi_rf_loop`` from this loop's output. A feedback carries at
        most one. Attaching here rather than to the station is what makes
        a loop nothing can run unrepresentable: the clock and the
        actuator come from the same object.
    reference_phase
        Centroid RF phase the loop regulates to [rad], e.g. the launch
        phase of the matched bunch; may be set later through the attribute
        of the same name.
    gain
        RF phase offset per radian of measured error [1]; ``0`` records
        without acting.
    n_delay
        Latency in controller samples: on a sample the output is built
        from the newest measurement at least this many samples old,
        whichever bunch left it. ``0`` acts from the first sample at or
        after the passage's cell.
    record
        Record to append to; a fresh one of its own if ``None``.
    name
        Loop name, for messages.

    Raises
    ------
    ValueError
        If ``n_delay`` is negative or the feedback already carries a loop.
    """

    def __init__(
        self,
        *,
        feedback: IQCavityFeedbackBase,
        reference_phase: float,
        gain: float,
        n_delay: int = 1,
        record: StationPhaseLoopRecord | None = None,
        name: str | None = None,
    ) -> None:
        if n_delay < 0:
            raise ValueError(f"n_delay={n_delay} must be >= 0")
        if getattr(feedback, "phase_loop", None) is not None:
            raise ValueError(
                f"{feedback} already carries a phase loop; it clocks one"
            )
        self._feedback = feedback
        #: Centroid RF phase the loop regulates to [rad].
        self.reference_phase = float(reference_phase)
        self._gain = float(gain)
        self._n_delay = int(n_delay)
        self._record = StationPhaseLoopRecord() if record is None else record
        self.name = name
        feedback.phase_loop = self

    @property
    def feedback(self) -> IQCavityFeedbackBase:
        """
        The cavity feedback that clocks this loop.

        Returns
        -------
        feedback
            The feedback this loop is attached to.
        """
        return self._feedback

    @property
    def station(self) -> RFStationBaseClass:
        """
        The RF station this loop regulates.

        Returns
        -------
        station
            Parent station of :attr:`feedback`, whose ``phi_rf_loop`` that
            feedback writes from this loop's output.
        """
        return self._feedback.parent_rf_station

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
    def n_delay(self) -> int:
        """
        Latency of the loop in controller samples.

        Returns
        -------
        n_delay
            Minimum age, in samples, of the measurement an output is built
            from.
        """
        return self._n_delay

    @property
    def record(self) -> StationPhaseLoopRecord:
        """
        The record of this station's measurements and applied offsets.

        Returns
        -------
        record
            The record this loop appends to.
        """
        return self._record

    def measure(
        self, phase: float, *, time: float, cell: int, applied: float
    ) -> float:
        """
        Record one passage: its centroid phase error and the offset it saw.

        Called by the station's cavity feedback once per passage, after it
        has fixed the offset that passage is kicked with.

        Parameters
        ----------
        phase
            Centroid RF phase of the passing bunch [rad].
        time
            The bunch's reference time at the passage [s].
        cell
            Coarse-grid cell of the passage on the feedback's cell clock.
        applied
            ``phi_rf_loop`` in force at the passage's kick [rad].

        Returns
        -------
        error
            The wrapped centroid phase error [rad].
        """
        error = wrap_phase(phase - self.reference_phase)
        self._record.times.append(float(time))
        self._record.cells.append(int(cell))
        self._record.errors.append(error)
        self._record.corrections.append(float(applied))
        return error

    def offsets_for_cells(
        self,
        first_cell: int,
        n_cells: int,
        *,
        controller_update_interval: int,
        carried: float,
    ) -> NumpyArray:
        """
        The loop's output over a run of coarse cells, held between samples.

        Parameters
        ----------
        first_cell
            Cell clock value of the first cell of the run.
        n_cells
            Cells in the run.
        controller_update_interval
            Cells per controller sample; a cell is a sample when its cell
            clock value is a multiple of it.
        carried
            Offset in force before the run [rad], held until the first
            sample.

        Returns
        -------
        offsets
            One offset per cell [rad]: on a sample cell ``-gain`` times the
            error of the newest measurement at least ``n_delay`` samples
            old (``0`` if there is none), otherwise the previous cell's.
        """
        offsets = np.empty(n_cells)
        value = float(carried)
        record = self._record
        delay_cells = self._n_delay * controller_update_interval
        for local in range(n_cells):
            cell = first_cell + local
            if cell % controller_update_interval == 0:
                index = record.newest_at_or_before(cell - delay_cells)
                used = record.errors[index] if index is not None else 0.0
                value = -self._gain * used
            offsets[local] = value
        return offsets
