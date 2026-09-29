# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Holds the base class `ExecutionModel`.

Notes
-----
Authors:
S. Lauber
L. Thiele
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from blond.core.simulation.simulation import Simulation

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass
    from blond.handle_results.observables import ObservablesOncePerTurnBase

    CallbackTypeHint = Callable[["Simulation", BeamBaseClass], None]

logger = logging.getLogger(__name__)


def flush_before_readout(
    observe: Sequence, callbacks: Sequence, turn_i: int
) -> None:
    """
    Run queued kernel calls if anything reads the beam this turn.

    Without an active observable or callback the queue may span the turn
    boundary, fusing the end of one turn with the start of the next.

    Parameters
    ----------
    observe
        Observables of the main loop.
    callbacks
        Callbacks of the main loop, each with ``each_turn_i``.
    turn_i
        Current turn.
    """
    from blond.core.backends.backend import backend

    if any(o.is_active_this_turn(turn_i=turn_i) for o in observe) or any(
        turn_i % callback.each_turn_i == 0 for callback in callbacks
    ):
        backend.specials.flush()


class ExecutionModel(ABC):  # pragma: no cover
    """Base class to define execution strategies of a simulation."""

    @abstractmethod  # pragma: no cover
    def mainloop(
        self,
        simulation: Simulation,
        beams: tuple[BeamBaseClass, ...],
        n_turns: int,
        observe: tuple[ObservablesOncePerTurnBase, ...] = (),
        show_progressbar: bool = True,
        callbacks: Sequence[CallbackTypeHint] | CallbackTypeHint | None = None,
        until_section_index: int = -1,
    ) -> None:
        """
        Execute the beam dynamics simulation.

        Parameters
        ----------
        simulation
            Adapter for Simulation object.
        beams
            The beam to simulate.
        n_turns
            Number of turns to simulate.
        observe
            List of observables to protocol of whats happening inside
            the simulation.
        show_progressbar
            If True, will show a progress bar indicating how many turns have
            been completed and other metrics.
        callbacks
            Optional user-defined functions `[callback_1, callback_2, ...]`.
            called at the end of each turn.
            Useful for custom data collection or live plotting. Default is None.

            The callback can be defined as follows.
            The rate at with which this function is
            called can be set by `each_turn_i`.
            >>> from blond import Beam, Simulation
            >>> def my_callback(simulation: Simulation, beam: Beam) -> None:
            >>>     ...
            >>> my_callback.each_turn_i = 2
            .
        until_section_index
            Section index until which to run the simulation. Default is -1.

        Notes
        -----
        This method assumes that ``Simulation.finalize(...)`` was executed
        before.
        """
        pass
