# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Holds the `MainloopCounterRotatingBeams` class.

Notes
-----
Authors:
S. Lauber
L. Thiele
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from tqdm import tqdm  # type: ignore

from blond import Simulation
from blond.core.simulation.execution_models.base import ExecutionModel
from blond.generals.warnings_ import NotTestedWarning

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass
    from blond.handle_results.observables import ObservablesOncePerTurnBase

    CallbackTypeHint = Callable[["Simulation", BeamBaseClass], None]

logger = logging.getLogger(__name__)


class MainloopCounterRotatingBeams(ExecutionModel):
    """Executor where one beams rotates forward, and the second backwards."""

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
        Execute the beam dynamics simulation for counter-rotating beams.

        Parameters
        ----------
        simulation
            Adapter for Simulation object.
        beams
            Tuple of two beams (co-rotating, counter-rotating).
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
        until_section_index
            Section index until which to run the simulation. Default is -1.

        Examples
        --------
        >>> from blond import Beam, Simulation
        >>> def my_callback(simulation: Simulation, beam: Beam) -> None:
        ...     ...
        >>> my_callback.each_turn_i = 2
            .
        """
        assert (
            beams[0].is_counter_rotating,
            beams[1].is_counter_rotating,
        ) == (
            False,
            True,
        ), "First beam must be normal, second beam must be counter-rotating"
        warnings.warn("Untested code", NotTestedWarning, stacklevel=2)

        if callbacks is not None:
            warnings.warn(
                "Callbacks are currently not supported for simulations"
                " with counter-rotating beams.",
                UserWarning,
                stacklevel=2,
            )

        if n_turns != 1 and until_section_index != -1:
            warnings.warn(
                f"n_turns is ignored since until_section_index was {until_section_index}",
                stacklevel=1,
            )

        logger.info("Starting simulation mainloop...")
        iterator = range(n_turns)
        if show_progressbar:
            iterator = tqdm(iterator)  # Add TQDM display to iteration
        simulation.turn_i.value = 0

        num_elements = len(simulation._ring.elements.elements)

        for turn_i in iterator:
            for element_ind, element in enumerate(
                simulation._ring.elements.elements
            ):
                simulation.turn_i.value = turn_i
                simulation.section_i.value = element.section_index

                if simulation.section_i.value >= until_section_index != -1:
                    return

                if element.is_active_this_turn(turn_i=simulation.turn_i.value):
                    element.track(beams[0])  # [0] is expected to be corotating

                element_counterrot = simulation.ring.elements.elements[
                    num_elements - element_ind - 1
                ]
                if element_counterrot.is_active_this_turn(
                    turn_i=simulation.turn_i.value
                ):
                    element_counterrot.track(beams[1])

            for observable in observe:
                if observable.is_active_this_turn(
                    turn_i=simulation.turn_i.value
                ):
                    observable.update()
