# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Holds the `MainloopSingleBeam` class.

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

from blond.core.simulation.execution_models.base import ExecutionModel

if TYPE_CHECKING:  # pragma: no cover
    from blond import Simulation
    from blond.core.beam.base import BeamBaseClass
    from blond.handle_results.observables import ObservablesOncePerTurnBase

    CallbackTypeHint = Callable[["Simulation", BeamBaseClass], None]

logger = logging.getLogger(__name__)


class MainloopSingleBeam(ExecutionModel):
    """Most basic executor for a single beam."""

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
        Execute the beam dynamics simulation for only one beam.

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
        assert len(beams) == 1, f"{beams=}"

        if n_turns != 1 and until_section_index != -1:
            warnings.warn(
                f"n_turns is ignored since until_section_index was {until_section_index}",
                stacklevel=1,
            )

        beam = beams[0]
        logger.info("Starting simulation mainloop...")
        callbacks = simulation._sanitize_callbacks(callbacks)

        iterator = range(
            simulation.turn_i.value, simulation.turn_i.value + n_turns
        )
        if show_progressbar:
            # Get particle count for throughput calculation
            n_particles = beam._dt.global_size
            iterator = tqdm(iterator, desc="BLonD3 mainloop")  # Add TQDM
            # display to iteration
        for turn_i in iterator:
            if show_progressbar:
                self._add_progressbar_info(
                    iterator, n_particles, show_progressbar
                )

            simulation.turn_i.value = turn_i

            simulation._calculate_current_t_rev(reference=beam.reference)
            for element in simulation._ring.elements.elements:
                simulation.section_i.value = element.section_index
                if simulation.section_i.value >= until_section_index != -1:
                    return
                if element.is_active_this_turn(turn_i=simulation.turn_i.value):
                    element.track(beam=beam)
            for observable in observe:
                if observable.is_active_this_turn(
                    turn_i=simulation.turn_i.value
                ):
                    observable.update()
            for callback in callbacks:
                if (turn_i % callback.each_turn_i) == 0:  # NOQA duck-typing
                    callback(simulation, beam)

        # make possible to run two main-loops after each other
        simulation.turn_i.value += 1
        simulation.section_i.value = 0

    @staticmethod
    def _add_progressbar_info(
        iterator: tqdm, n_particles: int, show_progressbar: bool
    ):
        # Update tqdm with particle throughput metrics
        if show_progressbar and hasattr(iterator, "format_dict"):
            # Get current iteration rate (it/s) from tqdm
            format_dict = iterator.format_dict
            rate = format_dict.get("rate", None)
            if rate is not None and rate > 0 and n_particles > 0:
                # Calculate particles per second per iteration
                sec_per_particle = 1 / rate / n_particles
                # Format in scientific notation for readability
                particles_str = f"{sec_per_particle * 1e9:f}ns per particle"

                iterator.set_postfix_str(f"{particles_str}", refresh=False)
