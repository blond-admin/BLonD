# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
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

    @staticmethod
    def _check_two_beam_profile_placement(simulation: Simulation) -> None:
        """
        Validate live-profile placement for two counter-rotating beams.

        A shared profile is histogrammed in place by whichever beam tracked
        it last. With two counter-rotating beams every element is tracked
        once per beam per turn -- the co-rotating beam through the elements
        in forward order, the counter-rotating beam in reverse order -- so a
        consumer (RF station, wakefield) only reads the *correct* beam's
        histogram if its profile is tracked as a ring element on **both
        sides** of the consumer: each beam then re-histograms the profile
        immediately before reaching the consumer in *its* traversal order.

        For every element that consumes a live (``active=True``) profile
        this check therefore requires the same profile object to appear in
        the one-turn element list both before and after the consumer.
        Frozen profiles (``active=False``) are deliberately static and are
        skipped.

        Parameters
        ----------
        simulation
            Adapter for the Simulation object holding the ring elements.

        Raises
        ------
        ValueError
            When a consumer's live profile is never tracked, or is tracked
            on only one side of the consumer.
        """
        elements = simulation._ring.elements.elements
        for index, element in enumerate(elements):
            # Candidate profiles this element consumes: its own, its local
            # wakefield's, and those of any attached cavity feedbacks.
            candidates = [getattr(element, "profile", None)]
            local_wakefield = getattr(element, "_local_wakefield", None)
            if local_wakefield is not None:
                candidates.append(getattr(local_wakefield, "profile", None))
            for feedback in getattr(element, "cavity_feedback_list", []):
                candidates.append(getattr(feedback, "profile", None))
            profiles = []
            for candidate in candidates:
                if candidate is None or any(candidate is p for p in profiles):
                    continue
                profiles.append(candidate)
            for profile in profiles:
                MainloopCounterRotatingBeams._check_one_profile(
                    elements, index, element, profile
                )

    @staticmethod
    def _check_one_profile(elements, index, element, profile) -> None:
        """
        Validate the placement of one consumed profile (see caller).

        Parameters
        ----------
        elements
            The one-turn ring element list.
        index
            Index of the consuming element in ``elements``.
        element
            The consuming element (for the error message).
        profile
            The profile object the element consumes.

        Raises
        ------
        ValueError
            When the live profile is never tracked, or tracked on only one
            side of the consumer.
        """
        if not getattr(profile, "active", False):
            # Frozen histogram: both beams read the same static line
            # density by construction -- nothing to validate.
            return
        tracked_before = any(e is profile for e in elements[:index])
        tracked_after = any(e is profile for e in elements[index + 1 :])
        if not tracked_before and not tracked_after:
            raise ValueError(
                f"{type(element).__name__} (element {index}) consumes a "
                "live profile that is never tracked as a ring element. "
                "With two counter-rotating beams the profile must be "
                "placed as an element on BOTH sides of its consumer "
                "(profile, consumer, profile), so each beam re-histograms "
                "it immediately before the consumer in its own traversal "
                "order -- or freeze the histogram with "
                "``profile.active = False`` if a static line density is "
                "intended."
            )
        if not (tracked_before and tracked_after):
            missing = "before" if not tracked_before else "after"
            raise ValueError(
                f"{type(element).__name__} (element {index}) consumes a "
                f"live profile that is tracked only on one side (missing "
                f"{missing} the consumer). With two counter-rotating "
                "beams the profile must be placed as an element on BOTH "
                "sides of its consumer (profile, consumer, profile): the "
                "counter-rotating beam traverses the elements in reverse "
                "order, so a one-sided profile is histogrammed with the "
                "wrong beam for one of the two passages. Alternatively "
                "freeze the histogram with ``profile.active = False``."
            )

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
        ), (
            "First beam must be co-rotating, second beam must be counter-rotating."
        )
        warnings.warn("Untested code", NotTestedWarning, stacklevel=2)

        self._check_two_beam_profile_placement(simulation)

        if callbacks is not None:
            warnings.warn(
                "Callbacks are only called once per turn and receive the first beam as an argument.",
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
        simulation.turn_counter.value = 0

        callbacks = simulation._sanitize_callbacks(callbacks)

        num_elements = len(simulation._ring.elements.elements)

        for turn_i in iterator:
            for element_ind, element in enumerate(
                simulation._ring.elements.elements
            ):
                simulation.turn_counter.value = turn_i
                section = element.section_index

                if section >= until_section_index != -1:
                    return

                if element.is_active_this_turn(
                    turn_i=simulation.turn_counter.value
                ):
                    element.track(beams[0])  # [0] is expected to be corotating

                element_counterrot = simulation.ring.elements.elements[
                    num_elements - element_ind - 1
                ]
                if element_counterrot.is_active_this_turn(
                    turn_i=simulation.turn_counter.value
                ):
                    element_counterrot.track(beams[1])

            for observable in observe:
                if observable.is_active_this_turn(
                    turn_i=simulation.turn_counter.value
                ):
                    observable.update()

            for callback in callbacks:
                if (turn_i % callback.each_turn_i) == 0:  # NOQA duck-typing
                    callback(simulation, beams[0])
