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

        A shared profile is histogrammed *in place* by whichever beam
        tracked it last. With two counter-rotating beams every element is
        tracked once per beam per turn: within iteration ``k`` the
        co-rotating beam tracks ``elements[k]`` and *then* the
        counter-rotating beam tracks ``elements[N - 1 - k]``. A consumer
        (RF station, wakefield) reads the *correct* beam's histogram only
        if, in this interleaved schedule, the most recent beam to
        re-histogram its profile was the beam doing the reading.

        The naive ``(profile, consumer, profile)`` sandwich does **not**
        guarantee this: the two beams' in-place writes interleave, so the
        other beam's histogram can fall between a beam's own write and its
        read (this is even true for the minimal three-element sandwich).
        Rather than approximate the requirement with a placement rule,
        this check *replays the exact interleave* over the one-turn
        element list and rejects any layout in which some consumer reads
        the wrong beam's line density. Frozen profiles (``active=False``)
        are deliberately static and are exempt.

        Parameters
        ----------
        simulation
            Adapter for the Simulation object holding the ring elements.

        Raises
        ------
        ValueError
            When a consumed live profile is never tracked as a ring
            element, or when the counter-rotating interleave makes a
            consumer read the other beam's histogram (clobbering).
        """
        elements = simulation._ring.elements.elements

        # Map each element index -> the distinct live profiles it consumes.
        consumed: dict[int, list] = {}
        for index, element in enumerate(elements):
            profiles = MainloopCounterRotatingBeams._live_consumed_profiles(
                element
            )
            if profiles:
                consumed[index] = profiles

        if not consumed:
            return

        # A consumed live profile that never appears as a ring element is
        # never histogrammed for either beam -- clear, specific error.
        for index, profiles in consumed.items():
            element = elements[index]
            for profile in profiles:
                if not any(e is profile for e in elements):
                    raise ValueError(
                        f"{type(element).__name__} (element {index}) "
                        "consumes a live profile that is never tracked as "
                        "a ring element, so it is never histogrammed for "
                        "either counter-rotating beam. Add the profile as "
                        "a ring element on a safe placement, give each "
                        "beam its own profile instance, or freeze it with "
                        "``profile.active = False`` for a static line "
                        "density."
                    )

        MainloopCounterRotatingBeams._reject_clobbering_layout(
            elements, consumed
        )

    @staticmethod
    def _live_consumed_profiles(element) -> list:
        """
        Return the distinct live profiles an element consumes.

        Candidate profiles are the element's own ``profile``, its local
        wakefield's ``profile``, and those of any attached cavity
        feedbacks. Frozen profiles (``active=False``) are excluded: their
        histogram is static, so both beams read the same line density and
        there is nothing to validate.

        Parameters
        ----------
        element
            A ring element that may consume one or more profiles.

        Returns
        -------
        list
            The distinct live (``active=True``) profile objects consumed.
        """
        candidates = [getattr(element, "profile", None)]
        local_wakefield = getattr(element, "_local_wakefield", None)
        if local_wakefield is not None:
            candidates.append(getattr(local_wakefield, "profile", None))
        for feedback in getattr(element, "cavity_feedback_list", []):
            candidates.append(getattr(feedback, "profile", None))
        profiles: list = []
        for candidate in candidates:
            if candidate is None:
                continue
            if not getattr(candidate, "active", False):
                continue  # frozen histogram: static, exempt
            if any(candidate is p for p in profiles):
                continue
            profiles.append(candidate)
        return profiles

    @staticmethod
    def _reject_clobbering_layout(elements, consumed: dict) -> None:
        """
        Reject layouts where the interleave corrupts a consumer's read.

        Replays the exact mainloop schedule -- within iteration ``k`` the
        co-rotating beam tracks ``elements[k]`` then the counter-rotating
        beam tracks ``elements[N - 1 - k]`` -- tracking, per shared live
        profile, which beam most recently histogrammed it. A profile
        element writes (its ``owner`` becomes the tracking beam); a
        consumer reads and must see its own beam as the current owner. The
        schedule is run for two turns so the owner reaches steady state
        (identical every turn), and reads are checked on the second turn.

        Parameters
        ----------
        elements
            The one-turn ring element list.
        consumed
            Mapping of element index to the live profiles it consumes, as
            produced by :meth:`_live_consumed_profiles`.

        Raises
        ------
        ValueError
            On the first consumer that reads the other beam's histogram.
        """
        corruption = MainloopCounterRotatingBeams._first_corrupted_read(
            elements, consumed
        )
        if corruption is None:
            return

        index, beam = corruption
        element = elements[index]
        raise ValueError(
            f"{type(element).__name__} (element {index}) consumes a live "
            "profile that the two counter-rotating beams histogram in "
            "place in an interleaving order which corrupts it: replaying "
            "the mainloop's forward-then-counter schedule, the "
            f"{beam} beam reads the profile after the other beam "
            "re-histogrammed it, so it sees the wrong beam's line "
            "density. Note the minimal (profile, consumer, profile) "
            "sandwich is NOT safe -- the counter beam's interleaved "
            "histogram falls between the co-rotating beam's write and its "
            "read (and symmetrically for the counter beam). Freeze the "
            "histogram with ``profile.active = False`` for a static line "
            "density, or give each beam its own profile instance, or use "
            "a placement verified safe by this exact interleave replay."
        )

    @staticmethod
    def _first_corrupted_read(elements, consumed: dict):
        """
        Replay the interleave; return the first corrupted read or None.

        Builds the ordered per-turn schedule -- co-rotating ``elements[k]``
        then counter-rotating ``elements[N - 1 - k]`` -- and replays it for
        two turns (the histogram owner reaches its periodic steady state
        after one turn), checking consumer reads on the second turn.

        Parameters
        ----------
        elements
            The one-turn ring element list.
        consumed
            Mapping of element index to the live profiles it consumes.

        Returns
        -------
        tuple or None
            ``(index, beam)`` of the first consumer that reads the other
            beam's histogram, or ``None`` when every read is correct.
        """
        num_elements = len(elements)
        live_ids = {
            id(profile)
            for profiles in consumed.values()
            for profile in profiles
        }
        owner: dict[int, str | None] = dict.fromkeys(live_ids)

        schedule = []
        for element_ind in range(num_elements):
            schedule.append(("co-rotating", element_ind))
            schedule.append(
                ("counter-rotating", num_elements - 1 - element_ind)
            )

        for turn in range(2):
            checking = turn == 1
            for beam, index in schedule:
                tracked = elements[index]
                if id(tracked) in live_ids:
                    owner[id(tracked)] = beam  # histogram in place
                if not (checking and index in consumed):
                    continue
                for profile in consumed[index]:
                    if owner[id(profile)] != beam:
                        return index, beam
        return None

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
