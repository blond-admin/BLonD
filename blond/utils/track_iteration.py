# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Module to wrap tracking into an iterator with option to call
user specified functions every n turns**

:Authors: **Simon Albright**
"""

# Futurisation
from __future__ import annotations

# General imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable, Callable, Protocol, Any, Self

    from .types import Trackable

    class Predicate(Protocol):
        def __call__(
            self,
            _map: Iterable[Trackable],
            turn_number: int,
            *args: Any,
            **kwargs: Any,
        ): ...


class TrackIteration:
    """
    Class to provide an iterator for tracking with an option to run passed
    functions every n turns

    Parameters
    ----------

    track_map : iterable of objects
        Each object will be called on every turn with object.track()
    init_turn : integer
        The turn number tracking will start from, only used to initialise
        a turn counter
    final_turn : integer
        The last turn number to track next(TrackIteration) will raise
        StopIteration when turn_number == final_turn

    Attributes
    ----------
    function_list : list of functions to be called with specified interval
    """

    def __init__(
        self,
        track_map: Iterable[Trackable],
        init_turn: int = 0,
        final_turn: int = -1,
    ):
        if not all((hasattr(m, "track") for m in track_map)):
            raise AttributeError("All map objects must be trackable")

        self._map = list(track_map)
        if isinstance(init_turn, int):
            self.turn_number = init_turn
        else:
            raise TypeError("init_turn must be an integer")

        if isinstance(final_turn, int):
            self._final_turn = final_turn
        else:
            raise TypeError("final_turn must be an integer")

        self.function_list: list[tuple[Predicate, int]] = []

    def _track_turns(self, n_turns: int):
        """
        Function to track for specified number of turns
        calls next() function n_turns times
        """

        for i in range(n_turns):
            next(self)

    def add_function(
        self,
        predicate: Predicate,
        repetion_rate: int,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Takes a user defined callable and calls it every repetion_rate
        number of turns with predicate(track_map, turn_number, *args, **kwargs)
        """

        self.function_list.append(
            (self._partial(predicate, *args, **kwargs), repetion_rate)
        )

    def __next__(self) -> int:
        """
        First raises StopIteration if turn_number == final_turn

        Next calls track() from each element in trackMap list and raises
        StopIteration if no more turns available

        Finally iterates over each function specified in add_function
        and calls them with predicate(trackMap, turn_number) if
        turn_number % repetitionRate == 0
        """

        if self.turn_number == self._final_turn:
            raise StopIteration

        try:
            for m in self._map:
                m.track()
        # TODO:  Check it is the expected IndexError (caused by reaching n_turns)
        except IndexError:
            raise StopIteration

        self.turn_number += 1

        for func, rate in self.function_list:
            if self.turn_number % rate == 0:
                func(self._map, self.turn_number)

        return self.turn_number

    def __iter__(self) -> Self:
        """
        returns self
        """

        return self

    def __call__(self, n_turns: int = 1) -> int:
        """
        Makes object callable with option to specify number of tracked turns
        default tracks 1 turn
        """

        self._track_turns(n_turns)
        return self.turn_number

    def _partial(self, predicate: Callable, *args, **kwargs) -> Callable:
        """
        reimplementation of functools.partial to prepend
        rather than append to *args
        """

        def part_func(_map, turn):
            return predicate(_map, turn, *args, **kwargs)

        return part_func
