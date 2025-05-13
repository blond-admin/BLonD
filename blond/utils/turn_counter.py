# Futurisation
from __future__ import annotations

# General
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional


class TurnCounter:
    def __init__(self, name: str):
        self.name = name

        self.current_turn = 0
        self.current_section = 0

        self._max_turns = -1
        self._n_sections = -1

        self.initialised = False

    def __next__(self):
        if self.current_turn == (self._max_turns) and self.current_section == (
            self._n_sections - 1
        ):
            raise StopIteration

        if self.current_section == (self._n_sections - 1):
            self.current_section = 0
            self.current_turn += 1
        else:
            self.current_section += 1

        return self.current_turn, self.current_section

    def __iter__(self):
        return self

    def __str__(self):
        return (
            f"{self.name} - Turn {self.current_turn}/{self._max_turns}"
            f" - Section {self.current_section}/{self._n_sections}"
        )

    def __repr__(self):
        return self.__str__()

    def initialise(self, max_turns: int, n_sections: int):
        if self.initialised:
            raise RuntimeError("Counter already initialised")

        self._max_turns = int(max_turns)
        self._n_sections = int(n_sections)
        self.initialised = True


_DEFINED_COUNTERS: dict[str, TurnCounter] = {}


def get_turn_counter(name: Optional[str] = None) -> TurnCounter:
    if name is None:
        name = "BLonD"

    if name in _DEFINED_COUNTERS:
        return _DEFINED_COUNTERS[name]

    _DEFINED_COUNTERS[name] = TurnCounter(name)

    return _DEFINED_COUNTERS[name]
