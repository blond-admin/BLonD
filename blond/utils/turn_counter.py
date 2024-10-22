# Futurisation
from __future__ import annotation

# General
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict


class TurnCounter:

    def __init__(self, name: str):

        self.name = name

        self.current_turn = 0
        self.current_section = 0

    def __next__(self):

        if (self.current_turn == (self.max_turns-1)
            and self.current_section == (self.n_sections-1)):
            raise StopIteration

        if self.current_section == (self.n_sections-1):
            self.current_section = 0
            self.current_turn += 1
        else:
            self.current_section += 1

        return self.current_turn, self.current_section

    def __iter__(self):
        return self

    def __str__(self):
        return f"{self.name} - Turn {self.current_turn} - Section {self.current_section}"

    def __repr__(self):
        return self.__str__()


_DEFINED_COUNTERS : Dict[str, TurnCounter] = {}


def get_turn_counter(name: str = None) -> TurnCounter:

    if name is None:
        name = 'BLonD'

    if name in _DEFINED_COUNTERS:
        return _DEFINED_COUNTERS[name]

    _DEFINED_COUNTERS[name] = TurnCounter(name)

    return _DEFINED_COUNTERS[name]
