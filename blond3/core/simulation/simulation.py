from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import (
    Iterable,
    Tuple,
    Optional,
)

from tqdm import tqdm

from ..base import DynamicParameter
from ..ring.ring import Ring
from ...beam_preparation.base import MatchingRoutine
from ...handle_results.observables import Observables


class Simulation:
    def __init__(self, ring: Ring, ring_attributes: Optional[dict | Iterable] = None):
        super().__init__()
        if ring_attributes is not None:
            ring.magic_add(ring_attributes)
        ring.late_init(simulation=self)
        self.ring: Ring = ring
        self.turn_i = DynamicParameter(None)
        self.group_i = DynamicParameter(None)

    @cached_property
    def get_separatrix(self):
        return None

    @cached_property
    def get_hash(self):
        return None

    def print_one_turn_execution_order(self):
        self.ring.elements.print_order()

    def invalidate_cache(
        self,
        # turn i needed to be
        # compatible with subscription
        turn_i: int,
    ):
        self.__dict__.pop("get_separatrix", None)
        self.__dict__.pop("get_hash", None)

    def prepare_beam(
        self,
        preparation_routine: MatchingRoutine,
    ):
        preparation_routine.prepare_beam(ring=self.ring)

    def run_simulation(
        self,
        n_turns: Optional[int] = None,
        turn_i_init: int = 0,
        observe: Tuple[Observables, ...] = tuple(),
        show_progressbar: bool = True,
    ) -> None:
        if len(self.ring.beams) == 1:
            self._run_simulation_single_beam(
                n_turns=n_turns,
                turn_i_init=turn_i_init,
                observe=observe,
                show_progressbar=show_progressbar,
            )
        elif len(self.ring.beams) == 2:
            assert (
                self.ring.beams[0].is_counter_rotating,
                self.ring.beams[1].is_counter_rotating,
            ) == (
                False,
                True,
            ), "First beam must be normal, second beam must be counter-rotating"
            self._run_simulation_counterrotating_beam(
                n_turns=n_turns,
                turn_i_init=turn_i_init,
                observe=observe,
                show_progressbar=show_progressbar,
            )

    def _run_simulation_single_beam(
        self,
        n_turns: int,
        turn_i_init: int = 0,
        observe: Tuple[Observables, ...] = tuple(),
        show_progressbar: bool = True,
    ) -> None:
        for observable in observe:
            observable.late_init(
                simulation=self, n_turns=n_turns, turn_i_init=turn_i_init
            )
        iterator = range(turn_i_init, turn_i_init + n_turns)
        if show_progressbar:
            iterator = tqdm(iterator)  # Add TQDM display to iteration
        self.turn_i.on_change(self.invalidate_cache)
        for turn_i in iterator:
            self.turn_i.value = turn_i
            for element in self.ring.elements.elements:
                self.group_i.current_group = element.group
                if element.is_active_this_turn(turn_i=self.turn_i.value):
                    element.track(self.ring.beams[0])
            for observable in observe:
                if observable.is_active_this_turn(turn_i=self.turn_i.value):
                    observable.update(simulation=self)

        # reset counters to uninitialized again
        self.turn_i.value = None
        self.group_i.value = None

    def _run_simulation_counterrotating_beam(
        self,
        n_turns: int,
        turn_i_init: int = 0,
        observe: Tuple[Observables, ...] = tuple(),
        show_progressbar: bool = True,
    ) -> None:
        pass  # todo

    def load_results(
        self,
        n_turns: int,
        turn_i_init: int = 0,
        observe=Tuple[Observables, ...],
    ) -> SimulationResults:
        return
