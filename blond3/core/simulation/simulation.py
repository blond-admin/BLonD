from __future__ import annotations

from abc import abstractmethod, ABC
from functools import cached_property
from typing import (
    Optional,
)
from typing import (
    Tuple,
)

from numpy.typing import NDArray as NumpyArray
from tqdm import tqdm

from ..backend import backend
from ..base import BeamPhysicsRelevant, Preparable
from ..base import DynamicParameter
from ..beam.base import BeamBaseClass
from ..helpers import find_instances_with_method
from ..ring.helpers import get_elements, get_init_order
from ..ring.ring import Ring
from ...beam_preparation.base import MatchingRoutine
from ...cycles.base import EnergyCycle
from ...handle_results.observables import Observables


class TurnDependent(ABC):
    def __init__(self, simulation: Simulation):
        self._simulation = simulation

    @abstractmethod
    def by_turn(self, turn_i: int) -> backend.float:
        pass


class Gamma(TurnDependent):
    def by_turn(self, turn_i: int) -> backend.float:
        beam_energy = self._simulation._energy_cycle.beam_energy_by_turn[turn_i]
        beam_res_energy = self._simulation._beams[0].rest_mass
        return  # TODO


class Beta(TurnDependent):
    def by_turn(self, turn_i: int) -> backend.float:
        gamma = self._simulation.gamma.by_turn(turn_i=turn_i)
        return  # TODO


class Velocity(TurnDependent):
    def by_turn(self, turn_i: int) -> backend.float:
        from scipy.constants import speed_of_light as c0

        return self._simulation.beta.by_turn(turn_i=turn_i) * c0


class RevolutionFrequency(TurnDependent):
    def by_turn(self, turn_i: int):
        pass


class Simulation(Preparable):
    def __init__(
        self,
        ring: Ring,
        beams: Tuple[BeamBaseClass, ...],
        energy_cycle: NumpyArray | EnergyCycle,
    ):
        super().__init__()
        self._ring: Ring = ring
        assert self.beams != (), f"{self.beams=}"
        assert len(self._beams) <= 2, "Maximum two beams allowed"
        self._beams: Tuple[BeamBaseClass, ...] = beams
        self._energy_cycle: EnergyCycle = energy_cycle

        self.turn_i = DynamicParameter(None)
        self.group_i = DynamicParameter(None)

        self._gamma = Gamma(self)
        self._beta = Beta(self)
        self._velocity = Velocity(self)
        self._revolution_frequency = RevolutionFrequency(self)

        self._exec_on_init_simulation()

    @property
    def gamma(self):
        return self._gamma

    @property
    def beta(self):
        return self._beta

    @property
    def velocity(self):
        return self._velocity

    @property
    def revolution_frequency(self):
        return self._revolution_frequency

    def on_init_simulation(self, simulation: Simulation):
        pass

    def _exec_all_in_tree(self, method: str, **kwargs):
        instances = find_instances_with_method(self, f"{method}")
        ordered_classes = get_init_order(instances, f"{method}.requires")

        classes_check = set()
        for ins in instances:
            classes_check.add(type(ins))
        assert len(classes_check) == len(ordered_classes), "BUG"

        for cls in ordered_classes:
            for element in instances:
                if not type(element) == cls:
                    continue
                element.__dict__[f"{method}"](**kwargs)

    def _exec_on_init_simulation(self):
        self._exec_all_in_tree("on_init_simulation", simulation=self)

    def _exec_on_run_simulation(self, n_turns: int, turn_i_init: int):
        self._exec_all_in_tree(
            "on_run_simulation",
            simulation=self,
            n_turns=n_turns,
            turn_i_init=turn_i_init,
        )

    @staticmethod
    def from_locals(locals: dict):
        locals_list = locals.values()
        _rings = get_elements(locals_list, Ring)
        assert len(_rings) == 1, f"Found {len(_rings)} rings"
        ring = _rings[0]

        beams = get_elements(locals_list, BeamBaseClass)

        _energy_cycles = get_elements(locals_list, Ring)
        assert len(_energy_cycles) == 1, f"Found {len(_energy_cycles)} energy cycles"
        energy_cycle = _energy_cycles[0]

        elements = get_elements(locals_list, BeamPhysicsRelevant)
        ring.add_elements(elements=elements, reorder=True)

        sim = Simulation(ring=ring, beams=beams, energy_cycle=energy_cycle)
        return sim

    @property
    def ring(self):
        return self._ring

    @property
    def beams(self):
        return self._beams

    @property
    def energy_cycle(self):
        return self._energy_cycle

    @cached_property
    def get_separatrix(self):
        return None

    @cached_property
    def get_hash(self):
        return None

    def print_one_turn_execution_order(self):
        self._ring.elements.print_order()

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
        preparation_routine.on_prepare_beam(simulation=self)

    def run_simulation(
        self,
        n_turns: Optional[int] = None,
        turn_i_init: int = 0,
        observe: Tuple[Observables, ...] = tuple(),
        show_progressbar: bool = True,
    ) -> None:
        self._exec_on_run_simulation(
            n_turns=n_turns,
            turn_i_init=turn_i_init,
        )
        if len(self._beams) == 1:
            self._run_simulation_single_beam(
                n_turns=n_turns,
                turn_i_init=turn_i_init,
                observe=observe,
                show_progressbar=show_progressbar,
            )
        elif len(self._beams) == 2:
            assert (
                self._beams[0].is_counter_rotating,
                self._beams[1].is_counter_rotating,
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
        iterator = range(turn_i_init, turn_i_init + n_turns)
        if show_progressbar:
            iterator = tqdm(iterator)  # Add TQDM display to iteration
        self.turn_i.on_change(self.invalidate_cache)
        for turn_i in iterator:
            self.turn_i.value = turn_i
            for element in self._ring.elements.elements:
                self.group_i.current_group = element.group
                if element.is_active_this_turn(turn_i=self.turn_i.value):
                    element.track(self._beams[0])
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
