from __future__ import annotations

import abc
import copy
import inspect
import math
import os.path
import warnings
from abc import ABC
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from os import PathLike
from typing import (
    Iterable,
    Any,
    List,
    Tuple,
    Optional,
    TypeVar,
    Type,
    Callable,
    Literal,
)
from typing import Optional as LateInit

import numpy as np
from cupy.typing import NDArray as CupyArray
from numpy.typing import NDArray as NumpyArray, DTypeLike
from scipy.constants import m_p, c, e
from tqdm import tqdm

from blond3.backend import backend

CLASS_DIAGRAM_HACKS = 0
T = TypeVar("T")


def requires(argument: List):
    def decorator(function):
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        wrapper.requires = argument
        return wrapper

    return decorator


def get_elements(elements: Iterable, class_: Type[T]) -> Tuple[T, ...]:
    return tuple(filter(lambda x: isinstance(x, class_), elements))


def get_init_order(instances: Iterable[Any], dependency_attribute: str):
    graph, in_degree, all_classes = build_dependency_graph(
        instances, dependency_attribute
    )
    sorted_classes = topological_sort(graph, in_degree, all_classes)
    return sorted_classes


def build_dependency_graph(
    instances: Iterable[Any], dependency_attribute: str
) -> (defaultdict[Any, list], defaultdict[Any, int], set):
    """Function to build a dependency graph based on
    'late_init_dependencies' defined in classes"""

    graph = defaultdict(list)  # Directed graph: dependency -> list of dependent classes
    in_degree = defaultdict(
        int
    )  # Count of incoming edges (dependencies) for each class
    all_classes = set()  # Set to keep track of all involved classes

    # Iterate through the types (classes) of all given instances
    for cls in [type(o) for o in instances]:
        all_classes.add(cls)  # Register the class
        # Traverse the class's MRO (method resolution order) to get parent classes as well
        # For each dependency declared in 'late_init_dependencies'
        dependencies = set()
        for cls_ in inspect.getmro(cls):
            deps = get_dependencies(cls_, dependency_attribute)
            for dep_ in deps:
                dependencies.add(dep_)
        for dep in dependencies:
            graph[dep].append(cls)  # Add edge: dep -> cls
            in_degree[cls] += 1  # Increment in-degree count for the class
            all_classes.add(dep)  # Ensure the dependency class is also tracked
    return graph, in_degree, all_classes


def get_dependencies(cls_: type, dependency_attribute: str):
    if "." in dependency_attribute:
        if dependency_attribute.count(".") != 1:
            raise NotImplementedError(
                f"Only one . allowed in " f"{dependency_attribute=}"
            )
        atr1, atr2 = dependency_attribute.split(".")
        attr = getattr(cls_, atr1, None)
        if attr is not None:
            attr = getattr(attr, atr2, [])
            if not isinstance(attr, list):
                raise Exception(type(attr))
        else:
            attr = []
        if not isinstance(attr, list):
            raise Exception(type(attr))
    else:
        attr = getattr(cls_, dependency_attribute, [])
        if not isinstance(attr, list):
            raise Exception(type(attr))
    return attr


def topological_sort(
    graph: defaultdict[Any, list], in_degree: defaultdict[Any, int], all_classes: set
) -> list[Any]:
    """Function to perform topological sort on the dependency graph"""
    # Initialize queue with classes that have no dependencies (in-degree 0)
    queue = deque([cls for cls in all_classes if in_degree[cls] == 0])
    sorted_classes = []  # List to store the sorted order

    # Kahn's algorithm for topological sorting
    while queue:
        cls = queue.popleft()
        sorted_classes.append(cls)
        # Reduce in-degree of dependent classes
        for neighbor in graph[cls]:
            in_degree[neighbor] -= 1
            # If in-degree becomes 0, add to queue
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If not all classes are sorted, there is a cycle in the graph
    if len(sorted_classes) != len(all_classes):
        raise ValueError("Cyclic dependency detected")
    return sorted_classes


class Preparable(ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def late_init(self, simulation: Simulation, **kwargs) -> None:
        if CLASS_DIAGRAM_HACKS:
            self._prepare_simulation = Simulation()


class MainLoopRelevant(Preparable):
    def __init__(self):
        super().__init__()
        self.each_turn_i = 1

    def is_active_this_turn(self, turn_i: int):
        return turn_i % self.each_turn_i == 0


class BeamPhysicsRelevant(MainLoopRelevant):
    n_instances = 0

    def __init__(self, group: int = 0, name: Optional[str] = None):
        super().__init__()
        self._group = group
        if name is None:
            name = f"Unnamed-{type(self)}-{type(self).n_instances:3d}"
        self.name = name
        type(self).n_instances += 1

    @property
    def group(self):
        return self._group

    @abc.abstractmethod
    def track(self, beam: BeamBaseClass) -> None:
        if CLASS_DIAGRAM_HACKS:
            self.track_beam = BeamBaseClass()


class Losses(BeamPhysicsRelevant):
    def __init__(self):
        super().__init__()


class BoxLosses(Losses):
    def __init__(
        self,
        t_min: Optional[backend.float] = None,
        t_max: Optional[backend.float] = None,
        e_min: Optional[backend.float] = None,
        e_max: Optional[backend.float] = None,
    ):
        super().__init__()

        self.t_min = backend.float(t_min)
        self.t_max = backend.float(t_max)
        self.e_min = backend.float(e_min)
        self.e_max = backend.float(e_max)

    def track(self, beam: BeamBaseClass):
        backend.loss_box(
            beam.write_partial_flags(), self.t_min, self.t_max, self.e_min, self.e_max
        )

    def late_init(self, simulation: Simulation, **kwargs):
        pass


class SeparatrixLosses(Losses):
    def __init__(self):
        super().__init__()
        self._simulation: LateInit[Simulation] = None

    def late_init(self, simulation: Simulation, **kwargs):
        self._simulation = simulation

    def track(self, beam: BeamBaseClass):
        self._simulation.get_separatrix()  # TODO


class DynamicParameter:  # TODO add code generation for this method with type-hints
    def __init__(self, value_init):
        self._value = value_init
        self._observers: List[Callable[[Any], None]] = []

    def on_change(self, callback: Callable[[Any], None]):
        """Subscribe to changes on a specific parameter."""
        self._observers.append(callback)

    def _notify(self, value):
        """Notify all observers about a parameter change."""
        for callback in self._observers:
            callback(value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_val: T):
        if new_val != self._value:
            self._notify(new_val)
        self._value = new_val


class BeamPhysicsRelevantElements(ABC):
    def __init__(self):
        self.elements: Tuple[BeamPhysicsRelevant, ...] = ()

        if CLASS_DIAGRAM_HACKS:
            self._elements = BeamPhysicsRelevant()

    def add_element(self, element: BeamPhysicsRelevant):
        self.elements = (*self.elements, element)

    @property
    def n_elements(self):
        return len(self.elements)

    def get_elements(
        self, class_: Type[T], group: Optional[int] = None
    ) -> Tuple[T, ...]:
        elements = get_elements(self.elements, class_)
        if group is not None:
            elements = tuple(filter(lambda x: x.group == group, elements))
        return elements

    def reorder(self):
        pass

    def print_order(self):
        for element in self.elements:
            filtered_dict = {
                k: v for k, v in element.__dict__.items() if not k.startswith("_")
            }
            print(element.name, type(element), filtered_dict)


class ProgrammedCycle(Preparable):
    def __init__(self):
        super().__init__()


class EnergyCycle(ProgrammedCycle):
    def __init__(self, beam_energy_by_turn: NumpyArray):
        super().__init__()
        self._beam_energy_by_turn = beam_energy_by_turn.astype(backend.float)

    @property
    def beam_energy_by_turn(self):
        return self._beam_energy_by_turn

    @staticmethod
    def from_linspace(start, stop, turns, endpoint: bool = True):
        return EnergyCycle(
            beam_energy_by_turn=backend.linspace(
                start, stop, turns, endpoint=endpoint, dtype=backend.float
            )
        )

    def late_init(self, simulation: Simulation, **kwargs):
        pass


class RfParameterCycle(ProgrammedCycle):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_phase(self, turn_i: int):
        pass

    @abc.abstractmethod
    def get_effective_voltage(self, turn_i: int):
        pass


class NoiseGenerator(ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_noise(self, n_turns: int):
        pass


class VariNoise(NoiseGenerator):
    def get_noise(self, n_turns: int):
        pass


class ConstantProgram(RfParameterCycle):
    def __init__(self, phase: float, effective_voltage: float):
        super().__init__()
        self._phase = backend.float(phase)
        self._effective_voltage = backend.float(effective_voltage)

    def get_phase(self, turn_i: int):
        return self._phase

    def get_effective_voltage(self, turn_i: int):
        return self._effective_voltage

    def late_init(self, simulation: Simulation, **kwargs):
        pass


class RFNoiseProgram(RfParameterCycle):
    def __init__(
        self,
        phase: float,
        effective_voltage: float,
        phase_noise_generator: NoiseGenerator,
    ):
        super().__init__()
        self._phase = backend.float(phase)
        self._effective_voltage = backend.float(effective_voltage)
        self._phase_noise_generator = phase_noise_generator

        self._phase_noise: LateInit[NumpyArray] = None

    def late_init(self, simulation: Simulation, **kwargs):
        n_turns = len(simulation.ring.energy_cycle._beam_energy_by_turn)
        # TODO max_turns attribute
        self._phase_noise = self._phase_noise_generator.get_noise(
            n_turns=n_turns
        ).astype(backend.float)

    def get_phase(self, turn_i: int):
        return self._phase + self._phase_noise[turn_i]

    def get_effective_voltage(self, turn_i: int):
        return self._effective_voltage


class Ring(ABC):
    def __init__(self, circumference):
        super().__init__()
        self._circumference = backend.float(circumference)
        self._elements = BeamPhysicsRelevantElements()
        self._beams: Tuple[BeamBaseClass, ...] = ()
        self._energy_cycle: LateInit[EnergyCycle] = None
        self._t_rev = DynamicParameter(None)

        if CLASS_DIAGRAM_HACKS:
            self._beams = BeamBaseClass()  # TODO
            self._energy_cycle = EnergyCycle()  # TODO

    @property
    def beams(self):
        return self._beams

    @property
    def energy_cycle(self):
        return self._energy_cycle

    @property
    def elements(self):
        return self._elements

    @property
    def t_rev(self):
        return self._t_rev

    @property
    def circumference(self):
        return self._circumference

    @property
    def one_turn_pathlength(self):
        return

    def add_element(
        self,
        element: BeamPhysicsRelevant,
        reorder: bool = False,
        deepcopy: bool = False,
    ):
        if CLASS_DIAGRAM_HACKS:
            self._add_elemet = BeamPhysicsRelevant()  # TODO

        if deepcopy:
            element = copy.deepcopy(element)

        self.elements.add_element(element)

        if reorder:
            self.elements.reorder()

    def add_elements(
        self, elements: Iterable[BeamPhysicsRelevant], reorder: bool = False
    ):
        for element in elements:
            self.add_element(element=element)

        if reorder:
            self.elements.reorder()

    def magic_add(self, ring_attributes: dict | Iterable):
        from collections.abc import Iterable as Iterable_  # so isinstance works

        if isinstance(ring_attributes, dict):
            values = ring_attributes.values()
        elif isinstance(ring_attributes, Iterable_):
            values = ring_attributes
        else:
            raise ValueError(
                f"Cant handle {type(ring_attributes)=}, must be " f"`Iterable` instead"
            )
        for val in values:
            if isinstance(val, BeamBaseClass):
                self.add_beam(beam=val)
            elif isinstance(val, BeamPhysicsRelevant):
                self.add_element(element=val)
            elif isinstance(val, EnergyCycle):
                self.set_energy_cycle(val)
            else:
                pass

        # reorder elements for correct execution order
        self.elements.reorder()

    def set_energy_cycle(self, energy_cycle: NumpyArray | EnergyCycle):
        if CLASS_DIAGRAM_HACKS:
            self._energy_cycle = EnergyCycle()  # TODO
        if isinstance(energy_cycle, np.ndarray):
            energy_cycle = EnergyCycle(energy_cycle)
        self._energy_cycle = energy_cycle

    def add_beam(self, beam: BeamBaseClass):
        if CLASS_DIAGRAM_HACKS:
            self._beam = BeamBaseClass()  # TODO

        if len(self.beams) == 0:
            assert beam.is_counter_rotating is False
            self._beams = (beam,)

        elif len(self.beams) == 1:
            assert beam.is_counter_rotating is True
            self._beams = (self.beams[0], beam)
        else:
            raise NotImplementedError("No more than two beam allowed!")

    def late_init(self, simulation: Simulation, **kwargs):
        assert self.beams != (), f"{self.beams=}"
        assert self.energy_cycle is not None, f"{self.energy_cycle}"
        ordered_classes = get_init_order(self.elements.elements, "late_init.requires")
        for cls in ordered_classes:
            for element in self.elements.elements:
                if not type(element) == cls:
                    continue
                element.late_init(simulation=simulation)
        all_drifts = self.elements.get_elements(DriftBaseClass)
        sum_share_of_circumference = sum(
            [drift.share_of_circumference for drift in all_drifts]
        )
        assert sum_share_of_circumference == 1, (
            f"{sum_share_of_circumference=}, but should be 1. It seems the "
            f"drifts are not correctly configured."
        )
        simulation.turn_i.on_change(self.update_t_rev)

    def get_t_rev(self, turn_i):
        return self.circumference / beta_by_ekin(
            self.energy_cycle._beam_energy_by_turn[turn_i]
        )

    def update_t_rev(self, new_turn_i: int):
        self.t_rev.value = self.get_t_rev(new_turn_i)


class DriftBaseClass(BeamPhysicsRelevant, ABC):
    def __init__(self, share_of_circumference: float, group: int = 0):
        super().__init__(group=group)
        self.__share_of_circumference = backend.float(share_of_circumference)

    @property
    def share_of_circumference(self):
        return self.__share_of_circumference

    def track(self, beam: BeamBaseClass):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        pass


class DriftSimple(DriftBaseClass):
    def __init__(
        self,
        transition_gamma: float,
        share_of_circumference: float = 1.0,
        group: int = 0,
    ):
        super().__init__(share_of_circumference=share_of_circumference, group=group)
        self._transition_gamma = backend.float(transition_gamma)

    def late_init(self, simulation: Simulation, **kwargs):
        pass

    @property
    def transition_gamma(self):
        return self._transition_gamma

    def track(self, beam: BeamBaseClass):
        backend.drift_simple(
            beam.write_partial_dt(), beam.read_partial_dE(), self._transition_gamma
        )

    @cached_property
    def momentum_compaction_factor(self):
        return 1 / self._transition_gamma**2

    def invalidate_cache(self):
        self.__dict__.pop("momentum_compaction_factor", None)


class DriftSpecial(DriftBaseClass):
    def track(self, beam: BeamBaseClass):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        super().late_init(simulation=simulation)

    pass


class DriftXSuite(DriftBaseClass):
    def track(self, beam: BeamBaseClass):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        super().late_init(simulation=simulation)

    pass


class CavityBaseClass(BeamPhysicsRelevant, ABC):
    def __init__(
        self,
        rf_program: Optional[RfParameterCycle] = None,
        group: int = 0,
        local_wakefield: Optional[WakeField] = None,
    ):
        super().__init__(group=group)
        self._rf_program: RfParameterCycle = rf_program
        self._local_wakefield = local_wakefield
        self._turn_i_dynamic: LateInit[None]

        # TODO REMOVE
        if CLASS_DIAGRAM_HACKS:
            self._rf_program = RfParameterCycle()  # TODO REMOVE

    @abc.abstractmethod
    def derive_rf_program(self, simulation: Simulation):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        self.derive_rf_program(simulation=simulation)
        self._turn_i_dynamic = simulation.turn_i
        assert self._rf_program is not None

    def track(self, beam: BeamBaseClass):
        if self._local_wakefield is not None:
            self._local_wakefield.track(beam=beam)


class SingleHarmonicCavity(CavityBaseClass):
    def __init__(
        self,
        harmonic: int | float,
        rf_program: Optional[RfParameterCycle] = None,
        group: int = 0,
        local_wakefield: Optional[WakeField] = None,
    ):
        super().__init__(
            rf_program=rf_program, group=group, local_wakefield=local_wakefield
        )
        self._harmonic = harmonic

    @property
    def harmonic(self):
        return self._harmonic

    def derive_rf_program(self, simulation: Simulation):
        pass

    def track(self, beam: BeamBaseClass):
        super().track(beam=beam)
        backend.kick_single_harmonic(
            beam.read_partial_dt(),
            beam.write_partial_dE(),
            self._rf_program.get_phase(turn_i=self._turn_i_dynamic.value),
            self._rf_program.get_effective_voltage(turn_i=self._turn_i_dynamic.value),
        )

    def late_init(self, simulation: Simulation, **kwargs):
        super().late_init(simulation=simulation)
        pass


class MultiHarmonicCavity(CavityBaseClass):
    def track(self, beam: BeamBaseClass):
        pass

    def derive_rf_program(self, simulation: Simulation):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        pass


class Impedance(BeamPhysicsRelevant):
    def __init__(self, group: int = 0, profile: LateInit[ProfileBaseClass] = None):
        super().__init__(group=group)
        self._profile = profile

    @abc.abstractmethod
    def calc_induced_voltage(self):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        if self._profile is None:
            profiles = simulation.ring.elements.get_elements(
                ProfileBaseClass, group=self.group
            )
            assert len(profiles) == 1, (
                f"Found {len(profiles)} profiles in "
                f"{self.group=}, but can only handle one. Set the attribute "
                f"`your_impedance.profile` in advance or remove the second "
                f"profile from this group."
            )
            self._profile = profiles[0]
        else:
            pass


class WakeField(Impedance):
    def __init__(
        self,
        sources: Tuple[WakeFieldSource, ...],
        solver: Optional[WakeFieldSolver],
        group: int = 0,
        profile: LateInit[ProfileBaseClass] = None,
    ):
        super().__init__(group=group, profile=profile)

        self.solver = solver
        self.sources = sources

    def late_init(self, simulation: Simulation, **kwargs):
        super().late_init(simulation=simulation, **kwargs)
        assert len(self.sources) > 0, "Provide for at least one `WakeFieldSource`"
        self.solver.late_init(simulation=simulation, parent_wakefield=self)

    def _calc_induced_voltage(self):
        return self.solver.calc_induced_voltage()

    def track(self, beam: BeamBaseClass):
        induced_voltage = self._calc_induced_voltage()
        beam.kick_induced(self)


class WakeFieldSolver(Preparable):
    def late_init(self, simulation: Simulation, **kwargs):
        self._late_init(
            simulation=simulation, parent_wakefield=kwargs["parent_wakefield"]
        )

    @abc.abstractmethod
    def _late_init(self, simulation: Simulation, parent_wakefield: WakeField):
        pass

    @abc.abstractmethod
    def calc_induced_voltage(self):
        pass


class InductiveImpedanceSolver(WakeFieldSolver):
    def __init__(self):
        super().__init__()
        self._beam: LateInit[BeamBaseClass] = None
        self._Z: LateInit[NumpyArray] = None
        self._T_rev_dynamic: LateInit[DynamicParameter] = None

    def _late_init(self, simulation: Simulation, parent_wakefield: WakeField):
        self._parent_wakefield = parent_wakefield
        impedances: Tuple[InductiveImpedance] = parent_wakefield.sources
        assert all([isinstance(o, InductiveImpedance) for o in impedances])
        self._Z = np.array([o.Z_over_n for o in impedances])
        self._T_rev_dynamic = simulation.ring.t_rev

    def calc_induced_voltage(self):
        diff = self._parent_wakefield._profile.diff()
        return diff * self._Z * self._T_rev_dynamic.value


class PeriodicFreqSolver(WakeFieldSolver):
    def __init__(self, t_periodicity: float):
        super().__init__()
        self._t_periodicity = t_periodicity

        self._parent_wakefield: LateInit[WakeField] = None
        self._update_on_calc: LateInit[bool] = None
        self._n: LateInit[int] = None
        self._freq_x: LateInit[NumpyArray] = None
        self._freq_y: LateInit[NumpyArray] = None

    def _update_internal_data(self):
        self._n = int(
            math.ceil(self._t_periodicity / self._parent_wakefield._profile.dx)
        )
        self._freq_x = np.fft.rfftfreq(self._n, d=self._parent_wakefield._profile.dx)
        self._freq_y = np.zeros_like(self._freq_x)
        for source in self._parent_wakefield.sources:
            if isinstance(source, FreqDomain):
                self._freq_y += source.get_freq_y(freq_x=self._freq_x)
            else:
                raise Exception("Can only accept impedance that support `FreqDomain`")

    def _warning_callback(self, t_rev_new: float):
        tolerance = 0.1 / 100
        deviation = abs(1 - t_rev_new / self._t_periodicity)
        if deviation > tolerance:
            warnings.warn(
                f"The PeriodicFreqSolver was configured for "
                f"{self._t_periodicity=:.2e} s, but the actual Ring "
                f"periodicity is {t_rev_new:.2e} s, a deviation of {deviation} %."
            )

    @requires([WakeField])
    def _late_init(self, simulation: Simulation, parent_wakefield: WakeField):
        simulation.ring.t_rev.on_change(self._warning_callback)

        if parent_wakefield._profile is not None:
            is_static = isinstance(parent_wakefield._profile, StaticProfile)
            is_dynamic = isinstance(
                parent_wakefield._profile, DynamicProfileConstCutoff
            ) or isinstance(parent_wakefield._profile, DynamicProfileConstNBins)
            self._parent_wakefield = parent_wakefield
            self._update_internal_data()
            if is_static:
                self._update_on_calc = False
            elif is_dynamic:
                self._update_on_calc = True
            else:
                raise NotImplementedError(
                    f"Unrecognized type(profile) = "
                    f"{type(parent_wakefield._profile)}"
                )
        else:
            raise Exception(f"{parent_wakefield._profile=}")

    def calc_induced_voltage(self):
        if self._update_on_calc:
            self._update_internal_data()  # might cause performance issues :(

        induced_voltage = np.fft.irfft(
            self._freq_y * self._parent_wakefield._profile.beam_spectrum(self._n)
        )
        return induced_voltage


class SingleTurnWakeSolverTimeDomain(WakeFieldSolver):
    pass


class AnalyticSingleTurnResonatorSolver(WakeFieldSolver):
    pass


class MutliTurnResonatorSolver(WakeFieldSolver):
    pass


class WakeFieldSource(ABC):
    pass


class TimeDomain(ABC):
    pass


class FreqDomain(ABC):
    @abc.abstractmethod
    def get_freq_y(self, freq_x: NumpyArray):
        pass


class AnalyticWakeFieldSource(WakeFieldSource):
    pass


class DiscreteWakeFieldSource(WakeFieldSource):
    pass


class InductiveImpedance(AnalyticWakeFieldSource, FreqDomain):
    def get_freq_y(self, freq_x: NumpyArray):
        imp = np.zeros(len(freq_x), dtype=complex)
        imp[:] = 1j * self.Z_over_n * ring.f_rev  # TODO

    def __init__(self, Z_over_n: float):
        self.Z_over_n = Z_over_n


class Resonators(AnalyticWakeFieldSource, TimeDomain, FreqDomain):
    pass


class ImpedanceReader(ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def load_file(self, filepath: PathLike) -> Tuple[NumpyArray, NumpyArray]:
        return freq, amplitude  # NOQA


class ExampleImpedanceReader1(ImpedanceReader):
    def __init__(self):
        super().__init__()

    def load_file(self, filepath: PathLike) -> Tuple[NumpyArray, NumpyArray]:
        table = np.loadtxt(
            filepath,
            skiprows=1,
            dtype=complex,
            encoding="utf-8",
            converters={
                0: lambda s: complex(
                    bytes(s, encoding="utf-8").decode("UTF-8").replace("i", "j")
                ),
                1: lambda y: complex(
                    bytes(y, encoding="utf-8").decode("UTF-8").replace("i", "j")
                ),
            },
        )
        freq, amplitude = table[:, 0].real, table[:, 1]
        return freq, amplitude


class ModesExampleReader2(str, Enum):
    OPEN_LOOP = "open loop"
    CLOSED_LOOP = "closed loop"
    SHORTED = "shorted"


class ExampleImpedanceReader2(ImpedanceReader):
    def __init__(self, mode: ModesExampleReader2 = ModesExampleReader2.CLOSED_LOOP):
        super().__init__()
        self._mode = mode

    def load_file(self, filepath: PathLike) -> Tuple[NumpyArray, NumpyArray]:
        data = np.loadtxt(filepath, dtype=float, skiprows=1)
        data[:, 3] = np.deg2rad(data[:, 3])
        data[:, 5] = np.deg2rad(data[:, 5])
        data[:, 7] = np.deg2rad(data[:, 7])

        freq_x = data[:, 0]
        if self._mode.value == ModesExampleReader2.OPEN_LOOP.value:
            Re_Z = data[:, 4] * np.cos(data[:, 3])
            Im_Z = data[:, 4] * np.sin(data[:, 3])
        elif self._mode.value == ModesExampleReader2.CLOSED_LOOP.value:
            Re_Z = data[:, 2] * np.cos(data[:, 5])
            Im_Z = data[:, 2] * np.sin(data[:, 5])
        elif self._mode.value == ModesExampleReader2.SHORTED.value:
            Re_Z = data[:, 6] * np.cos(data[:, 7])
            Im_Z = data[:, 6] * np.sin(data[:, 7])
        else:
            raise NameError(f"{self._mode=}")
        scale = 13
        freq_y = scale * (Re_Z + 1j * Im_Z)

        return freq_x, freq_y


class ImpedanceTable(DiscreteWakeFieldSource):
    @abc.abstractmethod
    @staticmethod
    def from_file(filepath: PathLike, reader: ImpedanceReader):
        reader.load_file(filepath=filepath)


@dataclass(frozen=True)
class ImpedanceTableFreq(ImpedanceTable, FreqDomain):
    freq_x: NumpyArray
    freq_y: NumpyArray

    @staticmethod
    def from_file(filepath: PathLike, reader: ImpedanceReader):
        x_array, y_array = reader.load_file(filepath=filepath)
        return ImpedanceTableFreq(freq_x=x_array, freq_y=y_array)


@dataclass(frozen=True)
class ImpedanceTableTime(ImpedanceTable, TimeDomain):
    wake_x: NumpyArray
    wake_y: NumpyArray

    @staticmethod
    def from_file(filepath: PathLike | str, reader: ImpedanceReader):
        x_array, y_array = reader.load_file(filepath=filepath)
        return ImpedanceTableTime(wake_x=x_array, wake_y=y_array)


ImpedanceTableTime.from_file("yolo", ExampleImpedanceReader1())


class Feedback(BeamPhysicsRelevant):
    def __init__(self, group: int = 0):
        super().__init__(group=group)


class LocalFeedback(Feedback):
    def __init__(
        self,
        profile: ProfileBaseClass,
        cavity: SingleHarmonicCavity | MultiHarmonicCavity,
        group: int = 0,
    ):
        super().__init__(group=group)
        self.cavity = cavity
        self.profile = profile
        if CLASS_DIAGRAM_HACKS:
            self.cavity = CavityBaseClass()  # TODO

    def track(self, beam: BeamBaseClass):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        pass


RfFeedback = LocalFeedback  # just an alias name


class GlobalFeedback(Feedback):
    def __init__(self, profile: ProfileBaseClass, group: int = 0):
        super().__init__(group=group)
        self.profile = profile

    def track(self, beam: BeamBaseClass):
        pass

    # Use `requires` to automatically sort execution order of
    # `element.late_init` for all elements
    @requires([SingleHarmonicCavity])
    def late_init(self, simulation: Simulation, **kwargs):
        self.cavities = simulation.ring.elements.get_elements(SingleHarmonicCavity)


BeamFeedback = GlobalFeedback  # just an alias name


class GroupedFeedback(Feedback):
    def __init__(
        self,
        profile: ProfileBaseClass,
        cavities: List[SingleHarmonicCavity | MultiHarmonicCavity],
        group: int = 0,
    ):
        super().__init__(group=group)
        self.profile = profile
        self.cavities = cavities


class LhcBeamFeedBack(GlobalFeedback):
    def __init__(self, profile: ProfileBaseClass, group: int = 0):
        super().__init__(profile=profile, group=group)


class LhcRfFeedback(LocalFeedback):
    def __init__(
        self,
        profile: ProfileBaseClass,
        cavity: SingleHarmonicCavity | MultiHarmonicCavity,
        group: int = 0,
    ):
        super().__init__(profile=profile, cavity=cavity, group=group)


class SimulationResults(ABC):
    def __init__(self):
        super().__init__()
        self.observables: List[Observables] = []
        if CLASS_DIAGRAM_HACKS:
            self.observables = Observables()  # TODO remove


class ProfileBaseClass(BeamPhysicsRelevant):
    def __init__(self):
        super().__init__()
        self._hist_x: LateInit[NumpyArray | CupyArray] = None
        self._hist_y: LateInit[NumpyArray | CupyArray] = None

        self._beam_spectrum_buffer = {}

    @property
    def hist_x(self):
        return self._hist_x

    @property
    def hist_y(self):
        return self._hist_y

    @cached_property
    def diff_hist_y(self):
        return backend.gradient(self._hist_y, self.hist_step)

    @cached_property
    def hist_step(self):
        return backend.float(self._hist_x[1] - self._hist_x[0])

    @cached_property
    def cut_left(self):
        return backend.float(self._hist_x[0] - self.hist_step / 2.0)

    @cached_property
    def cut_right(self):
        return backend.float(self._hist_x[-1] + self.hist_step / 2.0)

    @cached_property
    def bin_edges(self):
        return backend.linspace(
            self.cut_left, self.cut_right, len(self._hist_x) + 1, backend.float
        )

    def track(self, beam: BeamBaseClass):
        if beam.is_distributed:
            raise NotImplementedError("Impleemt hisogram on distributed array")
        else:
            backend.histogram(
                beam.read_partial_dt(), self.cut_left, self.cut_right, self._hist_y
            )
        self.invalidate_cache()

    def late_init(self, simulation: Simulation, **kwargs):
        assert self._hist_x is not None
        assert self._hist_y is not None
        self.invalidate_cache()

    @staticmethod
    def get_arrays(cut_left: float, cut_right: float, n_bins: int):
        step = (cut_right - cut_left) / n_bins
        offset = step / 2
        hist_x = backend.linspace(
            cut_left + offset, cut_right - offset, n_bins, dtype=backend.float
        )
        hist_y = backend.zeros(n_bins, dtype=backend.float)
        return hist_x, hist_y

    @property
    def cutoff_frequency(self):
        return backend.float(1 / (2 * self.hist_step))

    def _calc_gauss(self):
        return

    @cached_property
    def gauss_fit_params(self):
        return self._calc_gauss()

    @cached_property
    def beam_spectrum(self, n_fft: int):
        if n_fft in self._beam_spectrum_buffer.keys():
            self._beam_spectrum_buffer = np.fft.irfft(self._hist_y, n_fft)
        else:
            np.fft.irfft(self._hist_y, n_fft, out=self._beam_spectrum_buffer[n_fft])

        return self._beam_spectrum_buffer

    def invalidate_cache(self):
        for attribute in (
            "gauss_fit_params",
            "beam_spectrum",
            "hist_step",
            "cut_left",
            "cut_right",
            "bin_edges",
        ):
            self.__dict__.pop(attribute, None)


class StaticProfile(ProfileBaseClass):
    def __init__(self, cut_left: float, cut_right: float, n_bins: int):
        super().__init__()
        self.hist_x, self.hist_y = ProfileBaseClass.get_arrays(
            cut_left=cut_left, cut_right=cut_right, n_bins=n_bins
        )

    @staticmethod
    def from_cutoff(cut_left: float, cut_right: float, cutoff_frequency: float):
        dt = 1 / (2 * cutoff_frequency)
        n_bins = int(math.ceil((cut_right - cut_left) / dt))
        return StaticProfile(cut_left=cut_left, cut_right=cut_right, n_bins=n_bins)

    @staticmethod
    def from_rad(
        cut_left_rad: float, cut_right_rad: float, n_bins: int, t_period: float
    ):
        rad_to_frac = 1 / (2 * np.pi)
        cut_left = cut_left_rad * rad_to_frac * t_period
        cut_right = cut_right_rad * rad_to_frac * t_period
        return StaticProfile(cut_left=cut_left, cut_right=cut_right, n_bins=n_bins)


class DynamicProfile(ProfileBaseClass):
    def __init__(self):
        super().__init__()

    def late_init(self, simulation: Simulation, **kwargs):
        self.update_attributes(beam=simulation.ring.beams[0])
        super().late_init(simulation=simulation)

    @abc.abstractmethod
    def update_attributes(self, beam: BeamBaseClass):
        pass

    def track(self, beam: BeamBaseClass):
        self.update_attributes(beam=beam)
        super().track(beam=beam)


class DynamicProfileConstCutoff(DynamicProfile):
    def __init__(self, timestep: float):
        super().__init__()
        self.timestep = timestep

    def update_attributes(self, beam: BeamBaseClass):
        cut_left = beam.dt_min()  # TODO caching of attribute acess
        cut_right = beam.dt_max()  # TODO caching of attribute acess
        n_bins = int(math.ceil((cut_right - cut_left) / self.timestep))
        self.hist_x, self.hist_y = ProfileBaseClass.get_arrays(
            cut_left=cut_left, cut_right=cut_right, n_bins=n_bins
        )


class DynamicProfileConstNBins(ProfileBaseClass):
    def __init__(self, n_bins: int):
        super().__init__()
        self.n_bins = int_from_float_with_warning(n_bins, warning_stacklevel=2)

    def update_attributes(self, beam: BeamBaseClass):
        cut_left = beam.dt_min()  # TODO caching of attribute acess
        cut_right = beam.dt_max()  # TODO caching of attribute acess
        self.hist_x, self.hist_y = ProfileBaseClass.get_arrays(
            cut_left=cut_left, cut_right=cut_right, n_bins=self.n_bins
        )


@dataclass(frozen=True)
class ParticleType:
    mass: float
    charge: float


proton = ParticleType(mass=m_p * c**2 / e, charge=1)


def int_from_float_with_warning(value: float | int, warning_stacklevel: int) -> int:
    if isinstance(value, int):
        return value
    elif isinstance(value, float):
        return_value = int(value)
        if value != return_value:
            warnings.warn(
                f"{value} has been converted to {return_value}",
                UserWarning,
                # so int_from_float_with_warning behaves as warning.warn
                # the `stacklevel` is adjusted
                stacklevel=warning_stacklevel + 1,
            )


class BeamFlags(int, Enum):
    LOST = 0
    ACTIVE = 1


class BeamBaseClass(Preparable, ABC):
    def __init__(
        self,
        n_particles: int | float,
        n_macroparticles: int | float,
        particle_type: ParticleType,
        is_counter_rotating: bool = False,
        is_distributed=False,
    ):
        super().__init__()

        self.__n_particles__init = int_from_float_with_warning(
            n_particles, warning_stacklevel=2
        )
        self.__n_macroparticles__init = int_from_float_with_warning(
            n_macroparticles, warning_stacklevel=2
        )
        self._is_distributed = is_distributed
        self.__particles = particle_type
        self._dE = None
        self._dt = None
        self._flags = None
        self._is_counter_rotating = is_counter_rotating

    @property
    def is_distributed(self):
        return self._is_distributed

    @property
    def is_counter_rotating(self):
        return self._is_counter_rotating

    @abc.abstractmethod
    def late_init(self, simulation: Simulation, **kwargs):
        pass

    @abc.abstractmethod
    @property
    def dt_min(self):
        pass

    @abc.abstractmethod
    @property
    def dt_max(self):
        pass

    @abc.abstractmethod
    @property
    def dE_min(self):
        pass

    @abc.abstractmethod
    @property
    def dE_min(self):
        pass

    @abc.abstractmethod
    @property
    def common_array_size(self):
        pass

    @abc.abstractmethod
    def invalidate_cache_dE(self):
        pass

    @abc.abstractmethod
    def invalidate_cache_dt(self):
        pass

    @abc.abstractmethod
    def invalidate_cache(self):
        self.invalidate_cache_dE()
        self.invalidate_cache_dt()

    def read_partial_dt(self):
        """Returns dt-array on current node (distributed computing ready)

        Note
        ----
        Depends on `is_distributed`
        """
        return self._dt

    def write_partial_dt(self):
        """Returns dt-array on current node (distributed computing ready)

        Note
        ----
        Depends on"""
        self.invalidate_cache_dt()
        return self._dt

    def read_partial_dE(self):
        """Returns dE-array on current node (distributed computing ready)

        Note
        ----
        Depends on `is_distributed`
        """
        return self._dE

    def write_partial_dE(self):
        """Returns dE-array on current node (distributed computing ready)

        Note
        ----
        Depends on `is_distributed`
        """
        self.invalidate_cache_dE()
        return self._dE

    def write_partial_flags(self):
        """Returns flags-array on current node (distributed computing ready)

        Note
        ----
        Depends on `is_distributed`
        """
        self.invalidate_cache_dt()
        self.invalidate_cache_dE()
        return self._flags


class Beam(BeamBaseClass):
    def __init__(
        self,
        n_particles: int | float,
        n_macroparticles: int | float,
        particle_type: ParticleType,
        is_counter_rotating: bool = False,
    ):
        super().__init__(
            n_particles=n_particles,
            n_macroparticles=n_macroparticles,
            particle_type=particle_type,
            is_counter_rotating=is_counter_rotating,
        )


class WeightenedBeam(BeamBaseClass):
    def __init__(
        self,
        n_particles: int | float,
        n_macroparticles: int | float,
        particle_type: ParticleType,
    ):
        super().__init__(n_particles, n_macroparticles, particle_type)
        self._weights: LateInit[NumpyArray] = None

    @staticmethod
    def from_beam(beam: Beam):
        pass


class Observables(MainLoopRelevant):
    def __init__(self, each_turn_i: int):
        super().__init__()
        self.each_turn_i = each_turn_i

        self._n_turns: LateInit[int] = None
        self._turn_i_init: LateInit[int] = None
        self._turns_array: LateInit[NumpyArray] = None

    @property
    def turns_array(self):
        return self._turns_array

    @abc.abstractmethod
    def update(self, simulation: Simulation):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        self._late_init(
            simulation=simulation,
            n_turns=kwargs["n_turns"],
            turn_i_init=kwargs["turn_i_init"],
        )

    @abc.abstractmethod
    def _late_init(self, simulation: Simulation, n_turns: int, turn_i_init: int):
        self._n_turns = n_turns
        self._turn_i_init = turn_i_init
        self._turns_array = np.arange(turn_i_init, turn_i_init + n_turns)

    @abc.abstractmethod
    def to_disk(self):
        pass

    @abc.abstractmethod
    def from_disk(self):
        pass


class ArrayRecorder(ABC):
    @abc.abstractmethod
    def write(self, newdata: NumpyArray):
        pass

    @abc.abstractmethod
    def get_valid_entries(self):
        pass

    @abc.abstractmethod
    def to_disk(self):
        pass

    @abc.abstractmethod
    def from_disk(self):
        pass


class DenseArrayRecorder(ArrayRecorder):
    def __init__(
        self,
        filepath: str,
        shape: int | Tuple[int, ...],
        dtype: Optional[DTypeLike] = None,
        order: Literal["C", "F"] = "C",
        overwrite=True,
    ):
        self._memory = np.empty(shape=shape, dtype=dtype, order=order)
        self._write_idx = 0
        self.filepath = filepath
        self.overwrite = overwrite
        if not self.overwrite:
            if os.path.exists(self.filepath):
                warnings.warn(f"{self.filepath} exists already!")

    def to_disk(self):
        if not self.overwrite:
            assert not os.path.exists(self.filepath)
        np.save(self.filepath, self.get_valid_entries())

    def from_disk(self):
        pass

    def write(self, newdata: NumpyArray):
        self._memory[self._write_idx] = newdata
        self._write_idx += 1

    def get_valid_entries(self):
        return self._memory[: self._write_idx]


class ChunkedArrayRecorder(ArrayRecorder):
    pass


class ProfileObservation(Observables):
    def __init__(self, each_turn_i: int, profile: LateInit[ProfileBaseClass] = None):
        super().__init__(each_turn_i=each_turn_i)
        self._profile = profile
        self._hist_ys: LateInit[DenseArrayRecorder] = None

    def _late_init(self, simulation: Simulation, n_turns: int, turn_i_init: int):
        super()._late_init(
            simulation=simulation, n_turns=n_turns, turn_i_init=turn_i_init
        )
        n_entries = n_turns // self.each_turn_i

        if self._profile is None:
            profiles = simulation.ring.elements.get_elements(ProfileBaseClass)
            if len(profiles) == 0:
                raise Exception("Please define a profile for your simulation!")
            elif len(profiles) > 1:
                raise Exception(
                    f"There are {len(profiles)} that can be observed."
                    f" Select one profile when initializing `ProfileObservation`."
                )
            profile = profiles[0]
            assert isinstance(profile, DynamicProfileConstNBins) or isinstance(
                profile, StaticProfile
            ), f"Only `DynamicProfileConstNBins` or `StaticProfile` allowed"
            self._profile = profile
        n_bins = len(self._profile._hist_y)

        self._hist_ys = DenseArrayRecorder(
            f"{simulation.get_hash}_hist_ys", (n_bins, n_entries)
        )

    def update(self, simulation: Simulation):
        self._hist_ys.write(self._profile._hist_y)

    @property
    def hist_ys(self):
        return self._hist_ys.get_valid_entries()


class BunchObservation(Observables):
    def __init__(self, each_turn_i: int):
        super().__init__(each_turn_i=each_turn_i)
        self._dts: LateInit[DenseArrayRecorder] = None
        self._dEs: LateInit[DenseArrayRecorder] = None
        self._flags: LateInit[DenseArrayRecorder] = None

    def _late_init(self, simulation: Simulation, n_turns: int, turn_i_init: int):
        super()._late_init(
            simulation=simulation, n_turns=n_turns, turn_i_init=turn_i_init
        )
        n_entries = n_turns // self.each_turn_i
        n_particles = simulation.ring.beams[0].common_array_size
        shape = (n_particles, n_entries)
        self._dts = DenseArrayRecorder(f"{simulation.get_hash}_dts", shape)
        self._dEs = DenseArrayRecorder(f"{simulation.get_hash}_dEs", shape)
        self._flags = DenseArrayRecorder(f"{simulation.get_hash}_flags", shape)

    def update(self, simulation: Simulation):
        self._dts.write(simulation.ring.beams[0]._dt)
        self._dEs.write(simulation.ring.beams[0]._dE)
        self._flags.write(simulation.ring.beams[0]._flags)

    @property
    def dts(self):
        return self._dts.get_valid_entries()

    @property
    def dEs(self):
        return self._dEs.get_valid_entries()

    @property
    def flags(self):
        return self._flags.get_valid_entries()


class CavityPhaseObservation(Observables):
    def __init__(self, each_turn_i: int, cavity: SingleHarmonicCavity):
        super().__init__(each_turn_i=each_turn_i)
        self._cavity = cavity
        self._phases: LateInit[DenseArrayRecorder] = None

    def _late_init(self, simulation: Simulation, n_turns: int, turn_i_init: int):
        super()._late_init(
            simulation=simulation, n_turns=n_turns, turn_i_init=turn_i_init
        )
        n_entries = n_turns // self.each_turn_i
        self._phases = DenseArrayRecorder(f"{simulation.get_hash}_phases", n_entries)

    def update(self, simulation: Simulation):
        self._phases.write(self._cavity._rf_program.get_phase(simulation.turn_i.value))

    @property
    def phases(self):
        return self._phases.get_valid_entries()


class BeamPreparationRoutine(ABC):
    pass

    @abc.abstractmethod
    def prepare_beam(
        self,
        ring: Ring,
    ):
        ring.beams
        ring.elements
        if CLASS_DIAGRAM_HACKS:
            self._ring = Ring()
        pass


class MatchingRoutine(BeamPreparationRoutine):
    pass


class EmittanceMatcher(MatchingRoutine):
    def __init__(self, some_emittance):
        self.some_emittance = some_emittance


class BiGaussian(MatchingRoutine):
    def __init__(
        self,
        rms_dt: float,
        reinsertion: bool,
        seed: int,
    ):
        pass

    def prepare_beam(self, ring: Ring):
        pass


class Simulation(ABC):
    def __init__(self, ring: Ring, ring_attributes: Optional[dict | Iterable] = None):
        super().__init__()
        if ring_attributes is not None:
            ring.magic_add(ring_attributes)
        ring.late_init(simulation=self)
        self.ring: Ring = ring
        self.turn_i = DynamicParameter(None)
        self.group_i = DynamicParameter(None)

        if CLASS_DIAGRAM_HACKS:
            self._ring = Ring()  # TODO remove

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
        if CLASS_DIAGRAM_HACKS:
            self._preparation_routine = MatchingRoutine()

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
