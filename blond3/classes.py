from __future__ import annotations

import abc
import inspect
import math
import warnings
from abc import ABC
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from typing import Iterable, Any, List, Tuple, Optional, TypeVar, Type
from typing import Optional as LateInit

import numpy as np
from cupy.typing import NDArray as CupyArray
from numpy.typing import NDArray as NumpyArray
from scipy.constants import m_p, c, e
from tqdm import tqdm

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


class BeamPhysicsRelevantElements(ABC):
    def __init__(self):
        self.elements: Tuple[BeamPhysicsRelevant, ...] = ()

        if CLASS_DIAGRAM_HACKS:
            self._elements = BeamPhysicsRelevant()

    def add_element(self, element: BeamPhysicsRelevant):
        self.elements = (*self.elements, element)

    def get_elements(
        self, class_: Type[T], group: Optional[int] = None
    ) -> Tuple[T, ...]:
        elements = get_elements(self.elements, class_)
        if group is not None:
            elements = tuple(filter(lambda x: x.group == group, elements))
        return elements

    def reorder(self):
        pass


class ProgrammedCycle(ABC):
    def __init__(self):
        super().__init__()


class EnergyCycle(ProgrammedCycle):
    def __init__(self, beam_energy_by_turn):
        super().__init__()
        self.beam_energy_by_turn = beam_energy_by_turn


class RfParameterCycle(ProgrammedCycle):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def late_init(self, simulation: Simulation, **kwargs):
        if CLASS_DIAGRAM_HACKS:
            self.simulation = Simulation()  # TODO remove

    @abc.abstractmethod
    def get_phase(self, turn_i: int):
        pass

    @abc.abstractmethod
    def get_effective_voltage(self, turn_i: int):
        pass


class ConstantProgram(RfParameterCycle):
    def __init__(self, phase: float, effective_voltage: float):
        super().__init__()
        self._phase = phase
        self._effective_voltage = effective_voltage

    def get_phase(self, turn_i: int):
        return self._phase

    def get_effective_voltage(self, turn_i: int):
        return self._effective_voltage

    def late_init(self, simulation: Simulation, **kwargs):
        pass


class Counter(ABC):
    def __init__(self):
        super().__init__()


class TurnCounter(Counter):
    def __init__(self, current_turn: int):
        super().__init__()
        self.current_turn: int = current_turn


class GroupCounter(Counter):
    def __init__(self, current_group: int):
        super().__init__()
        self.current_group: int = current_group


class ElementCounter(Counter):
    pass


class Ring(ABC):
    def __init__(self, circumference):
        super().__init__()
        self.__circumference = circumference
        self.elements = BeamPhysicsRelevantElements()
        self.beams: Tuple[Beam, ...] = ()
        self.energy_cycle: LateInit[EnergyCycle] = None

        if CLASS_DIAGRAM_HACKS:
            self._beams = Beam()  # TODO
            self._energy_cycle = EnergyCycle()  # TODO

    @property
    def circumference(self):
        return self.__circumference

    @property
    def one_turn_pathlength(self):
        return

    def add_element(self, element: BeamPhysicsRelevant):
        if CLASS_DIAGRAM_HACKS:
            self._add_elemet = BeamPhysicsRelevant()  # TODO
        self.elements.add_element(element)

    def add_elements(
        self, elements: Iterable[BeamPhysicsRelevant], reorder: bool = False
    ):
        for element in elements:
            self.add_element(element=element)

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
            if isinstance(val, Beam):
                self.add_beam(beam=val)
            elif isinstance(val, BeamPhysicsRelevant):
                self.add_element(element=val)
            elif isinstance(val, EnergyCycle):
                self.energy_cycle = val
            else:
                pass

        # reorder elements for correct execution order
        self.elements.reorder()

    def set_energy_cycle(self, energy_cycle: NumpyArray | EnergyCycle):
        if CLASS_DIAGRAM_HACKS:
            self.energy_cycle = EnergyCycle()  # TODO
        if isinstance(energy_cycle, np.ndarray):
            energy_cycle = EnergyCycle(energy_cycle)
        self.energy_cycle = energy_cycle

    def add_beam(self, beam: Beam):
        if CLASS_DIAGRAM_HACKS:
            self.beam = Beam()  # TODO
        self.beams = (*self.beams, beam)

    def late_init(self, simulation: Simulation, **kwargs):
        assert self.beams != (), f"{self.beams=}"
        assert self.energy_cycle is not None, f"{self.energy_cycle}"
        ordered_classes = get_init_order(self.elements.elements, "late_init.requires")
        for cls in ordered_classes:
            for element in self.elements.elements:
                if not type(element) == cls:
                    continue
                element.late_init(simulation=simulation)
        all_drifts = self.elements.get_elements(Drift)
        sum_share_of_circumference = sum(
            [drift.share_of_circumference for drift in all_drifts]
        )
        assert sum_share_of_circumference == 1, (
            f"{sum_share_of_circumference=}, but should be 1. It seems the "
            f"drifts are not correctly configured."
        )


class Preparable(ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def late_init(self, simulation: Simulation, **kwargs):
        if CLASS_DIAGRAM_HACKS:
            self._prepare_simulation = Simulation()


class MainLoopRelevant(Preparable):
    def __init__(self):
        super().__init__()
        self.each_turn_i = 1

    def is_active_this_turn(self, turn_i: int):
        return turn_i % self.each_turn_i == 0


class BeamPhysicsRelevant(MainLoopRelevant):
    def __init__(self, group: int = 0):
        super().__init__()
        self.__group = group

    @property
    def group(self):
        return self.__group

    @abc.abstractmethod
    def track(self, beam: Beam):
        if CLASS_DIAGRAM_HACKS:
            self.track_beam = Beam()


class Drift(BeamPhysicsRelevant, ABC):
    def __init__(self, share_of_circumference: float, group: int = 0):
        super().__init__(group=group)
        self.__share_of_circumference = share_of_circumference

    @property
    def share_of_circumference(self):
        return self.__share_of_circumference

    def track(self, beam: Beam):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        pass


class DriftSimple(Drift):
    def __init__(
        self,
        transition_gamma: float,
        share_of_circumference: float = 1.0,
        group: int = 0,
    ):
        super().__init__(share_of_circumference=share_of_circumference, group=group)
        self._transition_gamma = transition_gamma

    @property
    def momentum_compaction_factor(self):
        return 1 / self._transition_gamma**2

    def track(self, beam: Beam):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        pass


class DriftSpecial(Drift):
    def track(self, beam: Beam):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        pass

    pass


class DriftXSuite(Drift):
    def track(self, beam: Beam):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        pass

    pass


class CavityBaseClass(BeamPhysicsRelevant, ABC):
    def __init__(self, rf_program: Optional[RfParameterCycle] = None, group: int = 0):
        super().__init__(group=group)
        self.rf_program: RfParameterCycle = rf_program

        # TODO REMOVE
        if CLASS_DIAGRAM_HACKS:
            self.rf_program = RfParameterCycle()  # TODO REMOVE

    @abc.abstractmethod
    def derive_rf_program(self, simulation: Simulation):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        self.derive_rf_program(simulation=simulation)
        assert self.rf_program is not None


class Cavity(CavityBaseClass):
    def __init__(
        self,
        harmonic: int | float,
        rf_program: Optional[RfParameterCycle] = None,
        group: int = 0,
    ):
        super().__init__(rf_program=rf_program, group=group)
        self._harmonic = harmonic

    @property
    def harmonic(self):
        return self._harmonic

    def derive_rf_program(self, simulation: Simulation):
        pass

    def track(self, beam: Beam):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        super().late_init(simulation=simulation)
        pass


class LumpedCavities(CavityBaseClass):
    def track(self, beam: Beam):
        pass

    def derive_rf_program(self, simulation: Simulation):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        pass


class Impedance(BeamPhysicsRelevant):
    def __init__(self, group: int = 0, profile: LateInit[Profile] = None):
        super().__init__(group=group)
        self._profile = profile

    @abc.abstractmethod
    def calc_induced_voltage(self):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        if self._profile is None:
            profiles = simulation.ring.elements.get_elements(Profile, group=self.group)
            assert len(profiles) == 1, (
                f"Found {len(profiles)} profiles in "
                f"{self.group=}, but can only handle one. Set the attribute "
                f"`your_impedance.profile` in advance or remove the second "
                f"profile from this group."
            )
            self._profile = profiles[0]
        else:
            pass


class InductiveImpedance(Impedance):
    def __init__(
        self,
        Z_over_n: float,
        mode: str,
        group: int = 0,
        profile: LateInit[Profile] = None,
    ):
        super().__init__(group=group, profile=profile)
        self.Z_over_n = Z_over_n
        self.deriv_mode = mode

    def calc_induced_voltage(self):
        pass

    def track(self, beam: Beam):
        pass


class WakeField(Impedance):
    def __init__(
        self,
        solver: Optional[WakeFieldSolver],
        group: int = 0,
        sources: Tuple[WakeFieldSource, ...] = tuple(),
        profile: LateInit[Profile] = None,
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

    def track(self, beam: Beam):
        induced_voltage = self._calc_induced_voltage()
        beam.kick(induced_voltage)


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


class PeriodicFreqSolver(WakeFieldSolver):
    def __init__(self, t_periodicity: float):
        super().__init__()
        self._t_periodicity = t_periodicity
        self._n = None
        self._profile: LateInit[Profile] = None
        self._freq_x: LateInit[NumpyArray] = None
        self._freq_y: LateInit[NumpyArray] = None

    @requires([WakeField])
    def _late_init(self, simulation: Simulation, parent_wakefield: WakeField):
        if parent_wakefield._profile is not None:
            self._profile = parent_wakefield._profile
            self._n = int(math.ceil(self._t_periodicity / self._profile.dx))
            self._freq_x = np.fft.rfftfreq(self._n, d=self._profile.dx)
            self._freq_y = np.zeros_like(self._freq_x)
            for source in parent_wakefield.sources:
                if isinstance(source, FreqDomain):
                    self._freq_y += source.get_freq_y(freq_x=self._freq_x)
                else:
                    raise Exception(
                        "Can only accept impedance that support `FreqDomain`"
                    )
        else:
            raise Exception()

    def calc_induced_voltage(self):
        induced_voltage = np.fft.irfft(
            self._freq_y * self._profile.beam_spectrum(self._n)
        )
        return induced_voltage


class SingleTurnWakeSolverTimeDomain(WakeFieldSolver):
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


class DiscreteFieldSource(WakeFieldSource):
    pass


class Resonator(AnalyticWakeFieldSource, TimeDomain, FreqDomain):
    pass


class ImpedanceReader(ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def load_file(self, filepath: PathLike) -> Tuple[NumpyArray, NumpyArray]:
        pass


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
        if self._mode == ModesExampleReader2.OPEN_LOOP:
            Re_Z = data[:, 4] * np.cos(data[:, 3])
            Im_Z = data[:, 4] * np.sin(data[:, 3])
        elif self._mode == ModesExampleReader2.CLOSED_LOOP:
            Re_Z = data[:, 2] * np.cos(data[:, 5])
            Im_Z = data[:, 2] * np.sin(data[:, 5])
        elif self._mode == ModesExampleReader2.SHORTED:
            Re_Z = data[:, 6] * np.cos(data[:, 7])
            Im_Z = data[:, 6] * np.sin(data[:, 7])
        else:
            raise NameError(f"{self._mode=}")
        scale = 13
        freq_y = scale * (Re_Z + 1j * Im_Z)

        return freq_x, freq_y


class ImpedanceTable(DiscreteFieldSource):
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
        self, profile: Profile, cavity: Cavity | LumpedCavities, group: int = 0
    ):
        super().__init__(group=group)
        self.cavity = cavity
        self.profile = profile
        if CLASS_DIAGRAM_HACKS:
            self.cavity = CavityBaseClass()  # TODO

    def track(self, beam: Beam):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        pass


RfFeedback = LocalFeedback  # just an alias name


class GlobalFeedback(Feedback):
    def __init__(self, profile: Profile, group: int = 0):
        super().__init__(group=group)
        self.profile = profile

    def track(self, beam: Beam):
        pass

    # Use `requires` to automatically sort execution order of
    # `element.late_init` for all elements
    @requires([Cavity])
    def late_init(self, simulation: Simulation, **kwargs):
        self.cavities = simulation.ring.elements.get_elements(Cavity)


BeamFeedback = GlobalFeedback  # just an alias name


class GroupedFeedback(Feedback):
    def __init__(
        self, profile: Profile, cavities: List[Cavity | LumpedCavities], group: int = 0
    ):
        super().__init__(group=group)
        self.profile = profile
        self.cavities = cavities


class LhcBeamFeedBack(GlobalFeedback):
    def __init__(self, profile: Profile, group: int = 0):
        super().__init__(profile=profile, group=group)


class LhcRfFeedback(LocalFeedback):
    def __init__(
        self, profile: Profile, cavity: Cavity | LumpedCavities, group: int = 0
    ):
        super().__init__(profile=profile, cavity=cavity, group=group)


class SimulationResults(ABC):
    def __init__(self):
        super().__init__()
        self.observables: List[Observables] = []
        if CLASS_DIAGRAM_HACKS:
            self.observables = Observables()  # TODO remove


class Profile(BeamPhysicsRelevant):
    def __init__(self):
        super().__init__()
        self.hist_x: LateInit[NumpyArray | CupyArray] = None
        self.hist_y: LateInit[NumpyArray | CupyArray] = None
        self.bin_edges: LateInit[NumpyArray | CupyArray] = None
        self.cut_left: LateInit[float] = None
        self.cut_right: LateInit[float] = None
        self.dx: LateInit[float] = None
        self.beam_spectrum_buffer = {}

    def track(self, beam: Beam):
        histogram(beam.dt, start=self.cut_left, stop=self.cut_right, out=self.hist_y)

    def beam_spectrum(self, n_fft: int):
        if n_fft in self.beam_spectrum_buffer.keys():
            self.beam_spectrum_buffer = np.fft.irfft(self.hist_y, n_fft)
        else:
            np.fft.irfft(self.hist_y, n_fft, out=self.beam_spectrum_buffer[n_fft])

        return self.beam_spectrum_buffer

    def late_init(self, simulation: Simulation, **kwargs):
        assert self.hist_x is not None
        assert self.dx is not None
        assert self.hist_y is not None
        assert self.bin_edges is not None
        assert self.cut_left is not None
        assert self.cut_right is not None

    @staticmethod
    def get_arrays(cut_left: float, cut_right: float, n_bins: int):
        bin_edges = cut_left + (cut_right - cut_left) * np.arange(
            n_bins + 1, dtype=np.float64
        )
        hist_x = bin_edges[:-1] + (bin_edges[1] - bin_edges[0])
        hist_y = np.zeros(n_bins, dtype=np.float64)
        return hist_x, hist_y, bin_edges

    @property
    def cutoff_frequency(self):
        return 1 / (2 * self.dx)


class StaticProfile(Profile):
    def __init__(self, cut_left: float, cut_right: float, n_bins: int):
        super().__init__()
        self.hist_x, self.hist_y, self.bin_edges = Profile.get_arrays(
            cut_left=cut_left, cut_right=cut_right, n_bins=n_bins
        )
        self.dx = self.hist_x[1] - self.hist_x[0]
        self.start = float(self.bin_edges.min())
        self.stop = float(self.bin_edges.max())

    @staticmethod
    def from_cutoff(cut_left: float, cut_right: float, cutoff_frequency: float):
        dt = 1 / (2 * cutoff_frequency)
        n_bins = int(math.ceil((cut_right - cut_left) / dt))
        return StaticProfile(cut_left=cut_left, cut_right=cut_right, n_bins=n_bins)


class DynamicProfile(Profile):
    def __init__(self):
        super().__init__()

    def late_init(self, simulation: Simulation, **kwargs):
        self.update_attributes(beam=simulation.ring.beams[0])
        super().late_init(simulation=simulation)

    @abc.abstractmethod
    def update_attributes(self, beam: Beam):
        pass

    def track(self, beam: Beam):
        self.update_attributes(beam=beam)
        super().track(beam=beam)


class DynamicProfileConstCutoff(DynamicProfile):
    def __init__(self, timestep: float):
        super().__init__()
        self.timestep = timestep

    def update_attributes(self, beam: Beam):
        cut_left = beam.dt.min()  # TODO caching of attribute acess
        cut_right = beam.dt.max()  # TODO caching of attribute acess
        n_bins = int(math.ceil((cut_right - cut_left) / self.timestep))
        self.hist_x, self.hist_y, self.bin_edges = Profile.get_arrays(
            cut_left=cut_left, cut_right=cut_right, n_bins=n_bins
        )
        self.dx = self.hist_x[1] - self.hist_x[0]
        self.start = cut_left
        self.stop = cut_right


class DynamicProfileConstNBins(Profile):
    def __init__(self, n_bins: int):
        super().__init__()
        self.n_bins = int_from_float_with_warning(n_bins, warning_stacklevel=2)

    def update_attributes(self, beam: Beam):
        cut_left = beam.dt.min()  # TODO caching of attribute acess
        cut_right = beam.dt.max()  # TODO caching of attribute acess
        self.hist_x, self.hist_y, self.bin_edges = Profile.get_arrays(
            cut_left=cut_left, cut_right=cut_right, n_bins=self.n_bins
        )
        self.dx = self.hist_x[1] - self.hist_x[0]
        self.start = cut_left
        self.stop = cut_right


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


class Beam(Preparable):
    def __init__(
        self,
        n_particles: int | float,
        n_macroparticles: int | float,
        particle_type,
    ):
        self.__n_particles = int_from_float_with_warning(
            n_particles, warning_stacklevel=2
        )
        self.__n_macroparticles = int_from_float_with_warning(
            n_macroparticles, warning_stacklevel=2
        )
        self.__particles = particle_type
        super().__init__()
        self.dE = None
        self.dt = None

    def late_init(self, simulation: Simulation, **kwargs):
        pass

    @property
    def n_macroparticles(self):
        return self.__n_macroparticles


class Observables(MainLoopRelevant):
    def __init__(self, each_turn_i: int):
        super().__init__()
        self.each_turn_i = each_turn_i

    @abc.abstractmethod
    def update(self, simulation: Simulation):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        self._late_init(simulation=simulation, n_turns=kwargs["n_turns"])

    @abc.abstractmethod
    def _late_init(self, simulation: Simulation, n_turns: int):
        pass


class ProfileObservation(Observables):
    def __init__(self, each_turn_i: int):
        super().__init__(each_turn_i=each_turn_i)


class BunchObservation(Observables):
    def __init__(self, each_turn_i: int):
        super().__init__(each_turn_i=each_turn_i)
        self._dts: LateInit[NumpyArray] = None
        self._dEs: LateInit[NumpyArray] = None
        self._flags: LateInit[NumpyArray] = None
        self._write_index = 0

    def _late_init(self, simulation: Simulation, n_turns: int):
        n_entries = n_turns // self.each_turn_i
        n_particles = simulation.ring.beams[0].n_macroparticles
        self._dts = np.empty((n_particles, n_entries))
        self._dEs = np.empty((n_particles, n_entries))
        self._flags = np.empty((n_particles, n_entries))

    def update(self, simulation: Simulation):
        self._dts[:, self._write_index] = simulation.ring.beams[0].dt
        self._dEs[:, self._write_index] = simulation.ring.beams[0].dE
        self._flags[:, self._write_index] = simulation.ring.beams[0].flags

        self._write_index += 1

    @property
    def dts(self):
        return self._dts[: self._write_index]

    @property
    def dEs(self):
        return self._dEs[: self._write_index]

    @property
    def flags(self):
        return self._flags[: self._write_index]


class CavityPhaseObservation(Observables):
    def __init__(self, each_turn_i: int, cavity: Cavity):
        super().__init__(each_turn_i=each_turn_i)
        self._cavity = cavity
        self._phases: LateInit[NumpyArray] = None
        self._write_index = 0

    def _late_init(self, simulation: Simulation, n_turns: int):
        n_entries = n_turns // self.each_turn_i
        self._phases = np.empty(n_entries)

    def update(self, simulation: Simulation):
        self._phases[self._write_index] = self._cavity.rf_program.get_phase(
            simulation.turn_counter.current_turn
        )

        self._write_index += 1

    @property
    def phases(self):
        return self._phases[: self._write_index]


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
        self.generate_hash()
        ring.late_init(simulation=self)
        self.ring: Ring = ring
        self.turn_counter: TurnCounter | None = None
        self.group_counter: GroupCounter | None = None

        if CLASS_DIAGRAM_HACKS:
            self._ring = Ring()  # TODO remove

    def generate_hash(self):
        pass

    def prepare_beam(
        self,
        preparation_routine: MatchingRoutine,
    ):
        preparation_routine.prepare_beam(ring=self.ring)
        if CLASS_DIAGRAM_HACKS:
            self._preparation_routine = MatchingRoutine()

    def run_simulation(
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
        self.turn_counter = TurnCounter(turn_i_init)
        self.group_counter = GroupCounter(self.ring.elements.elements[0].group)
        for turn_i in iterator:
            self.turn_counter.current_turn = turn_i
            for element in self.ring.elements.elements:
                group_onchange = self.group_counter.current_group != element.group
                self.group_counter.current_group = element.group
                if element.is_active_this_turn(turn_i=self.turn_counter.current_turn):
                    element.track(self.ring.beams[0])
            for observable in observe:
                if observable.is_active_this_turn(
                    turn_i=self.turn_counter.current_turn
                ):
                    observable.update(simulation=self)

        # reset counters to uninitialized again
        self.turn_counter = None
        self.group_counter = None

    def load_results(
        self,
        n_turns: int,
        turn_i_init: int = 0,
        observe=Tuple[Observables, ...],
    ) -> SimulationResults:
        return
