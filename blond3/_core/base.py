from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Literal

import numpy as np

from .backends.backend import backend

if TYPE_CHECKING:  # pragma: no cover
    from os import PathLike

    from .beam.base import BeamBaseClass
    from .simulation.simulation import Simulation
    from typing import (
        Optional,
        TypeVar,
        List,
        Callable,
        Any,
        Type,
    )
    from numpy.typing import NDArray as NumpyArray

    T = TypeVar("T")


class Preparable(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        pass

    @abstractmethod
    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        beam
            Simulation beam object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        pass


class MainLoopRelevant(Preparable):
    """
    Base class for objects that are relevant for the simulation main loop

    Attributes
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.



    """

    def __init__(self) -> None:
        super().__init__()
        self.each_turn_i = 1
        self.active = True

    def is_active_this_turn(self, turn_i: int) -> bool:
        """Whether the element is active or not"""
        if self.active:
            return turn_i % self.each_turn_i == 0
        else:
            return False


class Schedulable:
    """
    Base class for objects with schedule parameters

    Attributes
    ----------
    schedules
        Dictionary to update a certain attribute by some value
        via `apply_schedules`
    """

    def __init__(self):
        super().__init__()
        self.schedules: dict[str, _Scheduled] = {}

    def schedule(
        self,
        attribute: str,
        value: float | int | NumpyArray | Tuple[NumpyArray, NumpyArray],
        mode: Literal["per-turn", "constant"] | None = None,
    ):
        """
        Schedule a parameter to be changed during simulation

        Notes
        -----
        Can be constant, per turn or interpolated in time


        Parameters
        ----------
        attribute
            Attribute that shall be changed by scheduler
        value
            Values to be set during schedule
        mode
            Required when arrays are handed over
            "per-turn" or "constant"
        """
        assert hasattr(self, attribute), (
            f"Attribute {attribute} doesnt exist, choose from" f" {vars(self)}"
        )

        self.schedules[attribute] = get_scheduler(value, mode=mode)

    def schedule_from_file(
        self,
        attribute: str,
        filename: str | PathLike,
        mode: Literal["per-turn", "constant"] | None = None,
        **kwargs_loadtxt,
    ):
        """
        Schedule a parameter to be changed during simulation

        Notes
        -----
        Can be constant, per turn or interpolated in time


        Parameters
        ----------
        attribute
            Attribute that shall be changed by scheduler
        filename
            Filename to read the parameters from
        mode
            Required when arrays are handed over
            "per-turn" or "constant"
        kwargs_loadtxt
            Additional keyword arguments to be passed to `numpy.loadtxt`
        """
        assert hasattr(self, attribute), (
            f"Attribute {attribute} doesnt exist, choose from" f" {vars(self)}"
        )
        values = np.loadtxt(filename, **kwargs_loadtxt)
        self.schedules[attribute] = get_scheduler(values, mode=mode)

    def apply_schedules(self, turn_i: int, reference_time: float):
        """Set value of schedule to the target parameter for current turn/time

        Parameters
        ----------
        turn_i
            Currently turn index
        reference_time
            Current time, in [s]
        """
        for attribute, schedule in self.schedules.items():
            self.__setattr__(
                attribute,
                schedule.get_scheduled(turn_i=turn_i, reference_time=reference_time),
            )


class BeamPhysicsRelevant(MainLoopRelevant):
    """
    Main loop element with relevance for beam physics

    Parameters
    ----------
    section_index
        Section index to group elements into sections
    name
        User given name of the element

    Attributes
    ----------
    name
        User given name of the element
    """

    n_instances = 0

    def __init__(self, section_index: int = 0, name: Optional[str] = None):
        super().__init__()
        self._section_index = section_index
        if name is None:
            name = f"Unnamed-{type(self).__name__}-" f"{type(self).n_instances:03d}"
        self.name = name
        type(self).n_instances += 1

    @property  # as readonly attributes
    def section_index(self) -> int:
        """Section index to group elements into sections"""
        return self._section_index

    @abstractmethod
    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        pass


class _Scheduled:
    @abstractmethod
    def get_scheduled(self, turn_i: int, reference_time: float):
        """Get the value of the schedule for the current turn/time

        Parameters
        ----------
        turn_i
            Currently turn index
        reference_time
            Current time, in [s]
        """
        pass


class ScheduledConstant(_Scheduled):
    def __init__(self, value: float | int | NumpyArray):
        """Schedule a value that never changes

        Parameters
        ----------
        value
            A constant value
        """
        super().__init__()
        if isinstance(value, np.ndarray):
            self.value = value.astype(backend.float)
        else:
            self.value = backend.float(value)

    def get_scheduled(
        self,
        turn_i: int,
        reference_time: float,
    ) -> float | int | NumpyArray:
        """Get the constant value

        Parameters
        ----------
        turn_i
            Currently turn index
        reference_time
            Current time, in [s]
        """
        return self.value


class ScheduledArray(_Scheduled):
    def __init__(self, values: NumpyArray):
        """Schedule values that change per turn

        Parameters
        ----------
        values
            Values per turn
            (indexing is done via self.values[turn_i])
        """
        super().__init__()
        self.values = values.astype(backend.float)

    def get_scheduled(
        self,
        turn_i: int,
        reference_time: float,
    ) -> NumpyArray:
        """Get the value of the schedule for the current turn

        Parameters
        ----------
        turn_i
            Currently turn index
        reference_time
            Current time, in [s]
        """

        return self.values[turn_i]


class ScheduledInterpolation(_Scheduled):
    def __init__(self, times: NumpyArray, values: NumpyArray):
        """Schedule values that change along time"""

        super().__init__()
        self.times = times
        self.values = values  # TODO values.astype(backend.float)

    def get_scheduled(
        self,
        turn_i: int,
        reference_time: float,
    ):
        """Get the value of the schedule for the current time

        Parameters
        ----------
        turn_i
            Currently turn index
        reference_time
            Current time, in [s]
        """
        return np.interp(reference_time, self.times, self.values)


def get_scheduler(
    value: float | int | NumpyArray | Tuple[NumpyArray, NumpyArray],
    mode: Literal["per-turn", "constant"] | None = None,
):
    """Auto-select the correct class of the schedulers

    Parameters
    ----------
    value
        Can be constant, per turn or interpolated in time
    mode
        Required when arrays are handed over
        "per-turn" or "constant"
    """
    if isinstance(value, int) or isinstance(value, float):
        return ScheduledConstant(value=value)
    elif isinstance(value, np.ndarray):
        assert mode is not None
        if mode == "per-turn":
            return ScheduledArray(values=value)
        elif mode == "constant":
            return ScheduledConstant(value=value)
    elif isinstance(value, np.ndarray):
        return ScheduledInterpolation(times=value[0], values=value[1])


class DynamicParameter:  # TODO add code generation for this method with type-hints
    def __init__(self, value_init):
        """Changeable parameter tact can be subscribed on_change

        Parameters
        ----------
        value_init
            Initial parameter that is set as parameter.value
        """
        self._value = value_init
        self._observers: List[Callable[[Any], None]] = []

    def on_change(self, callback: Callable[[Any], None]):
        """Subscribe to changes on a specific parameter.

        Parameters
        ----------
        callback
            User defined callback `def my_callback(new_value): ...`
        """
        self._observers.append(callback)

    def _notify(self, value):
        """Execute all callbacks of subscribed observers"""
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


class HasPropertyCache(object):
    """Helper objet to use @cached_property() for class methods"""

    def _invalidate_cache(self, props: Tuple[str, ...]):
        """Delete the stored values of functions with @cached_property"""
        for prop in props:
            self.__dict__.pop(prop, None)

    @abstractmethod
    def invalidate_cache(self):
        """Delete the stored values of functions with @cached_property"""
        pass
