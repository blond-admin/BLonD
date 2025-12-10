# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Internal mix-ins to define all `Simulation` relevant classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from os import PathLike
    from typing import Any, TypeVar

    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation

    T = TypeVar("T")


class Preparable(ABC):
    """Internal Mix-in for a class to make it preparable by the `Simulation` object."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod  # pragma: no cover
    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        """
        pass

    @abstractmethod  # pragma: no cover
    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation `Beam` object.
        n_turns
            Number of turns to simulate.
        turn_i_init
            Initial turn to execute simulation.
        **kwargs
            Additional keyword arguments.
        """
        pass


class MainLoopRelevant(Preparable):
    """
    Base class for objects that are relevant for the simulation main loop.

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
        """
        Whether the element is active or not.

        Parameters
        ----------
        turn_i
            Current index of turn.

        Returns
        -------
        bool
            True if the element is active this turn, False otherwise.
        """
        if self.active:
            return turn_i % self.each_turn_i == 0
        else:
            return False


class Schedulable:
    """
    Base class for objects with schedule parameters.

    Attributes
    ----------
    schedules
        Dictionary to update a certain attribute by some value
        via `apply_schedules`
    """

    def __init__(self) -> None:
        super().__init__()
        self.schedules: dict[str, _Scheduled] = {}
        self.schedule_active = False

    def schedule(
        self,
        attribute: str,
        value: float | int | NumpyArray | tuple[NumpyArray, NumpyArray],
        mode: Literal["per-turn", "constant"] | None = None,
    ) -> None:
        """
        Schedule a parameter to be changed during simulation.

        Parameters
        ----------
        attribute
            Attribute that shall be changed by scheduler.
        value
            Values to be set during schedule.
        mode
            Required when arrays are handed over.
            "per-turn" or "constant".

        Notes
        -----
        Can be constant, per turn or interpolated in time.
        """
        assert hasattr(self, attribute), (
            f"Attribute {attribute} doesnt exist, choose from {vars(self)}"
        )
        self.schedules[attribute] = get_scheduler(value, mode=mode)
        self.schedule_active = True

    def schedule_from_file(
        self,
        attribute: str,
        filename: str | PathLike,
        mode: Literal["per-turn", "constant"] | None = None,
        **kwargs_loadtxt,
    ) -> None:
        """
        Schedule a parameter to be changed during simulation.

        Parameters
        ----------
        attribute
            Attribute that shall be changed by scheduler.
        filename
            Filename to read the parameters from.
        mode
            Required when arrays are handed over.
            "per-turn" or "constant".
        **kwargs_loadtxt
            Additional keyword arguments to be passed to `numpy.loadtxt`.

        Notes
        -----
        Can be constant, per turn or interpolated in time.
        """
        assert hasattr(self, attribute), (
            f"Attribute {attribute} doesnt exist, choose from {vars(self)}"
        )
        values = np.loadtxt(filename, **kwargs_loadtxt)
        self.schedules[attribute] = get_scheduler(values, mode=mode)
        self.schedule_active = True

    def apply_schedules(
        self,
        turn_i: int,
        reference_time: float,
    ) -> None:
        """
        Set value of schedule to the target parameter for current turn/time.

        Parameters
        ----------
        turn_i
            Currently turn index.
        reference_time
            Current time, in [s].
        """
        for attribute, schedule in self.schedules.items():
            self.__setattr__(
                attribute,
                schedule.get_scheduled(
                    turn_i=turn_i, reference_time=reference_time
                ),
            )


class SimulationElementBase(MainLoopRelevant, ABC):
    """
    Abstract base class for all elements participating in the main simulation loop.

    Elements derived from this class are executed as part of the simulation's
    main turn-by-turn loop. They can be:

      * :class:`BeamPhysicsRelevant` — modify the beam state (e.g., drifts, rf stations, kicks)
      * :class:`BeamObservationElement — record or analyze beam information without modifying it

    Subclasses must implement:
      - ``on_init_simulation(simulation)``: called once before the simulation loop starts.
      - ``on_run_simulation(simulation, beam, n_turns, turn_i_init, **kwargs)``:
        called during each iteration of the main simulation loop.

    Parameters
    ----------
    section_index
        Identifier used to group elements that belong to the same section of the ring.
    name
        Optional human-readable name for the element.
    **kwargs
        Additional keyword arguments passed to the parent initializer.
    """

    def __init__(
        self, section_index: int = 0, name: str | None = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._section_index = section_index
        if name is None:
            name = (
                f"Unnamed-{type(self).__name__}-{type(self).n_instances:03d}"
                if hasattr(type(self), "n_instances")
                else f"Unnamed-{type(self).__name__}"
            )
        self.name = name

    @property  # as readonly attributes
    def section_index(self) -> int:
        """
        Section index to group elements into sections.

        Returns
        -------
        int
            The section index.
        """
        return self._section_index

    @abstractmethod  # pragma: no cover
    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        """
        pass

    @abstractmethod  # pragma: no cover
    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation `Beam` object.
        n_turns
            Number of turns to simulate.
        turn_i_init
            Initial turn to execute simulation.
        **kwargs
            Additional keyword arguments.
        """
        pass

    def info_string(self, prefix="") -> str:
        """
        Print the state of the object.

        Parameters
        ----------
        prefix
            Add this to the start of the line.

        Returns
        -------
        str
            The state of the object.
        """
        from blond.core.ring.beam_physics_relevant_elements import (
            pretty_string,  # prevent circular import
        )

        filtered_dict = {
            k: pretty_string(v)
            for k, v in self.__dict__.items()
            if (not k.startswith("_")) and (k != "name")
        }
        content = (
            f"{prefix}{self.name:{40 - len(prefix)}s} {(type(self).__name__):20s} "
            f"{str(self.section_index):13s} {filtered_dict}"
        )
        return content

    @abstractmethod  # pragma: no cover
    def track(self, beam: BeamBaseClass) -> None:
        """
        Apply the element's physics effect to the beam.

        Parameters
        ----------
        beam
            The beam object whose state will be updated by this element.
        """
        pass


class BeamPhysicsRelevant(SimulationElementBase):
    """
    Abstract base class for elements that modify the beam state during tracking.

    This class defines the interface for all *physics-relevant* elements in the
    simulation — that is, elements which actively change the beam's longitudinal
    or transverse coordinates (e.g., drifts, rf stations, kicks).

    Each subclass must implement the :meth:`track` method, which applies its
    specific transformation to the beam state during each simulation turn.

    Parameters
    ----------
    section_index
        Identifier grouping elements that belong to the same section of the ring.
        Defaults to 0.
    name
        Human-readable name for the element. If not provided, a unique name is
        automatically generated.
    **kwargs
        Additional keyword arguments passed to the parent.
    """

    n_instances = 0

    def __init__(
        self, section_index: int = 0, name: str | None = None, **kwargs
    ) -> None:
        super().__init__(section_index, name)
        type(self).n_instances += 1


class BeamObservationElement(SimulationElementBase):
    """
    Abstract base class for elements that observe the beam state during tracking.

    Subclasses must implement the :meth:`track` method, which is called during
    each simulation step to access the beam data and record or process relevant
    quantities.

    Parameters
    ----------
    section_index
        Identifier grouping elements that belong to the same section of the ring.
        Defaults to 0.
    name
        Human-readable name for the element. If not provided, a unique name is
        automatically generated.
    **kwargs
        Additional keyword arguments passed to the parent :class:`SimulationElementBase`.
    """

    n_instances = 0

    def __init__(
        self, section_index: int = 0, name: str | None = None, **kwargs
    ) -> None:
        super().__init__(section_index=section_index, name=name, **kwargs)
        type(self).n_instances += 1

    @abstractmethod  # pragma: no cover
    def track(self, beam: BeamBaseClass) -> None:
        """
        Inspect the beam state without modifying it.

        Parameters
        ----------
        beam
            The beam object to be inspected or recorded.
        """
        pass


class UserDefinedElement(BeamPhysicsRelevant, ABC):
    """
    Element that can be defined by the user.

    Notes
    -----
    The ``track()`` method must be implemented.

    Examples
    --------
    >>> from blond import backend
    >>> class TimeRandomizer(UserDefinedElement):
    ...     def __init__(self):
    ...         super().__init__()
    ...
    ...     def track(self, beam: BeamBaseClass):
    ...         dt = beam.write_partial_dt()
    ...         dt += backend.random.rand(len(dt))
    """

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        """
        pass

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation `Beam` object.
        n_turns
            Number of turns to simulate.
        turn_i_init
            Initial turn to execute simulation.
        **kwargs
            Additional keyword arguments.
        """
        pass


class _Scheduled:
    @abstractmethod  # pragma: no cover
    def get_scheduled(
        self,
        turn_i: int,
        reference_time: float,
    ):
        """
        Get the value of the schedule for the current turn/time.

        Parameters
        ----------
        turn_i
            Currently turn index.
        reference_time
            Current time, in [s].
        """
        pass


class ScheduledConstant(_Scheduled):
    """
    Schedule a value that never changes.

    Parameters
    ----------
    value
        A constant value.
    """

    def __init__(self, value: float | int | NumpyArray) -> None:
        super().__init__()
        self.value = value

    def get_scheduled(
        self,
        turn_i: int,
        reference_time: float,
    ) -> float | int | NumpyArray:
        """
        Get the constant value.

        Parameters
        ----------
        turn_i
            Currently turn index.
        reference_time
            Current time, in [s].

        Returns
        -------
        value
            The constant value.
        """
        return self.value


class ScheduledArray(_Scheduled):
    """
    Schedule values that change per turn.

    Parameters
    ----------
    values
        Values per turn.
        (indexing is done via self.values[turn_i]).
    """

    def __init__(self, values: NumpyArray) -> None:
        super().__init__()
        self.values = values

    def get_scheduled(
        self,
        turn_i: int,
        reference_time: float,
    ) -> NumpyArray:
        """
        Get the value of the schedule for the current turn.

        Parameters
        ----------
        turn_i
            Currently turn index.
        reference_time
            Current time, in [s].

        Returns
        -------
        value
            The scheduled value for the current turn.
        """
        return self.values[turn_i]


class ScheduledInterpolation(_Scheduled):
    """
    Schedule values that change along time.

    Parameters
    ----------
    times
        Values along the times axis, in [s].
    values
        Values along the values axis.
    interpolator
        Interpolation routine to get time in between the base values.
        Default: `numpy.interp`.

    See Also
    --------
    blond.generals.interpolation.interp_linear : NumPy's `interp` function
    blond.generals.interpolation.interp_makima : Modified Akima Interpolation
    blond.generals.interpolation.interp_pchip : Piecewise Cubic Hermite Interpolating Polynomial
    """

    def __init__(
        self,
        times: NumpyArray,
        values: NumpyArray,
        interpolator: Callable = np.interp,
    ) -> None:
        super().__init__()
        self.times = times
        self.values = values
        self.interpolator = interpolator

    def get_scheduled(
        self,
        turn_i: int,
        reference_time: float,
    ):
        """
        Get the value of the schedule for the current time.

        Parameters
        ----------
        turn_i
            Currently turn index.
        reference_time
            Current time, in [s].

        Returns
        -------
        value
            The interpolated value for the current time.
        """
        return self.interpolator(reference_time, self.times, self.values)


def get_scheduler(
    value: float | int | NumpyArray | tuple[NumpyArray, NumpyArray],
    mode: Literal["per-turn", "constant"] | None = None,
) -> _Scheduled:
    """
    Auto-select the correct class of the schedulers.

    Parameters
    ----------
    value
        Can be constant, per turn or interpolated in time.
    mode
        Required when arrays are handed over.
        "per-turn" or "constant".

    Returns
    -------
    scheduler
        The appropriate scheduler instance.
    """
    if isinstance(value, int | float):
        return ScheduledConstant(value=value)
    elif isinstance(value, np.ndarray):
        assert mode is not None
        if mode == "per-turn":
            return ScheduledArray(values=value)
        elif mode == "constant":
            return ScheduledConstant(value=value)
        else:
            raise TypeError(type(value))
    elif isinstance(value, tuple):
        return ScheduledInterpolation(times=value[0], values=value[1])
    else:
        raise TypeError(type(value))


class DynamicParameter:  # TODO add code generation for this method with type-hints
    """
    Changeable parameter tact can be subscribed on_change.

    Parameters
    ----------
    value_init
        Initial parameter that is set as parameter.value.
    """

    def __init__(self, value_init: Any) -> None:
        self._value = value_init
        self._observers: list[Callable[[Any], None]] = []

    def on_change(self, callback: Callable[[Any], None]) -> None:
        """
        Subscribe to changes on a specific parameter.

        Parameters
        ----------
        callback
            User defined callback `def my_callback(new_value): ...`.
        """
        self._observers.append(callback)

    def _notify(self, value: Any) -> None:
        """
        Execute all callbacks of subscribed observers.

        Parameters
        ----------
        value
            The new value to notify observers about.
        """
        for callback in self._observers:
            callback(value)

    @property
    def value(self):
        """
        Get the current value.

        Returns
        -------
        value
            The current value.
        """
        return self._value

    @value.setter
    def value(self, new_val: T) -> None:
        """
        Set the current value.

        Parameters
        ----------
        new_val
            The new value to set.
        """
        if new_val != self._value:
            self._notify(new_val)
        self._value = new_val


class HasPropertyCache:
    """Helper objet to use @cached_property() for class methods."""

    def _invalidate_cache(self, props: tuple[str, ...]) -> None:
        """
        Delete the stored values of functions with @cached_property.

        Parameters
        ----------
        props
            Tuple of property names to invalidate.
        """
        for prop in props:
            self.__dict__.pop(prop, None)

    @abstractmethod  # pragma: no cover
    def invalidate_cache(self):
        """Delete the stored values of functions with @cached_property."""
        pass
