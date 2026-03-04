# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Internal mix-ins to define all `Simulation` relevant classes."""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from scipy.interpolate import interp1d

from blond.core.backends.backend import backend
from blond.generals.cupy import no_cupy_import

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from os import PathLike
    from typing import Any, TypeVar

    from cupy.typing import NDArray as CupyArray
    from numpy.typing import ArrayLike
    from numpy.typing import NDArray as NumpyArray
    from scipy.interpolate import (
        Akima1DInterpolator,
        PchipInterpolator,
    )

    from blond.core.beam.base import BeamBaseClass
    from blond.core.reference_clock.reference_clock import ReferenceCoordinates
    from blond.core.simulation.simulation import Simulation
    from blond.generals.protocols import AnyInterpolator
    from blond.handle_results.observables import ObservablesOncePerTurnBase

    T = TypeVar("T")

logger = logging.getLogger(__name__)


class Preparable(ABC):
    """
    Internal Mix-in for a class to make it preparable by the `Simulation` object.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments for method
        resolution order of inheriting elements.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

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
        **kwargs
            Additional keyword arguments.
        """
        pass


class MainLoopRelevant(Preparable):
    """
    Base class for objects that are relevant for the simulation main loop.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments for method
        resolution order of inheriting elements.

    Attributes
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
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

    Parameters
    ----------
    **kwargs
        Additional keyword arguments for method
        resolution order of inheriting elements.

    Attributes
    ----------
    schedules
        Dictionary to update a certain attribute by some value
        via `apply_schedules`
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.intended_for_scheduling = set()
        self.schedules: dict[str, SchedulerBaseClass] = {}
        self.schedule_active = False

    def _add_intended_schedule(self, *names: str) -> None:
        """
        Add a variable name to the intended schedules.

        When scheduling anything different as an intended variable,
        this class will issue a `UserWarning`.

        Parameters
        ----------
        *names
            Names of a variable.
        """
        for name in names:
            self.intended_for_scheduling.add(str(name))

    def schedule(
        self,
        attribute: str,
        value: ScheduledArray
        | ScheduledInterpolation
        | NumpyArray
        | tuple[NumpyArray, NumpyArray],
    ) -> None:
        """
        Schedule a parameter to change dynamically during the simulation.

        This method allows you to define how an attribute of the object
        evolves over time. The scheduling can be done using various types
        of input for convenience or precise control.

        Parameters
        ----------
        attribute
            The name of the attribute to be scheduled.
            Must be an existing attribute of the object.

        value
            The schedule definition for the attribute.
            Can be provided in one of several forms:

            1. **Convenient input options**:
                - `NumpyArray`: Automatically cast to `ScheduledArray`.
                - `tuple[NumpyArray, NumpyArray]`: Automatically cast to `ScheduledInterpolation`.

            2. **Explicit scheduling objects**:
                - `ScheduledArray`: Full control over array-based scheduling.
                - `ScheduledInterpolation`: Full control over interpolation-based scheduling.

        Raises
        ------
        AssertionError
            If the specified attribute does not exist on the object.

        Notes
        -----
        - Once a schedule is applied, the `schedule_active` flag is set to True.
        - For convenience, non-explicit types are automatically converted using `get_scheduler`.
        """
        if attribute not in self.intended_for_scheduling:
            warnings.warn(
                f"'{attribute}' is not intended to be scheduled. "
                f"This can result in bugs. Use at your own risk.",
                UserWarning,
                stacklevel=2,
            )
        assert hasattr(self, attribute), (
            f"Attribute {attribute} doesnt exist, choose from {vars(self)}"
        )
        if isinstance(value, SchedulerBaseClass):
            # explicit declaration
            self.schedules[attribute] = value
        else:
            # should allow easier user input, but is less explicit
            self.schedules[attribute] = get_scheduler(value)
        self.schedule_active = True

        self.apply_schedules(turn_i=0, reference_time=0)

    def schedule_from_file(
        self,
        attribute: str,
        filename: str | PathLike,
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
        self.schedules[attribute] = get_scheduler(values)
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
            value = schedule.get_scheduled(
                turn_i=turn_i, reference_time=reference_time
            )
            self.__setattr__(
                attribute,
                value,
            )
            logger.debug(f"Wrote {self}.{attribute} = {value}")


class SimulationElementBase(MainLoopRelevant, ABC):
    """
    Abstract base class for all elements participating in the main simulation loop.

    Elements derived from this class are executed as part of the simulation's
    main turn-by-turn loop. They can be:

      * :class:`BeamPhysicsRelevant` — modify the beam state (e.g., drifts, rf stations, kicks)
      * :class:`BeamObservationElement — record or analyze beam information without modifying it

    Subclasses must implement:
      - ``on_init_simulation(simulation)``: called once before the simulation loop starts.
      - ``on_run_simulation(simulation, beam, n_turns, **kwargs)``:
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

    n_instances = 0

    def __init__(
        self, section_index: int = 0, name: str | None = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        type(self).n_instances += 1

        self._section_index = section_index
        if name is None:
            name = (
                f"Unnamed-{type(self).__name__}-{type(self).n_instances:03d}"
                if hasattr(type(self), "n_instances")
                else f"Unnamed-{type(self).__name__}"
            )
        self.name = name
        self.observables: dict[
            int, ObservablesOncePerTurnBase
        ] = {}  # one observable per beam id

    def add_observable(
        self, beam: BeamBaseClass, observable: ObservablesOncePerTurnBase
    ) -> None:
        """
        Add the observable to the self.observables dict at the id of the beam.

        Parameters
        ----------
        beam
            Beam, which should be tracked by this observable.
        observable
            Observable to be added to the list.
        """
        if id(beam) in self.observables:
            raise ValueError(f"Observable for {id(beam)=} already set.")
        self.observables[id(beam)] = observable

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
    def _track(self, beam: BeamBaseClass) -> None:
        """
        Apply the element's physics effect to the beam.

        Parameters
        ----------
        beam
            The beam object whose state will be updated by this element.
        """
        pass

    def track(self, beam: BeamBaseClass) -> None:
        """
        Check if the element is active this turn and then call _track.

        Additionally, if any observables are attached to this element via
        self.observables, they will be called.

        Parameters
        ----------
        beam
            The beam object whose state will be updated by this element.
        """
        if self.active:
            self._track(beam=beam)
        if id(beam) in self.observables:
            self.observables[id(beam)].update()


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

    def __init__(
        self, section_index: int = 0, name: str | None = None, **kwargs
    ) -> None:
        super().__init__(section_index, name, **kwargs)


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

    def __init__(
        self, section_index: int = 0, name: str | None = None, **kwargs
    ) -> None:
        super().__init__(section_index=section_index, name=name, **kwargs)


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
    ...     def _track(self, beam: BeamBaseClass):
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
        **kwargs
            Additional keyword arguments.
        """
        pass


# n.b.:  runtime_checkable will check the method is present, but does
# not validate the signature.
@runtime_checkable
class _Trackable(Protocol):
    def track(self, beam): ...


class UnsafeUserElement(UserDefinedElement):
    """
    Class to wrap around an arbitrary user defined element.

    Used to sanitise non-standard objects defined by the user, should
    not be used for production code.

    The given `.track` method will be called on every turn, and the
    element will be taken as part of section 0.  For any other
    behaviour, inheriting from `UserDefinedElement` is essential.

    Parameters
    ----------
    element
        The element defined by the user, must implement a
        `.track(self, beam)` method.

    Examples
    --------
    >>> class Test:
    ...    def track(self, beam):
    ...        print("This is a test")
    >>> ring.add_element(Test())
    """

    def __init__(self, element: _Trackable):
        if not isinstance(element, _Trackable):
            raise TypeError(
                "Arbitrary user elements must at minimum "
                "define a `.track(self, beam)` method."
            )
        else:
            warnings.warn(
                f"Element {element} (class name {element.__class__.__name__}) "
                "is not recognised, attempting to coerce it to a usable form, "
                "but results are not guaranteed. Inheriting from "
                "`UserDefinedElement` is strongly recommended.",
                stacklevel=2,
            )

        super().__init__()
        self._element = element

    def _track(self, beam: BeamBaseClass):
        self._element.track(beam)


class SchedulerBaseClass(ABC):
    """Base class to create objects used for scheduling of parameters."""

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


class ScheduledArray(SchedulerBaseClass):
    """
    Schedule values that change per turn.

    Parameters
    ----------
    values
        Values per turn.
        (indexing is done via self.values[turn_i]).
    """

    def __init__(self, values: NumpyArray | CupyArray) -> None:
        super().__init__()
        self.values = values

    def get_scheduled(
        self,
        turn_i: int,
        reference_time: float,
    ) -> NumpyArray | CupyArray:
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
        value = self.values[turn_i]
        if no_cupy_import.is_cupy_array(value):
            value = value.get()

        return value


class ScheduledInterpolation(SchedulerBaseClass):
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
        Default: `scipy.interpolate.interp1d`.
    **kwargs
        Optional keyword arguments for the interpolator.

    See Also
    --------
    scipy.interpolate.interp1d : 1D interpolator similar to `np.interp`.
    scipy.interpolate.Akima1DInterpolator : Modified Akima Interpolation.
    scipy.interpolate.PchipInterpolator : Piecewise Cubic Hermite Interpolating Polynomial.

    Examples
    --------
    Using the Akima interpolation

    >>> import scipy
    >>> t_arr = np.linspace(0, 10)
    >>> vals = np.linspace(-10, 0)
    >>> scheduler = ScheduledInterpolation(
    ...     times=t_arr,
    ...     values=vals,
    ...     interpolator=scipy.interpolate.Akima1DInterpolator,
    ...     method="makima",
    ... )

    Using the Akima interpolation
    >>> import scipy
    >>> t_arr = np.linspace(0, 10)
    >>> vals = np.linspace(-10, 0)
    >>> scheduler = ScheduledInterpolation(
    ...     times=t_arr,
    ...     values=vals,
    ...     interpolator=scipy.interpolate.PchipInterpolator,
    ...     method="makima",
    ... )
    """

    def __init__(
        self,
        times: NumpyArray | CupyArray,
        values: NumpyArray | CupyArray,
        interpolator: type[
            Akima1DInterpolator
            | PchipInterpolator
            | interp1d
            | AnyInterpolator
        ] = interp1d,
        **kwargs,
    ) -> None:
        super().__init__()
        self.interpolator = interpolator(times, values, **kwargs)

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
        return self.interpolator(reference_time)


def get_scheduler(
    value: ArrayLike,
) -> SchedulerBaseClass:
    """
    Auto-select the correct class of the schedulers.

    Parameters
    ----------
    value
        Array - per turn
        (Array, Array) - time vs value, to be interpolated.

    Returns
    -------
    scheduler
        The appropriate scheduler instance.
    """
    value = backend._asarray_if_needed(value)
    match value.shape:
        case (_,):
            schedule = ScheduledArray(values=value)
        case _, _:
            schedule = ScheduledInterpolation(times=value[0], values=value[1])
        case _:
            raise ValueError(
                "Scheduler input must be either 1D for turn-based or 2D"
                " for time-based"
            )

    return schedule


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


class AltersReference(ABC):
    """Base class for objects that alter the reference coordinate system."""

    @abstractmethod  # pragma: no cover
    def track_reference(self, reference: ReferenceCoordinates, **kwargs):
        """
        Update the coordinates of the reference coordinate system.

        Parameters
        ----------
        reference
            The object that holds the reference time [s] and total energy [eV].
        **kwargs
            Allows more arguments in the method definition outside the
            abstract class.

        Returns
        -------
        change
            Change of reference time or energy.
        """
        pass
