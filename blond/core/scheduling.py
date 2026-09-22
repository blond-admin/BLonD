# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Scheduler objects to change parameters during the simulation."""

from __future__ import annotations

import numbers
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import interp1d

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from typing import Any

    from numpy.typing import NDArray as NumpyArray
    from scipy.interpolate import (
        Akima1DInterpolator,
        PchipInterpolator,
    )

    from blond.generals.protocols import AnyInterpolator


class ScheduledBaseClass(ABC):
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


class ScheduledArray(ScheduledBaseClass):
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


class ScheduledInterpolation(ScheduledBaseClass):
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
    blond.generals.interpolators.DerivativeInterpolator : Smooth derivative.

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

    Using the PCHIP interpolation

    >>> import scipy
    >>> t_arr = np.linspace(0, 10)
    >>> vals = np.linspace(-10, 0)
    >>> scheduler = ScheduledInterpolation(
    ...     times=t_arr,
    ...     values=vals,
    ...     interpolator=scipy.interpolate.PchipInterpolator,
    ... )
    """

    def __init__(
        self,
        times: NumpyArray,
        values: NumpyArray,
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
        value = self.interpolator(reference_time)

        # Guard against 0D arrays being returned by interpolator
        if not isinstance(value, numbers.Number) and value.shape == ():
            value = value[()]

        return value


class ScheduledFunctional(ScheduledBaseClass):
    """
    Schedule values computed by a user-supplied callable.

    The callable is evaluated once per turn and its return value is
    written to the scheduled attribute. This allows the value to follow
    an arbitrary functional relationship, e.g. a closed-form expression
    in the turn index or time (which can be built from a `sympy`
    expression via `sympy.lambdify`).

    Parameters
    ----------
    function
        Callable evaluated each turn. It is called with the keyword
        arguments `turn_i` and `reference_time` (matching
        `get_scheduled`) and must return the scheduled value. Arguments
        that are not needed can simply be ignored.

    Examples
    --------
    A sinusoidal voltage program in time:

    >>> import numpy as np
    >>> scheduler = ScheduledFunctional(
    ...     lambda turn_i, reference_time: 6e6 * np.sin(2 * np.pi * reference_time)
    ... )

    A value that depends on the turn index:

    >>> scheduler = ScheduledFunctional(
    ...     lambda turn_i, reference_time: 1e3 * turn_i
    ... )
    """

    def __init__(
        self,
        function: Callable[..., Any],
    ) -> None:
        super().__init__()
        self.function = function

    def get_scheduled(
        self,
        turn_i: int,
        reference_time: float,
    ) -> Any:
        """
        Get the value of the schedule by evaluating the callable.

        Parameters
        ----------
        turn_i
            Currently turn index.
        reference_time
            Current time, in [s].

        Returns
        -------
        value
            The value returned by the callable for the current turn/time.
        """
        return self.function(turn_i=turn_i, reference_time=reference_time)


def get_scheduler(
    value: NumpyArray | tuple[NumpyArray, NumpyArray] | Callable[..., Any],
) -> ScheduledBaseClass:
    """
    Auto-select the correct class of the schedulers.

    Parameters
    ----------
    value
        Array - per turn
        (Array, Array) - time vs value, to be interpolated.
        Callable - evaluated per turn via `ScheduledFunctional`.

    Returns
    -------
    scheduler
        The appropriate scheduler instance.
    """
    if isinstance(value, np.ndarray):
        return ScheduledArray(values=value)
    elif isinstance(value, tuple):
        return ScheduledInterpolation(times=value[0], values=value[1])
    elif callable(value):
        return ScheduledFunctional(function=value)
    else:
        raise TypeError(type(value))
