# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Interpolation routines that resemble the `np.interp` arguments."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

from typing import Any, Protocol


class AnyInterpolator(Protocol):
    """
    A simplified Protocol for 1D interpolation.

    A simplified Protocol for 1D interpolation methods like `interp1d`, `Akima1DInterpolator`,
    and `PchipInterpolator`, with only __init__ and __call__ methods.

    Parameters
    ----------
    x
        Array of input data points.
    y
        Array of output values corresponding to x.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(self, x: NumpyArray, y: NumpyArray, **kwargs) -> None: ...

    def __call__(self, x: float | NumpyArray) -> NumpyArray:
        """
        Interpolate at new points x.

        Parameters
        ----------
        x
            The points at which to evaluate the interpolated values.

        Returns
        -------
        y
            Interpolated values at the given points.
        """
        ...


class ScheduleFunction(Protocol):
    """
    A callable that computes a scheduled value per turn.

    Used by `ScheduledFunctional`, which calls it with the keyword
    arguments `turn_i` and `reference_time`, so the parameters must
    carry exactly these names.
    """

    def __call__(self, *, turn_i: int, reference_time: float) -> Any:
        """
        Compute the scheduled value.

        Parameters
        ----------
        turn_i
            Current turn index.
        reference_time
            Current time, in [s].

        Returns
        -------
        value
            The scheduled value for this turn/time.
        """
        ...
