# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public License version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.md.
# In applying this license, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Function(s) for pre-processing input data**

:Authors: **Simon Albright**
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

import numpy as np

from . import exceptions as blond_exceptions
from . import bmath as bm

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray
    from typing import Any, Iterable, TypeVar

    T = TypeVar("T")


def check_input(variable: Any, msg: str, *args) -> tuple[bool, Any]:
    """Check input and return InputDataError exception with user defined
    message if False

    Args:
        variable (_type_): _description_
        msg (_type_): _description_

    Raises:
        blond_exceptions.InputDataError: _description_

    Returns:
        _type_: _description_
    """
    result = check_data_dimensions(variable, *args)

    if result[0]:
        return result
    raise blond_exceptions.InputDataError(msg)


def check_data_dimensions(input_data: Any, *args) -> tuple[bool, Any]:
    """
    General function to check if input_data is number, or nD array
    for each member of args the input_data is checked
    0 is taken as checking if input_data is a number
    integers are taken as the length of a list-like input
    tuples or lists are taken as dimensions
    [[1, 2], [1, 2]] will return true for length = 2 or dims = (2, 2)

    Args:
        input_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    success = False

    for arg in args:
        if arg == 0:
            if _check_number(input_data):
                success = True

        elif isinstance(arg, int):
            if _check_length(input_data, arg):
                success = True

        else:
            if _check_dimensions(input_data, arg):
                success = True

        if success:
            break

    return success, type(input_data)


def _check_number(input_data: Any) -> bool:
    """returns True if input_data can be cast to int

    Args:
        input_data (_type_): _description_

    Raises:
        TypeError: _description_

    Returns:
        _type_: _description_
    """
    try:
        int(input_data)
        if isinstance(input_data, np.ndarray):
            raise TypeError
        return True
    except (TypeError, ValueError):
        return False


def _check_length(input_data: NumpyArray | list | tuple, length: int) -> bool:
    """Returns True if len(input_data) == length
    Should this return True if n-dim > 1?

    Args:
        input_data (_type_): _description_
        length (_type_): _description_

    Raises:
        TypeError: _description_

    Returns:
        _type_: _description_
    """
    if not _check_number(length):
        raise TypeError("Length must be numeric")

    try:
        return len(input_data) == length
    except TypeError:
        return False


def _check_dimensions(
    input_data: NumpyArray | list | tuple, dim: Iterable[int]
) -> bool:
    """
    Casts input_data to numpy array and dimensions to tuple
    compares shape of array to tuple and returns True if equal.
    If dim[n] == -1 it is set to input_data.shape[n] to allow comprehension
    of arrays with arbitrary length in one or more dimensions.
    Args:
        input_data (_type_): _description_
        dim (_type_): _description_

    Raises:
        TypeError: _description_

    Returns:
        _type_: _description_
    """

    try:
        iter(dim)
    except TypeError:
        dim = [dim]

    input_shape = np.array(input_data).shape

    try:
        if -1 in dim:
            try:
                dim = [
                    input_shape[i] if dim[i] == -1 else dim[i]
                    for i in range(len(dim))
                ]
            except IndexError:
                return False
    except TypeError as exc:
        raise TypeError("dim must be number or iterable of numbers") from exc

    return input_shape == tuple(dim)


def interp_if_array(new_x: float, value: T | Iterable[Iterable[T]]) -> T:
    """
    Interpolate value at new_x if it is a 2-array.  If it is a number,
    return the same value

    Args:
        new_x (float): The new x at which to interpolate
        value (T | Iterable[Iterable[T]]): Either a number or an array
                                           to be interpolated.

    Returns:
        T: The interpolated value
    """

    if isinstance(value, numbers.Number):
        return value
    else:
        return bm.interp(new_x, value[0], value[1])
