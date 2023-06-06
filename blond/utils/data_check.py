# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public License version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.md.
# In applying this license, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Function(s) for pre-processing input data**

:Authors: **Simon Albright**
'''

import numpy as np

import blond.utils.exceptions as blExcept


def check_input(variable, msg, *args):
    """Check input and return InputDataError exception with user defined
    message if False

    Args:
        variable (_type_): _description_
        msg (_type_): _description_

    Raises:
        blExcept.InputDataError: _description_

    Returns:
        _type_: _description_
    """
    result = check_data_dimensions(variable, *args)

    if result[0]:
        return result
    raise blExcept.InputDataError(msg)


def check_data_dimensions(input_data, *args):
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


def _check_number(input_data):
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


def _check_length(input_data, length):
    """ Returns True if len(input_data) == length
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


def _check_dimensions(input_data, dim):
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
                dim = [input_shape[i] if dim[i] == -1
                       else dim[i] for i in range(len(dim))]
            except IndexError:
                return False
    except TypeError as exc:
        raise TypeError("dim must be number or iterable of numbers") from exc

    return input_shape == tuple(dim)
