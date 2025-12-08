# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Functions that help dealing with iterables.

Authors
-------
Simon Lauber
"""

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def all_equal(iterable: Iterable[T]) -> bool:
    """
    Check if all elements in the iterable are equal.

    Parameters
    ----------
    iterable : Iterable[T]
        An iterable containing elements to be compared.

    Returns
    -------
    all_equal
        True if all elements are equal or the iterable is empty,
        False otherwise.

    Examples
    --------
    >>> all_equal([1, 1, 1])
    >>> True

    >>> all_equal([1, 2, 1])
    >>> False

    >>> all_equal([])
    >>> True
    """
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        return True  # Empty iterable → considered all equal
    return all(x == first for x in iterator)
