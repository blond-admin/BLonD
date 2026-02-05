# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helper functions to deal with the late-init methods of `Simulation`."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING
from unittest.mock import Mock

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any, TypeVar

    T = TypeVar("T")

logger = logging.getLogger(__name__)


def int_from_float_with_warning(
    value: float | int, warning_stacklevel: int
) -> int:
    """
    Make int from float, warn if there are fractional digits.

    Parameters
    ----------
    value
        Some float value, potentially with fractional values.
    warning_stacklevel
        `warnings.warn` parameter.

    Returns
    -------
    int_value
        Integer value converted from input.
    """
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
        return return_value
    else:
        raise TypeError(type(value))


def _find(
    root: Any,
    is_wanted: Callable[[object], bool],
    skip_properties: bool,
) -> Any:
    """
    Find all instances within root that match ``is_wanted``.

    This method does a tree walk on all objects within root.
    Each found object is evaluated with ``is_wanted(obj)``
    and depending on this returned.

    Parameters
    ----------
    root
        Base instance to be inspected.
        All attributes are recursively scanned.
    is_wanted
        Function that identifies what is searched for.
    skip_properties
        If `True`, only attributes that are not a ``@property`` will be
        investigated.

    Returns
    -------
    found_instances
        Set of instances that have been idientified via `is_wanted()`.

    Examples
    --------
    Class attributes that should not be searched for `method_name`
    can be omitted by placing `skip_find_instances_attributes` into the class
    definition.
    >>> class ItsComplicated:
    ...     skip_find_instances_attributes = ["problem"]
    ...
    ...     @property
    ...     def problem(self): # won't be accessed by `find_instances_with_method()`
    ...         raise NotImplementedError()
    ...
    ...     @property # will be accessed
    ...     def not_a_problem(self):
    ...         pass
    """
    found = set()
    seen = set()

    def _walk(
        obj: Any,
        skip_list,
        where,
    ):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        is_mock = isinstance(obj, Mock)
        if hasattr(obj, "skip_find_instances_attributes") and not is_mock:
            skip_list.extend(obj.skip_find_instances_attributes)

        # Check if object has the desired method
        if is_wanted(obj):
            logger.info(f"Found {obj} at {where}")
            found.add(obj)

        # Recurse into object attributes or container elements
        if isinstance(obj, dict):
            for key, value in obj.items():
                _walk(key, skip_list, where + str(key))
                _walk(value, skip_list, where + str(value))
        elif isinstance(obj, (list, tuple, set)):  # NOQA: UP038
            for item in obj:
                _walk(item, skip_list, where + str(item))
        elif hasattr(obj, "__dict__"):
            # checks if is python class
            for attr_name in obj.__dict__ if skip_properties else dir(obj):
                if attr_name in skip_list or (
                    # prevent infinite recursion in mock object
                    is_mock and attr_name in str(obj)
                ):
                    continue
                # Skip built-in attributes or private class attributes
                if attr_name.startswith("__") and attr_name.endswith("__"):
                    continue
                try:
                    attr = getattr(obj, attr_name)
                except Exception:
                    continue  # Skip attributes that raise errors on access
                _walk(attr, skip_list, where + str(attr))

    _walk(
        root,
        skip_list=[
            "_mock_children",  # prevent infinite recursion in mock object
            "return_value",  # prevent infinite recursion in mock object
        ],
        where="",
    )

    return found


def find_instances_with_method(root: Any, method_name: str) -> Any:
    """
    Find all instances within root that have a callable `methodname`.

    This method does a tree walk on all objects within root.
    Class attributes that should not be searched for `method_name`
    can be omitted by placing `skip_find_instances_attributes` into the class
    definition. An example is given below.

    Parameters
    ----------
    root
        Base instance to be inspected.
        All attributes are recursively scanned
        for classes with a method `methodname`.
    method_name
        Name of the method to be searched for.

    Returns
    -------
    found_instances
        Set of instances that have the specified method.

    See Also
    --------
    _find: Internal method used.

    Examples
    --------
    Class attributes that should not be searched for `method_name`
    can be omitted by placing `skip_find_instances_attributes` into the class
    definition.
    >>> class ItsComplicated:
    ...     skip_find_instances_attributes = ["problem"]
    ...
    ...     @property
    ...     def problem(self): # won't be accessed by `find_instances_with_method()`
    ...         raise NotImplementedError()
    ...
    ...     @property # will be accessed
    ...     def not_a_problem(self):
    ...         pass
    """

    def _has_method(obj):
        return (
            hasattr(obj, method_name)
            and callable(getattr(obj, method_name))
            and not isinstance(obj, type)
        )

    found = _find(root=root, is_wanted=_has_method, skip_properties=False)
    return found


def find_instances_by_class(root: Any, class_: type[T]) -> T:
    """
    Find all instances within root that are ``isinstance`` of `class_`.

    This method does a tree walk on all objects within root.

    Parameters
    ----------
    root
        Base instance to be inspected.
        All attributes are recursively scanned
        for ``isinstance(attribute, class_)``.
    class_
        Class type to search for.

    Returns
    -------
    found_instances
        Set of instances that are a ``isinstance(element, class_)``.

    See Also
    --------
    _find: Internal method used.
    """

    def _matches_class(obj):
        return isinstance(obj, class_) and not isinstance(obj, type)

    found = _find(root=root, is_wanted=_matches_class, skip_properties=True)
    return found
