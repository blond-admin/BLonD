# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helper functions to deal with the late-init methods of `Simulation`."""

from __future__ import annotations

import contextlib
import logging
import types
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING
from unittest import mock
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

    def _walk(  # noqa: PLR0912
        obj: Any,
        skip_list,
        where,
    ):
        if id(obj) in seen:
            return
        # Never descend into imported modules: a module is a shared global
        # namespace, so crawling its globals reaches arbitrary third-party
        # state (e.g. pytest's ``mark``) instead of the simulation tree, and
        # is never what we are looking for.
        if isinstance(obj, types.ModuleType):
            return
        # todo remove no cover when Python11 is the main CI tester.
        #  in Python10 this line is never hit..
        if (  # pragma: no cover
            type(obj) is mock._Call
        ):  # prevent crash on `hash(obj)` with mocks...
            return

        seen.add(id(obj))
        is_mock = isinstance(obj, Mock)
        if hasattr(obj, "skip_find_instances_attributes") and not is_mock:
            # Guard against objects whose ``__getattr__`` fabricates any
            # attribute (e.g. pytest's ``MarkGenerator``): the value may be
            # absent or not iterable, in which case there is nothing to skip.
            with contextlib.suppress(TypeError):
                skip_list.extend(obj.skip_find_instances_attributes)

        # Check if object has the desired method
        if is_wanted(obj):
            # Lazy %-args, not an f-string: the message is only rendered
            # if INFO is actually enabled, and ``obj`` here can be a
            # whole beam or profile.
            logger.info("Found %s at %s", obj, where)
            found.add(obj)

        # Recurse into object attributes or container elements.
        #
        # The breadcrumb is built from NAMES -- keys, indices, attribute
        # names -- and never from the values being walked.  Interpolating
        # the value instead made every numpy array in the tree render
        # itself to text on every visit, to build a string for a log line
        # that is almost never emitted: ~0.9 s of a 5-turn two-beam RCS
        # run, a third of it inside ``numpy.arrayprint``.  Names are also
        # the more useful path ("root.profile.hist_y" over a wall of
        # digits).
        if isinstance(obj, dict):
            for key, value in obj.items():
                _walk(key, skip_list, f"{where}[key]")
                _walk(value, skip_list, f"{where}[{key!s:.40}]")
        elif isinstance(obj, (list, tuple, set)):  # NOQA: UP038
            for index, item in enumerate(obj):
                _walk(item, skip_list, f"{where}[{index}]")
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
                _walk(attr, skip_list, f"{where}.{attr_name}")

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
    """

    def _matches_class(obj):
        return isinstance(obj, class_) and not isinstance(obj, type)

    found = _find(root=root, is_wanted=_matches_class, skip_properties=True)
    return found
