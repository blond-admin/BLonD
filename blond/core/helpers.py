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
    >>> class ItsComplicated:
    ...     skip_find_instances_attributes = ["problem"]
    ...
    ...     @property
    ...     def problem(self): # wont be accessed by `find_instances_with_method()`
    ...         raise NotImplementedError()
    ...
    ...     @property # will be accessed
    ...     def not_a_problem(self):
    ...         pass
    """
    found = set()
    seen = set()
    import os
    import sys

    print("TRACE:", sys.gettrace())

    def walk(obj: Any, skip_list, where):
        if id(obj) in seen:
            return
        print(
            "PID", os.getpid(), "PPID", os.getppid(), "TRACE", sys.gettrace()
        )
        print(f"obj {obj}")
        print(f"where {where}")
        # if not sys.gettrace():
        #     import faulthandler
        #
        #     faulthandler.dump_traceback(all_threads=True)
        #     sys.exit(-1)
        seen.add(id(obj))
        is_mock = isinstance(obj, Mock)
        if hasattr(obj, "skip_find_instances_attributes") and not is_mock:
            skip_list.extend(obj.skip_find_instances_attributes)

        # Check if object has the desired method
        if (
            hasattr(obj, method_name)
            and callable(getattr(obj, method_name))
            and not isinstance(obj, type)
        ):
            logger.info(f"Found {obj}.{method_name}(..) at {where}")
            found.add(obj)

        # Recurse into object attributes or container elements
        if isinstance(obj, dict):
            for key, value in obj.items():
                walk(key, skip_list, where + str(key))
                walk(value, skip_list, where + str(value))
        elif isinstance(obj, (list, tuple, set)):  # NOQA: UP038
            for item in obj:
                walk(item, skip_list, where + str(item))
        elif (
            hasattr(obj, "__dict__") and not is_mock
        ):  # checks if is python class
            for attr_name in dir(obj):
                if attr_name in skip_list:
                    continue
                # Skip built-in attributes or private class attributes
                if attr_name.startswith("__") and attr_name.endswith("__"):
                    continue
                try:
                    attr = getattr(obj, attr_name)
                except Exception:
                    # print(e.with_traceback(None))
                    continue  # Skip attributes that raise errors on access
                print(attr)
                print(where + str(attr))
                print("ill walk on")
                walk(attr, skip_list, where + str(attr))

    walk(root, skip_list=[], where="")
    # sys.exit(-1)
    return found
