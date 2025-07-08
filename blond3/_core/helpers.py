from __future__ import annotations

import warnings
from collections.abc import Iterable as AbstractIterable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any, Union, TypeVar, Iterable
    from numpy.typing import NDArray as NumpyArray

    T = TypeVar("T")


def int_from_float_with_warning(value: float | int, warning_stacklevel: int) -> int:
    """Make int from float, warn if there are fractional digits"""
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


def safe_index(x: Union[T, NumpyArray[T]], idx: int) -> T:
    """Access index, even if x is a float or int."""
    if isinstance(x, np.ndarray):
        return x[idx]
    else:
        return x  # scalar: ignore index


def float_or_array_typesafe(something: float | Iterable, dtype: T) -> T | NumpyArray[T]:
    """Apply dtype for iterable or number."""
    if isinstance(something, AbstractIterable):
        something = np.array(something, dtype=dtype)
    else:
        something = dtype(something)
    return something


def find_instances_with_method(root: Any, method_name: str):
    """Find all instances within root that have a callable `methodname`"""
    found = set()
    seen = set()

    def walk(obj: Any):
        if id(obj) in seen:
            return
        seen.add(id(obj))

        # Check if object has the desired method
        if hasattr(obj, method_name) and callable(getattr(obj, method_name)):
            found.add(obj)

        # Recurse into object attributes or container elements
        if isinstance(obj, dict):
            for key, value in obj.items():
                walk(key)
                walk(value)
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                walk(item)
        elif hasattr(obj, "__dict__"):  # checks if is python class
            for attr_name in dir(obj):
                # Skip built-in attributes
                if attr_name.startswith("__") and attr_name.endswith("__"):
                    continue
                try:
                    attr = getattr(obj, attr_name)
                except Exception:
                    continue  # Skip attributes that raise errors on access
                walk(attr)

    walk(root)
    return found
