# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Late-initialised attributes that stay non-optional for type checkers.

Many BLonD objects are built in two steps: ``__init__`` stores the user
parameters, and a later step (``on_init_simulation``, ``setup_beam``,
...) fills attributes that depend on the rest of the simulation.
Typing those as ``T | None`` forces every later reader to narrow with
an ``assert``.  `_LateInit` declares them as plain ``T`` instead, and
turns a read before they are filled into a `NotInitialisedError` that
names what fills them.

Only use it for attributes that are *always* filled eventually; an
attribute that may legitimately stay absent stays ``T | None``.
"""

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, overload

from .exceptions_ import NotInitialisedError

_T = TypeVar("_T")

# Names declared on each class, in declaration order.  Weakly keyed so
# dynamically created classes are not kept alive by the registry.
_declared: weakref.WeakKeyDictionary[type, tuple[str, ...]] = (
    weakref.WeakKeyDictionary()
)


class _LateInit(Generic[_T], ABC):
    def __init__(self) -> None:
        # ``None`` in the instance ``__dict__`` shadows any class
        # docstring, so Sphinx does not repeat it on every attribute;
        # describe attributes in the owner's ``Attributes`` section.
        self.__doc__ = None

    def __set_name__(self, owner: type[Any], name: str) -> None:
        """
        Remember the attribute name and register it on the owner.

        Parameters
        ----------
        owner
            Class owning the attribute.
        name
            Name of the attribute.
        """
        self._pub_name = name
        _declared[owner] = (*_declared.get(owner, ()), name)

    # The overloads make instance reads plain ``T``; without them type
    # checkers see ``T | _LateInit[T]``.
    @overload
    def __get__(self, instance: None, owner: type[Any]) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type[Any] | None) -> _T: ...

    @abstractmethod
    def __get__(
        self, instance: object | None, owner: type[Any] | None = None
    ) -> _T | Self: ...

    if TYPE_CHECKING:
        # Checker-only stubs.  Defining these at runtime would make this
        # a data descriptor, routing every read of a filled attribute
        # through the Python-level ``__get__`` (several times slower).
        def __set__(self, instance: object, value: _T) -> None:
            """
            Fill the attribute.

            Parameters
            ----------
            instance
                Object owning the attribute.
            value
                New value of the attribute.
            """

        def __delete__(self, instance: object) -> None:
            """
            Reset the attribute to not initialised.

            Parameters
            ----------
            instance
                Object owning the attribute.
            """


class InitalisedInternally(_LateInit[_T]):
    """
    Attribute BLonD fills while setting up or starting a run.

    Filled by a lifecycle hook, and re-filled on every run. Must hold a
    value before the first turn, so reading one unfilled is a
    sequencing fault in BLonD rather than a mistake in the input
    script.

    Parameters
    ----------
    set_by
        What fills the attribute, for the error message.
    """

    def __init__(self, set_by: str) -> None:
        super().__init__()
        self._set_by = set_by

    # Repeated from `_LateInit`: an override without them makes an
    # instance read type as ``T | Self`` instead of plain ``T``.
    @overload
    def __get__(self, instance: None, owner: type[Any]) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type[Any] | None) -> _T: ...

    def __get__(
        self, instance: object | None, owner: type[Any] | None = None
    ) -> _T | Self:
        """
        Read the attribute.

        Parameters
        ----------
        instance
            Object owning the attribute, or `None` for class access.
        owner
            Class owning the attribute.

        Returns
        -------
        value
            The attribute value, or the descriptor itself on class
            access.

        Raises
        ------
        NotInitialisedError
            If the attribute has not been filled yet.
        """
        if instance is None:
            return self
        # Only reached while the instance ``__dict__`` lacks the name.
        inst_class = type(instance).__name__
        raise NotInitialisedError(
            f"{inst_class}.{self._pub_name} is not "
            f"initialised yet; it will be initialised when "
            f"{inst_class}.{self._set_by} is called.",
            name=self._pub_name,
            obj=instance,
        )


class AssignedDuringTracking(_LateInit[_T]):
    """
    Attribute filled by tracking, once the simulation runs.

    Carries no value before the first turn, so it is exempt from any
    check that setup left everything filled. Should be cleared when a
    new run starts: nothing re-fills it, so a value from an earlier run
    would otherwise persist into the next one.
    """

    # Repeated from `_LateInit`: an override without them makes an
    # instance read type as ``T | Self`` instead of plain ``T``.
    @overload
    def __get__(self, instance: None, owner: type[Any]) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type[Any] | None) -> _T: ...

    def __get__(
        self, instance: object | None, owner: type[Any] | None = None
    ) -> _T | Self:
        """
        Read the attribute.

        Parameters
        ----------
        instance
            Object owning the attribute, or `None` for class access.
        owner
            Class owning the attribute.

        Returns
        -------
        value
            The attribute value, or the descriptor itself on class
            access.

        Raises
        ------
        NotInitialisedError
            If the attribute has not been filled yet.
        """
        if instance is None:
            return self
        # Only reached while the instance ``__dict__`` lacks the name.
        inst_class = type(instance).__name__
        raise NotInitialisedError(
            f"{inst_class}.{self._pub_name} is only assigned while "
            f"tracking; it has no value until the simulation has run.",
            name=self._pub_name,
            obj=instance,
        )


class ToBeDefined(_LateInit[_T]):
    """
    Attribute supplied from the input script.

    Must hold a value before the first turn, but is assigned from
    outside BLonD and is never cleared. Reading one unfilled means the
    input script left it out.
    """

    # Repeated from `_LateInit`: an override without them makes an
    # instance read type as ``T | Self`` instead of plain ``T``.
    @overload
    def __get__(self, instance: None, owner: type[Any]) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type[Any] | None) -> _T: ...

    def __get__(
        self, instance: object | None, owner: type[Any] | None = None
    ) -> _T | Self:
        """
        Read the attribute.

        Parameters
        ----------
        instance
            Object owning the attribute, or `None` for class access.
        owner
            Class owning the attribute.

        Returns
        -------
        value
            The attribute value, or the descriptor itself on class
            access.

        Raises
        ------
        NotInitialisedError
            If the attribute has not been filled yet.
        """
        if instance is None:
            return self
        # Only reached while the instance ``__dict__`` lacks the name.
        inst_class = type(instance).__name__
        raise NotInitialisedError(
            f"{inst_class}.{self._pub_name} has not been defined. "
            f"Use '{inst_class}.schedule(attribute={self._pub_name!r}, [..])' "
            "to assign it a value.",
            name=self._pub_name,
            obj=instance,
        )


def late_init_attributes(obj: object) -> tuple[str, ...]:
    """
    List the attributes an object declares as late-initialised.

    Parameters
    ----------
    obj
        Object or class to inspect.

    Returns
    -------
    tuple of str
        Attribute names, least-derived class first and in declaration
        order within each class, with no duplicates. Empty if nothing
        is declared.
    """
    owner = obj if isinstance(obj, type) else type(obj)
    return tuple(
        dict.fromkeys(
            name
            for base in reversed(owner.__mro__)
            for name in _declared.get(base, ())
        )
    )
