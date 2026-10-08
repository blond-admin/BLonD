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
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, overload

from .exceptions_ import NotInitialisedError

_T = TypeVar("_T")

# Names declared on each class, in declaration order.  Weakly keyed so
# dynamically created classes are not kept alive by the registry.
_declared: weakref.WeakKeyDictionary[type, tuple[str, ...]] = (
    weakref.WeakKeyDictionary()
)


class SetBy(StrEnum):
    """
    Routes by which a late-initialised attribute gets its value.

    Used as the ``set_by`` of a `_LateInit` subclass. Each names the
    route a user is most likely to take, first, and any alternatives
    after it. ``{owner}`` is replaced by the class of the object being
    read and ``{attribute}`` by the attribute name, both when the error
    is raised. A route none of these describe takes a plain `str`.
    """

    SIMULATION = "`Simulation(...)`"
    RUN_SIMULATION = "`Simulation.run_simulation(...)`"
    SETUP_BEAM = "`Simulation.prepare_beam(...)` or `Beam.setup_beam(...)`"
    UPDATE_ATTRIBUTES = "`{owner}.update_attributes(...)`"
    ARGUMENT = "the `{attribute}` argument of `{owner}(...)`"
    ARGUMENT_OR_SCHEDULE = (
        "the `{attribute}` argument of `{owner}(...)`, assigning "
        "`{owner}.{attribute}`, or "
        "`{owner}.schedule(attribute={attribute!r}, ...)`"
    )


class _LateInit(Generic[_T], ABC):
    def __init__(self, set_by: SetBy | str) -> None:
        self._set_by = set_by
        # ``None`` in the instance ``__dict__`` shadows any class
        # docstring, so Sphinx does not repeat it on every attribute;
        # describe attributes in the owner's ``Attributes`` section.
        self.__doc__ = None

    def _filler(self, owner_name: str) -> str:
        """
        Render ``set_by`` for the object the attribute was read on.

        Parameters
        ----------
        owner_name
            Name of the class of the object being read.

        Returns
        -------
        str
            The text naming what fills the attribute.
        """
        return str(self._set_by).format(
            owner=owner_name, attribute=self._pub_name
        )

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
            f"{inst_class}.{self._pub_name} is not initialised yet; "
            f"it is filled by {self._filler(inst_class)}.",
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
            f"tracking; {self._filler(inst_class)} has not run yet.",
            name=self._pub_name,
            obj=instance,
        )


class ToBeDefined(_LateInit[_T]):
    """
    Attribute supplied from the input script.

    Must hold a value before the first turn, and is never cleared.
    Nothing fills it unless the input script asks for it, so reading
    one unfilled means the script left it out.
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
            f"Set it via {self._filler(inst_class)}.",
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


def unfilled(obj: object) -> tuple[str, ...]:
    """
    List the late-initialised attributes that are not filled yet.

    Parameters
    ----------
    obj
        Instance to inspect.

    Returns
    -------
    tuple of str
        Names missing from the instance ``__dict__``, in the order
        `late_init_attributes` gives them. Empty if all are filled.
    """
    return tuple(
        name for name in late_init_attributes(obj) if name not in vars(obj)
    )


def check_filled(obj: object, *names: str) -> None:
    """
    Check the named attributes, reporting every unfilled one at once.

    Parameters
    ----------
    obj
        Object owning the attributes.
    *names
        Names of the attributes to check, in the order to report them.

    Raises
    ------
    NotInitialisedError
        If any attribute is not filled yet.  One unfilled attribute
        raises the descriptor's own error; several are combined into a
        single error naming each attribute and what fills it.
    """
    errors = []
    for name in names:
        try:
            getattr(obj, name)
        except NotInitialisedError as exc:
            # Only an unfilled attribute is collected.  An
            # ``AttributeError`` from anywhere else is a bug, so it
            # propagates instead of being folded into the summary.
            errors.append(exc)

    match len(errors):
        case 0:
            return
        case 1:
            raise errors[0]
        case count:
            raise NotInitialisedError(
                f"{type(obj).__name__} has {count} attributes that are "
                "not initialised:\n"
                + "\n".join(f"  - {exc}" for exc in errors),
                obj=obj,
            )
