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


from .exceptions_ import NotInitialisedError

from typing import TYPE_CHECKING,TypeVar,  Generic, overload, Any, Self

_T = TypeVar("_T")

class _LateInit(Generic[_T]):

    def __init__(self, set_by: str) -> None:
        self._set_by = set_by
        # ``None`` in the instance ``__dict__`` shadows any class
        # docstring, so Sphinx does not repeat it on every attribute;
        # describe attributes in the owner's ``Attributes`` section.
        self.__doc__ = None

    def __set_name__(self, owner: type[Any], name: str) -> None:
        """
        Remember the attribute name the descriptor is assigned to.

        Parameters
        ----------
        owner
            Class owning the attribute.
        name
            Name of the attribute.
        """
        self._pub_name = name

    # The overloads make instance reads plain ``T``; without them type
    # checkers see ``T | _LateInit[T]``.
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
        raise NotInitialisedError(
            f"{type(instance).__name__}.{self._pub_name} is not "
            f"initialised yet; it is filled by {self._set_by}.",
            name=self._pub_name,
            obj=instance,
        )

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
