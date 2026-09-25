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
an ``assert``.  `LateInit` declares them as plain ``T`` instead, and
turns a read before they are filled into a `NotInitialisedError` that
names what fills them.

Only use it for attributes that are *always* filled eventually; an
attribute that may legitimately stay absent stays ``T | None``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

from .exceptions_ import BLonDException

_T = TypeVar("_T")

# ``set_by`` texts for attributes filled by the simulation lifecycle hooks
BY_INIT_SIMULATION = "`Simulation(...)` via `on_init_simulation()`"
BY_WAKEFIELD_INIT_SIMULATION = (
    "`Simulation(...)` via `on_wakefield_init_simulation()`"
)
BY_RUN_SIMULATION = (
    "`Simulation.run_simulation(...)` via `on_run_simulation()`"
)


class NotInitialisedError(BLonDException, AttributeError):
    """
    A late-initialised attribute was read before it was filled.

    Subclasses `AttributeError`, so ``hasattr`` reports `False` for an
    attribute that is not filled yet.
    """


class LateInit(Generic[_T]):
    """
    Descriptor for an attribute that is set after ``__init__``.

    Declared on the class as
    ``_dE: LateInit[NumpyArray] = LateInit("Beam.setup_beam()")``.
    Instance reads type as ``T`` (never ``T | None``), so code that runs
    after initialisation needs no narrowing ``assert``.  Reading it too
    early raises `NotInitialisedError` naming what fills it.  ``del``
    resets a filled attribute to unfilled; `reset_late_init` also
    accepts attributes that are not filled.

    Parameters
    ----------
    set_by
        What fills the attribute, for the error message, e.g.
        ``"on_init_simulation()"`` or ``"Beam.setup_beam()"``.
    doc
        Description of the attribute, used as its docstring.

    Notes
    -----
    At runtime this is a *non-data* descriptor: it defines only
    ``__get__``, so once the attribute is written the instance
    ``__dict__`` takes precedence and reads are ordinary attribute
    lookups, never a Python-level call.  The descriptor is only
    consulted while the attribute is unfilled.  ``__set__`` and
    ``__delete__`` exist for type checkers only.

    The value is stored in the instance ``__dict__`` under the
    attribute's own name, so ``vars()``, ``copy.deepcopy`` and pickling
    keep working.  This does not work on classes defining
    ``__slots__``.
    """

    def __init__(self, set_by: str, doc: str | None = None) -> None:
        self._set_by = set_by
        self.__doc__ = doc

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
        self._name = name

    # The overloads make instance reads plain ``T``; without them type
    # checkers see ``T | LateInit[T]``.
    @overload
    def __get__(self, instance: None, owner: type[Any]) -> LateInit[_T]: ...

    @overload
    def __get__(self, instance: object, owner: type[Any] | None) -> _T: ...

    def __get__(
        self, instance: object | None, owner: type[Any] | None = None
    ) -> _T | LateInit[_T]:
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
            f"{type(instance).__name__}.{self._name} is not "
            f"initialised yet; it is filled by {self._set_by}.",
            name=self._name,
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


def reset_late_init(instance: object, *names: str) -> None:
    """
    Reset late-initialised attributes to unfilled.

    Unlike ``del``, this also accepts attributes that are not filled,
    so it suits resetting state at the start of a run.

    Parameters
    ----------
    instance
        Object owning the attributes.
    *names
        Names of the `LateInit` attributes to reset.

    Raises
    ------
    TypeError
        If a name is not a `LateInit` attribute of the instance's class.
    """
    for name in names:
        if not isinstance(getattr(type(instance), name, None), LateInit):
            raise TypeError(
                f"{type(instance).__name__}.{name} is not a LateInit "
                "attribute."
            )
        instance.__dict__.pop(name, None)


def unfilled(instance: object) -> tuple[str, ...]:
    """
    List the late-initialised attributes that are not filled yet.

    Parameters
    ----------
    instance
        Object owning the attributes.

    Returns
    -------
    names
        Names of the unfilled `LateInit` attributes of the instance's
        class and its bases, base-class attributes last.
    """
    names: dict[str, None] = {}
    for cls in type(instance).__mro__:
        for name, attribute in vars(cls).items():
            if isinstance(attribute, LateInit) and name not in names:
                names[name] = None
    return tuple(name for name in names if name not in instance.__dict__)


def check_filled(instance: object) -> None:
    """
    Check that all late-initialised attributes are filled.

    Parameters
    ----------
    instance
        Object owning the attributes.

    Raises
    ------
    NotInitialisedError
        For the first `LateInit` attribute that is not filled yet,
        naming what fills it.
    """
    for name in unfilled(instance):
        getattr(instance, name)  # raises NotInitialisedError
