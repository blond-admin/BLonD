# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Weak back-references for objects pointing back to their owner.

Several objects keep a reference to the object owning them, which forms a
reference cycle. Reference counting cannot free a cycle, so its objects
(and the arrays they hold) stay in memory until a full cyclic garbage
collection, which loops creating simulations rarely trigger. Declaring
these back-references with `WeakBackReference` breaks the cycles.

The offenders, i.e. every back-reference declared with these helpers:

- ``Preparable._simulation``: the `Simulation` owns its elements.
- ``WakeFieldSolver._simulation`` and ``WakeFieldSolver._parent_wakefield``:
  the `WakeField` owns its solver.
- ``RFStationBaseClass._ring``: the `Ring` owns its RF stations.
- ``RFStationBaseClass._beam_feedback``: the `Ring` owns the beam
  feedback, which refers back to its RF stations.
- ``IntensityEffectManager._parent_simulation``: the `Simulation` owns its
  manager.
"""

# FIXME: ideally these cycles would not exist at all. A child should not
#  store its owner, but receive what it needs (the ring, the simulation
#  context, ...) as arguments of the calls that use it. These helpers only
#  patch the symptom of that architecture.

from __future__ import annotations

import weakref
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

# Instance attribute naming the back-references the instance owns
_STRONGLY_HELD = "_strongly_held_back_references"


def hold_strongly(instance: Any, *names: str) -> None:
    """
    Make back-references of one instance hold their referents strongly.

    For stand-ins that nothing else owns, e.g. those created by the
    ``headless`` constructors, which run without a `Simulation`.

    Parameters
    ----------
    instance
        Object owning the referents.
    *names
        Names of its `WeakBackReference` attributes to hold strongly.
    """
    instance.__dict__.setdefault(_STRONGLY_HELD, set()).update(names)
    for name in names:  # convert references assigned before
        setattr(instance, name, getattr(instance, name))


class _Reference:
    """
    Weak or strong reference; `deepcopy` points it to the copy.

    Parameters
    ----------
    referent
        Object to refer to.
    strong
        Whether to hold `referent` strongly.
    """

    __slots__ = ("_referent", "_strong")

    def __init__(self, referent: Any, strong: bool) -> None:
        self._strong = strong
        self._referent = referent if strong else weakref.ref(referent)

    def _get(self) -> Any:
        return self._referent if self._strong else self._referent()

    def __deepcopy__(self, memo: dict) -> _Reference:
        """
        Refer to the copy of the referent within the same `deepcopy`.

        Parameters
        ----------
        memo
            Objects already copied, as passed by `deepcopy`.

        Returns
        -------
        reference
            Reference to the copied referent.
        """
        referent = self._get()
        if referent is None:
            return self
        return _Reference(deepcopy(referent, memo), self._strong)


class WeakBackReference:
    """
    Attribute holding a weak reference to an object owning its instance.

    Assigning and reading it works as for a plain attribute. Reads give
    None if unset and raise `ReferenceError` once the referent was freed,
    e.g. when an element is still used after its `Simulation` was dropped.
    Under `deepcopy`, the copy refers to the copy of the referent.

    Examples
    --------
    >>> class Element:
    ...     _simulation = WeakBackReference()
    """

    def __set_name__(self, owner: type, name: str) -> None:
        """
        Store the reference under the attribute name.

        Parameters
        ----------
        owner
            Class declaring the attribute.
        name
            Name of the attribute.
        """
        self._name = name

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        """
        Return the referent.

        Parameters
        ----------
        instance
            Instance the attribute is read from, None if read from the class.
        owner
            Class of `instance`.

        Returns
        -------
        referent
            The referent, None if unset, or this descriptor when read from
            the class.

        Raises
        ------
        ReferenceError
            If the referent was freed.
        """
        if instance is None:
            return self
        reference = instance.__dict__.get(self._name)
        if reference is None:
            return None
        referent = reference._get()
        if referent is None:
            raise ReferenceError(
                f"{type(instance).__name__}.{self._name} refers to an "
                "object that was freed. It is held weakly, as it owns this "
                "object (e.g. the Simulation owning its elements). Keep a "
                "reference to it while using this object."
            )
        return referent

    def __set__(self, instance: Any, value: Any) -> None:
        """
        Store a reference to `value`, weak unless `hold_strongly` was used.

        Parameters
        ----------
        instance
            Instance the attribute is set on.
        value
            Object to refer to, or None.
        """
        strong = self._name in instance.__dict__.get(_STRONGLY_HELD, ())
        instance.__dict__[self._name] = (
            None if value is None else _Reference(value, strong)
        )
