# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Holds the `IntensityEffectManager`."""

from __future__ import annotations

import weakref
from copy import deepcopy
from typing import TYPE_CHECKING

from blond.physics.impedances.base import WakeField
from blond.physics.profiles import ProfileBaseClass

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.simulation.simulation import Simulation


class IntensityEffectManager:
    """
    Activate/deactivate Wakes and Profiles globally.

    Parameters
    ----------
    simulation
        `Simulation` context manager.
    """

    def __init__(self, simulation: Simulation) -> None:
        # Weak, as the simulation owns this manager: a strong back-reference
        # forms a cycle that keeps the simulation alive until a full GC.
        self._parent_simulation_ref = weakref.ref(simulation)

    @property
    def _parent_simulation(self) -> Simulation:
        return self._parent_simulation_ref()

    def __deepcopy__(self, memo: dict) -> IntensityEffectManager:
        """
        Copy the manager so that it manages the copied simulation.

        `deepcopy` keeps a weakref as is, which would leave the copy
        managing the original simulation.

        Parameters
        ----------
        memo
            Objects already copied, as passed by `copy.deepcopy`.

        Returns
        -------
        intensity_effect_manager
            Manager of the copied simulation.
        """
        return IntensityEffectManager(
            simulation=deepcopy(self._parent_simulation, memo)
        )

    def has_wakefields(self):
        """
        Check if there are any `WakeField` instances in the `Simulation`.

        Returns
        -------
        has_wakefields
            True if there is any WakeField.
        """
        wakefields = self._parent_simulation.ring.elements.get_elements(
            WakeField, recursive=True
        )
        return len(wakefields) > 0

    def is_active_wakefields(self) -> bool:  # TODO testcae
        """
        Check whehther all `Wakefields` are active or inactive.

        Returns
        -------
        is_active
            True if wakefields are active, False otherwise.

        Raises
        ------
        AssertionError
            If there is a mixed state of `Wakefields` being active/inactive.
        """
        wakefields = self._parent_simulation.ring.elements.get_elements(
            WakeField, recursive=True
        )
        actives = {wakefield.active for wakefield in wakefields}
        if len(actives) == 0:
            return False
        assert len(actives) == 1, (
            "Cant handle mixed states of wakefields being active/inactive."
        )
        return actives.pop()

    def set_wakefields(self, active: bool) -> None:
        """
        Activate/deactivate `WakeField`.

        Parameters
        ----------
        active
            True or False, so that simulation can skip the elements.
        """
        wakefields = self._parent_simulation.ring.elements.get_elements(
            WakeField, recursive=True
        )
        for wakefield in wakefields:
            wakefield.active = active

    def set_profiles(self, active: bool) -> None:
        """
        Activate/deactivate `ProfileBaseClass`.

        Parameters
        ----------
        active
            True or False, so that simulation can skip the elements.
        """
        profiles = self._parent_simulation.ring.elements.get_elements(
            ProfileBaseClass, recursive=True
        )
        for profile in profiles:
            profile.active = active
        # Deactivate the `Profiles` of the WakeFields.
        # The frozen wakefields can still affect the beam.

    def is_active_profiles(self) -> bool:  # TODO testcae
        """
        Check whether all `Profiles` are active or inactive.

        Returns
        -------
        is_active
            True if profiles are active, False otherwise.

        Raises
        ------
        AssertionError
            If there is a mixed state of `Wakefields` being active/inactive.
        """
        profiles = self._parent_simulation.ring.elements.get_elements(
            ProfileBaseClass, recursive=True
        )
        actives = {profile.active for profile in profiles}

        if len(actives) == 0:
            return False
        assert len(actives) == 1, (
            "Cant handle mixed states of Profiles being active/inactive."
        )
        return actives.pop()

    def freeze_wakefields(self) -> None:
        """Freeze the wakefields, so that they applied the kick to what has happened before."""
        # Deactivate all updates of wake fields.
        # They need to be previously calculated by a successful run with an
        # actual line density, as the probe beam has no line density.
        wakefields = self._parent_simulation.ring.elements.get_elements(
            WakeField, recursive=True
        )
        for wakefield in wakefields:
            wakefield.update_induced_voltage = False
            wakefield.profile.active = False

    def unfreeze_wakefields(self) -> None:
        """Unfreeze the wakefields, so that all intensity effects are active."""
        # Deactivate all updates of wake fields.
        # They need to be previously calculated by a successful run with an
        # actual line density, as the probe beam has no line density.
        wakefields = self._parent_simulation.ring.elements.get_elements(
            WakeField, recursive=True
        )
        for wakefield in wakefields:
            wakefield.update_induced_voltage = True
            wakefield.profile.active = True
