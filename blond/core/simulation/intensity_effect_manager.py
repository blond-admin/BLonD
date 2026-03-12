# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Holds the `IntensityEffectManager`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from blond.physics.impedances.base import WakeField
from blond.physics.profiles import ProfileBaseClass

if TYPE_CHECKING:  # pragma: no cover
    from blond import Simulation


class IntensityEffectManager:
    """
    Activate/deactivate Wakes and Profiles globally.

    Parameters
    ----------
    simulation
        `Simulation` context manager.
    """

    def __init__(self, simulation: Simulation) -> None:
        self._parent_simulation = simulation

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
