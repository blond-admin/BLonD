from __future__ import annotations

from typing import TYPE_CHECKING

from blond import WakeField
from blond.physics.profiles import ProfileBaseClass

if TYPE_CHECKING:  # pragma: no cover
    from blond import Simulation


class IntensityEffectManager:
    def __init__(self, simulation: Simulation) -> None:
        self._parent_simulation = simulation

    def has_wakefields(self):
        """
        Checks if there are any WakeFields in the Simulation

        Returns
        -------
        True if there is any WakeField
        """
        wakefields = self._parent_simulation.ring.elements.get_elements(
            WakeField
        )
        return len(wakefields) > 0

    def set_wakefields(self, active: bool) -> None:
        """
        Activate/deactivate `WakeField`

        Parameters
        ----------
        active
            True or False, so that simulation can skip the elements
        """
        wakefields = self._parent_simulation.ring.elements.get_elements(
            WakeField
        )
        for wakefield in wakefields:
            wakefield.active = active

    def set_profiles(self, active: bool) -> None:
        """
        Activate/deactivate `ProfileBaseClass`

        Parameters
        ----------
        active
            True or False, so that simulation can skip the elements

        """
        profiles = self._parent_simulation.ring.elements.get_elements(
            ProfileBaseClass
        )
        for profile in profiles:
            profile.active = active

        wakefields = self._parent_simulation.ring.elements.get_elements(
            WakeField
        )
        for wakefield in wakefields:
            wakefield.profile.active = active
