from __future__ import annotations

from typing import TYPE_CHECKING

from blond3 import WakeField
from blond3.physics.profiles import ProfileBaseClass

if TYPE_CHECKING:  # pragma: no cover
    from blond3 import Simulation


class IntensityEffectManager:
    def __init__(self, simulation: Simulation) -> None:
        self._parent_simulation = simulation

    def set_wakefields(self, active: bool) -> None:
        """
        Activate/deactivate `WakeField`

        Parameters
        ----------
        active
            True or False, so that simulation can skip the elements
        """
        wakefields = self._parent_simulation.ring.elements.get_elements(WakeField)
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
        profiles = self._parent_simulation.ring.elements.get_elements(ProfileBaseClass)
        for profile in profiles:
            profile.active = active
