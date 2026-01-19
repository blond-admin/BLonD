# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from blond.core.base import BeamPhysicsRelevant
from blond.core.ring.helpers import requires

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation
    from blond.physics.cavities import (
        MultiHarmonicRFStation,
        RFStationBaseClass,
        SingleHarmonicRFStation,
    )
    from blond.physics.profiles import ProfileBaseClass


class FeedbackBaseClass(BeamPhysicsRelevant):
    def __init__(
        self,
        section_index: int = 0,
        name: str | None = None,
    ):
        super().__init__(section_index=section_index, name=name)


class LocalFeedback(FeedbackBaseClass):
    def __init__(
        self,
        profile: ProfileBaseClass,
        section_index: int = 0,
        name: str | None = None,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self._parent_rf_station: (
            SingleHarmonicRFStation | MultiHarmonicRFStation | None
        ) = None
        self.profile = profile

    def set_parent_rf_station(self, rf_station: RFStationBaseClass):
        assert self._parent_rf_station is None, (
            "This feedback has already one owner!"
        )
        self._parent_rf_station = rf_station

    @abstractmethod  # pragma: no cover
    def track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        pass


RFFeedback = LocalFeedback  # just an alias name


class GlobalFeedback(FeedbackBaseClass):
    def __init__(
        self,
        profile: ProfileBaseClass,
        section_index: int = 0,
        name: str | None = None,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self.profile = profile
        self.cavities: list[RFStationBaseClass] | None = None

    # Use `requires` to automatically sort execution order of
    # `element.on_init_simulation` for all elements
    @requires(["SingleHarmonicRFStation"])
    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called

        simulation
            `Simulation` context manager
        """
        self.cavities = simulation.ring.elements.get_elements(
            SingleHarmonicRFStation
        )


BeamFeedback = GlobalFeedback  # just an alias name


class GroupedFeedback(FeedbackBaseClass):
    def __init__(
        self,
        profile: ProfileBaseClass,
        cavities: list[SingleHarmonicRFStation | MultiHarmonicRFStation],
        section_index: int = 0,
        name: str | None = None,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self.profile = profile
        self.cavities = cavities
