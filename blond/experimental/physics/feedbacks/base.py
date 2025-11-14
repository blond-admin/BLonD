from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from blond._core.base import BeamPhysicsRelevant
from blond._core.ring.helpers import requires

if TYPE_CHECKING:  # pragma: no cover
    from blond._core.beam.base import BeamBaseClass
    from blond._core.simulation.simulation import Simulation
    from blond.physics.cavities import (
        MultiHarmonicRfStation,
        RfStationBaseClass,
        SingleHarmonicRfStation,
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
    """What should this do?  # TODO: docm/clarification

    Currently we assume that this acts on the cavity voltage and
    is not concerned about how the beam is reacting to it.
    """
    def __init__(
        self,
        section_index: int = 0,
        name: str | None = None,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self._parent_cavity: (
                SingleHarmonicRfStation | MultiHarmonicRfStation | None
        ) = None

    def set_parent_cavity(self, cavity: RfStationBaseClass):
        assert self._parent_cavity is None, (
            "This feedback has already one owner!"
        )
        self._parent_cavity = cavity


RfFeedback = LocalFeedback  # just an alias name


class GlobalFeedback(FeedbackBaseClass):
    """What should this do?  # TODO: docm/clarification

    Currently we assume that this acts on the whole beam through the cavity.

    --> Induced voltage acts on last bunches different than on first --> adjust frequency of cavity
    """
    def __init__(
        self,
        section_index: int = 0,
        name: str | None = None,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self.cavities: list[RfStationBaseClass] | None = None

    # Use `requires` to automatically sort execution order of
    # `element.on_init_simulation` for all elements
    @requires(["SingleHarmonicRfStation", "MultiHarmonicRfStation"])  # check if this can work with RFStationBaseClass
    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        self.cavities = simulation.ring.elements.get_elements(
            SingleHarmonicRfStation
        )

BeamFeedback = GlobalFeedback  # just an alias name


class GroupedFeedback(FeedbackBaseClass):
    def __init__(
        self,
        profile: ProfileBaseClass,
        cavities: list[SingleHarmonicRfStation | MultiHarmonicRfStation],
        section_index: int = 0,
        name: str | None = None,
    ):
        raise NotImplementedError("Not used at the moment, needs to be implemented and refined")
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self.profile = profile
        self.cavities = cavities
