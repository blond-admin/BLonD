from __future__ import annotations

from typing import (
    List,
)

from ..cavities import MultiHarmonicCavity, SingleHarmonicCavity
from ..profiles import ProfileBaseClass
from ..._core.base import BeamPhysicsRelevant
from ..._core.beam.base import BeamBaseClass
from ..._core.ring.helpers import requires
from ..._core.simulation.simulation import Simulation


class FeedbackBaseClass(BeamPhysicsRelevant):
    def __init__(self, section_index: int = 0):
        super().__init__(section_index=section_index)


class LocalFeedback(FeedbackBaseClass):
    def __init__(
        self,
        profile: ProfileBaseClass,
        cavity: SingleHarmonicCavity | MultiHarmonicCavity,
        section_index: int = 0,
    ):
        super().__init__(section_index=section_index)
        self.cavity = cavity
        self.profile = profile

    def track(self, beam: BeamBaseClass):
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass


RfFeedback = LocalFeedback  # just an alias name


class GlobalFeedback(FeedbackBaseClass):
    def __init__(self, profile: ProfileBaseClass, section_index: int = 0):
        super().__init__(section_index=section_index)
        self.profile = profile

    def track(self, beam: BeamBaseClass):
        pass

    # Use `requires` to automatically sort execution order of
    # `element.on_init_simulation` for all elements
    @requires(["SingleHarmonicCavity"])
    def on_init_simulation(self, simulation: Simulation) -> None:
        self.cavities = simulation.ring.elements.get_elements(SingleHarmonicCavity)


BeamFeedback = GlobalFeedback  # just an alias name


class GroupedFeedback(FeedbackBaseClass):
    def __init__(
        self,
        profile: ProfileBaseClass,
        cavities: List[SingleHarmonicCavity | MultiHarmonicCavity],
        section_index: int = 0,
    ):
        super().__init__(section_index=section_index)
        self.profile = profile
        self.cavities = cavities
