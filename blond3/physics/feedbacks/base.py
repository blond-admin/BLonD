from __future__ import annotations

from typing import (
    List,
)

from ..cavities import MultiHarmonicCavity, SingleHarmonicCavity
from ..profiles import ProfileBaseClass
from ...core.base import BeamPhysicsRelevant
from ...core.beam.base import BeamBaseClass
from ...core.ring.helpers import requires
from ...core.simulation.base import Simulation


class Feedback(BeamPhysicsRelevant):
    def __init__(self, group: int = 0):
        super().__init__(group=group)


class LocalFeedback(Feedback):
    def __init__(
        self,
        profile: ProfileBaseClass,
        cavity: SingleHarmonicCavity | MultiHarmonicCavity,
        group: int = 0,
    ):
        super().__init__(group=group)
        self.cavity = cavity
        self.profile = profile

    def track(self, beam: BeamBaseClass):
        pass

    def late_init(self, simulation: Simulation, **kwargs):
        pass


RfFeedback = LocalFeedback  # just an alias name


class GlobalFeedback(Feedback):
    def __init__(self, profile: ProfileBaseClass, group: int = 0):
        super().__init__(group=group)
        self.profile = profile

    def track(self, beam: BeamBaseClass):
        pass

    # Use `requires` to automatically sort execution order of
    # `element.late_init` for all elements
    @requires([SingleHarmonicCavity])
    def late_init(self, simulation: Simulation, **kwargs):
        self.cavities = simulation.ring.elements.get_elements(SingleHarmonicCavity)


BeamFeedback = GlobalFeedback  # just an alias name


class GroupedFeedback(Feedback):
    def __init__(
        self,
        profile: ProfileBaseClass,
        cavities: List[SingleHarmonicCavity | MultiHarmonicCavity],
        group: int = 0,
    ):
        super().__init__(group=group)
        self.profile = profile
        self.cavities = cavities
