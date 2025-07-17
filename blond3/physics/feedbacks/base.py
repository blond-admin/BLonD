from __future__ import annotations

from typing import (
    List,
    TYPE_CHECKING,
)

from ..profiles import ProfileBaseClass
from ..._core.base import BeamPhysicsRelevant
from ..._core.beam.base import BeamBaseClass
from ..._core.ring.helpers import requires
from ..._core.simulation.simulation import Simulation

if TYPE_CHECKING:  # pragma: no cover
    from ..cavities import (
        MultiHarmonicCavity,
        SingleHarmonicCavity,
        CavityBaseClass,
        Type,
    )


class FeedbackBaseClass(BeamPhysicsRelevant):
    def __init__(self, section_index: int = 0):
        super().__init__(section_index=section_index)


class LocalFeedback(FeedbackBaseClass):
    def __init__(
        self,
        profile: ProfileBaseClass,
        section_index: int = 0,
    ):
        super().__init__(section_index=section_index)
        self._owner: SingleHarmonicCavity | MultiHarmonicCavity | None = None
        self.profile = profile

    def set_owner(self, cavity: Type[CavityBaseClass]):
        assert self._owner is None
        self._owner = cavity

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        pass


RfFeedback = LocalFeedback  # just an alias name


class GlobalFeedback(FeedbackBaseClass):
    def __init__(self, profile: ProfileBaseClass, section_index: int = 0):
        super().__init__(section_index=section_index)
        self.profile = profile

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        pass

    # Use `requires` to automatically sort execution order of
    # `element.on_init_simulation` for all elements
    @requires(["SingleHarmonicCavity"])
    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
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
