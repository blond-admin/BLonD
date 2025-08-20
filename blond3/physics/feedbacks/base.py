from __future__ import annotations

from abc import abstractmethod
from typing import (
    List,
    TYPE_CHECKING,
)

from ..._core.base import BeamPhysicsRelevant
from ..._core.ring.helpers import requires

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional, Optional as LateInit

    from ..cavities import (
        MultiHarmonicCavity,
        SingleHarmonicCavity,
        CavityBaseClass,
    )
    from ..profiles import ProfileBaseClass
    from ..._core.beam.base import BeamBaseClass

    from ..._core.simulation.simulation import Simulation


class FeedbackBaseClass(BeamPhysicsRelevant):
    def __init__(
        self,
        section_index: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__(section_index=section_index, name=name)


class LocalFeedback(FeedbackBaseClass):
    def __init__(
        self,
        profile: ProfileBaseClass,
        section_index: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self._parent_cavity: SingleHarmonicCavity | MultiHarmonicCavity | None = None
        self.profile = profile

    def set_parent_cavity(self, cavity: CavityBaseClass):
        assert self._parent_cavity is None, "This feedback has already one owner!"
        self._parent_cavity = cavity

    @abstractmethod  # pragma: no cover
    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        pass


RfFeedback = LocalFeedback  # just an alias name


class GlobalFeedback(FeedbackBaseClass):
    def __init__(
        self,
        profile: ProfileBaseClass,
        section_index: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self.profile = profile
        self.cavities: LateInit[List[CavityBaseClass]] = None

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
        name: Optional[str] = None,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self.profile = profile
        self.cavities = cavities
