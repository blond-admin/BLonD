from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from blond._core.base import BeamPhysicsRelevant
from blond._core.ring.helpers import requires

if TYPE_CHECKING:  # pragma: no cover
    from blond._core.beam.base import BeamBaseClass
    from blond._core.simulation.simulation import Simulation
    from blond.physics.cavities import (
        CavityBaseClass,
        MultiHarmonicCavity,
        SingleHarmonicCavity,
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
        self._parent_cavity: (
            SingleHarmonicCavity | MultiHarmonicCavity | None
        ) = None
        self.profile = profile

    def set_parent_cavity(self, cavity: CavityBaseClass):
        assert self._parent_cavity is None, (
            "This feedback has already one owner!"
        )
        self._parent_cavity = cavity

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


RfFeedback = LocalFeedback  # just an alias name


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
        self.cavities: list[CavityBaseClass] | None = None

    # Use `requires` to automatically sort execution order of
    # `element.on_init_simulation` for all elements
    @requires(["SingleHarmonicCavity"])
    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        self.cavities = simulation.ring.elements.get_elements(
            SingleHarmonicCavity
        )


BeamFeedback = GlobalFeedback  # just an alias name


class GroupedFeedback(FeedbackBaseClass):
    def __init__(
        self,
        profile: ProfileBaseClass,
        cavities: list[SingleHarmonicCavity | MultiHarmonicCavity],
        section_index: int = 0,
        name: str | None = None,
    ):
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self.profile = profile
        self.cavities = cavities
