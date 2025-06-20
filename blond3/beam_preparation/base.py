from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..core.simulation.simulation import Simulation


class BeamPreparationRoutine(ABC):
    @abstractmethod
    def on_prepare_beam(
        self,
        simulation: Simulation,
    ) -> None:
        pass


class MatchingRoutine(BeamPreparationRoutine, ABC):
    pass
