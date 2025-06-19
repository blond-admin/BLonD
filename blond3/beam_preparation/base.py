from __future__ import annotations

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:  # pragma: no cover
    from blond3.core.ring.ring import Ring


class BeamPreparationRoutine(ABC):
    @abstractmethod
    def prepare_beam(
        self,
        ring: Ring,
    ):
        pass


class MatchingRoutine(BeamPreparationRoutine):
    pass
