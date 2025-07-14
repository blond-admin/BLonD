from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .._core.simulation.simulation import Simulation


class BeamPreparationRoutine(ABC):
    """Base class to write beam preparation routines"""

    @abstractmethod
    def prepare_beam(
        self,
        simulation: Simulation,
    ) -> None:
        """Populates the `Beam` object with macro-particles

        Parameters
        ----------
        simulation
            Simulation context manager
        """
        pass


class MatchingRoutine(BeamPreparationRoutine, ABC):
    pass
