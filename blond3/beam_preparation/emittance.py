from __future__ import annotations

from typing import TYPE_CHECKING

from .base import MatchingRoutine
from .._core.beam.base import BeamBaseClass
from .._core.simulation.simulation import Simulation

if TYPE_CHECKING:  # pragma: no cover
    pass


class EmittanceMatcher(MatchingRoutine):
    def __init__(self, some_emittance: float):
        raise NotImplementedError("To be developed")  # TODO
        super().__init__()
        self.some_emittance = some_emittance

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """Populates the `Beam` object with macro-particles

        Parameters
        ----------
        simulation
            Simulation context manager
        """
        super().prepare_beam(
            simulation=simulation,
            beam=beam,
        )
