from __future__ import annotations

from .base import MatchingRoutine
from ..core.simulation.simulation import Simulation


class EmittanceMatcher(MatchingRoutine):
    def __init__(self, some_emittance: float):
        super().__init__()
        self.some_emittance = some_emittance

    def on_prepare_beam(self, simulation: Simulation) -> None:
        pass
