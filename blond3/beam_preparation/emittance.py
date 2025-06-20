from __future__ import annotations

from .base import MatchingRoutine
from .. import Simulation


class EmittanceMatcher(MatchingRoutine):
    def __init__(self, some_emittance: float):
        super().__init__()
        self.some_emittance = some_emittance

    def prepare_beam(self, simulation: Simulation) -> None:
        pass
