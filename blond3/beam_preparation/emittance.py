from __future__ import annotations

from .base import MatchingRoutine


class EmittanceMatcher(MatchingRoutine):
    def __init__(self, some_emittance: float):
        self.some_emittance = some_emittance
