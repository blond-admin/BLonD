from __future__ import annotations

from typing import TYPE_CHECKING

from .base import MatchingRoutine

if TYPE_CHECKING:  # pragma: no cover
    from blond3.core.ring.base import Ring


class BiGaussian(MatchingRoutine):
    def __init__(
        self,
        rms_dt: float,
        reinsertion: bool,
        seed: int,
    ):
        pass

    def prepare_beam(self, ring: Ring):
        pass
