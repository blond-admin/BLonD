from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from typing import List

    from ..handle_results.observables import Observables


class SimulationResults(ABC):
    def __init__(self):
        super().__init__()
        self.observables: List[Observables] = []
