from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from .._core.base import Preparable

if TYPE_CHECKING:
    pass


class ProgrammedCycle(Preparable, ABC):
    """Programmed cycle of parameters"""

    def __init__(self):
        super().__init__()
