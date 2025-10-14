"""Base classes for `NoiseGenerator`.

Authors
-------
Simon Lauber
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


class NoiseGenerator(ABC):
    """Base class for noise generation."""

    def __init__(self):
        super().__init__()

    @abstractmethod  # pragma: no cover
    def get_noise(self, n_turns: int) -> NumpyArray:
        """Generate noise for n turns."""
        pass
