from __future__ import annotations

from abc import abstractmethod, ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


class NoiseGenerator(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_noise(self, n_turns: int) -> NumpyArray:
        pass
