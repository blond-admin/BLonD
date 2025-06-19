from __future__ import annotations

from abc import abstractmethod, ABC


class NoiseGenerator(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_noise(self, n_turns: int):
        pass

