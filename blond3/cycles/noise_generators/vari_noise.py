from __future__ import annotations

from .base import NoiseGenerator


class VariNoise(NoiseGenerator):
    def get_noise(self, n_turns: int):
        raise NotImplementedError()  # TODO
        pass
