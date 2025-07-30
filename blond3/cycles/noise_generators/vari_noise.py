from __future__ import annotations

import warnings

import numpy as np

from .base import NoiseGenerator
from ..._core.backends.backend import backend


class VariNoise(NoiseGenerator):
    def get_noise(self, n_turns: int):
        warnings.warn("VariNoise needs to be implemented!")
        return np.ones(n_turns, dtype=backend.float)
