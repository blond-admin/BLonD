from __future__ import annotations

import warnings

import numpy as np

from ..._core.backends.backend import backend
from .base import NoiseGenerator


class VariNoise(NoiseGenerator):
    def get_noise(self, n_turns: int):
        warnings.warn("VariNoise needs to be implemented!")
        return np.ones(n_turns, dtype=backend.float)
