from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from ..._core.backends.backend import backend
from .base import NoiseGenerator

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


class VariNoise(NoiseGenerator):
    def get_noise(self, n_turns: int) -> NumpyArray:
        warnings.warn("VariNoise needs to be implemented!")
        return np.ones(n_turns, dtype=backend.float)
