# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from blond.cycles.noise_generators.base import NoiseGenerator

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from numpy import ndarray

    NumpyArray = ndarray[Any]


class VariNoise(NoiseGenerator):
    def get_noise(self, n_turns: int) -> NumpyArray:
        warnings.warn("VariNoise needs to be implemented!")
        return np.ones(n_turns)
