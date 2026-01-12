# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Base classes for :class:`~blond.cycles.noise_generators.base.NoiseGenerator`."""

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
        """
        Generate noise for n turns.

        Parameters
        ----------
        n_turns
            Number of turns to generate noise for.

        Returns
        -------
        noise
            Generated noise array.
        """
        pass
