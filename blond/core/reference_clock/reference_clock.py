# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helper class that holds the reference to the coordinate system."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import speed_of_light as c0

from blond.core.base import HasPropertyCache

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.particle_types import ParticleType


class ReferenceCoordinates(HasPropertyCache):
    """Helper class that holds the reference to the coordinate system."""

    def __init__(self, time, total_energy, particle_type):
        self.time = time
        self._total_energy = total_energy
        self._particle_type = particle_type

    @property
    def particle_type(self) -> ParticleType:
        """
        Particle type that the reference coordinates belong to.

        Returns
        -------
        particle_type
            Type of particles, e.g. protons.
        """
        return self._particle_type

    @property
    def total_energy(self) -> float:
        """
        Total beam energy [eV].

        Returns
        -------
        total_energy
            Total beam energy [eV].
        """
        if self._total_energy is None:
            raise ValueError(
                "Beam is not properly set up, please set `total_energy` first!"
            )
        return self._total_energy

    @total_energy.setter
    def total_energy(self, total_energy: float) -> None:
        """
        Total beam energy [eV].

        Parameters
        ----------
        total_energy
            Total beam energy [eV].
        """
        self.invalidate_cache_reference()
        self._total_energy = total_energy

    @cached_property
    def gamma(self) -> float:
        """
        Beam reference gamma a.k.a. Lorentz factor [].

        Returns
        -------
        gamma
            Beam reference gamma a.k.a. Lorentz factor [].
        """
        # total_energy in eV and mass_inv in [c²/eV]
        if self._total_energy is None:
            raise ValueError(
                "Beam is not properly set up, please set `total_energy` first!"
            )
        val = self._total_energy * self._particle_type.mass_inv
        return val

    @cached_property
    def beta(self) -> float:
        """
        Beam reference fraction of speed of light (v/c0) [].

        Returns
        -------
        beta
            Beam reference fraction of speed of light (v/c0) [].
        """
        gamma = self.gamma
        val = np.sqrt(1.0 - 1.0 / (gamma * gamma))
        return val

    @cached_property
    def velocity(self) -> float:
        """
        Beam reference speed [m/s].

        Returns
        -------
        velocity
            Beam reference speed [m/s].
        """
        return self.beta * c0

    def invalidate_cache_reference(self) -> None:
        """Reset cache of `cached_property` attributes."""
        super()._invalidate_cache(
            (
                "gamma",
                "beta",
                "velocity",
            )
        )
