# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helper class that holds the reference to the beam coordinate system."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.core.reference_clock.reference_clock_numba import beta as beta_nb
from blond.core.reference_clock.reference_clock_numba import gamma as gamma_nb
from blond.core.reference_clock.reference_clock_numba import (
    velocity as velocity_nb,
)

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.particle_types import ParticleType


class ReferenceCoordinates:
    """
    Helper class that holds the reference to the coordinate system.

    Parameters
    ----------
    time
        Reference time, in [s].
    total_energy
        Reference total energy, in [eV].
    particle_type
        The type of particle in the beam (e.g., protons, electrons).
        This determines properties like mass and charge.
    """

    def __init__(
        self,
        time: float,
        total_energy: float | None,
        particle_type: ParticleType,
    ):
        self.time = time  # no property syntax
        self._total_energy = total_energy  # only read access
        self._particle_type = particle_type  # only read access

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
        self._total_energy = total_energy

    @property
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
        return gamma_nb(self._total_energy, self._particle_type.mass_inv)

    @property
    def beta(self) -> float:
        """
        Beam reference fraction of speed of light (v/c0) [].

        Returns
        -------
        beta
            Beam reference fraction of speed of light (v/c0) [].
        """
        beta = beta_nb(self._total_energy, self._particle_type.mass_inv)
        assert not np.isnan(beta), f"{beta=}"
        return beta

    @property
    def velocity(self) -> float:
        """
        Beam reference speed [m/s].

        Returns
        -------
        velocity
            Beam reference speed [m/s].
        """
        return velocity_nb(self._total_energy, self._particle_type.mass_inv)
