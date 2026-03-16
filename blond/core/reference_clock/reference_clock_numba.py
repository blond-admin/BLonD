# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Numba-compiled math functions used for improved performance."""

from __future__ import annotations

import numba as nb
import numpy as np
from scipy.constants import speed_of_light as c0


@nb.njit(
    nb.float64(nb.float64, nb.float64),
    fastmath=True,
    cache=True,
    inline="always",
)  # pragma: no cover
def gamma(total_energy: float, mass_inv: float) -> float:
    """
    Beam reference gamma a.k.a. Lorentz factor [].

    Parameters
    ----------
    total_energy
        In [eV].
    mass_inv
        Inverse mass, in [c²/eV].

    Returns
    -------
    gamma
        Beam reference gamma a.k.a. Lorentz factor [].
    """
    # total_energy in eV and mass_inv in [c²/eV]
    return total_energy * mass_inv


@nb.njit(
    nb.float64(nb.float64, nb.float64),
    fastmath=True,
    cache=True,
    inline="always",
)  # pragma: no cover
def beta(total_energy: float, mass_inv: float) -> float:
    """
    Beam reference fraction of speed of light (v/c0) [].

    Parameters
    ----------
    total_energy
        In [eV].
    mass_inv
        Inverse mass, in [c²/eV].

    Returns
    -------
    beta
         Beam reference fraction of speed of light (v/c0) [].
    """
    gamma_ = gamma(total_energy, mass_inv)
    val = np.sqrt(1.0 - 1.0 / (gamma_ * gamma_))
    return val


@nb.njit(
    nb.float64(nb.float64, nb.float64),
    fastmath=True,
    cache=True,
    inline="always",
)  # pragma: no cover
def velocity(total_energy: float, mass_inv: float) -> float:
    """
    Beam reference speed [m/s].

    Parameters
    ----------
    total_energy
        In [eV].
    mass_inv
        Inverse mass, in [c²/eV].

    Returns
    -------
    velocity
        Beam reference speed [m/s].
    """
    return beta(total_energy, mass_inv) * c0
