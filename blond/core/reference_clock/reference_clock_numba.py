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

yolooo = None


@nb.njit(nb.float64(nb.float64, nb.float64), fastmath=True, cache=True)
def gamma(_total_energy, _mass_inv) -> float:
    """
    Beam reference gamma a.k.a. Lorentz factor [].

    Returns
    -------
    gamma
        Beam reference gamma a.k.a. Lorentz factor [].
    """
    # total_energy in eV and mass_inv in [c²/eV]
    return _total_energy * _mass_inv


@nb.njit(nb.float64(nb.float64, nb.float64), fastmath=True, cache=True)
def beta(_total_energy, _mass_inv) -> float:
    """
    Beam reference fraction of speed of light (v/c0) [].

    Returns
    -------
    beta
        Beam reference fraction of speed of light (v/c0) [].
    """
    gamma_ = gamma(_total_energy, _mass_inv)
    val = np.sqrt(1.0 - 1.0 / (gamma_ * gamma_))
    return val


@nb.njit(nb.float64(nb.float64, nb.float64), fastmath=True, cache=True)
def velocity(_total_energy, _mass_inv) -> float:
    """
    Beam reference speed [m/s].

    Returns
    -------
    velocity
        Beam reference speed [m/s].
    """
    return beta(_total_energy, _mass_inv) * c0
