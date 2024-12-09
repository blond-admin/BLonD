# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Module to generate distributions**

:Authors: **Danilo Quartullo**, **Helga Timko**, **Alexandre Lasheen**,
          **Juan F. Esteban Mueller**, **Theodoros Argyropoulos**,
          **Joel Repond**
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .methods import _get_dE_from_dt, populate_bunch
from ....utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Optional

    from ....input_parameters.ring import Ring
    from ....input_parameters.rf_parameters import RFStation
    from ...beam import Beam


@handle_legacy_kwargs
def parabolic(ring: Ring, rf_station: RFStation, beam: Beam,
              bunch_length: float, bunch_position: Optional[float] = None,
              bunch_energy: Optional[float] = None,
              energy_spread: Optional[float] = None, seed: int = 1234):
    r"""Generate a bunch of particles distributed as a parabola in phase space.

    Parameters
    ----------
    ring : class
        A Ring type class
    rf_station : class
        An RFStation type class
    beam : class
        A Beam type class
    bunch_length : float
        The length in time [s] of the bunch
    bunch_position : float (optional)
        The position in time [s] of the center of mass of the bunch
    bunch_energy : float (optional)
        The position in energy [eV] of the center of mass of the bunch
        (relative to the synchronous energy)
    energy_spread : float (optional)
        The spread in energy [eV] of the bunch
    seed : int (optional)
        Fixed seed to have a reproducible distribution

    """

    # Getting the position and spread if not defined by user
    counter = rf_station.counter[0]
    omega_rf = rf_station.omega_rf[0, counter]
    phi_s = rf_station.phi_s[counter]
    phi_rf = rf_station.phi_rf[0, counter]
    eta0 = rf_station.eta_0[counter]

    # RF wave is shifted by Pi below transition
    if eta0 < 0:
        phi_rf -= np.pi

    if bunch_position is None:
        bunch_position = (phi_s - phi_rf) / omega_rf

    if bunch_energy is None:
        bunch_energy = 0

    if energy_spread is None:
        energy_spread = _get_dE_from_dt(ring, rf_station, bunch_length)

    # Generating time and energy arrays
    time_array = np.linspace(bunch_position - bunch_length / 2,
                             bunch_position + bunch_length / 2,
                             100)
    energy_array = np.linspace(bunch_energy - energy_spread / 2,
                               bunch_energy + energy_spread / 2,
                               100)

    # Getting Hamiltonian on a grid
    dt_grid, deltaE_grid = np.meshgrid(
        time_array, energy_array)

    # Bin sizes
    bin_dt = float(time_array[1] - time_array[0])
    bin_energy = float(energy_array[1] - energy_array[0])

    # Density grid
    isodensity_lines = (((dt_grid - bunch_position)
                         / bunch_length * 2) ** 2.
                        + ((deltaE_grid - bunch_energy)
                           / energy_spread * 2) ** 2.)
    density_grid = 1 - isodensity_lines ** 2.
    density_grid[density_grid < 0] = 0
    density_grid /= np.sum(density_grid)

    populate_bunch(beam, dt_grid, deltaE_grid, density_grid, bin_dt,
                   bin_energy, seed)
