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

import warnings
from typing import TYPE_CHECKING

import numpy as np

from ....utils import bmath as bm
from ....utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Optional
    from numpy.typing import NDArray

    from ....input_parameters.ring import Ring
    from ....input_parameters.rf_parameters import RFStation
    from ...beam import Beam


@handle_legacy_kwargs
def populate_bunch(beam: Beam, time_grid: NDArray, deltaE_grid: NDArray,
                   density_grid: NDArray, time_step: float,
                   deltaE_step: float, seed: Optional[int]) -> None:
    """Populates bunch from the particle density in phase space.

    Method to populate the bunch using a random number generator from the
    particle density in phase space.

    Parameters
    ----------
    beam
        Class containing the beam properties.
    time_grid
        Used to fill beam.dt using probability from density_grid
    deltaE_grid
        Used to fill beam.dE using probability from density_grid
    density_grid
        Probability distribution of `time_grid` and `deltaE_grid`
    time_step
        beam.dt is randomly displaced by  +- 0.5 * time_step
    deltaE_step
        beam.dE is randomly displaced by  +- 0.5 * deltaE_step
    seed
        Random seed
    """
    # Initialise the random number generator
    np.random.seed(seed=seed)
    # Generating particles randomly inside the grid cells according to the
    # provided density_grid
    indexes = np.random.choice(np.arange(0, np.size(density_grid)),
                               beam.n_macroparticles, p=density_grid.flatten())

    # Randomize particles inside each grid cell (uniform distribution)
    beam.dt = (np.ascontiguousarray(time_grid.flatten()[indexes] +
                                    (np.random.rand(beam.n_macroparticles)
                                     - 0.5) * time_step)).astype(
        dtype=bm.precision.real_t, order='C', copy=False)
    beam.dE = (np.ascontiguousarray(deltaE_grid.flatten()[indexes] +
                                    (np.random.rand(beam.n_macroparticles)
                                     - 0.5) * deltaE_step)).astype(
        dtype=bm.precision.real_t, order='C', copy=False)


@handle_legacy_kwargs
def _get_dE_from_dt(ring: Ring, rf_station: RFStation, dt_amplitude: float) \
        -> float:
    r"""A routine to evaluate the dE amplitude from dt following a single
    RF Hamiltonian.

    Parameters
    ----------
    ring : class
        A Ring type class
    rf_station : class
        An RFStation type class
    dt_amplitude : float
        Full amplitude of the particle oscillation in [s]

    Returns
    -------
    dE_amplitude : float
        Full amplitude of the particle oscillation in [eV]

    """

    warnings.filterwarnings("once")
    if ring.n_sections > 1:
        warnings.warn("WARNING in bigaussian(): the usage of several" +
                      " sections is not yet implemented. Ignoring" +
                      " all but the first!")
    if rf_station.n_rf > 1:
        warnings.warn("WARNING in bigaussian(): the usage of multiple RF" +
                      " systems is not yet implemented. Ignoring" +
                      " higher harmonics!")

    counter = rf_station.counter[0]

    harmonic = rf_station.harmonic[0, counter]
    energy = rf_station.energy[counter]
    beta = rf_station.beta[counter]
    omega_rf = rf_station.omega_rf[0, counter]
    phi_s = rf_station.phi_s[counter]
    phi_rf = rf_station.phi_rf[0, counter]
    eta0 = rf_station.eta_0[counter]

    # RF wave is shifted by Pi below transition
    if eta0 < 0:
        phi_rf -= np.pi

    # Calculate dE_amplitude from dt_amplitude using single-harmonic Hamiltonian
    voltage = rf_station.charge * \
              rf_station.voltage[0, counter]
    eta0 = rf_station.eta_0[counter]

    phi_b = omega_rf * dt_amplitude + phi_s
    dE_amplitude = np.sqrt(voltage * energy * beta ** 2
                           * (np.cos(phi_b) - np.cos(phi_s) + (phi_b - phi_s)
                              * np.sin(phi_s))
                           / (np.pi * harmonic * np.fabs(eta0)))

    return dE_amplitude
