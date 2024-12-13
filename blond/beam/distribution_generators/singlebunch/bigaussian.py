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

from .methods import _get_dE_from_dt
from ....trackers.utilities import is_in_separatrix
from ....utils import bmath as bm
from ....utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Literal, Optional

    from ....input_parameters.ring import Ring
    from ....input_parameters.rf_parameters import RFStation
    from ...beam import Beam


@handle_legacy_kwargs
def bigaussian(ring: Ring, rf_station: RFStation, beam: Beam, sigma_dt: float,
               sigma_dE: Optional[float] = None, seed: int = 1234,
               reinsertion: bool = False) -> None:
    r"""Function generating a Gaussian beam in time and energy coordinates.

    Function generating a Gaussian beam both in time and energy
    coordinates. Fills Beam.dt and Beam.dE arrays.

    Parameters
    ----------
    ring : class
        A Ring type class
    rf_station : class
        An RFStation type class
    beam : class
        A Beam type class
    sigma_dt : float
        R.m.s. extension of the Gaussian in time
    sigma_dE : float (optional)
        R.m.s. extension of the Gaussian in energy; default is None and will
        match the energy coordinate according to bucket height and sigma_dt
    seed : int (optional)
        Fixed seed to have a reproducible distribution
    reinsertion : bool (optional)
        Re-insert particles that are generated outside the separatrix into the
        bucket; default in False

    """

    if sigma_dE is None:
        sigma_dE = _get_dE_from_dt(ring, rf_station, sigma_dt)
    counter = rf_station.counter[0]
    omega_rf = rf_station.omega_rf[0, counter]
    phi_s = rf_station.phi_s[counter]
    phi_rf = rf_station.phi_rf[0, counter]
    eta0 = rf_station.eta_0[counter]

    # RF wave is shifted by Pi below transition
    if eta0 < 0:
        phi_rf -= np.pi

    beam.sigma_dt = sigma_dt
    beam.sigma_dE = sigma_dE

    # Generate coordinates. For reproducibility, a separate random number stream is used for dt and dE
    rng_dt = np.random.default_rng(seed)
    rng_dE = np.random.default_rng(seed + 1)

    beam.dt = sigma_dt * rng_dt.normal(size=beam.n_macroparticles).astype(dtype=bm.precision.real_t, order='C',
                                                                          copy=False) + \
              (phi_s - phi_rf) / omega_rf
    beam.dE = sigma_dE * \
              rng_dE.normal(size=beam.n_macroparticles).astype(
                  dtype=bm.precision.real_t, order='C')

    # Re-insert if necessary
    if reinsertion:

        itemindex = bm.where(is_in_separatrix(ring, rf_station, beam,
                                              beam.dt, beam.dE) == False)[0]
        while itemindex.size > 0:
            beam.dt[itemindex] = sigma_dt * rng_dt.normal(size=itemindex.size).astype(dtype=bm.precision.real_t,
                                                                                      order='C', copy=False) \
                                 + (phi_s - phi_rf) / omega_rf

            beam.dE[itemindex] = sigma_dE * rng_dE.normal(
                size=itemindex.size).astype(dtype=bm.precision.real_t, order='C')

            itemindex = bm.where(is_in_separatrix(ring, rf_station, beam,
                                                  beam.dt, beam.dE) == False)[0]
