# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Functions and classes to interface BLonD with xsuite.

:Authors: **Birk Emil Karlsen-Baeck**, **Thom Arnoldus van Rijswijk**, **Helga Timko**, **Elleanor Lamb**
"""

import numpy as np
from scipy.constants import c
from xtrack import Particles, ZetaShift, ReferenceEnergyIncrease

from blond.core.beam.base import BeamBaseClass, BeamFlags
from numpy.typing import NDArray
from typing import Sequence, Union

FloatOrArray = Union[float, NDArray[np.floating]]


def xsuite_to_blond_transform(
    zeta: FloatOrArray,
    ptau: FloatOrArray,
    beta0: float,
    energy0: float,
    omega_rf: float,
    phi_s: float = 0,
):
    """
    Convert Xsuite coordinates to BLonD.

    Parameters
    ----------
    particles : Particles
        Particles to be tracked.
    beam : BeamBaseClass
        Beam to be tracked.
    """
    dE = ptau * beta0 * energy0
    dt = -zeta / (beta0 * c) + phi_s / omega_rf
    return dt, dE


def blond_to_xsuite_transform(
    dt: FloatOrArray,
    de: FloatOrArray,
    beta0: float,
    energy0: float,
    omega_rf: float,
    phi_s: float = 0,
):
    """
    Convert BLonD coordinates to Xsuite.

    Parameters
    ----------
    particles : Particles
        Particles to be tracked.
    beam : BeamBaseClass
        Beam to be tracked.
    """
    ptau = de / (beta0 * energy0)
    zeta = -(dt - phi_s / omega_rf) * beta0 * c
    return zeta, ptau


class BLonDElement3:
    """
    Wrapper to allow BLonD3 elements to be tracked inside Xsuite.

    Updates the longitudinal coordinates.
    """

    def __init__(
        self, trackable, beam: BeamBaseClass, update_zeta: bool = False
    ):
        """
        Initialise element.

        Parameters
        ----------
        trackable : BLonD3 element
            Any BLonD3 element with a `track(beam)` method.
            eg `RfStationBaseClass` or similar.
        update_zeta : bool
            Whether to convert Xsuite zeta -> BLonD dt and back.
        """
        self.beam = beam
        self.trackable = trackable
        self.update_zeta = update_zeta
        self.orbit_shift = ZetaShift(dzeta=0)

    def track(self, particles: Particles):
        """
        Track the BLonD element.

        Parameters
        ----------
        particles : Particles
            Particles to be tracked.
        beam : BeamBaseClass
            Beam to be tracked.
        """
        # Convert xsuite -> blond
        xsuite_to_blond_transform(particles, self.beam)

        self.trackable.track(self.beam)  # calls the BLonD track method

        # Convert blond -> xsuite
        blond_to_xsuite_transform(particles, self.beam)


class EnergyUpdate:
    """
    Class to update energy of Particles class turn-by-turn with the ReferenceEnergyIncrease function
    from xtrack. Additionally, it updates the frequency of the xtrack cavity in the line.
    Intended to be used without BLonD-Xsuite interface.

    """

    def __init__(self, momentum: Sequence):
        self.momentum = momentum

        init_p0c = self.momentum[1] - self.momentum[0]

        self.xsuite_energy_update = ReferenceEnergyIncrease(Delta_p0c=init_p0c)

    def track(self, particles: Particles):
        mask_alive = particles.state > 0

        # Use the still alive particles to find the current turn momentum
        p0c_before = particles.p0c[mask_alive]

        # Find the momentum for the next turn
        p0c_after = self.momentum[particles.at_turn[mask_alive][0]]

        # Update the energy increment
        self.xsuite_energy_update.Delta_p0c = p0c_after - p0c_before[0]

        # Apply the energy increment to the particles
        self.xsuite_energy_update.track(particles)
