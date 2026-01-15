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
from xtrack import Particles, ZetaShift
from blond.core.beam.base import BeamBaseClass, BeamFlags

class BlondElement3:
    """
    Wrapper to allow BLonD3 elements to be tracked inside Xsuite.

    Updates the longitudinal coordinates.
    """

    def __init__(self, trackable,
                 beam: BeamBaseClass,
                 update_zeta: bool = False ):
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

    def xsuite_to_blond(self, particles: Particles, beam: BeamBaseClass):
        """
        Convert Xsuite coordinates to BLonD.

        Parameters
        ----------
        particles : Particles
            Particles to be tracked.
        beam : BeamBaseClass
            Beam to be tracked.
        """
        beam._dE[:] = particles.beta0 * particles.energy0 * particles.ptau
        beam._dt[:] = -particles.zeta / (particles.beta0 * 3e8)


        active_mask = particles.state > 0
        beam._flags[:] = np.where(active_mask, BeamFlags.ACTIVE.value, BeamFlags.LOST.value)


    def blond_to_xsuite(self, particles: Particles, beam: BeamBaseClass):
        """
        Convert BLonD coordinates to Xsuite.

        Parameters
        ----------
        particles : Particles
            Particles to be tracked.
        beam : BeamBaseClass
            Beam to be tracked.
        """
        particles.ptau = beam._dE / (particles.beta0 * particles.energy0)
        if self.update_zeta:
            particles.zeta = -beam._dt * particles.beta0 * 3e8

        # Mark lost particles in Xsuite
        lost_mask = (beam._flags != BeamFlags.ACTIVE.value) & (particles.state > 0)
        particles.state[lost_mask] = -500


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
        self.xsuite_to_blond(particles, self.beam)

        self.trackable.track(self.beam) # calls the BLonD track method

        # Convert blond -> xsuite
        self.blond_to_xsuite(particles, self.beam)
