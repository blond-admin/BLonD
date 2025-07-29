# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Various beam phase loops with optional synchronisation/frequency/radial loops
for the CERN machines**

:Authors: **Helga Timko**, **Alexandre Lasheen**
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .base import LocalFeedback
from ..._core.backends.backend import backend

if TYPE_CHECKING:  # pragma: no cover
    from ..profiles import ProfileBaseClass
    from ..._core.beam.base import BeamBaseClass
    from typing import Optional


class Blond2BeamFeedback(LocalFeedback):
    """One-turn beam phase loop

    One-turn beam phase loop for different machines with different hardware.
    Use 'period' for a phase loop that is active only in certain turns.
    The phase loop acts directly on the RF frequency of all harmonics and
    affects the RF phase as well.

    Parameters
    ----------
    profile
        Base class to calculate the beam profile
    PL_gain
        Phase loop gain. Implementation depends on machine.
    window_coefficient
        Band-pass filter window coefficient for beam phase calculation.
    time_offset
        Determines from which RF-buckets the band-pass filter starts to acts
    delay
        Number of turns that the feedback starts acting later
    section_index
        Section index to group elements into sections
    name
        User given name of the element

    Attributes
    ----------
    profile
        Base class to calculate the beam profile
    delay
        Number of turns that the feedback starts acting later
    alpha
        Band-pass filter window coefficient for beam phase calculation.
    time_offset
        Determines from which RF-buckets the band-pass filter starts to acts
    gain
        Phase loop gain. Implementation depends on machine.
    drho
        Phase loop frequency correction of the main RF system.
    domega_rf
        Phase loop frequency correction of the main RF system.
    phi_beam
        Beam phase measured at the main RF frequency.
    dphi
        Phase difference between beam and RF.
    reference
        Reference signal for secondary loop to test step response.
    """

    def __init__(
        self,
        profile: ProfileBaseClass,
        PL_gain: float,
        window_coefficient: float = 0.0,
        time_offset: Optional[float] = None,
        delay: int = 0,
        section_index: int = 0,
        name: Optional[str] = None,
    ):
        """One-turn beam phase loop base class

        One-turn beam phase loop for different machines with different hardware.
        Use 'period' for a phase loop that is active only in certain turns.
        The phase loop acts directly on the RF frequency of all harmonics and
        affects the RF phase as well.

        Parameters
        ----------
        profile
            Base class to calculate the beam profile
        PL_gain
            Phase loop gain. Implementation depends on machine.
        window_coefficient
            Band-pass filter window coefficient for beam phase calculation.
        time_offset
            Determines from which RF-buckets the band-pass filter starts to acts
        delay
            # TODO UNKNOWN
        section_index
            Section index to group elements into sections
        name
            User given name of the element
        """
        super().__init__(
            profile=profile,
            section_index=section_index,
            name=name,
        )
        self.profile = profile

        self.delay = delay

        #: | *Band-pass filter window coefficient for beam phase calculation.*
        self.alpha = window_coefficient

        # determines from which RF-buckets the band-pass filter starts to acts
        self.time_offset = time_offset

        #: | *Phase loop gain. Implementation depends on machine.*
        self.gain = PL_gain

        #: | *Relative radial displacement [1], for radial loop.*
        self.drho = 0.0

        #: | *Phase loop frequency correction of the main RF system.*
        self.domega_rf = 0.0

        #: | *Beam phase measured at the main RF frequency.*
        self.phi_beam = 0.0

        #: | *Phase difference between beam and RF.*
        self.dphi = 0.0

        #: | *Reference signal for secondary loop to test step response.*
        self.reference = 0.0

        self.RFnoise = None  # FIXME remove this!

    @abstractmethod
    def update_domega_rf(self, beam: BeamBaseClass) -> None:
        pass

    def update_phi_beam(self):
        """Beam phase measured at the main RF frequency and phase

        Beam phase measured at the main RF frequency and phase. The beam is
        convolved with the window function of the band-pass filter of the
        machine. The coefficients of sine and cosine components determine the
        beam phase, projected to the range -Pi/2 to 3/2 Pi. Note that this beam
        phase is already w.r.t. the instantaneous RF phase.
        """

        # Main RF frequency at the present turn
        omega_rf = self._parent_cavity._omega_rf[0] + self._parent_cavity.delta_omega_rf
        phi_rf = self._parent_cavity.phi_rf[0] + self._parent_cavity.delta_phi_rf

        if self.time_offset is None:
            coeff = backend.specials.beam_phase(
                self.profile.hist_x,
                self.profile.hist_y,
                self.alpha,
                omega_rf,
                phi_rf,
                self.profile.hist_step,
            )
        else:
            indexes = self.profile.hist_x >= self.time_offset
            coeff = backend.specials.beam_phase(
                self.profile.hist_x[indexes],
                self.profile.hist_y[indexes],
                self.alpha,
                omega_rf,
                phi_rf,
                self.profile.hist_step,
            )

        # Project beam phase to (pi/2,3pi/2) range
        self.phi_beam = np.arctan(coeff) + np.pi

    def update_dphi(self, beam: BeamBaseClass):
        """
        *Phase difference between beam and RF phase of the main RF system.
        Optional: add RF phase noise through dphi directly.*
        """

        # Correct for design stable phase
        self.dphi = self.phi_beam - self._parent_cavity.phi_s

        # TODO fix this code
        # Possibility to add RF phase noise through the PL
        if self.RFnoise is not None:
            if self.noiseFB is not None:
                self.dphi += self.noiseFB.x * self.RFnoise.dphi[current_turn]
            else:
                if self.machine == "PSB":
                    self.dphi = self.dphi
                else:
                    self.dphi += self.RFnoise.dphi[current_turn]
