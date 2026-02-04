# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


"""
**Various beam phase loops with optional synchronisation/frequency/radial loops
for the CERN machines**

Authors
-------
Helga Timko
Alexandre Lasheen
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from blond import Simulation
from blond.core.backends.backend import backend
from blond.experimental.physics.feedbacks.base import (
    GlobalFeedback,
    LocalFeedback,
)

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass
    from blond.physics.cavities import RfStationBaseClass
    from blond.physics.profiles import ProfileBaseClass


class BeamFeedbackBase(GlobalFeedback):
    _parent_rf_station: RfStationBaseClass

    def __init__(
        self,
        profile: ProfileBaseClass,
        delay: int = 0,
        window_coefficient: float = 0.0,
        time_offset: float | None = None,
        current_thres=None,
    ):
        super().__init__(profile=profile)
        self.delay = delay
        self.window_coefficient = window_coefficient
        self.time_offset = time_offset
        self.current_thres = current_thres

        self.domega_rf = 0

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        if (
            self.current_thres is None
            and self.cavities[0]._cavity_feedback is not None
        ):
            raise RuntimeError(
                "The filled slots in the machine is needed to compute the cavity sum phase"
            )

    @abstractmethod
    def get_beam_attribute(self, beam: BeamBaseClass):
        # could be mean energy, mean phase or whatever
        pass

    @abstractmethod
    def apply_corrections(self, beam: BeamBaseClass):
        # shift the RF station phase or so
        pass

    def beam_phase(self):
        # Main RF frequency at the present turn
        counter = self.cavities[0]._turn_i
        omega_rf = self.cavities[0].omega_rf_actual[0]
        phi_rf = self.cavities[0].phi_rf_actual[0]

        if self.time_offset is None:
            coeff = backend.specials.beam_phase(
                self.profile.hist_x,
                self.profile.hist_y,
                self.window_coefficient,
                omega_rf,
                phi_rf,
                self.profile.hist_step,
            )
        else:
            indexes = self.profile.hist_x >= self.time_offset
            coeff = backend.specials.beam_phase(
                self.profile.hist_x[indexes],
                self.profile.hist_y,
                self.window_coefficient,
                omega_rf,
                phi_rf,
                self.profile.hist_step,
            )

        # Project beam phase to (pi/2,3pi/2) range
        self.phi_beam = np.arctan(coeff) + np.pi

    def phase_difference(
        self, beam: BeamBaseClass, RFnoise=None, noiseFB=None
    ):
        """
        *Phase difference between beam and RF phase of the main RF system.
        Optional: add RF phase noise through dphi directly.*
        """

        # Correct for design stable phase
        counter = self.cavities[0]._turn_i.value
        self.dphi = self.phi_beam - self.cavities[
            0
        ].calc_phi_s_single_harmonic(beam, enable_rf_phase=False)

        # Phase offset due to beam loading
        if self.cavities[0]._cavity_feedback is not None:
            filled_slots = (
                np.abs(
                    self.cavities[0]
                    ._cavity_feedback[0]
                    .I_BEAM_COARSE[
                        -self.cavities[0]._cavity_feedback[0].n_coarse :
                    ]
                )
                > self.current_thres
            )

            gap_phase_in_slots = (
                self.cavities[0]
                ._cavity_feedback[0]
                .gap_voltage_phase[filled_slots]
            )
            # voltage difference
            if len(gap_phase_in_slots) > 0:
                phi_mean = np.mean(gap_phase_in_slots)
            else:
                phi_mean = 0
            self.dphi = self.dphi + phi_mean

        # Possibility to add RF phase noise through the PL
        if RFnoise is not None:
            if noiseFB is not None:
                self.dphi += noiseFB.x * RFnoise.dphi[counter]
            else:
                self.dphi += RFnoise.dphi[counter]

    def track(self, beam: BeamBaseClass):
        self.get_beam_attribute(  # could be mean energy, mean phase or whatever
            beam=beam,
        )
        self.apply_corrections(
            beam=beam,
        )

        if self.cavities[0]._turn_i.value >= self.delay:
            # TODO incorrect for simulations that start later
            # domega_rf is updated later
            # this means domega_rf is effectively from last turn
            omega_increment = (
                self.domega_rf
                * self.cavities[0].harmonic[:]
                / self.cavities[0].harmonic[
                    self.cavities[0].main_harmonic_idx
                ]  # dynamically updated by `update_domega_rf`
            )
            self.cavities[0].delta_omega_rf = omega_increment


class Blond2BeamFeedback(LocalFeedback):
    """
    One-turn beam phase loop

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
        time_offset: float | None = None,
        delay: int = 0,
        section_index: int = 0,
        name: str | None = None,
    ):
        """
        One-turn beam phase loop base class

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

        self.alpha = window_coefficient

        self.time_offset = time_offset

        self.gain = PL_gain

        self.drho = 0.0

        self.domega_rf = 0.0

        self.phi_beam = 0.0

        self.dphi = 0.0

        self.reference = 0.0

        self.RFnoise = None  # FIXME remove this!

    @abstractmethod  # pragma: no cover
    def update_domega_rf(self, beam: BeamBaseClass) -> None:
        pass

    def update_phi_beam(self):
        """
        Beam phase measured at the main RF frequency and phase

        Beam phase measured at the main RF frequency and phase. The beam is
        convolved with the window function of the band-pass filter of the
        machine. The coefficients of sine and cosine components determine the
        beam phase, projected to the range -Pi/2 to 3/2 Pi. Note that this beam
        phase is already w.r.t. the instantaneous RF phase.
        """
        # Main RF frequency at the present turn
        omega_rf = (
            self._parent_rf_station._omega_rf[0]
            + self._parent_rf_station.delta_omega_rf
        )
        phi_rf = (
            self._parent_rf_station.phi_rf[0]
            + self._parent_rf_station.delta_phi_rf
        )

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
        Phase difference between beam and RF phase of the main RF system.
        Optional: add RF phase noise through dphi directly.
        """
        # Correct for design stable phase
        self.dphi = self.phi_beam - self._parent_rf_station.phi_s

        # TODO fix this code
        # Possibility to add RF phase noise through the PL
        if self.RFnoise is not None:
            if self.noiseFB is not None:
                self.dphi += self.noiseFB.x * self.RFnoise.dphi[current_turn]
            elif self.machine == "PSB":
                self.dphi = self.dphi
            else:
                self.dphi += self.RFnoise.dphi[current_turn]
