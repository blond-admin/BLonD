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

Notes
-----
Authors:
Helga Timko
Alexandre Lasheen
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from blond import Simulation
from blond.physics.feedbacks.base import (
    GlobalFeedback,
)

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass
    from blond.physics.profiles import ProfileBaseClass


class BeamFeedbackBase(GlobalFeedback):
    _parent_rf_station: RFStationBaseClass

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

        self.dphi: float = 0.0

        self.phi_beam: float = 0.0

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs,
    ) -> None:
        if (
            self.current_thres is None
            and self.cavities[0].any_feedback_not_none
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
        omega_rf = self.cavities[0].get_main_harmonic_omega_rf()
        phi_rf = self.cavities[0].get_main_harmonic_phi_rf()

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
        self.dphi = self.phi_beam - self.cavities[0].calc_phi_s_main_harmonic(
            beam
        )

        # Phase offset due to beam loading
        if self.cavities[0].any_feedback_not_none:
            filled_slots = (
                np.abs(
                    self.cavities[0]
                    .cavity_feedback_list[0]
                    .I_BEAM_COARSE[
                        -self.cavities[0].cavity_feedback_list[0].n_coarse :
                    ]
                )
                > self.current_thres
            )

            gap_phase_in_slots = (
                self.cavities[0]
                .cavity_feedback_list[0]
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

    def _track(self, beam: BeamBaseClass):
        self.get_beam_attribute(
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
                / self.cavities[
                    0
                ].get_main_harmonic()  # dynamically updated by `update_domega_rf`
            )
            self.cavities[0].delta_omega_rf = omega_increment
