# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

import numpy as np

from blond import Simulation
from blond.core.backends.backend import backend
from blond.core.beam.base import BeamBaseClass
from blond.experimental.physics.feedbacks.beam_feedback import (
    BeamFeedbackBase,
)
from blond.physics.profiles import ProfileBaseClass


class SPSBeamControl(BeamFeedbackBase):
    def __init__(
        self,
        profile: ProfileBaseClass,
        k_phi_n: float | NumpyArray,
        k_phi_nm1: float | NumpyArray,
        k_eps_n: float | NumpyArray,
        k_z_n: float | NumpyArray,
        k_a_n: float | NumpyArray,
        k_b_n: float | NumpyArray,
        phi_sync: float | NumpyArray,
        global_gain: float | NumpyArray,
        action_delay: int,
        *args,
        **kwargs,
    ):
        super().__init__(profile=profile, *args, **kwargs)

        self.k_phi_n = k_phi_n
        self.k_phi_nm1 = k_phi_nm1
        self.k_eps_n = k_eps_n
        self.k_z_n = k_z_n
        self.k_a_n = k_a_n
        self.k_b_n = k_b_n
        self.action_delay = action_delay

        self.phi_sync = phi_sync
        self.global_gain = global_gain

        self.dphi_z1 = 0
        self.dphi_z2 = 0
        self.dphi_z3 = 0
        self.epsilon_z1 = 0
        self.epsilon_z2 = 0
        self.epsilon_z3 = 0
        self.Zeta = 0
        self.Alpha = 0
        self.Alpha_z1 = 0
        self.Alpha_z2 = 0
        self.Alpha_z3 = 0

        self.domega_rf = 0.0
        self.dphi = 0.0
        self.reference = 0.0

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs,
    ) -> None:
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
            **kwargs,
        )

        def convert_to_array(parameter, delay_action=0):
            delay_action = np.concatenate(
                (np.zeros(delay_action), np.ones(n_turns + 1 - delay_action))
            )
            return parameter * np.ones(n_turns + 1) * delay_action

        if isinstance(self.k_phi_nm1, float):
            self.k_phi_nm1 = convert_to_array(
                self.k_phi_nm1, self.action_delay
            )

        if isinstance(self.k_phi_n, float):
            self.k_phi_n = convert_to_array(self.k_phi_n, self.action_delay)

        if isinstance(self.k_eps_n, float):
            self.k_eps_n = convert_to_array(self.k_eps_n)

        if isinstance(self.k_z_n, float):
            self.k_z_n = convert_to_array(self.k_z_n)

        if isinstance(self.k_a_n, float):
            self.k_a_n = convert_to_array(self.k_a_n)

        if isinstance(self.k_b_n, float):
            self.k_b_n = convert_to_array(self.k_b_n)

        if isinstance(self.phi_sync, float):
            self.phi_sync = convert_to_array(self.phi_sync)

        if isinstance(self.global_gain, float):
            self.global_gain = convert_to_array(self.global_gain)

    def get_beam_attribute(self, beam: BeamBaseClass):
        self.beam_phase()

    def apply_corrections(self, beam: BeamBaseClass):
        counter = self.cavities[0]._turn_i.value

        t_rev = float(
            (2 * np.pi * self.cavities[0].harmonic[0])
            / self.cavities[0].calc_main_harmonic_omega_rf_design(
                beam.reference.beta, self.cavities[0]._ring.circumference
            )
        )

        # Phase loop
        self.beam_phase()
        self.phase_difference(beam)

        self.domega_dphi = (
            -self.k_phi_n[counter] * self.dphi_z2
            - self.k_phi_nm1[counter] * self.dphi_z3
        )

        # Synchro Loop
        self.epsilon = self.cavities[0].phi_rf - self.phi_sync[counter]
        self.Zeta += self.epsilon_z1
        self.domega_sync = (
            -self.k_eps_n[counter] * self.epsilon
            - self.k_z_n[counter] * self.Zeta
        )

        # Frequency Loop
        self.domega_freq = (
            -self.k_a_n[counter] * self.Alpha_z1
            - self.k_b_n[counter] * self.Alpha_z2
        )

        # Total frequency correction
        self.domega_rf = self.domega_dphi + self.domega_sync + self.domega_freq

        # Update some parameters for the next turn
        self.Alpha_z3 = self.Alpha_z2
        self.Alpha_z2 = self.Alpha_z1
        self.Alpha_z1 = self.Alpha
        self.Alpha = self.domega_rf * t_rev
        self.epsilon_z3 = self.epsilon_z2
        self.epsilon_z2 = self.epsilon_z1
        self.epsilon_z1 = self.epsilon
        self.dphi_z3 = self.dphi_z2
        self.dphi_z2 = self.dphi_z1
        self.dphi_z1 = self.dphi

        # Apply global gain
        self.domega_rf *= self.global_gain[counter]
