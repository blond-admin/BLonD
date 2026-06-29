# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Implementation of the SPS beam control.

Notes
-----
Authors:
Danilo Quartullo
Leandro Intelisano
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

import numpy as np

from blond import Simulation
from blond.core.beam.base import BeamBaseClass
from blond.physics.feedbacks.beam_feedback import BeamFeedbackBase


class SPSBeamControl(BeamFeedbackBase):
    r"""
    Class for the SPS beam control.

    This class implements the feedbacks present in the beam
    control of the rf system in the Super Proton Synchrotron.

    Parameters
    ----------
    k_phi_n
        Feedback gain for the phase loop error from the previous turn.
    k_phi_nm1
        Feedback gain for the phase loop error from two turn prior.
    k_eps_n
        Feedback gain for the synchronization loop error.
    k_z_n
        Feedback gain for the integration of synchronization loop error.
    k_a_n
        Feedback gain for the frequency loop error.
    k_b_n
        Feedback gain for the integration of the frequency loop error.
    phi_sync
        Synchronous phase of the beam [rad].
    pl_gain
        Beam-phase loop gain of the beam control.
    action_delay
        Delay of the action of the beam-phase loop from the first turn.
    delay_turns
        The delay [turns] between measurement at correction from the beam control.
    current_thres
        Beam current threshold for gating of the profiles.
    *args
        Variable positional arguments.
    **kwargs
        Variable keyword arguments.
    """

    def __init__(
        self,
        k_phi_n: float | NumpyArray,
        k_phi_nm1: float | NumpyArray,
        k_eps_n: float | NumpyArray,
        k_z_n: float | NumpyArray,
        k_a_n: float | NumpyArray,
        k_b_n: float | NumpyArray,
        phi_sync: float | NumpyArray,
        pl_gain: float | NumpyArray,
        action_delay: int,
        delay_turns: int = 2,
        current_thres: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.k_phi_n = k_phi_n
        self.k_phi_nm1 = k_phi_nm1
        self.k_eps_n = k_eps_n
        self.k_z_n = k_z_n
        self.k_a_n = k_a_n
        self.k_b_n = k_b_n
        self.action_delay = action_delay

        self.phi_sync = phi_sync
        self.pl_gain = pl_gain

        self.delay_turns = delay_turns

        self.domega_rf_corr = [0.0] * self.delay_turns

        # Internal feedback parameters
        self.dphi_prev = 0
        self.epsilon = 0
        self.epsilon_prev = 0
        self.zeta = 0
        self.alpha = 0
        self.alpha_prev = 0

        self.domega_rf = 0.0
        self.dphi = 0.0

        # Frequency corrections
        self.domega_dphi = 0.0
        self.domega_sync = 0.0
        self.domega_freq = 0.0

        self.current_thres = current_thres

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs,
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation `Beam` object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
            **kwargs,
        )
        if (
            self.current_thres is None
            and self.main_cavities[0].any_feedback_not_none
        ):
            raise RuntimeError(
                "The filled slots in the machine is needed to compute the cavity sum phase"
            )

        def convert_to_array(parameter: float, delay_action: int = 0):
            result = np.zeros(n_turns + 1)
            result[delay_action:] = parameter
            return result

        def ensure_array_length(_value, _name, _delay):
            if isinstance(_value, float):
                return convert_to_array(_value, _delay)

            if len(_value) < n_turns + 1:
                raise ValueError(
                    f"Array `{_name}` is not the correct length, `n_turns + 1` or longer"
                )

            return _value

        fields_with_delay = [
            ("k_phi_nm1", self.action_delay),
            ("k_phi_n", self.action_delay),
        ]

        fields_no_delay = [
            "k_eps_n",
            "k_z_n",
            "k_a_n",
            "k_b_n",
            "phi_sync",
            "pl_gain",
        ]

        # handle delayed parameters
        for name, delay in fields_with_delay:
            value = getattr(self, name)
            setattr(self, name, ensure_array_length(value, name, delay))

        # handle non-delayed parameters
        for name in fields_no_delay:
            value = getattr(self, name)
            setattr(self, name, ensure_array_length(value, name, 0))

    def get_beam_attribute(self, beam: BeamBaseClass):
        """
        Calculate the beam phase.

        This method implements the measurement of the beam, which is the
        input for the feedback. In the case of the SPS beam control, this
        is the rf component phase of the beam.

        Parameters
        ----------
        beam
            A beam object to extract the beam attribute from.
        """
        self.beam_phase()

    def compute_correction(self, beam: BeamBaseClass):
        """
        Calculate the frequency correction from the beam control.

        This method implements the feedback systems in the SPS beam control, i.e.
        the beam-phase loop, the synchronization loop and the frequency loop.

        Parameters
        ----------
        beam
            A beam object to extract the beam attribute from.
        """
        counter = self.cavities[0]._turn_counter.value

        t_rev = float(
            (2 * np.pi * self.cavities[0].get_main_harmonic())
            / self.cavities[0].get_main_harmonic_omega_rf_design()
        )

        # Phase difference
        self.phase_difference(beam)
        self.cavity_sum_phase(self.current_thres)

        # Take into account the synchronous phase
        self.dphi = (
            self.dphi
            + np.pi
            - self.main_cavities[0].calc_phi_s_main_harmonic(beam)
        )

        # Phase loop
        self.domega_dphi = (
            -self.k_phi_n[counter] * self.dphi
            - self.k_phi_nm1[counter] * self.dphi_prev
        )

        # Synchro Loop
        self.epsilon = (
            self.cavities[0].get_main_harmonic_phi_rf()
            - self.phi_sync[counter]
        )
        self.zeta += self.epsilon_prev
        self.domega_sync = (
            -self.k_eps_n[counter] * self.epsilon
            - self.k_z_n[counter] * self.zeta
        )

        # Frequency Loop
        self.domega_freq = (
            -self.k_a_n[counter] * self.alpha
            - self.k_b_n[counter] * self.alpha_prev
        )

        # Total frequency correction
        self.domega_rf_corr = [
            self.domega_dphi + self.domega_sync + self.domega_freq
        ] + self.domega_rf_corr[:-1]

        self.domega_rf = self.domega_rf_corr[-1]

        # Update some parameters for the next turn
        self.alpha_prev = self.alpha
        self.alpha = self.domega_rf * t_rev
        self.epsilon_prev = self.epsilon
        self.dphi_prev = self.dphi

        # Apply global gain
        self.domega_rf *= self.pl_gain[counter]
