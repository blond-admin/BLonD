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
    from blond.physics.profiles import ProfileBaseClass

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
    profile
        Any Profile object which exposes the x- and y-axis of the beam line density.
    phase_noise
        Option to add phase noise through the beam control.
    k_phi_n
        Feedback gain for the phase loop error from the previous turn.
        Use ``beam_control.schedule("k_phi_n", ...)`` to influence
        the parameter along the simulated cycle.
    k_phi_nm1
        Feedback gain for the phase loop error from two turn prior.
        Use ``beam_control.schedule("k_phi_nm1", ...)`` to influence
        the parameter along the simulated cycle.
    k_eps_n
        Feedback gain for the synchronization loop error.
        Use ``beam_control.schedule("k_eps_n", ...)`` to influence
        the parameter along the simulated cycle.
    k_z_n
        Feedback gain for the integration of synchronization loop error.
        Use ``beam_control.schedule("k_z_n", ...)`` to influence
        the parameter along the simulated cycle.
    k_a_n
        Feedback gain for the frequency loop error.
        Use ``beam_control.schedule("k_a_n", ...)`` to influence
        the parameter along the simulated cycle.
    k_b_n
        Feedback gain for the integration of the frequency loop error.
        Use ``beam_control.schedule("k_b_n", ...)`` to influence
        the parameter along the simulated cycle.
    phi_sync
        Synchronous phase of the beam [rad].
        Use ``beam_control.schedule("phi_sync", ...)`` to influence
        the parameter along the simulated cycle.
    pl_gain
        Beam-phase loop gain of the beam control.
        Use ``beam_control.schedule("pl_gain", ...)`` to influence
        the parameter along the simulated cycle.
    action_delay
        Delay of the action of the beam-phase loop from the first turn.
    delay_turns
        The delay [turns] between measurement at correction from the beam control.
    current_thres
        Beam current threshold for gating of the profiles.
    **kwargs
        Variable keyword arguments for the `BeamFeedbackBase`.

    Attributes
    ----------
    epsilon
        The difference between the RF phase and the reference [rad], i.e. the synchronization loop error.
    domega_phi
        Angular frequency correction [rad/s] from the beam-phase loop.
    domega_sync
        Angular frequency correction [rad/s] from the synchronization loop.
    domega_freq
        Angular frequency correction [rad/s] from the frequency loop.
    """

    def __init__(
        self,
        profile: ProfileBaseClass,
        phase_noise=None,
        k_phi_n: float = 0.0,
        k_phi_nm1: float = 0.0,
        k_eps_n: float = 0.0,
        k_z_n: float = 0.0,
        k_a_n: float = 0.0,
        k_b_n: float = 0.0,
        phi_sync: float = 0.0,
        pl_gain: float = 0.0,
        action_delay: int = 0,
        delay_turns: int = 2,
        current_thres: float | None = None,
        **kwargs,
    ):
        super().__init__(profile=profile, phase_noise=phase_noise, **kwargs)

        self.k_phi_n = k_phi_n
        self.k_phi_nm1 = k_phi_nm1
        self.k_eps_n = k_eps_n
        self.k_z_n = k_z_n
        self.k_a_n = k_a_n
        self.k_b_n = k_b_n
        self.action_delay = action_delay

        self.phi_sync = phi_sync
        self.pl_gain = pl_gain

        self._domega_rf_corr = [0.0] * delay_turns

        # Internal feedback parameters
        self._dphi_prev = 0.0
        self._epsilon_prev = 0.0
        self._zeta = 0.0
        self._alpha = 0.0
        self._alpha_prev = 0.0

        self.epsilon = 0.0  # synchro loop error [rad]

        # Frequency corrections
        self.domega_phi = 0.0
        self.domega_sync = 0.0
        self.domega_freq = 0.0

        self.current_thres = current_thres

        self._register_schedulable_variables(
            "k_phi_n",
            "k_phi_nm1",
            "k_eps_n",
            "k_z_n",
            "k_a_n",
            "k_b_n",
            "phi_sync",
            "pl_gain",
        )

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
            and self._main_cavities[0].any_feedback_not_none
        ):
            raise RuntimeError(
                "The filled slots in the machine is needed to compute the cavity sum phase"
            )

    def update_beam_attributes(self, beam: BeamBaseClass):
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

    def update_frequency_correction(self, beam: BeamBaseClass):
        """
        Calculate the frequency correction from the beam control.

        This method implements the feedback systems in the SPS beam control, i.e.
        the beam-phase loop, the synchronization loop and the frequency loop.

        Parameters
        ----------
        beam
            A beam object to extract the beam attribute from.
        """
        t_rev = float(
            (2 * np.pi * self.cavities[0].get_main_harmonic())
            / self.cavities[0].get_main_harmonic_omega_rf_design()
        )

        # Phase difference
        self.update_phase_error()
        self.cavity_sum_phase(self.current_thres)

        # Take into account the synchronous phase
        self.dphi = (
            self.dphi
            + np.pi
            - self._main_cavities[0].calc_phi_s_main_harmonic(beam)
        )

        # Phase loop
        self.domega_phi = (
            -self.k_phi_n * self.dphi - self.k_phi_nm1 * self._dphi_prev
        )

        # Synchro Loop
        self.epsilon = (
            self.cavities[0].get_main_harmonic_phi_rf() - self.phi_sync
        )
        self._zeta += self._epsilon_prev
        self.domega_sync = (
            -self.k_eps_n * self.epsilon - self.k_z_n * self._zeta
        )

        # Frequency Loop
        self.domega_freq = (
            -self.k_a_n * self._alpha - self.k_b_n * self._alpha_prev
        )

        # Total frequency correction
        self._domega_rf_corr = [
            self.domega_phi + self.domega_sync + self.domega_freq
        ] + self._domega_rf_corr[:-1]

        self.delta_omega_rf = self._domega_rf_corr[-1]

        # Update some parameters for the next turn
        self._alpha_prev = self._alpha
        self._alpha = self.delta_omega_rf * t_rev
        self._epsilon_prev = self.epsilon
        self._dphi_prev = self.dphi

        # Apply global gain
        self.delta_omega_rf *= self.pl_gain
