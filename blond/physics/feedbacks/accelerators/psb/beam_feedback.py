# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Implementation of the PSB beam control.

Notes
-----
Authors:
Danilo Quartullo
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.physics.feedbacks.beam_feedback import (
    BeamFeedbackBase,
)

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation
    from blond.physics.profiles import ProfileBaseClass


class PSBBeamControl(BeamFeedbackBase):
    r"""
    Class for the PSB beam control.

    This class implements the feedbacks present in the beam
    control of the rf system in the Proton Synchrotron Booster.

    Parameters
    ----------
    profile
        Any Profile object which exposes the x- and y-axis of the beam line density.
    phase_noise
        Option to add phase noise through the beam control.
    pl_gain
        The gain of the beam-phase loop.
        Use ``beam_control.schedule("pl_gain", ...)`` to influence
        the parameter along the simulated cycle.
    rl_gain_a
        The first gain of the radial loop.
        Use ``beam_control.schedule("rl_gain_a", ...)`` to influence
        the parameter along the simulated cycle.
    rl_gain_b
        The second gain of the radial loop.
        Use ``beam_control.schedule("rl_gain_b", ...)`` to influence
        the parameter along the simulated cycle.
    period
        Time [s] between the actions of the phase loop.
    coefficients
        Coefficients for the transfer function of the feedback.
    **kwargs
        Variable keyword arguments for the `BeamFeedbackBase`.
    """

    def __init__(
        self,
        profile: ProfileBaseClass,
        phase_noise=None,
        pl_gain: float = 0.0,
        rl_gain_a: float = 0.0,
        rl_gain_b: float = 0.0,
        period: float = 10.0e-6,
        coefficients: list[float] = None,
        **kwargs,
    ):
        super().__init__(profile=profile, phase_noise=phase_noise, **kwargs)

        self.pl_gain = pl_gain

        self.delta_omega_rf = 0.0
        self.dphi = 0.0
        self.reference = 0.0

        #: | *Radial loop gain, proportional [1] and integral [1/s].*
        self.rl_gain_a = rl_gain_a
        self.rl_gain_b = rl_gain_b

        #: | *Optional: PL & RL acting only in certain time intervals/turns.*
        self.dt = period

        # Counter of turns passed since last time the PL was active
        self.PL_counter = 0
        self.on_time = np.array([])

        #: | *Array of transfer function coefficients.*
        if coefficients is None:
            self.coefficients = [
                0.999019,
                -0.999019,
                0.0,
                1.0,
                -0.998038,
                0.0,
            ]
        else:
            self.coefficients = coefficients

        #: | *Memory of previous phase correction, for phase loop.*
        self.dphi_sum = 0.0
        self.dphi_av = 0.0
        self.dphi_av_prev = 0.0

        #: | *Memory of previous relative radial correction, for rad loop.*
        self.dR_over_R_prev = 0.0

        #: | *Phase loop frequency correction [1/s]*
        self.domega_PL = 0.0

        #: | *Radial loop frequency correction [1/s]*
        self.domega_RL = 0.0

        self.dR_over_R = 0

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

        self.precalculate_time(n_turns)

    def precalculate_time(self, n_turns: int):
        """
        Calculate the PL action before running the simuliaton.

        For machines like the PSB, where the PL acts only in certain time
        intervals, pre-calculate on which turns to act.

        Parameters
        ----------
        n_turns
            Number of turns of the simulation.
        """
        if self.dt > 0:
            n = self.delay + 1
            while n < n_turns + 1:
                summa = 0
                while summa < self.dt:
                    summa += (
                        self.cavities[0].get_main_harmonic_t_rf()
                        * self.cavities[0].get_main_harmonic()
                    )
                    n += 1
                self.on_time = np.append(self.on_time, n - 1)
        else:
            self.on_time = np.arange(n_turns + 1)

    def get_beam_attribute(self, beam: BeamBaseClass):
        """
        Calculate the beam phase.

        This method implements the measurement of the beam, which is the
        input for the feedback. In the case of the PSB beam control, this
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

        Phase and radial loops for PSB. See documentation on-line for details.

        Parameters
        ----------
        beam
            A beam object to extract the beam attribute from.
        """
        # Average phase error while frequency is updated
        counter = self._simulation.turn_counter.value

        self.phase_difference()

        self.dphi_sum += self.dphi

        # Phase and radial loop active on certain turns
        if counter == self.on_time[self.PL_counter] and counter >= self.delay:
            # Phase loop
            self.dphi_av = self.dphi_sum / (
                self.on_time[self.PL_counter]
                - self.on_time[self.PL_counter - 1]
            )

            self.domega_PL = 0.99803799 * self.domega_PL + self.pl_gain * (
                0.99901903 * self.dphi_av - 0.99901003 * self.dphi_av_prev
            )

            self.dphi_av_prev = self.dphi_av
            self.dphi_sum = 0.0

            # Radial loop
            self.dR_over_R = (
                self.cavities[0].get_main_harmonic_omega_rf()
                - self.cavities[0].get_main_harmonic_omega_rf_design()
            ) / (
                self.cavities[0].get_main_harmonic_omega_rf_design()
                * (
                    1.0
                    / (
                        self.cavities[0]._ring.momentum_compaction_factor
                        * beam.reference.gamma**2
                    )
                    - 1.0
                )
            )

            self.domega_RL = (
                self.domega_RL
                + self.rl_gain_a * (self.dR_over_R - self.dR_over_R_prev)
                + self.rl_gain_b * self.dR_over_R
            )

            self.dR_over_R_prev = self.dR_over_R

            # Counter to pick the next time step when the PL & RL will be active
            self.PL_counter += 1

        # Apply frequency correction
        self.delta_omega_rf = -self.domega_PL - self.domega_RL
