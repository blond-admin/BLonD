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

    The beam control has a beam-phase loop, which acts on the difference between
    the beam phase and the RF phase. The transfer function of the beam-phase loop is

    .. math::
        H(z) = g_{PL} \frac{b_0 + b_1 z^{-1}}{1 + a_1 z^{-1}}

    in z-domain. In time-domain, the beam-phase loop correction from turn :math:`n` to turn :math:`n + 1` is

    .. math::
        \Delta \omega_{PL}^{n + 1} = - a_1 \Delta \omega_{PL}^{n} +
        g_{PL} \left ( b_0 \Delta \varphi_{PL}^{n + 1} + b_1 \Delta \varphi_{PL}^{n} \right )

    with :math:`\Delta \omega_{PL}` being the beam-phase loop angular frequency correction and
    :math:`\varphi_{PL}` the phase-loop error.

    The beam control also has a radial loop for longer-term corrections. The radial position is done through
    a PI controller. The frequency correction from the radial loop :math:`\Delta \omega_{RL}` is given as

    .. math::
        \Delta \omega_{RL}^{n + 1} = \Delta \omega_{RL}^{n} +
        g_{RL,a} \left [ \left ( \frac{\Delta R}{R} \right )^{n} -
        \left ( \frac{\Delta R}{R} \right )^{n - 1} \right ]
        + g_{RL,b} \left ( \frac{\Delta R}{R} \right )^{n}

    with :math:`\Delta R/R` being the relative radial offset and :math:`g_{RL,a}` and :math:`g_{RL,b}` being the
    proportional and integral gains respectively.

    Both the beam-phase loop and the radial loop act only every 10 micro seconds. The total frequency
    correction from the PSB beam control is

    .. math::
        \Delta \omega_{rf}^{n} = \Delta \omega_{PL}^{n} + \Delta \omega_{RL}^{n}.

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
        The proportional gain of the radial loop.
        Use ``beam_control.schedule("rl_gain_a", ...)`` to influence
        the parameter along the simulated cycle.
    rl_gain_b
        The integral gain of the radial loop.
        Use ``beam_control.schedule("rl_gain_b", ...)`` to influence
        the parameter along the simulated cycle.
    period
        Time [s] between the actions of the phase loop.
    pl_a_1
        Coefficient for the denominator of the beam-phase loop transfer function.
        See documentation above.
    pl_b_0
        First coefficient for the numerator of the beam-phase loop transfer function.
        See documentation above.
    pl_b_1
        Second coefficient for the numerator of the beam-phase loop transfer function.
        See documentation above.
    **kwargs
        Variable keyword arguments for the `BeamFeedbackBase`.

    Attributes
    ----------
    domega_pl
        Angular frequency correction [rad/s] from the beam-phase loop.
    domega_rl
        Angular frequency correction [rad/s] from the radial loop.
    dr_over_r
        Relative radial offset [-] between the beam orbit and the reference orbit.
    """

    def __init__(
        self,
        profile: ProfileBaseClass,
        phase_noise=None,
        pl_gain: float = 0.0,
        rl_gain_a: float = 0.0,
        rl_gain_b: float = 0.0,
        period: float = 10.0e-6,
        pl_a_1: float = -0.99803799,
        pl_b_0: float = 0.99901903,
        pl_b_1: float = -0.99901003,
        **kwargs,
    ):
        super().__init__(profile=profile, phase_noise=phase_noise, **kwargs)

        self.pl_gain = pl_gain

        #: | *Radial loop gain, proportional [1] and integral [1/s].*
        self.rl_gain_a = rl_gain_a
        self.rl_gain_b = rl_gain_b

        #: | *Optional: PL & RL acting only in certain time intervals/turns.*
        self.dt = period

        # Counter of turns passed since last time the PL was active
        self._pl_counter = 0
        self._on_time = np.array([])

        #: | *Array of transfer function coefficients for the beam-phase loop.*
        self._a_1 = pl_a_1
        self._b_0 = pl_b_0
        self._b_1 = pl_b_1

        #: | *Memory of previous phase correction, for phase loop.*
        self._dphi_sum = 0.0
        self._dphi_av = 0.0
        self._dphi_av_prev = 0.0

        #: | *Memory of previous relative radial correction, for rad loop.*
        self._dr_over_r_prev = 0.0

        self.domega_pl = 0.0
        self.domega_rl = 0.0
        self.dr_over_r = 0.0

        self._register_schedulable_variables(
            "pl_gain", "rl_gain_a", "rl_gain_b"
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
            n = self._delay + 1
            while n < n_turns + 1:
                summa = 0
                while summa < self.dt:
                    summa += (
                        self.cavities[0].get_main_harmonic_t_rf()
                        * self.cavities[0].get_main_harmonic()
                    )
                    n += 1
                self._on_time = np.append(self._on_time, n - 1)
        else:
            self._on_time = np.arange(n_turns + 1)

    def update_beam_attributes(self, beam: BeamBaseClass):
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

    def update_frequency_correction(self, beam: BeamBaseClass):
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

        self.update_phase_error()

        self._dphi_sum += self.dphi

        # Phase and radial loop active on certain turns
        if (
            counter == self._on_time[self._pl_counter]
            and counter >= self._delay
        ):
            # Phase loop
            self._dphi_av = self._dphi_sum / (
                self._on_time[self._pl_counter]
                - self._on_time[self._pl_counter - 1]
            )

            self.domega_pl = -self._a_1 * self.domega_pl + self.pl_gain * (
                self._b_0 * self._dphi_av + self._b_1 * self._dphi_av_prev
            )

            self._dphi_av_prev = self._dphi_av
            self._dphi_sum = 0.0

            # Radial loop
            self.dr_over_r = (
                self._main_cavities[0].get_main_harmonic_omega_rf()
                - self._main_cavities[0].get_main_harmonic_omega_rf_design()
            ) / (
                self._main_cavities[0].get_main_harmonic_omega_rf_design()
                * (
                    1.0
                    / (
                        self._simulation.ring.momentum_compaction_factor
                        * beam.reference.gamma**2
                    )
                    - 1.0
                )
            )

            self.domega_rl = (
                self.domega_rl
                + self.rl_gain_a * (self.dr_over_r - self._dr_over_r_prev)
                + self.rl_gain_b * self.dr_over_r
            )

            self._dr_over_r_prev = self.dr_over_r

            # Counter to pick the next time step when the PL & RL will be active
            self._pl_counter += 1

        # Apply frequency correction
        self.delta_omega_rf = -self.domega_pl - self.domega_rl
