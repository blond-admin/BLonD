from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

from blond3._core.beam.base import BeamBaseClass
from blond3.physics.feedbacks.beam_feedback import Blond2BeamFeedback
from blond3.physics.profiles import ProfileBaseClass


class PsbBeamFeedback(Blond2BeamFeedback):
    def __init__(
        self,
        profile: ProfileBaseClass,
        PL_gain: float,
        RL_gain: Tuple[float, float] = (0.0, 0.0),
        period: float = 10.0e-6,
        coefficients: Tuple[float, ...] = (
            0.999019,
            -0.999019,
            0.0,
            1.0,
            -0.998038,
            0.0,
        ),
        window_coefficient: float = 0.0,
        time_offset: Optional[float] = None,
        delay: int = 0,
        section_index: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__(
            profile=profile,
            PL_gain=PL_gain,
            window_coefficient=window_coefficient,
            time_offset=time_offset,
            delay=delay,
            section_index=section_index,
            name=name,
        )

        self.gain = self.gain * np.ones(ring.n_turns + 1)

        #: | *Radial loop gain, proportional [1] and integral [1/s].*
        self.gain2 = list(RL_gain)

        self.gain2[0] = self.gain2[0] * np.ones(ring.n_turns + 1)
        self.gain2[1] = self.gain2[1] * np.ones(ring.n_turns + 1)

        #: | *Optional: PL & RL acting only in certain time intervals/turns.*
        self.dt = 0
        # | *Phase Loop sampling period [s]*
        self.dt = period

        # Counter of turns passed since last time the PL was active
        self.PL_counter = 0
        self.on_time = np.array([])

        self.precalculate_time(ring)

        #: | *Array of transfer function coefficients.*
        self.coefficients = list(coefficients)

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

    def precalculate_time(self, ring: Ring):
        """
        *For machines like the PSB, where the PL acts only in certain time
        intervals, pre-calculate on which turns to act.*
        """

        if self.dt > 0:
            n = self.delay + 1
            while n < ring.t_rev.size:
                summa = 0
                while summa < self.dt:
                    try:
                        summa += ring.t_rev[n]
                        n += 1
                    except Exception:
                        self.on_time = np.append(self.on_time, 0)
                        return
                self.on_time = np.append(self.on_time, n - 1)
        else:
            self.on_time = np.arange(ring.t_rev.size)

    def update_domega_rf(self, beam: BeamBaseClass) -> None:
        # Average phase error while frequency is updated
        self.update_phi_beam()
        self.update_dphi(beam=beam)

        self.dphi_sum += self.dphi

        # Phase and radial loop active on certain turns
        if current_turn == self.on_time[self.PL_counter] and current_turn >= self.delay:
            # Phase loop
            self.dphi_av = self.dphi_sum / (
                self.on_time[self.PL_counter] - self.on_time[self.PL_counter - 1]
            )

            if self.RFnoise is not None:
                self.dphi_av += self.RFnoise.dphi[current_turn]

            self.domega_PL = 0.99803799 * self.domega_PL + self.gain[current_turn] * (
                0.99901903 * self.dphi_av - 0.99901003 * self.dphi_av_prev
            )

            self.dphi_av_prev = self.dphi_av
            self.dphi_sum = 0.0

            # Radial loop
            self.dR_over_R = (self._parent_cavity.delta_omega_rf[0]) / (
                self._parent_cavity._omega_rf[0]
                * (1.0 / (self.ring.alpha_0[0] * self._parent_cavity.gamma**2) - 1.0)
            )

            self.domega_RL = (
                self.domega_RL
                + self.gain2[0][current_turn] * (self.dR_over_R - self.dR_over_R_prev)
                + self.gain2[1][current_turn] * self.dR_over_R
            )

            self.dR_over_R_prev = self.dR_over_R

            # Counter to pick the next time step when the PL & RL will be active
            self.PL_counter += 1

        # Apply frequency correction
        self.domega_rf = -self.domega_PL - self.domega_RL

    def track(self, beam: BeamBaseClass) -> None:
        """
        Phase and radial loops for PSB. See documentation on-line for details.
        """

        self.update_domega_rf(beam=beam)
