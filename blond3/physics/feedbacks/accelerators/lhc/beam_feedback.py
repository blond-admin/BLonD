from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from ...beam_feedback import Blond2BeamFeedback

if TYPE_CHECKING:  # pragma: no cover
    from ....._core.beam.base import BeamBaseClass
    from ....profiles import ProfileBaseClass
    from ....._core.simulation.simulation import Simulation
    from typing import Optional


class LhcBeamFeedback(Blond2BeamFeedback):
    def __init__(
        self,
        profile: ProfileBaseClass,
        PL_gain: float,
        window_coefficient: float = 0.0,
        SL_gain=0.0,
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
        #: | *Synchronisation loop gain.*
        self.gain2 = SL_gain

        #: | *LHC Synchroronisation loop recursion variable*
        self.lhc_y = 0

        if self.gain2 != 0:
            #: | *LHC Synchronisation loop coefficient [1]*
            self.lhc_a = 5.25 - self._parent_cavity.omega_s0 / (np.pi * 40.0)
            #: | *LHC Synchronisation loop time constant [turns]*
            self.lhc_t = (
                2 * np.pi * self._parent_cavity.Q_s * np.sqrt(self.lhc_a)
            ) / np.sqrt(
                1
                + self.gain
                / self.gain2
                * np.sqrt((1 + 1 / self.lhc_a) / (1 + self.lhc_a))
            )

        else:
            self.lhc_a = np.zeros(self._parent_cavity.n_turns + 1)
            self.lhc_t = np.zeros(self._parent_cavity.n_turns + 1)

    def track(self, beam: BeamBaseClass) -> None:
        """
        Calculation of the LHC RF frequency correction from the phase difference
        between beam and RF (actual synchronous phase). The transfer function is

        .. math::
            \\Delta \\omega_{rf}^{PL} = - g_{PL} (\\Delta\\varphi_{PL} + \\phi_{N})

        where the phase noise for the controlled blow-up can be optionally
        activated.
        Using 'gain2', a synchro loop can be activated in addition to remove
        long-term frequency drifts:

        .. math::
            \\Delta \\omega_{rf}^{SL} = - g_{SL} (y + a \\Delta\\varphi_{rf}) ,

        where we use the recursion

        .. math::
            y_{n+1} = (1 - \\tau) y_n + (1 - a) \\tau \\Delta\\varphi_{rf} ,

        with a and \tau being defined through the synchrotron frequency f_s and
        the synchrotron tune Q_s as

        .. math::
            a (f_s) \\equiv 5.25 - \\frac{f_s}{\\pi 40~\\text{Hz}} ,

        .. math::
            \\tau(f_s) \\equiv 2 \\pi Q_s \\sqrt{ \\frac{a}{1 + \\frac{g_{PL}}{g_{SL}} \\sqrt{\\frac{1 + 1/a}{1 + a}} }}
        """

        self.update_domega_rf(beam=beam)

    def update_domega_rf(self, beam: BeamBaseClass) -> None:
        dphi_rf = self._parent_cavity.delta_phi_rf[0]
        self.update_phi_beam()
        self.update_dphi(beam=beam)
        # Frequency correction from phase loop and synchro loop
        self.domega_rf = -self.gain * self.dphi - self.gain2 * (
            self.lhc_y + self.lhc_a[current_turn] * (dphi_rf + self.reference)
        )
        # Update recursion variable
        self.lhc_y = (1 - self.lhc_t[current_turn]) * self.lhc_y + (
            1 - self.lhc_a[current_turn]
        ) * self.lhc_t[current_turn] * (dphi_rf + self.reference)


class LhcFBeamFeedback(Blond2BeamFeedback):
    def __init__(
        self,
        profile: ProfileBaseClass,
        PL_gain: float,
        FL_gain: float = 0.0,
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

        #: | *Frequency loop gain.*
        self.gain2 = FL_gain

    def track(self, beam: BeamBaseClass) -> None:
        """
        Calculation of the LHC RF frequency correction from the phase difference
        between beam and RF (actual synchronous phase). The transfer function is

        .. math::
            \\Delta \\omega_{rf}^{PL} = - g_{PL} (\\Delta\\varphi_{PL} + \\phi_{N})

        where the phase noise for the controlled blow-up can be optionally
        activated.
        Using 'gain2', a frequency loop can be activated in addition to remove
        long-term frequency drifts:

        .. math::
            \\Delta \\omega_{rf}^{FL} = - g_{FL} (\\omega_{rf} - h \\omega_{0})
        """

        self.update_domega_rf(beam=beam)

    def update_domega_rf(self, beam: BeamBaseClass) -> None:
        self.update_phi_beam()
        self.update_dphi(beam=beam)

        # Frequency correction from phase loop and frequency loop
        self.domega_rf = -self.gain * self.dphi - self.gain2 * (
            self._parent_cavity.delta_omega_rf[0] + self.reference
        )


class SpsRlBeamFeedback(Blond2BeamFeedback):
    def __init__(
        self,
        profile: ProfileBaseClass,
        PL_gain: float,
        window_coefficient: float = 0.0,
        RL_gain: float = 0.0,
        sample_dE: int = 1,
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
        #: | *Frequency loop gain.*
        self.gain2 = RL_gain

        #: | *Number of particles to sample from dE for orbit calculation*
        self.sample_dE = sample_dE

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        from ..... import DriftSimple  # prevent cyclic import

        self._simulation = simulation  # todo declare
        self._drift = simulation.ring.elements.get_element(DriftSimple)

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        self.alpha_0 = self._drift.alpha_0
        self.beta = beam.reference_beta
        self.energy = beam.reference_total_energy

    def track(self, beam: BeamBaseClass) -> None:
        """
        Calculation of the SPS RF frequency correction from the phase difference
        between beam and RF (actual synchronous phase). The transfer function is

        .. math::
            \\Delta \\omega_{rf}^{PL} = - g_{PL} (\\Delta\\varphi_{PL} + \\phi_{N})

        where the phase noise for the controlled blow-up can be optionally
        activated.
        Using 'gain2', a radial loop can be activated in addition to remove
        long-term frequency drifts
        """

        self.update_domega_rf(beam=beam)

    def update_domega_rf(self, beam: BeamBaseClass) -> None:
        if self.reference != 0:
            self.radial_steering_from_freq()

        self.update_phi_beam()
        self.update_dphi(beam=beam)
        self.radial_difference(beam=beam)

        eta_0 = self._drift.eta_0(gamma=beam.reference_gamma)
        # Frequency correction from phase loop and radial loop
        self.domega_dphi = -self.gain * self.dphi  # TODO declare
        self.domega_dR = (
            -np.sign(eta_0)
            * self.gain2
            * (self.reference - self.drho)
            / (self._simulation.ring.circumference / (2 * np.pi))
        )

        self.domega_rf = self.domega_dphi + self.domega_dR

    def radial_difference(self, beam: BeamBaseClass):
        """
        *Radial difference between beam and design orbit.*
        """
        self.average_dE = np.mean(beam._dE[:: self.sample_dE])  # todo other access
        # FIXME why is all clipped to the first turn????
        self.drho = (
            self.alpha_0  # self._drift.alpha_0[0]
            * (self._simulation.ring.circumference / (2 * np.pi))
            * self.average_dE
            # / (self.ring.beta[0] ** 2.0 * self.ring.energy[0])
            / (self.beta**2.0 * self.energy)
        )

    def radial_steering_from_freq(self):
        """
        *Frequency and phase change for the current turn due to the radial steering program.*
        """
        raise NotImplementedError(
            "BLonD2 port that was already broken."
            " Who wants to use"
            " this code must fix this code."
        )
        self.radial_steering_domega_rf = (
            -self._parent_cavity._omega_rf[0]
            * self._parent_cavity.eta_
            / self.ring.alpha_0[0]
            * self.reference
            / self.ring.ring_radius
        )

        self._parent_cavity.delta_omega_rf += (
            self.radial_steering_domega_rf
            * self._parent_cavity.harmonic[:]
            / self._parent_cavity.harmonic[0]
        )

        # Update the RF phase of all systems for the next turn
        # Accumulated phase offset due to PL in each RF system
        # FIXME dphi_rf_steering never declared, this will crash
        self._parent_cavity.dphi_rf_steering += (
            (2.0 * np.pi)
            * (self._parent_cavity.harmonic[:] / self._parent_cavity._omega_rf[:])
            * (self._parent_cavity.delta_omega_rf[:])
        )

        # Total phase offset
        self._parent_cavity.delta_phi_rf[:] += self._parent_cavity.dphi_rf_steering


class SpsFBeamFeedback(Blond2BeamFeedback):
    def __init__(
        self,
        profile: ProfileBaseClass,
        PL_gain: float,
        FL_gain: float = 0.0,
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
        #: | *Frequency loop gain.*
        self.gain2 = FL_gain

    def track(self, beam: BeamBaseClass) -> None:
        """
        Calculation of the SPS RF frequency correction from the phase
        difference between beam and RF (actual synchronous phase). Same as
        LHC_F, except the calculation of the beam phase.
        """

        self.update_domega_rf(beam=beam)

    def update_domega_rf(self, beam: BeamBaseClass) -> None:
        self.beam_phase_sharpWindow()
        self.update_dphi(beam=beam)

        # Frequency correction from phase loop and frequency loop
        self.domega_dphi = -self.gain * self.dphi
        self.domega_df = -self.gain2 * (self._parent_cavity.delta_omega_rf[0])

        self.domega_rf = self.domega_dphi + self.domega_df

    def beam_phase_sharpWindow(self):
        """
        *Beam phase measured at the main RF frequency and phase. The beam is
        averaged over a window. The coefficients of sine and cosine components
        determine the beam phase, projected to the range -Pi/2 to 3/2 Pi.
        Note that this beam phase is already w.r.t. the instantaneous RF phase.*
        """

        # Main RF frequency at the present turn
        omega_rf = (
            self._parent_cavity._omega_rf[0] + self._parent_cavity.delta_omega_rf[0]
        )
        phi_rf = self._parent_cavity.phi_rf[0] + self._parent_cavity.delta_phi_rf

        if self.alpha != 0.0:
            indexes = np.logical_and(
                (self.time_offset - np.pi / omega_rf) <= self.profile.hist_x,
                self.profile.hist_x
                <= (-1 / self.alpha + self.time_offset - 2 * np.pi / omega_rf),
            )
        else:
            indexes = np.ones(self.profile.n_bins, dtype=bool)

        # Convolve with window function
        scoeff = np.trapezoid(
            np.sin(omega_rf * self.profile.hist_x[indexes] + phi_rf)
            * self.profile.hist_y[indexes],
            dx=self.profile.hist_step,
        )
        ccoeff = np.trapezoid(
            np.cos(omega_rf * self.profile.hist_x[indexes] + phi_rf)
            * self.profile.hist_y[indexes],
            dx=self.profile.hist_step,
        )

        # Project beam phase to (pi/2,3pi/2) range
        self.phi_beam = np.arctan(scoeff / ccoeff) + np.pi


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
