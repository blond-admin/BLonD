from __future__ import annotations

from typing import Optional

import numpy as np

from blond3 import Simulation
from blond3._core.beam.base import BeamBaseClass
from blond3.physics.feedbacks.beam_feedback import Blond2BeamFeedback
from blond3.physics.profiles import ProfileBaseClass


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
