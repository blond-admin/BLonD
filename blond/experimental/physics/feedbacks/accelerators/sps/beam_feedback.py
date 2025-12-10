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
    from typing import Any
    from numpy.typing import NDArray as NumpyArray

import numpy as np

from blond import Simulation
from blond.core.backends.backend import backend
from blond.core.beam.base import BeamBaseClass
from blond.experimental.physics.feedbacks.beam_feedback import (
    BeamFeedbackBase,
    Blond2BeamFeedback,
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
        turn_i_init: int,
        **kwargs,
    ) -> None:
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
            turn_i_init=turn_i_init,
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
            / self.cavities[0].get_main_harmonic_omega_rf_design(
                beam.reference_beta, self.cavities[0]._ring.circumference
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


class SpsRlBeamFeedback(Blond2BeamFeedback):
    def __init__(
        self,
        profile: ProfileBaseClass,
        PL_gain: float,
        window_coefficient: float = 0.0,
        RL_gain: float = 0.0,
        sample_dE: int = 1,
        time_offset: float | None = None,
        delay: int = 0,
        section_index: int = 0,
        name: str | None = None,
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
        """
        Lateinit method when `simulation.__init__` is called

        simulation
            `Simulation` context manager
        """
        from blond.physics.drifts import DriftSimple

        self._simulation = simulation  # todo declare
        self._drift = simulation.ring.elements.get_element(DriftSimple)

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        self.alpha_0 = self._drift.alpha_0
        self.beta = beam.reference_beta
        self.energy = beam.reference_total_energy

    def track(self, beam: BeamBaseClass) -> None:
        r"""
        Calculation of the SPS RF frequency correction from the phase difference
        between beam and RF (actual synchronous phase). The transfer function is

        .. math::
            \Delta \omega_{rf}^{PL} = - g_{PL} (\Delta\varphi_{PL} + \phi_{N})

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
        """Radial difference between beam and design orbit."""
        self.average_dE = np.mean(
            beam._dE[:: self.sample_dE]
        )  # todo other access
        # FIXME why is all clipped to the first turn????
        self.drho = (
            self.alpha_0  # self._drift.alpha_0[0]
            * (self._simulation.ring.circumference / (2 * np.pi))
            * self.average_dE
            # / (self.ring.beta[0] ** 2.0 * self.ring.energy[0])
            / (self.beta**2.0 * self.energy)
        )

    def radial_steering_from_freq(self):
        """Frequency and phase change for the current turn due to the radial steering program."""
        raise NotImplementedError(
            "BLonD2 port that was already broken."
            " Who wants to use"
            " this code must fix this code."
        )
        self.radial_steering_domega_rf = (
            -self._parent_rf_station._omega_rf[0]
            * self._parent_rf_station.eta_
            / self.ring.alpha_0[0]
            * self.reference
            / self.ring.ring_radius
        )

        self._parent_rf_station.delta_omega_rf += (
            self.radial_steering_domega_rf
            * self._parent_rf_station.harmonic[:]
            / self._parent_rf_station.harmonic[0]
        )

        # Update the RF phase of all systems for the next turn
        # Accumulated phase offset due to PL in each RF system
        # FIXME dphi_rf_steering never declared, this will crash
        self._parent_rf_station.dphi_rf_steering += (
            (2.0 * np.pi)
            * (
                self._parent_rf_station.harmonic[:]
                / self._parent_rf_station._omega_rf[:]
            )
            * (self._parent_rf_station.delta_omega_rf[:])
        )

        # Total phase offset
        self._parent_rf_station.delta_phi_rf[:] += (
            self._parent_rf_station.dphi_rf_steering
        )


class SpsFBeamFeedback(Blond2BeamFeedback):
    def __init__(
        self,
        profile: ProfileBaseClass,
        PL_gain: float,
        FL_gain: float = 0.0,
        window_coefficient: float = 0.0,
        time_offset: float | None = None,
        delay: int = 0,
        section_index: int = 0,
        name: str | None = None,
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
        self.domega_df = (
            -self.gain2 * (self._parent_rf_station.delta_omega_rf[0])
        )

        self.domega_rf = self.domega_dphi + self.domega_df

    def beam_phase_sharpWindow(self):
        """
        Beam phase measured at the main RF frequency and phase. The beam is
        averaged over a window. The coefficients of sine and cosine components
        determine the beam phase, projected to the range -Pi/2 to 3/2 Pi.
        Note that this beam phase is already w.r.t. the instantaneous RF phase.
        """
        # Main RF frequency at the present turn
        omega_rf = (
            self._parent_rf_station._omega_rf[0]
            + self._parent_rf_station.delta_omega_rf[0]
        )
        phi_rf = (
            self._parent_rf_station.phi_rf[0]
            + self._parent_rf_station.delta_phi_rf
        )

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
