# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Implementation of the PS beam control.

Notes
-----
Authors:
Oleksandr Naumenko
Oliver Muller Smedt
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from blond.physics.feedbacks.beam_feedback import (
    BeamFeedbackBase,
)

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation
    from blond.physics.profiles import ProfileBaseClass


class PSBeamControl(BeamFeedbackBase):
    r"""
    Class for the PS beam control.

    Beam Feedback subclass for the PS. Implements 2 loops with their respective gains. Each loop can be
    deactivated by setting the respective gain to 0.

    1. Phase Loop using PL_gain, using the phase difference
    between beam and RF (actual synchronous phase). The transfer function is

    .. math::
        \\Delta \\omega_{rf}^{PL} = - g_{PL} \\Delta \\varphi_{out}

    with the transfered phase being calculated as

    .. math::
        \\Delta \\varphi_{out} = g_{diff} (\\Delta\\varphi_{PL} - \\Delta \\varphi_{prev}) + g_{int} \\ Delta \\varphi_{out,prev}

    where the phase difference :math: \\Delta\\varphi_{PL} is calculated as

    .. math::
        \\Delta\\varph_{PL} = \\varphi_{beam} - (\\varphi_{RF}+\\varphi_{programmed offset})

    2. Radial loop using RL_gain: a radial loop to remove
    long-term frequency drifts:

    .. math::
        \\Delta \\omega_{rf}^{RL} =  g_{RL} \\Delta \\rho_{out} ,

    with

    .. math::
        \\Delta \\rho_{out} = (1-g_{internal}) \\Delta \\rho + g_{internal} \\Delta \\rho_{prev}

    Parameters
    ----------
    profile
        Any Profile object which exposes the x- and y-axis of the beam line density.
    below_transition
        Array of values for whether the current turns are above or below transition energy.
    pl_gain
        The gain of the beam-phase loop [rad/s].
    rl_gain
        The gain of the radial loop [1/m s].
    delay
        Delay (in units of turns) of the initial correction of the feedback system.
    window_coefficient
        Window coefficient for the calculation of the beam phase. This parameter will
        reduce the weight of later samples of the beam profile.
    time_offset
        Time offset for the calculation of the beam phase.
    sample_de
        Determines downsampling of macroparticles for mean energy calculation.
        Every <sample_dE>. particle is sampled.
    phase_noise
        Option to add phase noise through the beam control.
    gd_pl
        Hardware determined differential gain of the phase loop.
    gi_pl
        Hardware determined integral gain of the phase loop.
    g_rl
        Hardware determined radial loop gain parameter.
    radial_reference
        Reference radial offset for the radial loop.
    initialize_steady_state
        Option to initialize in steady state (True) or with values in the feedbacks.
    prev_in_phase
        Initial value for phase input of the feedback.
    prev_out_phase
        Initial value for the phase output of the feedback.
    prev_out_radial
        Initial value for the radial output of the feedback.
    """

    def __init__(
        self,
        profile: ProfileBaseClass,
        below_transition: ArrayLike,
        pl_gain: float = 0,
        rl_gain: float = 0,
        delay: int = 0,
        window_coefficient: float = 0.0,
        time_offset: float | None = None,
        sample_de: int = 1,
        phase_noise=None,
        gd_pl: float = 5.704,
        gi_pl: float = 1 - 8.66e-5,
        g_rl: float = 1 - 1.853e-1,
        radial_reference: float = 0,
        initialize_steady_state: bool = True,
        prev_in_phase: float | None = 0,
        prev_out_phase: float | None = 0,
        prev_out_radial: float | None = 0,
    ):
        super().__init__(
            profile=profile,
            delay=delay,
            window_coefficient=window_coefficient,
            time_offset=time_offset,
            sample_de=sample_de,
            phase_noise=phase_noise,
        )

        self.below_transition = below_transition

        self.pl_gain = pl_gain
        self.rl_gain = rl_gain

        self.gi_pl = gi_pl
        self.gd_pl = gd_pl
        self.g_rl = g_rl

        self.domega_rf = 0.0
        self.dphi = 0.0
        self.reference = 0.0

        self.radial_reference = radial_reference

        self.initialize_steady_state = initialize_steady_state

        # set initial gains to zero steady state is desired.
        # If steady state is chosen prev_in_phase gets overritten on the first turn to match the beam
        # prev_out_radial is just set to zero under the assumption that this is the steady state case
        if initialize_steady_state:
            self.prev_in_phase = 0
            self.prev_out_phase = 0
            self.prev_out_radial = 0
        else:  # put a use defined initial state
            self.prev_in_phase = prev_in_phase
            self.prev_out_phase = prev_out_phase
            self.prev_out_radial = prev_out_radial

        self.domega_dphi: float = 0.0
        self.domega_dr: float = 0.0

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

    def calculate_offsets(self, beam: BeamBaseClass):
        """
        Calculate the offsets of the beam.

        Parameters
        ----------
        beam
            A beam object to extract the beam attribute from.
        """
        self.phase_difference()
        self.radial_difference(beam=beam)

    def get_beam_attribute(self, beam: BeamBaseClass):
        """
        Calculate the beam phase.

        This method implements the measurement of the beam, which is the
        input for the feedback. In the case of the LHC beam control, this
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

        This method implements the feedback systems in the LHC beam control, i.e.
        the beam-phase loop and the synchronization loop.

        Parameters
        ----------
        beam
            A beam object to extract the beam attribute from.
        """
        counter = self.cavities[0]._turn_counter.value

        self.calculate_offsets(beam=beam)

        if (
            self.initialize_steady_state and counter == 1
        ):  # We are assuming we are initially in a steady state
            self.prev_in_phase = self.dphi

        # Frequency correction from phase loop
        dphi_out = (
            self.gd_pl * (self.dphi - self.prev_in_phase)
            + self.gi_pl * self.prev_out_phase
        )
        self.domega_dphi = -self.pl_gain * dphi_out
        self.prev_in_phase = self.dphi
        self.prev_out_phase = dphi_out

        # Frequency correction from radial loop
        drho_out = (
            1 - self.g_rl
        ) * self.drho + self.g_rl * self.prev_out_radial

        # Condition for stable gain on radial loop is dependent on whether we are below or above transition
        if self.below_transition[counter]:
            self.domega_dr = -self.rl_gain * drho_out
        else:
            self.domega_dr = self.rl_gain * drho_out

        self.prev_out_radial = drho_out

        self.domega_rf = self.domega_dphi + self.domega_dr
