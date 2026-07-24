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
    from blond.core.beam.base import BeamBaseClass
    from blond.physics.profiles import ProfileBaseClass


class PSBeamControl(BeamFeedbackBase):
    r"""
    Class for the PS beam control.

    Beam Feedback subclass for the PS. Implements 2 loops with their respective gains. Each loop can be
    deactivated by setting the respective gain to 0.

    1. Phase Loop using PL_gain, using the phase difference
    between beam and RF (actual synchronous phase). The transfer function is

    .. math::
        \Delta \omega_{rf}^{PL} = - g_{PL} \Delta \varphi_{out}

    with the transfered phase being calculated as

    .. math::
        \Delta \varphi_{out} = g_{diff} (\Delta\varphi_{PL} - \Delta \varphi_{prev}) + g_{int} \Delta \varphi_{out,prev}

    where the phase difference :math:`\Delta\varphi_{PL}` is calculated as

    .. math::
        \Delta\varph_{PL} = \varphi_{beam} - (\varphi_{RF}+\varphi_{programmed offset})

    2. Radial loop using RL_gain: a radial loop to remove
    long-term frequency drifts:

    .. math::
        \Delta \omega_{rf}^{RL} =  g_{RL} \Delta \rho_{out} ,

    with

    .. math::
        \Delta \rho_{out} = (1-g_{internal}) \Delta \rho + g_{internal} \Delta \rho_{prev}

    Parameters
    ----------
    profile
        Any Profile object which exposes the x- and y-axis of the beam line density.
    phase_noise
        Option to add phase noise through the beam control.
    sample_de
        Determines downsampling of macroparticles for mean energy calculation.
        Every <sample_dE>. particle is sampled.
    pl_gain
        The gain of the beam-phase loop [rad/s].
        Use ``beam_control.schedule("pl_gain", ...)`` to influence
        the parameter along the simulated cycle.
    rl_gain
        The gain of the radial loop [1/m s].
        Use ``beam_control.schedule("rl_gain", ...)`` to influence
        the parameter along the simulated cycle.
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
    **kwargs
        Variable keyword arguments for the `BeamFeedbackBase`.

    Attributes
    ----------
    domega_dphi
        Angular frequency correction [rad/s] from the beam-phase loop.
    domega_dr
        Angular frequency correction [rad/s] from the radial loop.
    """

    def __init__(
        self,
        profile: ProfileBaseClass,
        phase_noise=None,
        sample_de: int = 1,
        pl_gain: float = 0,
        rl_gain: float = 0,
        gd_pl: float = 5.704,
        gi_pl: float = 1 - 8.66e-5,
        g_rl: float = 1 - 1.853e-1,
        radial_reference: float = 0,
        initialize_steady_state: bool = True,
        prev_in_phase: float = 0,
        prev_out_phase: float = 0,
        prev_out_radial: float = 0,
        **kwargs,
    ):
        super().__init__(
            profile=profile,
            sample_de=sample_de,
            phase_noise=phase_noise,
            **kwargs,
        )

        self.pl_gain = pl_gain
        self.rl_gain = rl_gain

        self.gi_pl = gi_pl
        self.gd_pl = gd_pl
        self.g_rl = g_rl

        self.radial_reference = radial_reference

        self._initialize_steady_state = initialize_steady_state

        # set initial gains to zero steady state is desired.
        # If steady state is chosen _prev_in_phase gets overritten on the first turn to match the beam
        # _prev_out_radial is just set to zero under the assumption that this is the steady state case
        if initialize_steady_state:
            self._prev_in_phase = 0
            self._prev_out_phase = 0
            self._prev_out_radial = 0
        else:  # put a use defined initial state
            self._prev_in_phase = prev_in_phase
            self._prev_out_phase = prev_out_phase
            self._prev_out_radial = prev_out_radial

        self.domega_dphi = 0.0
        self.domega_dr = 0.0

        self._register_schedulable_variables("pl_gain", "rl_gain")

    def calculate_offsets(self, beam: BeamBaseClass):
        """
        Calculate the offsets of the beam.

        Parameters
        ----------
        beam
            A beam object to extract the beam attribute from.
        """
        self.update_phase_error()
        self.radial_difference(beam=beam)

    def update_beam_attributes(self, beam: BeamBaseClass):
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
        self.calculate_offsets(beam=beam)

    def update_frequency_correction(self, beam: BeamBaseClass):
        """
        Calculate the frequency correction from the beam control.

        This method implements the feedback systems in the LHC beam control, i.e.
        the beam-phase loop and the synchronization loop.

        Parameters
        ----------
        beam
            A beam object to extract the beam attribute from.
        """
        counter = self._simulation.turn_counter.value

        if (
            self._initialize_steady_state and counter == 1
        ):  # We are assuming we are initially in a steady state
            self._prev_in_phase = self.dphi

        # Frequency correction from phase loop
        dphi_out = (
            self.gd_pl * (self.dphi - self._prev_in_phase)
            + self.gi_pl * self._prev_out_phase
        )
        self.domega_dphi = -self.pl_gain * dphi_out
        self._prev_in_phase = self.dphi
        self._prev_out_phase = dphi_out

        # Frequency correction from radial loop
        drho_out = (
            1 - self.g_rl
        ) * self.drho + self.g_rl * self._prev_out_radial

        # Condition for stable gain on radial loop is dependent on whether we are below or above transition
        self.domega_dr = -self.rl_gain * drho_out

        self._prev_out_radial = drho_out

        self.delta_omega_rf = self.domega_dphi + self.domega_dr
