# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Implementation of the LHC beam control.

Notes
-----
Authors:
Helga Timko
Birk Emil Karlsen-Bæck
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


class LHCBeamControl(BeamFeedbackBase):
    r"""
    Class for the LHC beam control.

    This class implements the feedbacks present in the beam
    control of the rf system in the Large Hadron Collider.

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

    Parameters
    ----------
    profile
        Any Profile object which exposes the x- and y-axis of the beam line density.
    pl_gain
        The gain of the beam-phase loop.
    sl_gain
        The gain of the synchronization loop.
    delay
        Delay (in units of turns) of the initial correction of the feedback system.
    window_coefficient
        Window coefficient for the calculation of the beam phase. This parameter will
        reduce the weight of later samples of the beam profile.
    time_offset
        Time offset for the calculation of the beam phase.
    phase_noise
        Option to add phase noise through the beam control.
    current_thres
        Beam current threshold for gating of the profiles.
    """

    def __init__(
        self,
        profile: ProfileBaseClass,
        pl_gain: float,
        sl_gain: float,
        delay: int = 0,
        window_coefficient: float = 0.0,
        time_offset: float | None = None,
        phase_noise=None,
        current_thres: float = None,
    ):
        super().__init__(
            profile=profile,
            delay=delay,
            window_coefficient=window_coefficient,
            time_offset=time_offset,
            phase_noise=phase_noise,
        )

        self.pl_gain = pl_gain
        self.sl_gain = sl_gain

        self.lhc_y = 0

        self.domega_rf = 0.0
        self.dphi = 0.0
        self.reference = 0.0
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

        if self.sl_gain != 0:
            Q_s0 = self.cavities[0].calc_synchrotron_tune_main_harmonic(
                beam,
                np.pi,
                simulation.ring.calc_average_eta_0(beam.reference.gamma),
            ) * np.ones(n_turns + 1)

            omega_rf = self.cavities[
                0
            ].get_main_harmonic_omega_rf_design() * np.ones(n_turns + 1)

            harm = self.cavities[0].get_main_harmonic()

            omega_s0 = Q_s0 * omega_rf / harm

            #: | *LHC Synchronisation loop coefficient [1]*
            self.lhc_a = 5.25 - omega_s0 / (np.pi * 40.0)
            #: | *LHC Synchronisation loop time constant [turns]*
            self.lhc_t = (2 * np.pi * Q_s0 * np.sqrt(self.lhc_a)) / np.sqrt(
                1
                + self.pl_gain
                / self.sl_gain
                * np.sqrt((1 + 1 / self.lhc_a) / (1 + self.lhc_a))
            )
        else:
            self.lhc_a = np.zeros(n_turns + 1)
            self.lhc_t = np.zeros(n_turns + 1)

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
        dphi_rf = self.cavities[0].delta_phi_rf

        self.phase_difference(phase_noise=self.phase_noise)
        self.cavity_sum_phase(self.current_thres)

        # Take into account the synchronous phase
        self.dphi = (
            self.dphi
            + np.pi
            - self.main_cavities[0].calc_phi_s_main_harmonic(beam)
        )

        # Frequency correction from phase loop and synchro loop
        self.domega_rf = -self.pl_gain * self.dphi - self.sl_gain * (
            self.lhc_y + self.lhc_a[counter] * (dphi_rf + self.reference)
        )

        # Update recursion variable
        self.lhc_y = (1 - self.lhc_t[counter]) * self.lhc_y + (
            1 - self.lhc_a[counter]
        ) * self.lhc_t[counter] * (dphi_rf + self.reference)
