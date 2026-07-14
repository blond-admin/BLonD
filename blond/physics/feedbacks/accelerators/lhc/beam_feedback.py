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

from blond.acc_math.analytic.hamilton import (
    calc_synchrotron_tune_single_harmonic,
)
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
    phase_noise
        Option to add phase noise through the beam control.
    pl_gain
        The gain of the beam-phase loop.
        Use ``beam_control.schedule("pl_gain", ...)`` to influence
        the parameter along the simulated cycle.
    sl_gain
        The gain of the synchronization loop.
        Use ``beam_control.schedule("sl_gain", ...)`` to influence
        the parameter along the simulated cycle.
    current_thres
        Beam current threshold for gating of the profiles.
    **kwargs
        Variable keyword arguments for the `BeamFeedbackBase`.

    Attributes
    ----------
    lhc_y
        Recursion variable for the synchronization loop.
    lhc_a
        The synchronization loop coefficient.
    lhc_t
        The synchronization loop time constant.
    reference
        Reference phase [rad] for the synchronization loop.
    """

    def __init__(
        self,
        profile: ProfileBaseClass,
        phase_noise=None,  # TODO refine when phase noise is merged
        pl_gain: float = 0.0,
        sl_gain: float = 0.0,
        current_thres: float | None = None,
        **kwargs,
    ):
        super().__init__(profile=profile, phase_noise=phase_noise, **kwargs)

        self.pl_gain = pl_gain
        self.sl_gain = sl_gain

        self.lhc_y = 0.0

        self.lhc_a = 0.0
        self.lhc_t = 0.0

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
            and self._main_cavities[0].any_feedback_not_none
        ):
            raise RuntimeError(
                "The filled slots in the machine is needed to compute the cavity sum phase"
            )

        if self.sl_gain != 0:
            self.calculate_synchro_coefficients(beam=beam)

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
        dphi_rf = self._main_cavities[0].delta_phi_rf

        self.phase_difference(phase_noise=self.phase_noise)
        self.cavity_sum_phase(self.current_thres)

        # Take into account the synchronous phase
        self.dphi = (
            self.dphi
            + np.pi
            - self._main_cavities[0].calc_phi_s_main_harmonic(beam)
        )

        # Frequency correction from phase loop and synchro loop
        self.delta_omega_rf = -self.pl_gain * self.dphi - self.sl_gain * (
            self.lhc_y + self.lhc_a * (dphi_rf + self.reference)
        )

        # Update recursion variable
        self.lhc_y = (1 - self.lhc_t) * self.lhc_y + (
            1 - self.lhc_a
        ) * self.lhc_t * (dphi_rf + self.reference)

    def calculate_synchro_coefficients(self, beam: BeamBaseClass):
        """
        Calculate the coefficients for the LHC synchronization loop.

        Parameters
        ----------
        beam
            A beam object to extract the beam attribute from.
        """
        voltages = self.get_from_all_rf_stations(
            "get_main_harmonic_voltage", self._main_cavities
        )

        Q_s0 = calc_synchrotron_tune_single_harmonic(
            charge=beam.particle_type.charge,
            voltage=np.sum(voltages),
            beta=beam.reference.beta,
            energy=beam.reference.total_energy,
            phi_s=np.pi,
            harmonic=self._main_cavities[0].get_main_harmonic(),
            eta_0=self._simulation.ring.calc_average_eta_0(
                beam.reference.gamma
            ),
        )

        omega_rf = self._main_cavities[0].get_main_harmonic_omega_rf_design()

        harm = self._main_cavities[0].get_main_harmonic()

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
