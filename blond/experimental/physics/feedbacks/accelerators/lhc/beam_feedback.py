# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import backend
from blond.experimental.physics.feedbacks.beam_feedback import (
    BeamFeedbackBase,
    Blond2BeamFeedback,
)

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation
    from blond.physics.profiles import ProfileBaseClass


class LHCBeamControl(BeamFeedbackBase):
    def __init__(
        self,
        profile: ProfileBaseClass,
        pl_gain: float,
        sl_gain: float,
        *args,
        **kwargs,
    ):
        super().__init__(profile=profile, *args, **kwargs)

        self.pl_gain = pl_gain
        self.sl_gain = sl_gain

        self.lhc_y = 0

        self.domega_rf = 0.0
        self.dphi = 0.0
        self.reference = 0.0

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs,
    ) -> None:
        """Hook called when simulation.run_simulation starts."""
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
            **kwargs,
        )

        if self.sl_gain != 0:
            Q_s0 = self.cavities[0].calc_synchrotron_tune_single_harmonic(
                beam,
                np.pi,
                simulation.ring.calc_average_eta_0(beam.reference.gamma),
            ) * np.ones(n_turns + 1)

            omega_rf = self.cavities[0].get_main_harmonic_omega_rf_design(
                beam.reference.beta, simulation.ring.circumference
            ) * np.ones(n_turns + 1)

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
        self.beam_phase()

    def apply_corrections(self, beam: BeamBaseClass):
        counter = self.cavities[0]._turn_i.value
        dphi_rf = self.cavities[0].delta_phi_rf

        self.phase_difference(beam)

        # Frequency correction from phase loop and synchro loop
        self.domega_rf = -self.pl_gain * self.dphi - self.sl_gain * (
            self.lhc_y + self.lhc_a[counter] * (dphi_rf + self.reference)
        )

        # Update recursion variable
        self.lhc_y = (1 - self.lhc_t[counter]) * self.lhc_y + (
            1 - self.lhc_a[counter]
        ) * self.lhc_t[counter] * (dphi_rf + self.reference)


class LhcBeamFeedback(Blond2BeamFeedback):
    def __init__(
        self,
        profile: ProfileBaseClass,
        PL_gain: float,
        window_coefficient: float = 0.0,
        SL_gain=0.0,
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
        #: | *Synchronisation loop gain.*
        self.gain2 = SL_gain

        #: | *LHC Synchroronisation loop recursion variable*
        self.lhc_y = 0

        if self.gain2 != 0:
            #: | *LHC Synchronisation loop coefficient [1]*
            self.lhc_a = 5.25 - self._parent_rf_station.omega_s0 / (
                np.pi * 40.0
            )
            #: | *LHC Synchronisation loop time constant [turns]*
            self.lhc_t = (
                2 * np.pi * self._parent_rf_station.Q_s * np.sqrt(self.lhc_a)
            ) / np.sqrt(
                1
                + self.gain
                / self.gain2
                * np.sqrt((1 + 1 / self.lhc_a) / (1 + self.lhc_a))
            )

        else:
            self.lhc_a = np.zeros(self._parent_rf_station.n_turns + 1)
            self.lhc_t = np.zeros(self._parent_rf_station.n_turns + 1)

    def track(self, beam: BeamBaseClass) -> None:
        """Calculation of the LHC RF frequency correction from the phase difference
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
        dphi_rf = self._parent_rf_station.delta_phi_rf[0]
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
        """Calculation of the LHC RF frequency correction from the phase difference
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
            self._parent_rf_station.delta_omega_rf[0] + self.reference
        )
