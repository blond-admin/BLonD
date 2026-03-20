# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Base class for the implementation of beam-based rf feedback systems.

Notes
-----
Authors:
Helga Timko
Alexandre Lasheen
Birk Emil Karlsen-Bæck
Oleksandr Naumenko
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from blond import Simulation, backend
from blond.physics.feedbacks.base import (
    GlobalFeedback,
)

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass
    from blond.physics.cavities import RFStationBaseClass
    from blond.physics.profiles import ProfileBaseClass


class BeamFeedbackBase(GlobalFeedback):
    """
    Base class for beam-based rf feedback systems in synchrotron particle accelerators.

    This class is intended to come with the features common for most
    beam-based rf feedback systems. The concrete beam feedback for a specific
    synchrotron is meant to be a child class of this object.

    Parameters
    ----------
    profile
        Any Profile object which exposes the x- and y-axis of the beam line density.
    delay
        Delay (in unites of turns) of the initial correction of the feedback system.
    window_coefficient
        Window coefficient for the calculation of the beam phase.
    time_offset
        Time offset for the calculation of the beam phase.
    current_thres
        Beam current threshold for gating of the profiles.

    Attributes
    ----------
    domega_rf
        The frequency correction of the feedback.
    dphi
        The phase shift due to the frequency correction.
    phi_beam
        The beam phase [rad].
    """

    _parent_rf_station: RFStationBaseClass

    def __init__(
        self,
        profile: ProfileBaseClass,
        delay: int = 0,
        window_coefficient: float = 0.0,
        time_offset: float | None = None,
        current_thres=None,
    ):
        super().__init__(profile=profile)
        self.delay = delay
        self.window_coefficient = window_coefficient
        self.time_offset = time_offset
        self.current_thres = current_thres

        self.domega_rf = 0

        self.dphi: float = 0.0

        self.phi_beam: float = 0.0

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
        if (
            self.current_thres is None
            and self.cavities[0].any_feedback_not_none
        ):
            raise RuntimeError(
                "The filled slots in the machine is needed to compute the cavity sum phase"
            )

    @abstractmethod
    def get_beam_attribute(self, beam: BeamBaseClass):
        """
        Calculate the beam-based measurement.

        This method is intended to implement equivalent signal processing
        to the beam-based measurement done in the real beam-based feedback.
        This could, for instance, be the phase of the beam rf component, the
        radial position of the beam, etc.

        Parameters
        ----------
        beam
            A beam object to extract the beam attribute from.
        """
        # could be mean energy, mean phase or whatever
        pass

    @abstractmethod
    def compute_correction(self, beam: BeamBaseClass):
        """
        Calculate the frequency corrections from the feedback.

        This method is intended to implement the feedback itself and
        calculate the frequency corrections from the feedback based on
        the beam-based measurement.

        Parameters
        ----------
        beam
            A beam object to extract the beam attribute from.
        """
        # shift the RF station phase or so
        pass

    def beam_phase(self):
        """
        Calculate the phase of the rf component of the beam.

        This method uses the `backend.specials.beam_phase` function
        to calculate the rf component phase of the beam based on the
        profile object.
        """
        # Main RF frequency at the present turn
        omega_rf = self.cavities[0].get_main_harmonic_omega_rf()
        phi_rf = self.cavities[0].get_main_harmonic_phi_rf()

        if self.time_offset is None:
            coeff = backend.specials.beam_phase(
                self.profile.hist_x,
                self.profile.hist_y,
                self.window_coefficient,
                omega_rf,
                phi_rf,
                self.profile.hist_step,
            )
        else:
            indexes = self.profile.hist_x >= self.time_offset
            coeff = backend.specials.beam_phase(
                self.profile.hist_x[indexes],
                self.profile.hist_y,
                self.window_coefficient,
                omega_rf,
                phi_rf,
                self.profile.hist_step,
            )

        # Project beam phase to (pi/2,3pi/2) range
        self.phi_beam = np.arctan(coeff) + np.pi

    def phase_difference(
        self, beam: BeamBaseClass, RFnoise=None, noiseFB=None
    ):
        """
        Calculate phase difference between the beam and rf system.

        This method calculates the phase difference between the beam and the
        rf system. If a local feedback is used in the rf station, then the
        method is able to take into account the transient perturbation due to
        beam loading in the cavities.

        Parameters
        ----------
        beam
            The beam object used in the simulation.
        RFnoise
            Object containing rf phase noise.
        noiseFB
            Noise feedback object, e.g. used for controlled longitudinal emittance blow-up.
        """
        # Correct for design stable phase
        counter = self.cavities[0]._turn_i.value
        self.dphi = self.phi_beam - self.cavities[0].calc_phi_s_main_harmonic(
            beam
        )

        # TODO: Generalize to multiple harmonics and multiple rf stations
        # Phase offset due to beam loading
        if self.cavities[0].any_feedback_not_none:
            filled_slots = (
                np.abs(
                    self.cavities[0]
                    .cavity_feedback_list[0]
                    .I_BEAM_COARSE[
                        -self.cavities[0].cavity_feedback_list[0].n_coarse :
                    ]
                )
                > self.current_thres
            )

            gap_phase_in_slots = (
                self.cavities[0]
                .cavity_feedback_list[0]
                .gap_voltage_phase[filled_slots]
            )
            # voltage difference
            if len(gap_phase_in_slots) > 0:
                phi_mean = np.mean(gap_phase_in_slots)
            else:
                phi_mean = 0
            self.dphi = self.dphi + phi_mean

        # Possibility to add RF phase noise through the PL
        if RFnoise is not None:
            if noiseFB is not None:
                self.dphi += noiseFB.x * RFnoise.dphi[counter]
            else:
                self.dphi += RFnoise.dphi[counter]

    def _track(self, beam: BeamBaseClass):
        """
        Track method for the beam feedback.

        This method computes the beam-based measurement, computes
        the frequency corrections from the feedback and applies them
        to the parent rf stations.

        Parameters
        ----------
        beam
            The beam object used in the simulation.
        """
        self.get_beam_attribute(
            beam=beam,
        )
        self.compute_correction(
            beam=beam,
        )

        # TODO: generalize to multiple rf stations around the ring?
        if self.cavities[0]._turn_i.value >= self.delay:
            # TODO incorrect for simulations that start later
            # domega_rf is updated later
            # this means domega_rf is effectively from last turn
            omega_increment = (
                self.domega_rf
                * self.cavities[0].harmonic[:]
                / self.cavities[
                    0
                ].get_main_harmonic()  # dynamically updated by `update_domega_rf`
            )
            self.cavities[0].delta_omega_rf = omega_increment
