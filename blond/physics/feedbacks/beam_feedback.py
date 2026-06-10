# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
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
from itertools import compress
from typing import TYPE_CHECKING

import numpy as np

from blond import Simulation, backend
from blond.core.ring.helpers import requires
from blond.physics.feedbacks.base import (
    GlobalFeedback,
)

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

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
    sample_de
        Determines which particles to sample for mean energy calculation.
        Every <sample_dE>. particle is sampled.
    """

    _parent_rf_station: RFStationBaseClass

    def __init__(
        self,
        profile: ProfileBaseClass,
        delay: int = 0,
        window_coefficient: float = 0.0,
        time_offset: float | None = None,
        sample_de: int = 1,
    ):
        super().__init__(profile=profile)
        self.delay = delay
        self.window_coefficient = window_coefficient
        self.time_offset = time_offset
        self.sample_de = sample_de

        self.domega_rf = 0

        self.dphi: float = 0.0

        self.phi_beam: float = 0.0

        self.drho: float = 0.0
        self.average_de: float = 0.0
        self.main_rf_stations_mask: NumpyArray | None = None
        self.main_cavities: list[RFStationBaseClass] | None = None
        self.main_harmonic: int | None = None

    @requires(["RFStationBaseClass"])
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
        self.update_main_rf_stations()

    @abstractmethod  # pragma: no cover
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

    @abstractmethod  # pragma: no cover
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
        omega_rf = self.main_cavities[0].get_main_harmonic_omega_rf()

        # Calculate RF phase based on all the rf stations
        phi_rfs = self.get_from_all_rf_stations(
            "get_main_harmonic_phi_rf", cavity_list=self.main_cavities
        )
        voltages = self.get_from_all_rf_stations(
            "get_main_harmonic_voltage", cavity_list=self.main_cavities
        )
        phi_rf = np.angle(np.sum(voltages * np.exp(1j * phi_rfs)))

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

        self.phi_beam = np.arctan(coeff)

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
        counter = self.main_cavities[0]._turn_counter.value
        # TODO: a priori the beam control does not know about the synchronous phase?
        self.dphi = self.phi_beam

        """
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
        """

        # Possibility to add RF phase noise through the PL
        if RFnoise is not None:
            if noiseFB is not None:
                self.dphi += noiseFB.x * RFnoise.dphi[counter]
            else:
                self.dphi += RFnoise.dphi[counter]

    def cavity_sum_phase(self, current_thres: float):
        """
        Calculate the cavity sum phase when tracking with cavity feedbacks.

        This method sums the cavity gap voltage over all rf stations having
        the main harmonic and that have a cavity feedback model acting the
        main harmonic. Cavity sum phase is then added to `dphi`.

        Parameters
        ----------
        current_thres
            Beam current threshold for gating of the profiles.
        """
        filled_slots: NumpyArray | None = None
        cavity_sum: NumpyArray | None = None

        # iterate over rf stations on the main harmonic
        for cav in self.main_cavities:
            # Get cavity feedback on main harmonic for every rf station
            _cavity_feedback = cav.get_main_harmonic_cavity_feedback()

            # If the cavity is not None then add its contribution to the cavity sum
            if _cavity_feedback is not None and filled_slots is None:
                filled_slots = (
                    np.abs(
                        _cavity_feedback.I_BEAM_COARSE[
                            -_cavity_feedback.n_coarse :
                        ]
                    )
                    > current_thres
                )

                cavity_sum = _cavity_feedback.V_ANT_COARSE[
                    -_cavity_feedback.n_coarse :
                ]

            elif _cavity_feedback is not None and filled_slots is not None:
                cavity_sum += _cavity_feedback.V_ANT_COARSE[
                    -_cavity_feedback.n_coarse :
                ]

        if cavity_sum is not None:
            cavity_sum_phase = np.angle(cavity_sum)
            cavity_sum_phase = cavity_sum_phase[filled_slots]

            self.dphi = self.dphi + np.mean(cavity_sum_phase)

    def radial_difference(self, beam: BeamBaseClass):
        """
        Radial difference between beam and design orbit.

        Parameters
        ----------
        beam
            The beam object used in the simulation.
        """
        # Calculate alpha
        alpha = self.main_cavities[0]._ring.momentum_compaction_factor
        ring_radius = self.main_cavities[0]._ring.circumference / (2 * np.pi)

        # Correct for design orbit
        self.average_de = beam.read_partial_dE()[:: self.sample_de].mean()

        self.drho = (
            alpha
            * ring_radius
            * self.average_de
            / (beam.reference.beta**2.0 * beam.reference.total_energy)
        )

    def update_main_rf_stations(self, harmonic: int = None):
        """
        Update which rf stations are ones with the main harmonic.

        This function updates the mask over the rf stations associated
        with the beam control which have the global main harmonic.

        Parameters
        ----------
        harmonic
            The new main harmonic number. If no number is passed then
            the new main harmonic will be the main harmonic of the first
            rf station.
        """
        harmonics = self.get_from_all_rf_stations(
            method_or_attr="get_main_harmonic"
        )
        if harmonic is not None:
            self.main_harmonic = harmonic
        else:
            self.main_harmonic = harmonics[0]

        self.main_rf_stations_mask = np.zeros(harmonics.shape, dtype=bool)

        for i, harm in enumerate(harmonics):
            self.main_rf_stations_mask[i] = harm == self.main_harmonic

        if not np.any(self.main_rf_stations_mask):
            raise ValueError("No RF stations are on the main harmonic")

        self.main_cavities = list(
            compress(self.cavities, self.main_rf_stations_mask)
        )

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

        if self.cavities[0]._turn_counter.value >= self.delay:
            # TODO incorrect for simulations that start later
            for cav in self.cavities:
                # domega_rf is updated later
                # this means domega_rf is effectively from last turn
                omega_increment = (
                    self.domega_rf
                    * cav.harmonic
                    / self.main_harmonic  # dynamically updated by `update_domega_rf`
                )
                cav.delta_omega_rf = omega_increment
