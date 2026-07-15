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

import warnings
from abc import abstractmethod
from itertools import compress
from typing import TYPE_CHECKING

import numpy as np

from blond import Simulation, backend
from blond.core.base import (
    DynamicParameter,
    Schedulable,
)
from blond.core.ring.helpers import requires
from blond.physics.feedbacks.base import (
    GlobalFeedback,
)

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.physics.cavities import RFStationBaseClass
    from blond.physics.profiles import ProfileBaseClass


class BeamFeedbackBase(GlobalFeedback, Schedulable):
    """
    Base class for beam-based rf feedback systems in synchrotron particle accelerators.

    This class is intended to come with the features common for most
    beam-based rf feedback systems. The concrete beam feedback for a specific
    synchrotron is meant to be a child class of this object.

    Parameters
    ----------
    profile
        Any Profile object which exposes the x- and y-axis of the beam line density.
    phase_noise
        Option to add phase noise through the beam control.
    **kwargs
        Additional variable keyword arguments for the class. These are

        * delay: Delay (in units of turns) of the initial correction of the feedback system.

        * window_coefficient: Window coefficient for the calculation of the beam phase.
          This parameter will reduce the weight of later samples of the beam profile.

        * time_offset: Time offset [s] for the calculation of the beam phase.

        * sample_de: Determines downsampling of macroparticles for mean energy calculation.
          Every `sample_de` particle is sampled.

    Attributes
    ----------
    delta_omega_rf
        Angular frequency [rad/s] correction propagated to the RF station.
    dphi
        Phase difference [rad] between the beam and the RF system.
    phi_beam
        Phase of the beam [rad].
    drho
        Radial offset of the beam [m] with respect to reference orbit.
    average_de
        Average energy offset [eV] of the beam with respect to the synchronous energy.
    main_harmonic
        The harmonic number of the main RF system.
    """

    def __init__(
        self,
        profile: ProfileBaseClass,
        phase_noise=None,
        **kwargs,
    ):
        super().__init__(profile=profile)
        self._delay = kwargs.get("delay", 0)
        self.window_coefficient = kwargs.get("window_coefficient", 0.0)
        self.time_offset = kwargs.get("time_offset")
        self.sample_de = kwargs.get("sample_de", 1)
        self.phase_noise = phase_noise

        self.delta_omega_rf = 0.0
        self.dphi = 0.0

        self.phi_beam = 0.0

        self.drho = 0.0
        self.average_de = 0.0
        self.main_harmonic: int | None = None

        self._main_cavities: list[RFStationBaseClass] | None = None
        self._simulation: Simulation | None = None
        self._turn_counter: DynamicParameter | None = None

        self._first_turn_value_checked = False

    @requires(["RFStationBaseClass"])
    def on_init_simulation(self, simulation: Simulation, **kwargs) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        **kwargs
            Configure parameters collected by the MRO chain.
        """
        super().on_init_simulation(
            simulation,
            turn_counter=simulation.turn_counter,
            **kwargs,
        )

    def configure(
        self,
        *,
        turn_counter: DynamicParameter | None = None,
        **kwargs,
    ) -> None:
        """
        Store the runtime references needed during tracking.

        Parameters
        ----------
        turn_counter
            Live turn counter; accessed as ``turn_counter.value`` each track call.
        **kwargs
            Passed to the next level in the MRO chain.
        """
        super().configure(**kwargs)
        self._turn_counter = turn_counter

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
        self._first_turn_value_checked = False
        self._simulation = simulation

        self.check_main_rf_stations_with_cavity_feedback()

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
        omega_rf = self._main_cavities[0].get_main_harmonic_omega_rf()

        # Calculate RF phase based on the vectorial sum of all voltages on the main harmonic frequency
        phi_rfs = self.get_from_all_rf_stations(
            "get_main_harmonic_phi_rf", rf_station_list=self._main_cavities
        )
        voltages = self.get_from_all_rf_stations(
            "get_main_harmonic_voltage", rf_station_list=self._main_cavities
        )
        # Total RF phase for beam-phase calculation
        phi_rf = np.angle(np.sum(voltages * np.exp(1j * phi_rfs)))

        if self.time_offset is not None:
            mask = self.profile.hist_x >= self.time_offset
            hist_x = self.profile.hist_x[mask]
            hist_y = self.profile.hist_y[mask]
        else:
            hist_x = self.profile.hist_x
            hist_y = self.profile.hist_y

        coeff = backend.specials.beam_phase(
            hist_x,
            hist_y,
            self.window_coefficient,
            omega_rf,
            phi_rf,
            self.profile.hist_step,
        )

        self.phi_beam = np.arctan(coeff)

    def phase_difference(self, phase_noise=None):
        """
        Calculate phase difference between the beam and rf system.

        This method calculates the phase difference between the beam and the
        rf system.

        Parameters
        ----------
        phase_noise
            Option to add phase noise through the beam control.
        """
        # Correct for design stable phase
        counter = self._simulation.turn_counter.value
        self.dphi = self.phi_beam

        # Possibility to add RF phase noise through the PL
        if phase_noise is not None:
            self.dphi += phase_noise.dphi[counter]

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
        # TODO: Handling of simulations with some main rf stations without cavity FB and some with
        for cav in self._main_cavities:
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
        alpha = self._simulation.ring.momentum_compaction_factor
        ring_radius = self._simulation.ring.circumference / (2 * np.pi)

        # Correct for design orbit
        self.average_de = float(
            beam.read_partial_dE()[:: self.sample_de].mean()
        )

        self.drho = (
            alpha
            * ring_radius
            * self.average_de
            / (beam.reference.beta**2.0 * beam.reference.total_energy)
        )

    def update_main_rf_stations(self, new_main_harmonic: int = None):
        """
        Update which rf stations are ones with the main harmonic.

        This function updates the mask over the rf stations associated
        with the beam control which have the global main harmonic.

        Parameters
        ----------
        new_main_harmonic
            The new main harmonic number. If no number is passed then
            the new main harmonic will be the lowest main harmonic of the
            rf stations.
        """
        harmonics = self.get_from_all_rf_stations(
            method_or_attr="get_main_harmonic"
        )
        self.main_harmonic = (
            np.min(harmonics)
            if new_main_harmonic is None
            else new_main_harmonic
        )

        main_rf_stations_mask = harmonics == self.main_harmonic

        if not np.any(main_rf_stations_mask):
            raise ValueError("No RF stations are on the main harmonic")

        self._main_cavities = list(
            compress(self.cavities, main_rf_stations_mask)
        )

    def check_main_rf_stations_with_cavity_feedback(self):
        """
        Check if all main RF stations have a cavity feedback or not.

        This function checks all RF stations on the main harmonic for cavity feedbacks.
        If some have feedbacks and others do not then a warning is raised.
        """
        cavity_feedback_list = self.get_from_all_rf_stations(
            method_or_attr="get_main_harmonic_cavity_feedback",
            rf_station_list=self._main_cavities,
        )

        mask = np.array(
            [x is None for x in cavity_feedback_list.flat], dtype=bool
        ).reshape(cavity_feedback_list.shape)

        all_none = mask.all()
        all_not_none = (~mask).all()

        if not all_none or not all_not_none:
            warnings.warn(
                "Some RF stations acting on the main harmonic do not have a cavity feedback model."
                "Calculation of cavity sum phase might give incorrect results.",
                stacklevel=1,
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
        if self.schedule_active:
            self.apply_schedules(
                turn_i=self._turn_counter.value,
                reference_time=float(beam.reference.time),
            )

        self.get_beam_attribute(
            beam=beam,
        )
        # TODO: maybe later such that delta_omega_rf is returned from this method
        self.compute_correction(
            beam=beam,
        )

        # Check whether the method has encountered the first turn or not
        if not self._first_turn_value_checked:
            assert self._simulation.turn_counter.value == 0, (
                f"Expected first turn_counter value to be 0, "
                f"got {self._simulation.turn_counter.value}"
            )
            self._first_turn_value_checked = True

        if self._simulation.turn_counter.value >= self._delay:
            for cav in self.cavities:
                # delta_omega_rf is updated later
                # this means delta_omega_rf is effectively from last turn
                omega_increment = (
                    self.delta_omega_rf
                    * cav.harmonic
                    / self.main_harmonic  # dynamically updated by `update_delta_omega_rf`
                )
                cav.delta_omega_rf = omega_increment
