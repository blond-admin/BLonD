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
        The harmonic number of the main RF system. The type is int | None.
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
        self.main_harmonic = (
            None  # type is int | None, documented in class docstring
        )

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
    def update_beam_attributes(self, beam: BeamBaseClass):
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
    def update_frequency_correction(self, beam: BeamBaseClass):
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
            accessor=lambda rf: rf.get_main_harmonic_phi_rf(),
            rf_station_list=self._main_cavities,
        )
        voltages = self.get_from_all_rf_stations(
            accessor=lambda rf: rf.get_main_harmonic_voltage(),
            rf_station_list=self._main_cavities,
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

        # TODO: Maybe add this statement into the backend
        self.phi_beam = np.arctan(float(coeff))

    def update_phase_error(self, phase_noise=None):
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

        This method would sum the complex cavity gap (antenna) voltage
        over all rf stations on the main harmonic that carry a cavity
        feedback model, gate the per-slot phases of that sum with
        `current_thres` on the beam current, and add the mean phase over
        the filled slots to `dphi`.

        It is currently not implemented: the original body was ported
        from the deleted blond2 coarse-array cavity-feedback API and was
        never wired to the surviving ``IQCavityFeedbackTimingClass``.
        As long as no rf station on the main harmonic carries a cavity
        feedback on ANY of its harmonics, this method is a silent no-op
        -- that is a normal, supported configuration, and the beam
        controls call this method unconditionally every turn. As soon as
        one such feedback exists it raises `NotImplementedError`, rather
        than letting the phase loop run as if the cavity were
        unregulated.

        Parameters
        ----------
        current_thres
            Beam current threshold for gating of the profiles. Unused
            until the method is implemented; the signature is kept
            because the beam controls pass it unconditionally.

        Raises
        ------
        NotImplementedError
            If any rf station on the main harmonic has a cavity
            feedback attached to any of its harmonics.
        """
        # TODO: Couple the beam phase loop to the surviving cavity
        #   feedback (``IQCavityFeedbackTimingClass``); see the
        #   NotImplementedError message below for the two APIs involved.
        # TODO: Handling of simulations with some main rf stations
        #   without cavity FB and some with
        for cav in self._main_cavities:
            # ``any_feedback_not_none``, not
            # ``get_main_harmonic_cavity_feedback``: the latter reads
            # only the main-harmonic slot of ``cavity_feedback_list``,
            # so a feedback regulating a non-main harmonic of a
            # `MultiHarmonicRFStation` would slip through unnoticed.
            # It is also the predicate the LHC/SPS beam controls use to
            # demand `current_thres` at run start, so both guards now
            # fire for exactly the same configurations.
            if not cav.any_feedback_not_none:
                continue
            raise NotImplementedError(
                "cavity_sum_phase is not implemented for the surviving "
                "cavity feedback: the original implementation read the "
                "deleted LHC/SPS coarse-array API (I_BEAM_COARSE, "
                "V_ANT_COARSE, sliced to a fixed n_coarse per turn), "
                "which exists nowhere in the live tree. The surviving "
                "mucol feedback (IQCavityFeedbackTimingClass) instead "
                "exposes per-turn, variable-length grids as "
                "beam_current_forward_coarse_grid and "
                "antenna_voltage_coarse_grid and has no n_coarse. "
                "Coupling the beam phase loop to that per-turn-grid "
                "API is an open design task, not a rename. This is "
                "raised for a cavity feedback on ANY harmonic of the "
                "station, not only on the main harmonic."
            )

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
            the new main harmonic will be the lowest main harmonic of all
            rf stations in the ring.
        """
        harmonics = self.get_from_all_rf_stations(
            accessor=lambda rf: rf.get_main_harmonic()
        )

        # If no new harmonic is passed by the user the method updates
        # the new main harmonic to be the lowest one out of all the rf stations.
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
        A station counts as having a cavity feedback when one is
        attached to any of its harmonics, not only to its main one.
        """
        # A feedback attached to a non-main harmonic still regulates the
        # station's gap voltage, so ``any_feedback_not_none`` -- and not
        # the main-harmonic-only ``get_main_harmonic_cavity_feedback``
        # -- decides whether a station carries a cavity feedback model.
        has_cavity_feedback = self.get_from_all_rf_stations(
            accessor=lambda rf: rf.any_feedback_not_none,
            rf_station_list=self._main_cavities,
        )

        # Only a MIXTURE is suspicious: a uniformly equipped and a
        # uniformly unequipped set of stations are both well defined.
        mixed_configuration = bool(
            np.any(has_cavity_feedback) and not np.all(has_cavity_feedback)
        )

        if mixed_configuration:
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

        self.update_beam_attributes(
            beam=beam,
        )
        # TODO: maybe later such that delta_omega_rf is returned from this method
        self.update_frequency_correction(
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
                    self.delta_omega_rf * cav.harmonic / self.main_harmonic
                )
                cav.delta_omega_rf = omega_increment
