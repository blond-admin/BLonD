# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Base classes for the implementation of cavity feedbacks."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import interp1d

from blond.core.base import AltersReference, DynamicParameter, HasPropertyCache
from blond.core.helpers import int_from_float_with_warning
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.core.ring.helpers import requires
from blond.physics.cavities import (
    MultiHarmonicRFStation,
    RFStationBaseClass,
    SingleHarmonicRFStation,
)
from blond.physics.feedbacks.base import LocalFeedback
from blond.physics.feedbacks.helpers import (
    cartesian_to_polar,
    cavity_response_sparse_matrix,
    polar_to_cartesian,
    rf_beam_current,
)
from blond.physics.profiles import StaticProfile

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray as NumpyArray

    from blond import Beam, Ring, Simulation
    from blond.core.beam.base import BeamBaseClass

# TODO rewrite all docstrings


class IQCavityFeedback(LocalFeedback, HasPropertyCache):
    """
    Base class to design cavity feedbacks.

    Parameters
    ----------
    profile
        Beam profile the feedback acts on.
    n_cavities
        Number of cavities the feedback controls.
    n_rf_periods_per_coarse_grid
        Number of periods for the coarse grid.
    harmonic_index
        Index of the RF harmonic that should be controlled by the feedback.
    use_lowpass_filter
        Whether to apply a lowpass filter when calculating the beam current.
    name
        Name of the object.

    Attributes
    ----------
    n_cavities
        Number of cavities the feedback is working on.
    use_lowpass_filter
        Apply a low-pass filter to the RF beam current.
    harmonic_index
        The harmonic index the cavity feedback is working on.
    n_rf_periods_per_coarse_grid
        Sampling time in the model and the number of samples per turn.
    """

    def __init__(
        self,
        profile: StaticProfile,
        n_cavities: int,
        n_rf_periods_per_coarse_grid: int | float,
        harmonic_index: int,
        use_lowpass_filter: bool = False,
        name: str | None = None,
    ):
        assert isinstance(profile, StaticProfile), (
            "IQ cavity feedbacks require static profiles"
        )
        super().__init__(
            profile=profile,
            name=name,
        )

        # Number of cavities the feedback is working on
        assert n_cavities > 0, f"{n_cavities=}, but must be bigger 0."
        self.n_cavities = int_from_float_with_warning(
            n_cavities,
            warning_stacklevel=2,
        )

        # Apply a low-pass filter to the RF beam current
        self.use_lowpass_filter = use_lowpass_filter

        # The harmonic index the cavity feedback is working on
        self.harmonic_index = int_from_float_with_warning(
            harmonic_index,
            warning_stacklevel=2,
        )

        # Ratio between rf periods and coarse grid sampling period
        if type(n_rf_periods_per_coarse_grid) is not int:
            warnings.warn(
                "n_periods_coarse is not an integer; coupling between loops might break",
                stacklevel=2,
            )
        self.n_rf_periods_per_coarse_grid = n_rf_periods_per_coarse_grid

        # Update the coarse grid sampling
        self.n_samples_coarse: int | None = None

        self.alpha_sum: NumpyArray | None = None

        self.beam_current_forward_coarse_grid: NumpyArray | None = None
        self.beam_current_fine_grid: NumpyArray | None = None
        self.antenna_voltage_coarse_grid: NumpyArray | None = None
        self.antenna_voltage_fine_grid: NumpyArray | None = None
        self.generator_current_coarse_grid: NumpyArray | None = None
        self.generator_current_fine_grid: NumpyArray | None = None
        self.gap_voltage_phase: NumpyArray | None = None

    @requires(["RFStationBaseClass", "BeamBaseClass"])
    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Initialisation function at the start of the simulation.

        All array elements are defined based on the parameters of
        the parent rf station, which at this point in time is
        already fully initialised.

        Parameters
        ----------
        simulation
            Simulation object to initialise on.
        beam
            Beam object to initialise on.
        n_turns
            Number of turns in the simulation.
        **kwargs
            Unused in this function.
        """
        self.invalidate_cache()

        self.n_samples_coarse = np.floor(
            self.t_rev / self.sampling_time_coarse
        )  # TODO: round or ceil?; should this be changed during simulation?

        self.voltage_setpoint = np.zeros(self.profile.n_bins, dtype=complex)
        self.beam_current_forward_coarse_grid = np.zeros(
            self.n_samples_coarse, dtype=complex
        )
        self.beam_current_fine_grid = np.zeros(
            self.profile.n_bins, dtype=complex
        )
        self.antenna_voltage_coarse_grid = np.zeros(
            self.n_samples_coarse, dtype=complex
        )
        self.antenna_voltage_fine_grid = np.zeros(
            self.profile.n_bins, dtype=complex
        )
        self.generator_current_coarse_grid = np.zeros(
            self.n_samples_coarse, dtype=complex
        )
        self.generator_current_fine_grid = np.zeros(
            self.profile.n_bins, dtype=complex
        )
        self.gap_voltage_phase = np.zeros(self.n_samples_coarse)

        self.invalidate_cache()

    @abstractmethod  # pragma: no cover
    def update_feedback_variables(self) -> None:
        r"""
        Method to update the variables specific to the feedback.

        This is meant to be implemented in the child class by the user.
        """
        pass

    def get_voltage_from_parent_rf_station(self) -> float:
        """
        Convenience function to get the voltage from the parent RF station.

        Returns
        -------
        voltage
            Voltage from the parent RF station, either at harmonic_index or the only one.
        """
        if isinstance(self._parent_rf_station, SingleHarmonicRFStation):
            return self._parent_rf_station.voltage
        else:
            return self._parent_rf_station.voltage[self.harmonic_index]

    @property
    def time_coarse_grid(self) -> NumpyArray:
        """
        Time points of the coarse grid.

        Time points of the coarse grid. If self.n_periods_coarse is > 1, the
        points will be placed at the center of the bins, while for self.n_periods_coarse < 1,
        the first one will be placed at self.n_periods_coarse * 0.5.

        Returns
        -------
        time_coarse_grid
            Time points of the coarse grid [s].
        """
        if self.n_rf_periods_per_coarse_grid < 1:
            return (
                np.arange(self.n_samples_coarse) * self.sampling_time_coarse
                + 0.5 * self.t_rf * self.n_rf_periods_per_coarse_grid
            )
        else:
            return (
                np.arange(self.n_samples_coarse) * self.sampling_time_coarse
                + 0.5 * self.t_rf
            )

    @abstractmethod  # pragma: no cover
    def circuit_track(self, no_beam: bool = False) -> None:
        r"""
        Method to track circuit of the feedback.

        Parameters
        ----------
        no_beam
            Beam dependant parts of the feedback can be skipped if this is True.

        Notes
        -----
        This is meant to be implemented in the child class by the user.
        The only requirement for this method is that it has to update the
        V_ANT_FINE and V_SET arrays turn-by-turn.
        """
        pass

    def track_no_beam(self, n_pretrack: int | None = 1) -> None:
        r"""
        Tracking method of the cavity feedback without beam in the accelerator.

        Parameters
        ----------
        n_pretrack
            Number of turns to pretrack the feedback.
        """
        self.update_feedback_variables()
        for _ in range(n_pretrack):
            self.circuit_track(no_beam=True)

    def _track(self, beam: BeamBaseClass) -> None:
        r"""
        Tracking method of the cavity feedback.

        Parameters
        ----------
        beam
            Simulation `Beam` object.
        """
        self.invalidate_cache()
        # Update parameters from rest of BLonD classes
        self.update_feedback_variables()

        # Get rf beam current
        self.calculate_rf_beam_current(
            beam=beam,
            use_lowpass_filter=self.use_lowpass_filter,
        )

        # Tracking circuit model of feedback
        self.circuit_track()

        # Convert to amplitude and phase
        self.relative_voltage_correction, self.alpha_sum = cartesian_to_polar(
            IQ_vector=self.antenna_voltage_fine_grid,
        )

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.relative_voltage_correction /= (
            self.get_voltage_from_parent_rf_station()
        )
        self.phase_correction = self.alpha_sum - np.mean(
            np.angle(self.voltage_setpoint)
        )

        self.gap_voltage_phase = np.angle(
            self.antenna_voltage_coarse_grid / self.voltage_setpoint
        )

    def calculate_rf_beam_current(
        self,
        beam: BeamBaseClass,
        use_lowpass_filter: bool = False,
    ) -> None:
        r"""
        Calculate the IQ beam current for the coarse and fine grid.

        Parameters
        ----------
        beam
            Simulation `Beam` object.
        use_lowpass_filter
            Usage of low-pass filter in the calculation of the beam current.
        """
        # Beam current from profile
        (
            self.beam_current_fine_grid,
            self.beam_current_forward_coarse_grid[-self.n_samples_coarse :],
        ) = rf_beam_current(
            beam=beam,
            profile=self.profile,
            omega_c=self.omega_carrier,
            T_rev=self.t_rev,
            use_lowpass_filter=use_lowpass_filter,
            downsample={
                "Ts": self.sampling_time_coarse,
                "points": self.n_samples_coarse,
            },
            external_reference=True,
            dT=self.residual_time_shift_from_last_turn,
        )

        # Convert RF beam currents to be in units of Amperes
        self.beam_current_fine_grid = (
            self.beam_current_fine_grid / self.profile.hist_step
        )
        self.beam_current_forward_coarse_grid = (
            self.beam_current_forward_coarse_grid / self.sampling_time_coarse
        )

    def set_point_from_rfstation(self) -> NumpyArray:
        r"""
        Compute the setpoint in I/Q based on the RF voltage in the RFStation.

        Returns
        -------
        V_set
            Voltage setpoint in I/Q frame [V].
        """
        V_set = polar_to_cartesian(
            self.get_voltage_from_parent_rf_station() / self.n_cavities,
            0,
        )

        return V_set * np.ones(self.n_samples_coarse)

    @property
    def harmonic(self) -> float:
        """
        Harmonic number of the parent cavity at harmonic_index.

        Returns
        -------
        harmonic
            Harmonic number of the parent cavity at harmonic_index.
        """
        if isinstance(self._parent_rf_station, SingleHarmonicRFStation):
            return self._parent_rf_station.get_main_harmonic()
        else:
            return self._parent_rf_station.harmonic[self.harmonic_index]

    @property
    def delta_omega_rf(self) -> float:
        """
        Frequency deviation of the main harmonic of the parent cavity at harmonic_index.

        Returns
        -------
        delta_omega_rf
            Frequency deviation of the main harmonic of the parent cavity at harmonic_index.
        """
        if isinstance(self._parent_rf_station, SingleHarmonicRFStation):
            return self._parent_rf_station.delta_omega_rf
        else:
            return self._parent_rf_station.delta_omega_rf[self.harmonic_index]

    @property
    def omega_rf_design(self) -> float:
        """
        Design RF frequency of the parent cavity at harmonic_index.

        Returns
        -------
        omega_rf_design
            Design RF frequency of the parent cavity at harmonic_index.
        """
        if isinstance(self._parent_rf_station, SingleHarmonicRFStation):
            return self._parent_rf_station.omega_rf_design
        else:
            return self._parent_rf_station.omega_rf_design[self.harmonic_index]

    @property
    def omega_rf(self) -> float:
        """
        Actual RF frequency of the parent cavity at harmonic_index.

        Returns
        -------
        omega_rf_actual
            Actual RF frequency of the parent cavity at harmonic_index.
        """
        if isinstance(self._parent_rf_station, SingleHarmonicRFStation):
            return self._parent_rf_station.omega_rf
        else:
            return self._parent_rf_station.omega_rf[self.harmonic_index]

    @property
    def phi_rf(self) -> float:
        """
        Actual RF phase of the parent cavity at harmonic_index.

        Returns
        -------
        phi_rf_actual
            Actual RF phase of the parent cavity at harmonic_index.
        """
        if isinstance(self._parent_rf_station, SingleHarmonicRFStation):
            return self._parent_rf_station.phi_rf
        else:
            return self._parent_rf_station.phi_rf[self.harmonic_index]

    cached_props = (
        "t_rf",
        "omega_carrier",
        "sampling_time",
        "residual_phase_from_last_turn",
        "voltage_setpoint",
    )

    @property
    def t_rf(self) -> float:
        """
        Actual RF period of the parent cavity at harmonic_index.

        Returns
        -------
        t_rf
            Actual RF period of the parent cavity at harmonic_index.
        """
        return 1 / (self.omega_rf / (2 * np.pi))

    @property
    def omega_carrier(self) -> float:
        """
        Feedback carrier frequency.

        Returns
        -------
        omega_carrier
            Feedback carrier frequency.
        """
        return self.omega_rf / self.n_rf_periods_per_coarse_grid

    @property
    def t_rev(self) -> float:
        """
        Revolution time based on the harmonic and the design frequency.

        Returns
        -------
        t_rev
            Revolution time based on the harmonic and the design frequency.
        """
        return float((2 * np.pi * self.harmonic) / self.omega_rf_design)

    @property
    def sampling_time_coarse(self) -> float:
        """
        Sampling time on the coarse grid.

        Returns
        -------
        sampling_time_coarse
            Sampling time based on the number of periods per coarse grid and the design frequency.
        """
        return self.n_rf_periods_per_coarse_grid * 2 * np.pi / self.omega_rf

    @property
    def residual_time_shift_from_last_turn(
        self,
    ) -> float:  # TODO: this is the time and not the phase or?
        """
        Residual phase from last turn to current turn.

        Returns
        -------
        residual_phase_from_last_turn
            Residual phase from the last turn to current turn.
        """
        return (
            -self.phi_rf / self.omega_rf
        )  # TODO: this should be negative or positive?

    @property
    def voltage_setpoint(self) -> NumpyArray:
        """
        Voltage setpoint on the fine grid [V].

        Returns
        -------
        voltage_setpoint
            Voltage setpoint on the fine grid [V].
        """
        return (
            np.ones_like(self.antenna_voltage_coarse_grid)
            * self.get_voltage_from_parent_rf_station()
        )

    def invalidate_cache(self) -> None:
        """Delete the stored values of functions with @property."""
        self._invalidate_cache(IQCavityFeedback.cached_props)


class IQCavityFeedbackTimingClass(IQCavityFeedback):
    """
    Dummy.

    Parameters
    ----------
    profile
        Static profile the feedback should act on.
    R_over_Q
        Geometric shunt impedance of the cavity.
    Q_L
        Loaded quality factor of the cavity.
    generator_current
        Generator current [A].
    n_cavities
        Number of cavities connected to the feedback.
    initial_voltage
        Initial voltage [V].
    n_rf_periods_per_coarse_grid
        Number of rf periods, which should be displayed by one coarse gridpoint. Default is 1.
    detuning
        Cavity detuning in [rad/s].
    debug
        Save debugging parameters during runtime.
    """

    def __init__(
        self,
        profile,
        R_over_Q: float,
        Q_L: float,
        generator_current: float,
        n_cavities: int | float,
        initial_voltage: float = 30.0e6,
        n_rf_periods_per_coarse_grid: int = 1,
        detuning: float = 0.0,
        debug: bool = False,
    ):
        super().__init__(
            profile=profile,
            n_cavities=1,
            harmonic_index=1,
            n_rf_periods_per_coarse_grid=n_rf_periods_per_coarse_grid,
        )

        self.R_over_Q = R_over_Q
        self.Q_L = Q_L

        self.detuning = detuning
        self.rf_centers = np.zeros(0)
        self.rf_centers_lengths = np.zeros(0, dtype=int)
        self.residual_time_last_rf_centers_calculation = 0

        self.ring: Ring | None = None
        self.turn_i: DynamicParameter | None = None

        self.reference_altering_elements: (
            tuple[AltersReference, ...] | None
        ) = None
        self.reference_altering_elements_reverse: (
            tuple[AltersReference, ...] | None
        ) = None
        self.own_index_in_reference_list: int | None = None
        self.own_index_in_reference_list_reverse: int | None = None

        self.forward_tracking_omega_rf: float | None = None
        self.forward_tracking_time: float | None = None
        self.tracked_forward_until_element: AltersReference | None = None
        self.last_forward_tracking_freq: float | None = None
        self.residual_taps_last_rf_centers_calculation: int = 0

        self.reverse_tracking_time_array: NumpyArray | None = None
        self.reverse_tracking_omega_list: NumpyArray | None = None

        self.reference_state_until_tracked: ReferenceCoordinates | None = None
        self.reference_turn_offset: int = 0
        self.last_tracked_turn_frwrd: int = 0
        self.last_tracked_beam_state_frwrd: bool | None = None

        self.phase_offset_frwrd_next: float = 0.0
        self.phase_offset_frwrd: float = 0.0

        self.last_val_ant_voltage: float = 0.0
        self.last_val_beam_current: float = 0.0
        self.last_val_generator_current: float = 0.0
        self.last_rf_centers_entry: float | None = None

        self.init_voltage = initial_voltage

        self.n_cavities = n_cavities

        self.debug = debug

        self.generator_current_constant = generator_current

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Dummy.

        Parameters
        ----------
        simulation
            Simulation object to initialise on.
        """
        pass

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs,
    ) -> None:
        """
        Initialisation function at the start of the simulation.

        All array elements are defined based on the parameters of
        the parent rf station, which at this point in time is
        already fully initialised.

        Parameters
        ----------
        simulation
            Simulation object to initialise on.
        beam
            Beam object to initialise on.
        n_turns
            Number of turns in the simulation.
        **kwargs
            Unused in this function.
        """
        self.turn_i = simulation.turn_i
        self.ring = simulation.ring

        self.reference_altering_elements = self.ring.elements.get_elements(
            AltersReference
        )
        self.own_index_in_reference_list = (
            self.reference_altering_elements.index(self._parent_rf_station)
        )
        self.reference_altering_elements_reverse = (
            self.reference_altering_elements[::-1]
        )
        self.own_index_in_reference_list_reverse = (
            self.reference_altering_elements_reverse.index(
                self._parent_rf_station
            )
        )

        self.reference_state_until_tracked = deepcopy(beam.reference)
        self.phase_offset_frwrd_next = 0
        self.phase_offset_frwrd = 0

    def get_passed_time_forward_direction(self, beam: BeamBaseClass):  # noqa: PLR0912
        """
        Determine the slice of elements, which should be tracked in the forward direction.

        Parameters
        ----------
        beam
            Beam object to receive the reference frame.
        """
        next_reference_altering_element_index = -1

        dummy_reference = deepcopy(beam.reference)
        start_time = dummy_reference.time

        found = False

        own_index_tracking = (
            self.own_index_in_reference_list_reverse
            if beam.is_counter_rotating
            else self.own_index_in_reference_list
        )
        if beam.is_counter_rotating:
            forward_list = self.reference_altering_elements_reverse
        else:
            forward_list = self.reference_altering_elements

        # beam is tracked after the feedback, therefore we have to track the current element
        # the schedules are applied correctly though as this is done in the RFCavityBaseClass._track, which was already called
        for el_ind, element in enumerate(
            forward_list[own_index_tracking:]
        ):  # iterate through remaining current turn
            if isinstance(element, RFStationBaseClass) and el_ind != 0:
                found = True
                next_reference_altering_element_index = (
                    el_ind + own_index_tracking
                    # This will be the next element
                )
                self.last_tracked_turn_frwrd = deepcopy(self.turn_i.value)
                self.reference_turn_offset = -1
                break
            element: AltersReference
            if isinstance(element, RFStationBaseClass):
                element.track_reference(
                    dummy_reference, beam.is_counter_rotating
                )
            else:
                element.track_reference(dummy_reference)

        if not found:
            if own_index_tracking != 0:
                for el_ind, element in enumerate(
                    forward_list[:own_index_tracking]
                ):  # iterate through initial next turn
                    element: AltersReference

                    if not isinstance(element, RFStationBaseClass):
                        element.track_reference(dummy_reference)
                    else:
                        next_reference_altering_element_index = (
                            el_ind
                            + len(
                                self.reference_altering_elements
                            )  # This will be the next element
                        )
                        self.last_tracked_turn_frwrd = deepcopy(
                            self.turn_i.value + 1
                        )
                        self.reference_turn_offset = 0
                        break
            else:
                next_reference_altering_element_index = -1

        self.forward_tracking_time = dummy_reference.time - start_time
        self.forward_tracking_omega_rf = (
            (
                self._parent_rf_station.calc_omega_rf_design(
                    dummy_reference.beta, self.ring.circumference
                )
            )
            + self._parent_rf_station.delta_omega_rf
        )  # TODO: problematic with multi-section if the delta is changed in between sections
        self.tracked_forward_until_element = (
            forward_list[
                next_reference_altering_element_index % len(forward_list)
            ]
            if next_reference_altering_element_index != -1
            else self._parent_rf_station
        )
        self.reference_index_until_tracked = (
            self.reference_altering_elements.index(
                self.tracked_forward_until_element
            )
        )
        self.reference_index_until_tracked_reverse = (
            self.reference_altering_elements_reverse.index(
                self.tracked_forward_until_element
            )
        )
        self.last_tracked_beam_state_frwrd = beam.is_counter_rotating
        self.reference_state_until_tracked = dummy_reference

        if self.debug:
            if (
                next_reference_altering_element_index == -1
                or next_reference_altering_element_index
                >= len(self.reference_altering_elements)
            ):
                # either none were found or it is around two turns
                self.current_slice_elements_forward = (
                    self.reference_altering_elements[
                        self.own_index_in_reference_list :
                    ]
                )
                self.current_slice_elements_forward += (
                    self.reference_altering_elements[
                        0 : next_reference_altering_element_index
                        - len(self.reference_altering_elements)
                    ]
                )
            else:  # element is in the same turn
                self.current_slice_elements_forward = self.reference_altering_elements[
                    self.own_index_in_reference_list : next_reference_altering_element_index
                ]

    def get_time_omega_array_reverse_direction(self, beam: BeamBaseClass):  # noqa: PLR0912, PLR0915
        """
        Determine the slice of elements, which should be tracked in the reverse direction.

        Only gets called after the first turn.

        Parameters
        ----------
        beam
            Beam object to receive the reference frame.
        """
        time_list = []
        omega_list = []
        start_time = self.reference_state_until_tracked.time

        found = False

        if self.turn_i.value > self.last_tracked_turn_frwrd:
            reference_turn_offset = -1
        elif self.turn_i.value == self.last_tracked_turn_frwrd:
            reference_turn_offset = 0
        else:
            raise RuntimeError("Turn value not possible, was a turn skipped?")

        if self.last_tracked_beam_state_frwrd is not None:
            if self.last_tracked_beam_state_frwrd:  # last beam was counterrot
                start_index = self.reference_index_until_tracked_reverse
                reverse_tracking_list = (
                    self.reference_altering_elements_reverse
                )
            else:  # last beam was corot
                start_index = self.reference_index_until_tracked
                reverse_tracking_list = self.reference_altering_elements
        else:
            # first turn, nothing has been tracked yet.
            if beam.is_counter_rotating:  # counterrot
                reverse_tracking_list = (
                    self.reference_altering_elements_reverse
                )
            else:  # corot
                reverse_tracking_list = self.reference_altering_elements
            start_index = 0

        for element in reverse_tracking_list[
            start_index:
        ]:  # iterate through remaining last turn
            element: AltersReference  # TODO: are duplicate elements allowed in pipeline?
            if isinstance(
                element, RFStationBaseClass
            ):  # and element == self.tracked_forward_until_element:
                # Since we are in the previous turn, we need to decrease this manually
                # and increase it afterwards (only for cavities in case of scheduled acceleration).
                # this is not strictly true for all cases, but only cases, where the reference crosses the turn border on the forward tracking
                element._turn_i._value += reference_turn_offset
                element.track_reference(
                    self.reference_state_until_tracked,
                    beam.is_counter_rotating,
                )
            else:
                element.track_reference(
                    self.reference_state_until_tracked
                )  # no need for CR flag
            if isinstance(element, RFStationBaseClass):
                element._turn_i._value -= reference_turn_offset

            omega_list.append(
                self._parent_rf_station.calc_omega_rf_design(
                    self.reference_state_until_tracked.beta,
                    self.ring.circumference,
                )
            )
            time_list.append(self.reference_state_until_tracked.time)
            isclose = np.isclose(
                self.reference_state_until_tracked.time,
                beam.reference.time,
                rtol=1e-12,
                atol=0,
            )
            is_above = (
                self.reference_state_until_tracked.time > beam.reference.time
            )
            if isclose or is_above:  # counterrotation should break earlier
                if is_above:
                    raise RuntimeError("yorak")
                    warnings.warn(
                        "Inconsistency with references, is a delta_omega_rf applied to the rf_stations?",
                        stacklevel=1,
                    )
                found = True
                break

        if reverse_tracking_list is self.reference_altering_elements:
            until_index = self.own_index_in_reference_list
        else:
            until_index = self.own_index_in_reference_list_reverse

        if not found:
            for element in reverse_tracking_list[
                :until_index
            ]:  # iterate through initial current turn
                element: AltersReference
                if isinstance(element, RFStationBaseClass):
                    element.track_reference(
                        self.reference_state_until_tracked,
                        beam.is_counter_rotating,
                    )
                else:
                    element.track_reference(self.reference_state_until_tracked)
                omega_list.append(
                    self._parent_rf_station.calc_omega_rf_design(
                        self.reference_state_until_tracked.beta,
                        self.ring.circumference,
                    )
                )
                time_list.append(self.reference_state_until_tracked.time)
                if np.isclose(
                    self.reference_state_until_tracked.time,
                    beam.reference.time,
                    rtol=1e-12,
                    atol=0,
                ):  # counterrotation should break earlier
                    break

        if len(time_list) > 1:
            self.reverse_tracking_time_array = np.append(
                np.array(time_list[0] - start_time), np.diff(time_list)
            )
            self.reverse_tracking_omega_list = np.array(omega_list)
        else:
            self.reverse_tracking_time_array = np.array(time_list)
            self.reverse_tracking_omega_list = np.array(omega_list)

        self._unify_same_frequency_time_points_reverse()

        if self.debug:
            self.reference_time_after_reverse = (
                self.reference_state_until_tracked.time
            )
            self.current_beam_reference_time = beam.reference.time
            self.reference_energy_after_reverse = (
                self.reference_state_until_tracked.total_energy
            )
            self.current_beam_reference_energy = beam.reference.total_energy

    @property
    def n_points_coarse_grid(self):
        """
        Number of points on the coarse grid in this turn.

        Returns
        -------
        n_points_coarse_grid
            Number of points on the coarse grid in this turn.
        """
        return len(self.rf_centers)

    def get_t_rev(self):
        """
        Get revolution time from parent cavity.

        Returns
        -------
        t_rev
            Revolution time from the parent cavity.
        """
        if isinstance(self._parent_rf_station, SingleHarmonicRFStation):
            return (
                2
                * np.pi
                / self._parent_rf_station.omega_rf_design
                * self._parent_rf_station.harmonic
            )
        elif isinstance(self._parent_rf_station, MultiHarmonicRFStation):
            return (
                2
                * np.pi
                / self.omega_rf_design
                * self._parent_rf_station.harmonic
            )
        else:
            raise RuntimeError(
                f"Unknown cavity type {type(self._parent_rf_station)}"
            )

    @staticmethod
    def _get_time_to_next_rising_edge_zero(
        phi: float, frequency: float
    ) -> float:
        phi_modulated = np.mod(phi, 2 * np.pi)
        return np.mod(np.pi - phi_modulated, 2 * np.pi) / frequency

    def _generate_rf_centers(self, t_rf, omega_rf, phi_rf, until_time: float):
        time_to_next_falling_edge_zero = (
            self._get_time_to_next_rising_edge_zero(
                phi_rf,
                omega_rf,
            )
        )

        # 2nd part of if: floating precision would miss this in the last turn, hence has to be done this turn
        if time_to_next_falling_edge_zero <= 0 and not np.isclose(
            self.residual_taps_last_rf_centers_calculation, 1
        ):
            time_to_next_falling_edge_zero += t_rf

        step_width_rf_centers = t_rf * self.n_rf_periods_per_coarse_grid
        if (
            self.residual_taps_last_rf_centers_calculation != 0
            and self.n_rf_periods_per_coarse_grid != 1
        ):
            # while time_to_next_falling_edge_zero + self.residual_time_last_rf_centers_calculation < step_width_rf_centers:
            time_to_next_falling_edge_zero += t_rf * (
                self.n_rf_periods_per_coarse_grid
                - int(self.residual_taps_last_rf_centers_calculation)
                - 1
            )
        rf_centers = np.arange(
            start=time_to_next_falling_edge_zero,
            stop=until_time,  # ensure that the last value is taken even with float precision
            step=step_width_rf_centers,
        )

        if len(rf_centers) == 0:
            warnings.warn(
                f"no rf centers in turn {self.turn_i.value} at {self.section_index}",
                stacklevel=2,
            )
            return

        # reset with current turn
        self.residual_time_last_rf_centers_calculation = (
            until_time - rf_centers[-1]
        )
        self.residual_taps_last_rf_centers_calculation = (
            self.residual_time_last_rf_centers_calculation / t_rf
        )
        self.last_forward_tracking_freq = omega_rf
        return rf_centers

    def calculate_rf_centers_for_forward_direction(
        self, beam: BeamBaseClass
    ) -> None:
        """
        Calculate the centers of the rf buckets in the current turn.

        Parameters
        ----------
        beam
            Beam object to receive the reference frame.
        """
        self.get_passed_time_forward_direction(beam=beam)
        self.phase_offset_frwrd += self.phase_offset_frwrd_next
        self.phase_offset_frwrd_next = (
            2.0
            * np.pi
            * self.harmonic
            * self.delta_omega_rf
            / self._parent_rf_station.calc_omega_rf_design(
                beam_beta=self.reference_state_until_tracked.beta,
                ring_circumference=self.ring.circumference,
            )
        )

        new_rf_centers = self._generate_rf_centers(
            t_rf=(2 * np.pi / self.forward_tracking_omega_rf),
            # TODO: this is indeed necessary for the multi-section acceleration tracking, delta_omega hast to be applied somewhere else if applicable
            omega_rf=self.forward_tracking_omega_rf,
            phi_rf=self.phase_offset_frwrd,  # phase_offset_frwrd,
            until_time=self.forward_tracking_time,
        )

        self.rf_centers_lengths = np.append(
            self.rf_centers_lengths, len(new_rf_centers)
        )

        self.rf_centers = np.append(
            self.rf_centers,
            new_rf_centers,
        )
        pass

    def _unify_same_frequency_time_points_reverse(self):
        if len(self.reverse_tracking_time_array) > 1:
            time_arr_to_use = np.copy(self.reverse_tracking_time_array)
            omega_array_to_use = np.copy(self.reverse_tracking_omega_list)

            for omega_ind in range(1, len(omega_array_to_use)):
                if (
                    omega_array_to_use[omega_ind - 1]
                    == omega_array_to_use[omega_ind]
                ):
                    time_arr_to_use[omega_ind] += time_arr_to_use[
                        omega_ind - 1
                    ]
                    time_arr_to_use[omega_ind - 1] = 0

            mask = time_arr_to_use != 0
            self.reverse_tracking_time_array = time_arr_to_use[mask]
            self.reverse_tracking_omega_list = omega_array_to_use[mask]

    def calculate_rf_centers_for_reverse_direction(
        self, beam: BeamBaseClass
    ) -> None:
        """
        Dummy.

        Parameters
        ----------
        beam
            Beam object to receive the reference frame.
        """
        if (
            self.own_index_in_reference_list == 0
            and self.tracked_forward_until_element is None
        ):
            return
        if beam.reference.time == self.reference_state_until_tracked.time:
            return

        self.get_time_omega_array_reverse_direction(beam=beam)

        for time_ind, time in enumerate(self.reverse_tracking_time_array):
            # if time == 0:  # cavities may cause this in debug mode
            #     continue
            new_rf_centers = self._generate_rf_centers(
                t_rf=(2 * np.pi / self.reverse_tracking_omega_list[time_ind]),
                omega_rf=self.reverse_tracking_omega_list[time_ind],
                phi_rf=self.phi_rf,
                # TODO: not working atm with delta_omega since the calculation of phi_increment is not done correctly in parent rf cavity
                until_time=time,
            )
            self.rf_centers_lengths = np.append(
                self.rf_centers_lengths, len(new_rf_centers)
            )
            self.rf_centers = np.append(
                self.rf_centers,
                new_rf_centers,
            )

    def circuit_track(
        self,
        omega_input: float,
        no_beam: bool = False,
        start_index: int = 0,
        end_index: int = -1,
    ) -> None:
        """
        Dummy.

        Parameters
        ----------
        omega_input
            Frequency in the tracked segment.
        no_beam
            No beam in this segment.
        start_index
            Index of self.rf_centers at which to start computing the response.
        end_index
            Index of rf_centers until which to compute the response.
        """
        for rf_centers_idx in range(start_index, end_index):
            if rf_centers_idx == 0:
                if self.last_rf_centers_entry is None:
                    # first entry, just use frwrd direction
                    delta_t = (
                        self.rf_centers[rf_centers_idx + 1]
                        - self.rf_centers[rf_centers_idx]
                    )
                else:
                    delta_t = (
                        self.rf_centers[0]
                        + self.residual_time_last_rf_centers_calculation
                    )
            elif rf_centers_idx == start_index:
                delta_t = (
                    self.rf_centers[rf_centers_idx]
                    + self.residual_time_last_rf_centers_calculation
                )
            else:
                delta_t = (
                    self.rf_centers[rf_centers_idx]
                    - self.rf_centers[rf_centers_idx - 1]
                )
            assert delta_t > 0
            self.cavity_response(
                omega_input * delta_t,
                coarse_grid_index_to_update=rf_centers_idx,
                no_beam=no_beam,
            )

        if not no_beam:
            init_beam_time = self.profile.cut_left
            assert init_beam_time > 0, (
                f"{init_beam_time=} has to be > 0, shift profile."
            )

            # last entry is forward length
            antenna_voltage_init = interp1d(
                self.rf_centers[-self.rf_centers_lengths[-1] :],
                self.antenna_voltage_coarse_grid[
                    -self.rf_centers_lengths[-1] :
                ],
            )(init_beam_time)
            generator_current_init = interp1d(
                self.rf_centers[-self.rf_centers_lengths[-1] :],
                self.generator_current_coarse_grid[
                    -self.rf_centers_lengths[-1] :
                ],
            )(init_beam_time)

            # TODO: fix in case of RK application
            samples_per_rf_fine_grid = omega_input * self.profile.hist_step
            self.generator_current_fine_grid = np.interp(
                self.profile.hist_x,
                self.rf_centers[-self.rf_centers_lengths[-1] :],
                self.generator_current_coarse_grid[
                    -self.rf_centers_lengths[-1] :
                ],
            )

            relative_detuning = self.detuning / omega_input
            self.cavity_response_fine(
                antenna_voltage_init,
                0,
                generator_current_init,
                samples_per_rf_fine_grid,
                relative_detuning=relative_detuning,
            )

    def cavity_response(
        self,
        omega_times_T_s,
        coarse_grid_index_to_update,
        no_beam: bool = False,
    ):
        """
        Calculate antenna voltage on the coarse grid for a specific index.

        Parameters
        ----------
        omega_times_T_s
            Angular frequency times sampling time.
        coarse_grid_index_to_update
            Coarse grid index to update.
        no_beam
            If no beam is present, the beam current is set to 0.
        """
        if coarse_grid_index_to_update != 0:
            if no_beam:
                beam_current = 0
            else:
                forward_offset = (
                    len(self.rf_centers) - self.rf_centers_lengths[-1]
                )
                beam_current = self.beam_current_forward_coarse_grid[
                    coarse_grid_index_to_update - forward_offset
                ]
            self.antenna_voltage_coarse_grid[coarse_grid_index_to_update] = (
                self.generator_current_coarse_grid[
                    coarse_grid_index_to_update - 1
                ]
                * self.R_over_Q
                * omega_times_T_s
                + self.antenna_voltage_coarse_grid[
                    coarse_grid_index_to_update - 1
                ]
                * (
                    1
                    - 0.5 * omega_times_T_s / self.Q_L
                    + 1j * self.detuning * omega_times_T_s
                )
                - beam_current * 0.5 * self.R_over_Q * omega_times_T_s
            )
        else:
            self.antenna_voltage_coarse_grid[coarse_grid_index_to_update] = (
                self.last_val_generator_current
                * self.R_over_Q
                * omega_times_T_s
                + self.last_val_ant_voltage
                * (
                    1
                    - 0.5 * omega_times_T_s / self.Q_L
                    + 1j * self.detuning * omega_times_T_s
                )
                - self.last_val_beam_current
                * 0.5
                * self.R_over_Q
                * omega_times_T_s
            )

    def update_feedback_variables(self) -> None:
        """Dummy."""
        pass

    def reset_arrays(self):
        """Reset coarse grid arrays to match rf_centers length and save last values."""
        if self.antenna_voltage_coarse_grid is None:
            self.last_val_ant_voltage = self.init_voltage
        else:
            self.last_val_ant_voltage = self.antenna_voltage_coarse_grid[-1]
        self.antenna_voltage_coarse_grid = np.zeros(
            len(self.rf_centers), dtype=np.complex128
        )
        # TODO: update this when feedback part is implemented
        if self.generator_current_coarse_grid is None:
            self.last_val_generator_current = self.generator_current_constant
        else:
            self.last_val_generator_current = (
                self.generator_current_coarse_grid[-1]
            )

        self.generator_current_coarse_grid = (
            np.ones(len(self.rf_centers), dtype=np.complex128)
            * self.generator_current_constant
        )

    def _track(self, beam: Beam) -> None:
        """
        Dummy.

        Parameters
        ----------
        beam
            Beam to be tracked.
        """
        if len(self.rf_centers) != 0:
            self.last_rf_centers_entry = self.rf_centers[-1]

        self.rf_centers = np.zeros(0)
        self.rf_centers_lengths = np.zeros(0, dtype=int)

        if self.tracked_forward_until_element is not None:  # noqa: SIM102
            if (
                self.tracked_forward_until_element
                is not self._parent_rf_station
            ):  # otherwise, the full turn was already tracked
                self.calculate_rf_centers_for_reverse_direction(beam=beam)
        elif self._parent_rf_station._turn_i.value == 0:
            # at first call, this always needs to be tracked, since the values from the start of the simulation until now are not retrieved yet.
            self.calculate_rf_centers_for_reverse_direction(beam=beam)

        len_rev = len(self.rf_centers)

        remaining_delta_t_from_reverse_tracking = (
            self.residual_time_last_rf_centers_calculation
        )

        self.calculate_rf_centers_for_forward_direction(beam=beam)

        self.reset_arrays()
        for omega_index, omega_track in enumerate(
            self.reverse_tracking_omega_list
        ):
            start_index = np.sum(
                self.rf_centers_lengths[:omega_index], dtype=int
            )
            end_index = np.sum(
                self.rf_centers_lengths[: omega_index + 1], dtype=int
            )

            self.circuit_track(
                omega_input=omega_track,
                start_index=start_index,
                end_index=end_index,
                no_beam=True,
            )

        len_frwrd = len(self.rf_centers) - len_rev

        if self.debug:
            self.relative_voltage_correction = np.ones_like(
                self.profile.hist_x
            )
            self.phase_correction = np.zeros_like(self.profile.hist_x)
            return

        # default behavior
        self.calculate_rf_beam_current_partial(
            beam=beam,
            use_lowpass_filter=False,
            n_points=len_frwrd,
            remaining_delta_t_from_reverse_tracking=remaining_delta_t_from_reverse_tracking,
        )

        self.circuit_track(
            omega_input=self.forward_tracking_omega_rf,
            no_beam=False,
            start_index=len(self.rf_centers) - len_frwrd,
            end_index=len(self.rf_centers),
        )  # for all rf_centers

        # Convert to amplitude and phase
        self.relative_voltage_correction, self.alpha_sum = cartesian_to_polar(
            IQ_vector=self.antenna_voltage_fine_grid,
        )

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.relative_voltage_correction /= (
            self.get_voltage_from_parent_rf_station()
        )
        self.phase_correction = self.alpha_sum - np.mean(
            np.angle(self.voltage_setpoint)
        )

        self.gap_voltage_phase = np.angle(
            self.antenna_voltage_coarse_grid / self.voltage_setpoint
        )

        # dummy values

    def cavity_response_fine(
        self,
        initial_voltage_fine_grid: float,
        initial_voltage_gradient_fine_grid: float,
        initial_generator_current_fine_grid: float,
        samples_per_rf_fine_grid: float,
        relative_detuning: float,
    ):
        r"""
        ACS cavity response model in matrix form on the fine-grid.

        Parameters
        ----------
        initial_voltage_fine_grid : float
            Initial condition of the voltage on the fine grid.
        initial_voltage_gradient_fine_grid : float
            Initial condition of the voltage gradient on the fine grid.
        initial_generator_current_fine_grid : float
            Initial condition of the generator current on the fine grid.
        samples_per_rf_fine_grid
            Sample points per period on the fine grid.
        relative_detuning
            Cavity detuning relative to the center frequency.
        """
        # if self.fine_RK:
        #     _, self.antenna_voltage_fine_grid = (
        #         self.runge_kutta_tryout_2nd_order(
        #             dV_ant_init=initial_voltage_fine_grid,
        #             delta_omega=self.omega_detuning,
        #             V_init=initial_voltage_gradient_fine_grid,
        #             bin_centers=self.profile.hist_x,
        #             min_val=True,
        #             omega=self.omega_center,
        #         )
        #     )
        # else:

        self.antenna_voltage_fine_grid = cavity_response_sparse_matrix(
            I_beam=self.beam_current_fine_grid,
            I_gen=self.generator_current_fine_grid,
            V_ant_init=initial_voltage_fine_grid,
            I_gen_init=initial_generator_current_fine_grid,
            samples_per_rf=samples_per_rf_fine_grid,
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            relative_detuning=relative_detuning,
        )

        self.antenna_voltage_fine_grid *= self.n_cavities

    def calculate_rf_beam_current_partial(
        self,
        beam: BeamBaseClass,
        n_points: int,
        remaining_delta_t_from_reverse_tracking: float,
        use_lowpass_filter: bool = False,
    ) -> None:
        r"""
        Calculate the IQ beam current for the coarse and fine grid.

        Parameters
        ----------
        beam
            Simulation `Beam` object.
        n_points
            Number of points in the resulting coarse grid.
        remaining_delta_t_from_reverse_tracking
            Remaining time from the last rf_centers calculation, causes phase shift in beam current calculation.
        use_lowpass_filter
            Usage of low-pass filter in the calculation of the beam current.
        """
        if self.profile.active:
            self.profile.track(beam=beam)

        # Beam current from profile
        sampling_time_frwrd = (
            self.n_rf_periods_per_coarse_grid
            * 2
            * np.pi
            / self.forward_tracking_omega_rf
        )
        self.last_val_beam_current = (
            self.beam_current_forward_coarse_grid[-1]
            if self.beam_current_forward_coarse_grid is not None
            else 0
        )
        (
            self.beam_current_fine_grid,
            self.beam_current_forward_coarse_grid,
        ) = rf_beam_current(
            beam=beam,
            profile=self.profile,
            omega_c=self.forward_tracking_omega_rf,
            T_rev=self.forward_tracking_time,
            use_lowpass_filter=use_lowpass_filter,
            downsample={
                "Ts": sampling_time_frwrd,
                "points": n_points,
            },
            external_reference=True,
            dT=remaining_delta_t_from_reverse_tracking,
        )  # TODO: this is wrong --> adjust to rf_centers calculation

        # Convert RF beam currents to be in units of Amperes
        self.beam_current_fine_grid = (
            self.beam_current_fine_grid / self.profile.hist_step
        )
        self.beam_current_forward_coarse_grid = (
            self.beam_current_forward_coarse_grid / sampling_time_frwrd
        )
