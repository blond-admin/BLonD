# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Base classes for the implemntation of cavity feedbacks."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from blond.core.base import HasPropertyCache
from blond.core.helpers import int_from_float_with_warning
from blond.core.ring.helpers import requires
from blond.physics.cavities import SingleHarmonicRfStation
from blond.physics.feedbacks.base import LocalFeedback
from blond.physics.feedbacks.helpers import (
    cartesian_to_polar,
    polar_to_cartesian,
    rf_beam_current,
)
from blond.physics.profiles import StaticProfile

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray as NumpyArray

    from blond import Simulation
    from blond.core.beam.base import BeamBaseClass

# TODO rewrite all docstrings


class IQCavityFeedback(LocalFeedback, HasPropertyCache):
    """
    Base class to design cavity feedbacks.

    Parameters
    ----------
    profile
        Beam profile the feedback acts on
    n_cavities
        Number of cavities the feedback controls
    n_rf_periods_per_coarse_grid
        Number of periods for the coarse grid
    harmonic_index
        Index of the RF harmonic that should be controlled by the feedback
    use_lowpass_filter
        Whether to apply a lowpass filter when calculating the beam current
    name
        ----

    Attributes
    ----------
    n_cavities
        Number of cavities the feedback is working on
    use_lowpass_filter
        Apply a low-pass filter to the RF beam current
    harmonic_index
        The harmonic index the cavity feedback is working on
    n_rf_periods_per_coarse_grid
        Sampling time in the model and the number of samples per turn
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

        self.relative_voltage_correction: NumpyArray | None = None
        self.alpha_sum: NumpyArray | None = None
        self.phase_correction: NumpyArray | None = None

        self.beam_current_coarse_grid: NumpyArray | None = None
        self.beam_current_fine_grid: NumpyArray | None = None
        self.antenna_voltage_coarse_grid: NumpyArray | None = None
        self.antenna_voltage_fine_grid: NumpyArray | None = None
        self.generator_current_coarse_grid: NumpyArray | None = None
        self.generator_current_fine_grid: NumpyArray | None = None

    @requires(["RfStationBaseClass", "BeamBaseClass"])
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
        already fully intialised.

        Parameters
        ----------
        simulation
            Simulation object to initialise on
        beam
            beam object to initialise on
        n_turns
            Number of turns in the simulation
        kwargs
            Unused in this function

        """
        self.n_samples_coarse = round(
            self.t_rev / self.sampling_time_coarse
        )  # TODO: round or ceil?; should this be changed during simulation?

        self.voltage_setpoint = np.zeros(self.profile.n_bins, dtype=complex)
        self.beam_current_coarse_grid = np.zeros(
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
            voltage from the parent RF station, either at harmonic_index or the only one

        """
        if isinstance(self._parent_rf_station, SingleHarmonicRfStation):
            return self._parent_rf_station.voltage
        else:
            return self._parent_rf_station.voltage[self.harmonic_index]

    @cached_property
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
                + 0.5 * self.t_rf_actual * self.n_rf_periods_per_coarse_grid
            )
        else:
            return (
                np.arange(self.n_samples_coarse) * self.sampling_time_coarse
                + 0.5 * self.t_rf_actual
            )

    @abstractmethod  # pragma: no cover
    def circuit_track(self, no_beam: bool = False) -> None:
        r"""
        Method to track circuit of the feedback.

        Notes
        -----
        This is meant to be implemented in the child class by the user.
        The only requirement for this method is that it has to update the
        V_ANT_FINE and V_SET arrays turn-by-turn.

        Parameters
        ----------
        no_beam
            beam dependant parts of the feedback can be skipped if this is True
        """
        pass

    def track_no_beam(self, n_pretrack: int | None = 1) -> None:
        r"""Tracking method of the cavity feedback without beam in the accelerator."""
        self.update_feedback_variables()
        for _ in range(n_pretrack):
            self.circuit_track(no_beam=True)

    def track(self, beam: BeamBaseClass) -> None:
        r"""Tracking method of the cavity feedback.

        Parameters
        ----------
        beam
            Simulation `Beam` object

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
            IQ_vector=self.antenna_voltage_fine_grid[-self.profile.n_bins :],
        )

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.relative_voltage_correction /= (
            self.get_voltage_from_parent_rf_station()
        )
        self.phase_correction = self.alpha_sum - np.mean(
            np.angle(self.voltage_setpoint[-self.n_samples_coarse :])
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
            Simulation `Beam` object
        use_lowpass_filter
            usage of low-pass filter in the calculation of the beam current

        """
        # Beam current from profile
        (
            self.beam_current_fine_grid,
            self.beam_current_coarse_grid[-self.n_samples_coarse :],
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
            dT=self.residual_phase_from_last_turn,
        )

        # Convert RF beam currents to be in units of Amperes
        self.beam_current_fine_grid = (
            self.beam_current_fine_grid / self.profile.hist_step
        )
        self.beam_current_coarse_grid[-self.n_samples_coarse :] = (
            self.beam_current_coarse_grid[-self.n_samples_coarse :]
            / self.sampling_time_coarse
        )

    def set_point_from_rfstation(self) -> NumpyArray:
        r"""
        Computes the setpoint in I/Q based on the RF voltage in the RFStation.

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

    cached_props = (
        "harmonic",
        "omega_rf_design",
        "omega_rf_actual",
        "phi_rf_actual",
        "t_rf_actual",
        "omega_carrier",
        "sampling_time",
        "residual_phase_from_last_turn",
        "voltage_setpoint",
    )

    @cached_property
    def harmonic(self) -> float:
        """Harmonic number of the parent cavity at harmonic_index."""
        if isinstance(self._parent_rf_station, SingleHarmonicRfStation):
            return self._parent_rf_station.get_main_harmonic()
        else:
            return self._parent_rf_station.harmonic[self.harmonic_index]

    @cached_property
    def omega_rf_design(self) -> float:
        """Design RF frequency of the parent cavity at harmonic_index."""
        if isinstance(self._parent_rf_station, SingleHarmonicRfStation):
            return self._parent_rf_station.omega_rf_design
        else:
            return self._parent_rf_station.omega_rf_design[self.harmonic_index]

    @cached_property
    def omega_rf_actual(self) -> float:
        """Actual RF frequency of the parent cavity at harmonic_index."""
        if isinstance(self._parent_rf_station, SingleHarmonicRfStation):
            return self._parent_rf_station.omega_rf_actual
        else:
            return self._parent_rf_station.omega_rf_actual[self.harmonic_index]

    @cached_property
    def phi_rf_actual(self) -> float:
        """Actual RF phase of the parent cavity at harmonic_index."""
        if isinstance(self._parent_rf_station, SingleHarmonicRfStation):
            return self._parent_rf_station.phi_rf_actual
        else:
            return self._parent_rf_station.phi_rf_actual[self.harmonic_index]

    @cached_property
    def t_rf_actual(self) -> float:
        """Actual RF period of the parent cavity at harmonic_index."""
        return self.omega_rf_actual / (2 * np.pi)

    @cached_property
    def omega_carrier(self) -> float:
        """Feedback carrier frequency."""
        return self.omega_rf_actual / self.n_rf_periods_per_coarse_grid

    @cached_property
    def t_rev(self) -> float:
        """Revolution time based on the harmonic and the design freqeuncy."""
        return float((2 * np.pi * self.harmonic) / self.omega_rf_design)

    @cached_property
    def sampling_time_coarse(self) -> float:
        """Feedback carrier frequency."""
        return (
            self.n_rf_periods_per_coarse_grid
            * 2
            * np.pi
            / self.omega_rf_actual
        )

    @cached_property
    def residual_phase_from_last_turn(self) -> float:
        """Feedback carrier frequency."""
        return self.phi_rf_actual / self.omega_rf_actual

    @cached_property
    def voltage_setpoint(self) -> NumpyArray:
        """Voltage setpoint on the fine grid [V]."""
        return (
            np.ones_like(self.voltage_setpoint)
            * self.get_voltage_from_parent_rf_station()
        )

    def invalidate_cache(self) -> None:
        """Delete the stored values of functions with @cached_property."""
        self._invalidate_cache(IQCavityFeedback.cached_props)
