# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Base class for the implementation of local rf feedback systems.

Notes
-----
Authors:
Birk Emil Karlsen-Baeck
Helga Timko
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

import numpy as np

from blond.core.helpers import int_from_float_with_warning
from blond.core.ring.helpers import requires
from blond.physics.feedbacks.base import LocalFeedback
from blond.physics.feedbacks.buffers import (
    OneTurnBufferBase,
    TwoTurnBufferBase,
)
from blond.physics.feedbacks.helpers import (
    cartesian_to_polar,
    polar_to_cartesian,
    rf_beam_current,
)

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from numpy.typing import NDArray as NumpyArray

    from blond import Simulation, StaticProfile
    from blond.core.beam.base import BeamBaseClass


BufferCoarse = TypeVar(
    "BufferCoarse", bound=TwoTurnBufferBase | OneTurnBufferBase
)
"""TypeVar for the coarse-grid buffer type used by :class:`IQCavityFeedback`."""

BufferFine = TypeVar("BufferFine", bound=OneTurnBufferBase)
"""TypeVar for the fine-grid buffer type used by :class:`IQCavityFeedback`."""


class IQCavityFeedback(LocalFeedback, Generic[BufferCoarse, BufferFine]):
    """
    Base class for local rf feedback systems with IQ signal processing.

    This class is intended to come with the features common for most
    local rf feedback systems. The concrete cavity feedback for a specific
    synchrotron is meant to be a child class of this object.

    Parameters
    ----------
    profile
        Beam profile the feedback acts on.
    n_cavities
        Number of cavities the feedback controls.
    n_periods_coarse
        Number of periods for the coarse grid.
    harmonic_index
        Index of the RF harmonic that should be controlled by the feedback.
    use_lowpass_filter
        Whether to apply a lowpass filter when calculating the beam current.
    section_index
        Section index of the feedback.
    name
        Name of the feedback.
    """

    buffer_cls_coarse: ClassVar[type[BufferCoarse]]
    buffer_cls_fine: ClassVar[type[BufferFine]]

    def __init__(
        self,
        profile: StaticProfile,
        n_cavities: int,
        n_periods_coarse: int | float,
        harmonic_index: int,
        use_lowpass_filter: bool = False,
        section_index: int = 0,
        name: str | None = None,
    ):
        from blond import StaticProfile  # cyclic import

        assert isinstance(profile, StaticProfile)
        super().__init__(
            profile=profile,
            section_index=section_index,
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
        if type(n_periods_coarse) is not int:
            warnings.warn(
                "n_periods_coarse is not an integer; coupling between loops might break",
                stacklevel=1,
            )
        self.n_periods_coarse = n_periods_coarse

        self.omega_carrier_prev: float | None = None
        self.omega_carrier: float | None = None
        self.omega_rf: float | None = None
        self.t_rev: float | None = None

        # Present sampling time
        self.T_s_prev: float | None = None
        self.T_s: float | None = None

        # Update the coarse grid sampling
        self.n_coarse: int | None = None

        # Present coarse grid and save previous turn coarse grid
        self.rf_centers_prev: float | None = None

        # Residual part of last turn entering the current turn due to non-integer harmonic number
        self.dT: float | None = None

        self.rf_centers: NumpyArray | None = None

        self.alpha_sum: NumpyArray | None = None
        self.omega_carrier_prev: float | None = None
        self.T_s_prev: float | None = None
        self.rf_centers_prev: NumpyArray | None = None

        self.buffers_coarse: BufferCoarse | None = None
        self.buffers_fine: BufferFine | None = None

        self.gap_voltage_phase: NumpyArray | None = None

        self.dT: float | None = None

    @requires(["RFStationBaseClass", "BeamBaseClass"])
    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs: dict[str, Any],
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
            simulation=simulation, beam=beam, n_turns=n_turns, **kwargs
        )
        harmonic, omega_rf, phi_rf = self.get_harmonic_and_omega_rf_phi_rf()

        self.T_s = (self.n_periods_coarse * 2 * np.pi) / omega_rf
        # TODO REMWORK/REMOVE
        t_rev = float((2 * np.pi * harmonic) / omega_rf)
        # TODO REMWORK/REMOVE
        t_rf = t_rev / float(harmonic)

        self.n_coarse = round(t_rev / self.T_s)
        self.omega_carrier = omega_rf / self.n_periods_coarse
        # FIXME NO REDECLARATION!

        self.omega_rf = float(omega_rf)
        self.dT = 0

        # The least amount of arrays needed to feedback to the tracker object
        if self.n_periods_coarse < 1:
            self.rf_centers = (
                np.arange(self.n_coarse) * self.T_s
                + 0.5 * t_rf * self.n_periods_coarse
            )
        else:
            self.rf_centers = np.arange(self.n_coarse) * self.T_s + 0.5 * t_rf

        self.buffers_coarse = self.buffer_cls_coarse(
            samples_per_turn=self.n_coarse
        )
        self.buffers_fine = self.buffer_cls_fine(
            samples_per_turn=self.profile.n_bins
        )

        self.gap_voltage_phase = np.zeros(self.n_coarse)

    def set_hardware_commissioning(self, omega_rf: float, harmonic: int):
        """
        Method to prepare the cavity feedback model for transfer function measurements.

        This is meant to set the necessary feedback parameters to run the model
        standalone, e.g. to perform transfer function measurements.

        Parameters
        ----------
        omega_rf
            Angular frequency of the RF system.
        harmonic
            Harmonic number of the RF system.
        """
        self.T_s = (self.n_periods_coarse * 2 * np.pi) / omega_rf
        # TODO REMWORK/REMOVE
        t_rev = float((2 * np.pi * harmonic) / omega_rf)
        # TODO REMWORK/REMOVE
        t_rf = 2 * np.pi / omega_rf

        self.n_coarse = round(t_rev / self.T_s)
        self.omega_carrier = omega_rf / self.n_periods_coarse
        # FIXME NO REDECLARATION!

        self.omega_rf = float(omega_rf)
        self.dT = 0

        # The least amount of arrays needed to feedback to the tracker object
        self.rf_centers = np.arange(self.n_coarse) * self.T_s + 0.5 * t_rf

        self.buffers_coarse = self.buffer_cls_coarse(
            samples_per_turn=self.n_coarse
        )
        self.buffers_fine = self.buffer_cls_fine(
            samples_per_turn=self.profile.n_bins
        )

    @abstractmethod  # pragma: no cover
    def update_fb_variables(self) -> None:
        r"""
        Method to update the variables specific to the feedback.

        This is meant to be implemented in the child class by the user.
        """
        pass

    def get_harmonic_and_omega_rf_phi_rf(
        self,
    ) -> tuple[float, float, float]:
        """
        Convenience function to get the actual values, currently acting on the RF station.

        This function is necessary since the _parent_cavity can be either a multi or single harmonic cavity.
        One of the holds the values for phi_rf omega_rf and as arrays and one as floats.

        Returns
        -------
        harmonic
            Harmonic number for the harmonic index/only one.
        omega_rf
            Omega_rf for the harmonic index/only one.
        phi_rf
            Phi_rf for the harmonic index/only one.
        """
        harmonic = self._parent_rf_station.get_main_harmonic()
        omega_rf = self._parent_rf_station.get_main_harmonic_omega_rf()
        phi_rf = self._parent_rf_station.get_main_harmonic_phi_rf()

        return harmonic, omega_rf, phi_rf

    def get_harmonic_and_omega_rf_phi_rf_design(
        self,
    ) -> tuple[float, float, float]:
        """
        Convenience function to get the design values of the RF station.

        This function is necessary since the _parent_cavity can be either a multi or single harmonic cavity.
        One of the holds the values for phi_rf omega_rf and as arrays and one as floats.

        Returns
        -------
        harmonic
            Harmonic number for the harmonic index/only one.
        omega_rf_design
            Omega_rf_design for the harmonic index/only one.
        phi_rf_design
            Phi_rf_design for the harmonic index/only one.
        """
        harmonic = self._parent_rf_station.get_main_harmonic()
        omega_rf_design = (
            self._parent_rf_station.get_main_harmonic_omega_rf_design()
        )
        phi_rf_design = (
            self._parent_rf_station.get_main_harmonic_phi_rf_design()
        )

        return harmonic, omega_rf_design, phi_rf_design

    def get_voltage_from_parent_rf_station(self) -> float:
        """
        Convenience function to get the voltage from the parent RF station.

        Returns
        -------
        voltage
            Voltage from the parent RF station, either at harmonic_index or the only one.
        """
        return self._parent_rf_station.get_main_harmonic_voltage()

    def update_rf_variables(
        self, omega_rf: float | None = None, harmonic: float | None = None
    ) -> None:
        """
        Update variables from the other BLonD classes.

        This method updates the variables coming from the RF station the
        cavity feedback model is associated to.

        Parameters
        ----------
        omega_rf
            Angular frequency of the RF system.
        harmonic
            Harmonic number of the RF system.
        """
        if omega_rf is None or harmonic is None:
            harmonic, omega_rf, phi_rf = (
                self.get_harmonic_and_omega_rf_phi_rf()
            )
        else:
            phi_rf = 0.0

        # Present RF angular frequency
        self.omega_rf = omega_rf
        t_rev = float(  # TODO REMWORK/REMOVE
            2 * np.pi * harmonic / self.omega_rf
        )

        # Present carrier frequency: main RF frequency
        self.omega_carrier_prev = self.omega_carrier
        self.omega_carrier = self.omega_rf

        # Present sampling time
        self.T_s_prev = self.T_s
        self.T_s = self.n_periods_coarse * 2 * np.pi / self.omega_rf

        # Update the coarse grid sampling
        self.n_coarse = round(t_rev / self.T_s)

        # Present coarse grid and save previous turn coarse grid
        self.rf_centers_prev = np.copy(self.rf_centers)

        # Residual part of last turn entering the current turn due to non-integer harmonic number
        self.dT = -phi_rf / self.omega_rf

        self.rf_centers = (
            np.arange(self.n_coarse) + 0.5 / self.n_periods_coarse
        ) * self.T_s + self.dT

    @abstractmethod  # pragma: no cover
    def circuit_track(self, no_beam: bool = False) -> None:
        r"""
        Method to track circuit of the feedback.

        Parameters
        ----------
        no_beam
            Optional argument to track without calculating the
            beam-induced voltage. Flag used for pre-tracking of the model.

        Notes
        -----
        This is meant to be implemented in the child class by the user.
        The only requirement for this method is that it has to update the
        V_ANT_FINE and V_SET arrays turn-by-turn.
        """
        pass

    def track_no_beam(self, n_pretrack: int = 1) -> None:
        """
        Track the cavity feedback without beam in the accelerator.

        Meant to be called before the main tracking loop to have the feedback
        system in steady-state once the beam arrives.

        Parameters
        ----------
        n_pretrack
            Number of turns of pre-tracking.
        """
        self.update_fb_variables()
        for _i in range(n_pretrack):
            self.buffers_coarse.shift()
            self.circuit_track(no_beam=True)

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Tracking method of the cavity feedback.

        Parameters
        ----------
        beam
            Simulation `Beam` object.
        """
        # Update parameters from rest of BLonD classes
        self.update_rf_variables()
        self.update_fb_variables()
        self.buffers_coarse.shift()

        # Get rf beam current
        self.rf_beam_current(
            beam=beam,
            use_lowpass_filter=self.use_lowpass_filter,
        )

        # Tracking circuit model of feedback
        self.circuit_track()

        # Convert to amplitude and phase
        self.relative_amplitude_correction, self.alpha_sum = (
            cartesian_to_polar(
                IQ_vector=self.buffers_fine.v_ant,
            )
        )

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.relative_amplitude_correction /= (
            self.get_voltage_from_parent_rf_station()
        )
        self.phase_correction = self.alpha_sum - np.mean(
            np.angle(self.buffers_coarse.v_setpoint.curr)
        )

        self.gap_voltage_phase = np.angle(
            self.buffers_coarse.v_ant.curr
            / self.buffers_coarse.v_setpoint.curr
        )

    def rf_beam_current(
        self,
        beam: BeamBaseClass,
        use_lowpass_filter: bool = False,
    ) -> None:
        """
        Calculate RF beam current from beam profile.

        Parameters
        ----------
        beam
            Simulation `Beam` object.
        use_lowpass_filter
            Whether to apply a lowpass filter when calculating the beam current.
        """
        harmonic, omega_rf_design, _ = (
            self.get_harmonic_and_omega_rf_phi_rf_design()
        )
        t_rev = float(  # TODO REMWORK/REMOVE
            (2 * np.pi * harmonic) / omega_rf_design
        )
        # Beam current from profile
        (
            self.buffers_fine.i_beam,
            self.buffers_coarse.i_beam.curr,
        ) = rf_beam_current(
            beam=beam,
            profile=self.profile,
            omega_c=self.omega_carrier,
            T_rev=t_rev,
            use_lowpass_filter=use_lowpass_filter,
            downsample={"Ts": self.T_s, "points": self.n_coarse},
            external_reference=True,
            dT=self.dT,
        )

        # Convert RF beam currents to be in units of Amperes
        self.buffers_fine.i_beam = (
            self.buffers_fine.i_beam / self.profile.hist_step
        )
        self.buffers_coarse.i_beam.curr = (
            self.buffers_coarse.i_beam.curr / self.T_s
        )

    def set_point_from_rfstation(self) -> NumpyArray:
        """
        Compute the setpoint in I/Q based on the RF voltage in the RFStation.

        Returns
        -------
        v_set
            Setpoint voltage on the coarse-grid from parent RF station. A constant
            amplitude and phase is assumed.
        """
        V_set = polar_to_cartesian(
            self.get_voltage_from_parent_rf_station() / self.n_cavities,
            0,
        )

        return V_set * np.ones(self.n_coarse)
