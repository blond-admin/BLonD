# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from blond.core.helpers import int_from_float_with_warning
from blond.core.ring.helpers import requires
from blond.experimental.physics.feedbacks.helpers import (
    cartesian_to_polar,
    polar_to_cartesian,
    rf_beam_current,
)
from blond.physics.cavities import SingleHarmonicRfStation
from blond.physics.feedbacks.base import LocalFeedback

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray as NumpyArray

    from blond import Simulation, StaticProfile
    from blond.core.beam.base import BeamBaseClass

# TODO rewrite all docstrings


class IQCavityFeedback(LocalFeedback):
    """
    Base class to design cavity feedbacks.

    Parameters
    ----------
    profile
        Beam profile the feedback acts on
    n_cavities
        Number of cavities the feedback controls
    n_periods_coarse
        Number of periods for the coarse grid
    harmonic_index
        Index of the RF harmonic that should be controlled by the feedback
    use_lowpass_filter
        Whether to apply a lowpass filter when calculating the beam current
    section_index
        # TODO might be removed?
    name
        # TODO might be removed

    Attributes
    ----------
    n_cavities
        Number of cavities the feedback is working on
    use_lowpass_filter
        Apply a low-pass filter to the RF beam current
    harmonic_index
        The harmonic index the cavity feedback is working on
    n_periods_coarse
        Sampling time in the model and the number of samples per turn
    T_s
        xxx # TODO
    n_coarse
        xxx # TODO
    omega_carrier
        xxx # TODO
    omega_rf
        xxx # TODO
    dT
        xxx # TODO
    V_SET
        xxx # TODO
    I_BEAM_COARSE
        xxx # TODO
    I_BEAM_FINE
        xxx # TODO
    V_ANT_COARSE
        xxx # TODO
    V_ANT_FINE
        xxx # TODO
    I_GEN_COARSE
        xxx # TODO
    I_GEN_FINE
        xxx # TODO
    V_corr
        xxx # TODO
    alpha_sum
        xxx # TODO
    phi_corr
        xxx # TODO
    omega_carrier_prev
        xxx # TODO
    T_s_prev
        xxx # TODO
    rf_centers_prev
        xxx # TODO

    """

    # TODO docstring

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
        """
        Base class to design cavity feedbacks.

        Parameters
        ----------
        profile
            Beam profile the feedback acts on
        n_cavities
            Number of cavities the feedback controls
        n_periods_coarse
            Number of rf periods the coarse grid sampling period corresponds to
        harmonic_index
            Index of the RF harmonic that should be controlled by the feedback
        use_lowpass_filter
            Whether to apply a lowpass filter when calculating the beam current
        section_index
            # TODO migh be removed?
        name
            # TODO might be removed
        """
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
        if not type(n_periods_coarse) is int:
            warnings.warn(
                "n_periods_coarse is not an integer; coupling between loops might break"
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

        self.V_corr: NumpyArray | None = None
        self.alpha_sum: NumpyArray | None = None
        self.phi_corr: NumpyArray | None = None
        self.omega_carrier_prev: float | None = None
        self.T_s_prev: float | None = None
        self.rf_centers_prev: NumpyArray | None = None

        self.V_SET: NumpyArray | None = None
        self.I_BEAM_COARSE: NumpyArray | None = None
        self.I_BEAM_FINE: NumpyArray | None = None
        self.V_ANT_COARSE: NumpyArray | None = None
        self.V_ANT_FINE: NumpyArray | None = None
        self.I_GEN_COARSE: NumpyArray | None = None
        self.I_GEN_FINE: NumpyArray | None = None

        self.gap_voltage_phase: NumpyArray | None = None

        self.dT: float | None = None

    @requires(["RfStationBaseClass", "BeamBaseClass"])
    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        harmonic, omega_rf, phi_rf = self.get_harmonic_and_omega_rf_phi_rf()

        self.T_s = (self.n_periods_coarse * 2 * np.pi) / omega_rf
        # TODO REMWORK/REMOVE
        t_rev = float((2 * np.pi * harmonic) / omega_rf)
        # TODO REMWORK/REMOVE
        t_rf = t_rev / float(harmonic)

        self.n_coarse = round(t_rev / self.T_s)  # TODO: round or ceil?
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

        self.V_SET = np.zeros(2 * self.n_coarse, dtype=complex)
        self.I_BEAM_COARSE = np.zeros(2 * self.n_coarse, dtype=complex)
        self.I_BEAM_FINE = np.zeros(self.profile.n_bins, dtype=complex)
        self.V_ANT_COARSE = np.zeros(2 * self.n_coarse, dtype=complex)
        self.V_ANT_FINE = np.zeros(self.profile.n_bins, dtype=complex)
        self.I_GEN_COARSE = np.zeros(2 * self.n_coarse, dtype=complex)
        self.I_GEN_FINE = np.zeros(self.profile.n_bins, dtype=complex)

        self.gap_voltage_phase = np.zeros(self.n_coarse)

    def set_hardware_commissioning(self, omega_rf: float, harmonic: int):
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

        self.V_SET = np.zeros(2 * self.n_coarse, dtype=complex)
        self.I_BEAM_COARSE = np.zeros(2 * self.n_coarse, dtype=complex)
        self.I_BEAM_FINE = np.zeros(self.profile.n_bins, dtype=complex)
        self.V_ANT_COARSE = np.zeros(2 * self.n_coarse, dtype=complex)
        self.V_ANT_FINE = np.zeros(self.profile.n_bins, dtype=complex)
        self.I_GEN_COARSE = np.zeros(2 * self.n_coarse, dtype=complex)
        self.I_GEN_FINE = np.zeros(self.profile.n_bins, dtype=complex)

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
            harmonic number for the harmonic index/only one
        omega_rf
            omega_rf for the harmonic index/only one
        phi_rf
            phi_rf for the harmonic index/only one

        """

        if isinstance(self._parent_rf_station, SingleHarmonicRfStation):
            harmonic = self._parent_rf_station.harmonic
            omega_rf = self._parent_rf_station.omega_rf_actual
            phi_rf = self._parent_rf_station.phi_rf_actual
        else:
            harmonic = self._parent_rf_station.harmonic[self.harmonic_index]
            omega_rf = self._parent_rf_station.omega_rf_actual[
                self.harmonic_index
            ]
            phi_rf = self._parent_rf_station.phi_rf_actual[self.harmonic_index]
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
            harmonic number for the harmonic index/only one
        omega_rf_design
            omega_rf_design for the harmonic index/only one
        phi_rf_design
            phi_rf_design for the harmonic index/only one

        """
        if isinstance(self._parent_rf_station, SingleHarmonicRfStation):
            harmonic = self._parent_rf_station.harmonic
            omega_rf_design = self._parent_rf_station.omega_rf_design
            phi_rf_design = self._parent_rf_station.phi_rf_design
        else:
            harmonic = self._parent_rf_station.harmonic[self.harmonic_index]
            omega_rf_design = self._parent_rf_station.omega_rf_design[
                self.harmonic_index
            ]
            phi_rf_design = self._parent_rf_station.phi_rf_design[
                self.harmonic_index
            ]
        return harmonic, omega_rf_design, phi_rf_design

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

    def update_rf_variables(
        self, omega_rf: float | None = None, harmonic: float | None = None
    ) -> None:
        r"""Updating variables from the other BLonD classes."""
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
        self.omega_carrier = self.omega_rf / self.n_periods_coarse

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
            np.arange(self.n_coarse) + 0.5 * self.n_periods_coarse
        ) * self.T_s + self.dT

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
            # TODO
        """
        pass

    def track_no_beam(self, n_pretrack: int | None = 1) -> None:
        r"""Tracking method of the cavity feedback without beam in the accelerator."""
        self.update_fb_variables()
        for i in range(n_pretrack):
            self.circuit_track(no_beam=True)

    def track(self, beam: BeamBaseClass) -> None:
        r"""Tracking method of the cavity feedback.

        Parameters
        ----------
        beam
            Simulation `Beam` object

        """
        # Update parameters from rest of BLonD classes
        self.update_rf_variables()
        self.update_fb_variables()

        # Get rf beam current
        self.rf_beam_current(
            beam=beam,
            use_lowpass_filter=self.use_lowpass_filter,
        )

        # Tracking circuit model of feedback
        self.circuit_track()

        # Convert to amplitude and phase
        self.V_corr, self.alpha_sum = cartesian_to_polar(
            IQ_vector=self.V_ANT_FINE[-self.profile.n_bins :],
        )

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.V_corr /= self.get_voltage_from_parent_rf_station()
        self.phi_corr = self.alpha_sum - np.mean(
            np.angle(self.V_SET[-self.n_coarse :])
        )

        self.gap_voltage_phase = np.angle(
            self.V_ANT_COARSE[-self.n_coarse :] / self.V_SET[-self.n_coarse :]
        )

    def rf_beam_current(
        self,
        beam: BeamBaseClass,
        use_lowpass_filter: bool = False,
    ) -> None:
        r"""Calculate RF beam current from beam profile"""
        harmonic, omega_rf_design, _ = (
            self.get_harmonic_and_omega_rf_phi_rf_design()
        )
        t_rev = float(  # TODO REMWORK/REMOVE
            (2 * np.pi * harmonic) / omega_rf_design
        )
        # Beam current from profile
        self.I_BEAM_COARSE[: self.n_coarse] = self.I_BEAM_COARSE[
            -self.n_coarse :
        ]
        self.I_BEAM_FINE, self.I_BEAM_COARSE[-self.n_coarse :] = (
            rf_beam_current(
                beam=beam,
                profile=self.profile,
                omega_c=self.omega_rf,
                T_rev=t_rev,
                use_lowpass_filter=use_lowpass_filter,
                downsample={"Ts": self.T_s, "points": self.n_coarse},
                external_reference=True,
                dT=self.dT,
            )
        )

        # Convert RF beam currents to be in units of Amperes
        self.I_BEAM_FINE = self.I_BEAM_FINE / self.profile.hist_step
        self.I_BEAM_COARSE[-self.n_coarse :] = (
            self.I_BEAM_COARSE[-self.n_coarse :] / self.T_s
        )

    def set_point_from_rfstation(self) -> NumpyArray:
        r"""Computes the setpoint in I/Q based on the RF voltage in the RFStation"""
        V_set = polar_to_cartesian(
            self.get_voltage_from_parent_rf_station() / self.n_cavities,
            0,
        )

        return V_set * np.ones(self.n_coarse)
