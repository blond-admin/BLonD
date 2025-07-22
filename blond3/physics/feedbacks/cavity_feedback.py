from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .base import LocalFeedback
from .helpers import (
    rf_beam_current,
    cartesian_to_polar,
    polar_to_cartesian,
)
from ... import StaticProfile
from ..._core.helpers import int_from_float_with_warning

if TYPE_CHECKING:
    from typing import Optional, Optional as LateInit

    from numpy.typing import NDArray as NumpyArray
    from ..cavities import MultiHarmonicCavity
    from ..._core.beam.base import BeamBaseClass
    from ... import Simulation

# TODO rewrite all docstrings


class BirksCavityFeedback(LocalFeedback):
    """
    Base class to design cavity feedbacks

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

    # TODO remove after development
    _parent_cavity: MultiHarmonicCavity
    profile: StaticProfile

    def __init__(
        self,
        _parent_cavity: MultiHarmonicCavity,
        profile: StaticProfile,
        n_cavities: int,
        n_periods_coarse: int,
        harmonic_index: int,
        use_lowpass_filter: bool = False,
        section_index: int = 0,
        name: Optional[str] = None,
    ):
        """
        Base class to design cavity feedbacks

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
            # TODO migh be removed?
        name
            # TODO might be removed
        """
        assert isinstance(profile, StaticProfile)
        super().__init__(
            profile=profile,
            section_index=section_index,
            name=name,
        )
        self.set_parent_cavity(cavity=_parent_cavity)
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
        if self.harmonic_index > self._parent_cavity.n_rf - 1:
            raise RuntimeError(
                "ERROR in CavityFeedback: argument"
                " n_h is greater than the number of n_rf in RFStation"
            )

        # Sampling time in the model and the number of samples per turn
        self.n_periods_coarse = int(n_periods_coarse)

        self.T_s = (self.n_periods_coarse * 2 * np.pi) / self._parent_cavity._omega[
            self.harmonic_index
        ]
        # TODO REMWORK/REMOVE
        t_rev = float(
            (2 * np.pi * self._parent_cavity.harmonic[self.harmonic_index])
            / self._parent_cavity._omega[self.harmonic_index]
        )
        # TODO REMWORK/REMOVE
        t_rf = t_rev / float(self._parent_cavity.harmonic[self.harmonic_index])

        self.n_coarse = round(t_rev / self.T_s)
        self.omega_carrier = (
            self._parent_cavity._omega[self.harmonic_index] / self.n_periods_coarse
        )
        # FIXME NO REDECLARATION!

        self.omega_rf = float(self._parent_cavity._omega[self.harmonic_index])  #
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

        # TODO REWORK LATEINIT
        self.V_corr: LateInit = None
        self.alpha_sum: LateInit = None
        self.phi_corr: LateInit = None
        self.omega_carrier_prev: LateInit = None
        self.T_s_prev: LateInit = None
        self.rf_centers_prev: LateInit = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        pass

    @abstractmethod
    def update_fb_variables(self) -> None:
        r"""
        Method to update the variables specific to the feedback.

        This is meant to be implemented in the child class by the user.
        """
        pass

    def update_rf_variables(self) -> None:
        r"""Updating variables from the other BLonD classes"""

        # Present time step

        # Present RF angular frequency
        self.omega_rf = float(self._parent_cavity._omega[self.harmonic_index])
        t_rev = float(  # TODO REMWORK/REMOVE
            2
            * np.pi
            * self._parent_cavity.harmonic[self.harmonic_index]
            / self.omega_rf
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
        self.dT = -self._parent_cavity.phi_rf[self.harmonic_index] / self.omega_rf
        self.rf_centers = (
            np.arange(self.n_coarse) + 0.5 / self.n_periods_coarse
        ) * self.T_s + self.dT

    @abstractmethod
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

    def track_no_beam(self, n_pretrack: Optional[int] = 1) -> None:
        r"""
        Tracking method of the cavity feedback without beam in the accelerator
        """

        self.update_rf_variables()
        self.update_fb_variables()
        for i in range(n_pretrack):
            self.circuit_track(no_beam=True)

    def track(self, beam: BeamBaseClass) -> None:
        r"""
        Tracking method of the cavity feedback

        Parameters
        ----------
        beam
            Simulation beam object

        """
        # Update parameters from rest of BLonD classes
        self.update_rf_variables()
        self.update_fb_variables()

        # Get rf beam current
        (self.rf_beam_current(use_lowpass_filter=self.use_lowpass_filter),)

        # Tracking circuit model of feedback
        self.circuit_track()

        # Convert to amplitude and phase
        self.V_corr, self.alpha_sum = cartesian_to_polar(
            IQ_vector=self.V_ANT_FINE[-self.profile.n_bins :],
        )

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.V_corr /= self._parent_cavity.voltage[self.harmonic_index]
        self.phi_corr = self.alpha_sum - np.angle(
            np.interp(
                self.profile.hist_x, self.rf_centers, self.V_SET[-self.n_coarse :]
            )
        )

    def rf_beam_current(self, use_lowpass_filter: bool = False) -> None:
        r"""Calculate RF beam current from beam profile"""
        t_rev = float(  # TODO REMWORK/REMOVE
            (2 * np.pi * self._parent_cavity.harmonic[self.harmonic_index])
            / self.omega_rf
        )
        # Beam current from profile
        self.I_BEAM_COARSE[: self.n_coarse] = self.I_BEAM_COARSE[-self.n_coarse :]
        self.I_BEAM_FINE, self.I_BEAM_COARSE[-self.n_coarse :] = rf_beam_current(
            beam=self.beam,
            profile=self.profile,
            omega_c=self.omega_rf,
            T_rev=t_rev,
            use_lowpass_filter=use_lowpass_filter,
            downsample={"Ts": self.T_s, "points": self.n_coarse},
            external_reference=True,
            dT=self.dT,
        )

        # Convert RF beam currents to be in units of Amperes
        self.I_BEAM_FINE = self.I_BEAM_FINE / self.profile.hist_step
        self.I_BEAM_COARSE[-self.n_coarse :] = (
            self.I_BEAM_COARSE[-self.n_coarse :] / self.T_s
        )

    def set_point_from_rfstation(self) -> NumpyArray:
        r"""Computes the setpoint in I/Q based on the RF voltage in the RFStation"""

        V_set = polar_to_cartesian(
            self._parent_cavity.voltage[self.harmonic_index] / self.n_cavities,
            0,
        )

        return V_set * np.ones(self.n_coarse)
