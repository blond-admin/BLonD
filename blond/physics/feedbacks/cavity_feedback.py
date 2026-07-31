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

import dataclasses
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

import numpy as np

from blond.core.helpers import int_from_float_with_warning
from blond.core.ring.helpers import requires
from blond.physics.feedbacks.base import LocalFeedback
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


class TwoTurnArray:
    """
    Wrapper for a NumPy Array of dimension (2, N) array representing [previous turn, current turn].

    The class is intended to be used with local feedback systems.
    Indexing with a non-negative int/slice reads from CURR, as normal.
    Indexing with a negative int/slice transparently reaches back into
    PREV, as if CURR were conceptually preceded by PREV — no full
    concatenation needed for single-sample access.

    Parameters
    ----------
    n_samples
        Number of samples per turn.
    dtype
        The data-type stored in the array.
    """

    __slots__ = ("_data",)

    def __init__(self, n_samples: int, dtype=np.float64):
        self._data = np.zeros((2, n_samples), dtype=dtype)

    @property
    def n_samples(self) -> int:
        """
        Number of samples per turn.

        Returns
        -------
        n_samples
            Number of samples of each turn.
        """
        return self._data.shape[1]

    @property
    def prev(self) -> NumpyArray:
        """
        The array of the previous turn.

        Returns
        -------
        prev
            The array of values from the previous turn.
        """
        return self._data[0]

    @prev.setter
    def prev(self, array: NumpyArray):
        """
        Set the values of the previous-turn array.

        Parameters
        ----------
        array
            Array of values to set the previous-turn array with.
        """
        self._data[0] = array

    @property
    def curr(self) -> NumpyArray:
        """
        The array of the current turn.

        Returns
        -------
        prev
            The array of values from the current turn.
        """
        return self._data[1]

    @curr.setter
    def curr(self, array: NumpyArray):
        """
        Set the values of the current-turn array.

        Parameters
        ----------
        array
            Array of values to set the current-turn array with.
        """
        self._data[1] = array

    def shift(self) -> None:
        """Shift the current turn into the previous."""
        self._data[0] = self._data[1]

    @property
    def full(self) -> NumpyArray:
        """
        Flat array spanning the previous and current turn.

        Returns
        -------
        full
            Array spanning the previous and current turn.
        """
        return np.concatenate(self._data)

    def __getitem__(self, key: int | np.integer | slice):
        """
        Get elements from the current turn.

        Negative values for the key correspond to values
        indices in the previous turn.

        Parameters
        ----------
        key
            The key for obtaining the values on the array.

        Returns
        -------
        values
            The values corresponding to the key.
        """
        n = self.n_samples
        if isinstance(key, (int, np.integer)):
            if key >= 0:
                return self.curr[key]
            idx = n + key
            if idx < 0:
                raise IndexError(
                    f"index {key} reaches back further than one turn of history"
                )
            return self.prev[idx]

        if isinstance(key, slice):
            start = 0 if key.start is None else key.start
            stop = n if key.stop is None else key.stop
            step = 1 if key.step is None else key.step
            if start >= 0 and stop >= 0:
                return self.curr[start:stop:step]
            # boundary-crossing or fully negative slice: only concatenate
            # the (small) region actually needed
            concat = np.concatenate(self._data)
            lo = n + start if start < 0 else start
            hi = n + stop if stop < 0 else stop
            return concat[lo:hi:step]

        raise TypeError(f"unsupported index type: {type(key)}")

    def __setitem__(self, key, value) -> None:
        """
        Set elements in the two-turn array.

        Parameters
        ----------
        key
            The kay of the elements you want to change.
        value
            The new values of the elements corresponding to the key.
        """
        if isinstance(key, (int, np.integer)) and key < 0:
            raise IndexError("cannot write into previous-turn history")
        self.curr[key] = value

    def __len__(self) -> int:
        """
        The length of the turns.

        Returns
        -------
        n_samples
            The length of each turn in number of samples.
        """
        return self.n_samples

    def __repr__(self) -> str:
        """
        Printable representation of the TwoTurnArray.

        Returns
        -------
        info_string
            String showing previous and current turn elements.
        """
        return f"TwoTurnArray(prev={self.prev!r}, curr={self.curr!r})"


@dataclass
class BufferBase(ABC):
    """
    Base class for the buffer container used for the LocalFeedbacks.

    This class is intended to be used to store the buffers of a certain
    time-resolution inside the local feedbacks.

    Attributes
    ----------
    v_setpoint
        The setpoint voltage [V] buffer of the feedback system.
    v_ant
        Buffer containing the RF voltage measured by the antenna [V].
    i_beam
        Buffer containing the RF component of the beam current [A].
    i_gen
        Buffer containing the forward current [A] of the generator.
    """

    # Base parameters
    samples_per_turn: int

    # Base buffers needed for any CavityFeedback class
    v_setpoint: NumpyArray | TwoTurnArray = field(init=False)
    v_ant: NumpyArray | TwoTurnArray = field(init=False)
    i_beam: NumpyArray | TwoTurnArray = field(init=False)
    i_gen: NumpyArray | TwoTurnArray = field(init=False)

    def __post_init__(self):
        """Initialize the buffers."""
        self.v_setpoint = self._make_array(dtype=complex)
        self.v_ant = self._make_array(dtype=complex)
        self.i_beam = self._make_array(dtype=complex)
        self.i_gen = self._make_array(dtype=complex)

    @abstractmethod  # pragma: no cover
    def _make_array(self, dtype) -> NumpyArray | TwoTurnArray:
        """
        Return the array for a buffer with a certain data-type.

        Parameters
        ----------
        dtype
            The data-type the buffer should store.
        """
        raise NotImplementedError

    def shift(self):
        """Roll every two-turn array: curr -> prev, ready for a new curr."""
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if isinstance(val, TwoTurnArray):
                val.shift()


@dataclass
class OneTurnBufferBase(BufferBase):
    """
    Base class for buffers spanning a single turn used for the LocalFeedbacks.

    This class is intended to be used to store the buffers of a certain
    time-resolution inside the local feedbacks.

    Attributes
    ----------
    v_setpoint
        The setpoint voltage [V] buffer of the feedback system.
    v_ant
        Buffer containing the RF voltage measured by the antenna [V].
    i_beam
        Buffer containing the RF component of the beam current [A].
    i_gen
        Buffer containing the forward current [A] of the generator.
    """

    def _make_array(self, dtype) -> NumpyArray:
        """
        Return the array for a buffer with a certain data-type.

        Parameters
        ----------
        dtype
            The data-type the buffer should store.

        Returns
        -------
        array
            An array object initialized with the correct number of samples
            and data type.

        Notes
        -----
        These arrays will span a single turn only.
        """
        return np.zeros(self.samples_per_turn, dtype=dtype)


@dataclass
class TwoTurnBufferBase(BufferBase):
    """
    Base class for buffers spanning two turns used for the LocalFeedbacks.

    This class is intended to be used to store the buffers of a certain
    time-resolution inside the local feedbacks.

    Attributes
    ----------
    v_setpoint
        The setpoint voltage [V] buffer of the feedback system.
    v_ant
        Buffer containing the RF voltage measured by the antenna [V].
    i_beam
        Buffer containing the RF component of the beam current [A].
    i_gen
        Buffer containing the forward current [A] of the generator.
    """

    def _make_array(self, dtype) -> TwoTurnArray:
        """
        Return the array for a buffer with a certain data-type.

        Parameters
        ----------
        dtype
            The data-type the buffer should store.

        Returns
        -------
        array
            An array object initialized with the correct number of samples
            and data type.

        Notes
        -----
        These arrays will span two turns.
        """
        return TwoTurnArray(self.samples_per_turn, dtype=dtype)


BufferCoarse = TypeVar(
    "BufferCoarse", bound=TwoTurnBufferBase | OneTurnBufferBase
)
BufferFine = TypeVar("BufferFine", bound=OneTurnBufferBase)


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
