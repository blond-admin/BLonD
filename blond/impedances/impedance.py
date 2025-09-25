# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Module to compute intensity effects**

:Authors: **Juan F. Esteban Mueller**, **Danilo Quartullo**,
          **Alexandre Lasheen**, **Markus Schwarz**
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np
from scipy.constants import e

from ..toolbox.next_regular import next_regular
from ..utils import bmath as bm
from ..utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Optional, Callable, Literal, Any, Dict, Optional

    from numpy.typing import NDArray as NumpyArray
    from cupy.typing import NDArray as CupyArray

    NDArray = NumpyArray | CupyArray

    from ..beam.beam import Beam
    from ..beam.profile import Profile
    from .impedance_sources import _ImpedanceObject, Resonators
    from ..input_parameters.rf_parameters import RFStation
    from ..utils.types import DeviceType, BeamProfileDerivativeModes

    MtwModeTypes = Literal["freq", "time"]

if TYPE_CHECKING:
    from typing import Literal


class TotalInducedVoltage:
    r"""
    Object gathering all the induced voltage contributions. The input is a
    list of objects able to compute induced voltages (InducedVoltageTime,
    InducedVoltageFreq, InductiveImpedance). All the induced voltages will
    be summed in order to reduce the computing time. All the induced
    voltages should have the same slicing resolution.

    Parameters
    ----------
    beam : Beam
        Beam object
    profile : Profile
        Profile object
    induced_voltage_list : _InducedVoltage list
        List of objects for which induced voltages have to be calculated

    Attributes
    ----------
    beam : Beam
        Copy of the Beam object in order to access the beam info
    profile : Profile
        Copy of the Profile object in order to access the profile info
    induced_voltage_list : _InducedVoltage list
        List of objects for which induced voltages have to be calculated
    induced_voltage : float array
        Array to store the computed induced voltage [V]
    time_array : float array
        Time array corresponding to induced_voltage [s]
    """

    @handle_legacy_kwargs
    def __init__(
        self,
        beam: Beam,
        profile: Profile,
        induced_voltage_list: list[_InducedVoltage],
    ):
        """
        Constructor.
        """
        # Copy of the Beam object in order to access the beam info.
        self.beam = beam

        # Copy of the Profile object in order to access the profile info.
        self.profile = profile

        # Induced voltage list.
        self.induced_voltage_list = induced_voltage_list

        # Induced voltage from the sum of the wake sources in V
        self.induced_voltage: NDArray = np.zeros(
            int(self.profile.n_slices), dtype=bm.precision.real_t, order="C"
        )

        # Time array of the wake in s
        self.time_array: NDArray = self.profile.bin_centers

    def reprocess(self):
        """
        Reprocess the impedance contributions. To be run when profile changes
        """
        for induced_voltage_object in self.induced_voltage_list:
            induced_voltage_object.process()

    def induced_voltage_sum(self):
        """
        Method to sum all the induced voltages in one single array.
        """
        # For MPI, to avoid calculating beam spectrum multiple times
        beam_spectrum_dict = {}
        temp_induced_voltage = 0

        for induced_voltage_object in self.induced_voltage_list:
            induced_voltage_object.induced_voltage_generation(
                beam_spectrum_dict
            )
            temp_induced_voltage += induced_voltage_object.induced_voltage[
                : self.profile.n_slices
            ]

        self.induced_voltage = temp_induced_voltage.astype(
            dtype=bm.precision.real_t, order="C", copy=False
        )

    def track(self):
        """
        Track method to apply the induced voltage kick on the beam.
        """

        self.induced_voltage_sum()
        bm.linear_interp_kick(
            dt=self.beam.dt,
            dE=self.beam.dE,
            voltage=self.induced_voltage,
            bin_centers=self.profile.bin_centers,
            charge=self.beam.particle.charge,
            acceleration_kick=0.0,
        )

    @handle_legacy_kwargs
    def track_ghosts_particles(self, ghost_beam: Beam):
        bm.linear_interp_kick(
            dt=ghost_beam.dt,
            dE=ghost_beam.dE,
            voltage=self.induced_voltage,
            bin_centers=self.profile.bin_centers,
            charge=self.beam.particle.charge,
            acceleration_kick=0.0,
        )

    def to_gpu(self, recursive: bool = True):
        """
        Transfer all necessary arrays to the GPU
        """
        # Check if to_gpu has been invoked already
        if hasattr(self, "_device") and self._device == "GPU":
            return

        if recursive:
            # transfer recursively objects
            for obj in self.induced_voltage_list:
                obj.to_gpu()

        import cupy as cp

        self.induced_voltage = cp.array(self.induced_voltage)
        self.time_array = cp.array(self.time_array)

        # to make sure it will not be called again
        self._device: DeviceType = "GPU"

    def to_cpu(self, recursive=True):
        """
        Transfer all necessary arrays back to the CPU
        """
        # Check if to_cpu has been invoked already
        if hasattr(self, "_device") and self._device == "CPU":
            # todo statement shouldn't be required
            return

        if recursive:
            # transfer recursively objects
            for obj in self.induced_voltage_list:
                obj.to_cpu()

        import cupy as cp

        self.induced_voltage = cp.asnumpy(self.induced_voltage)
        self.time_array = cp.asnumpy(self.time_array)
        # to make sure it will not be called again
        self._device: DeviceType = "CPU"


class _InducedVoltage:
    r"""
    Induced voltage parent class. Only for internal use (inheritance), not to
    be directly instantiated.

    Parameters
    ----------
    beam: Beam
        Beam object
    profile : Profile
        Profile object
    frequency_resolution : float, optional
        Frequency resolution of the impedance [Hz]
    wake_length : float, optional
        Wake length [s]
    multi_turn_wake : boolean, optional
        Multi-turn wake enable flag
    mtw_mode : str
        Multi-turn wake mode can be 'freq' or 'time' (default)
    rf_station : RFStation, optional
        RFStation object for turn counter and revolution period
    use_regular_fft : boolean
        use the next_regular function to ensure regular number for FFT
        calculations (default is True for efficient calculations, for
        better control of the sampling frequency False is preferred)

    Attributes
    ----------
    beam: Beam
        Copy of the Beam object in order to access the beam info
    profile : Profile
        Copy of the Profile object in order to access the profile info
    induced_voltage : float array
        Induced voltage from the sum of the wake sources in V
    wake_length_input : float
        Wake length [s]
    rf_params : RFStation
        RFStation object for turn counter and revolution period
    multi_turn_wake : boolean
        Multi-turn wake enable flag
    mtw_mode : str
        Multi-turn wake mode can be 'freq' or 'time' (default)
    use_regular_fft : boolean
        User set value to use (default) or not regular numbers for FFTs
    """

    @handle_legacy_kwargs
    def __init__(
        self,
        beam: Beam,
        profile: Profile,
        frequency_resolution: Optional[float] = None,
        wake_length: Optional[float] = None,
        multi_turn_wake: bool = False,
        mtw_mode: Optional[MtwModeTypes] = "time",  # todo fix
        rf_station: Optional[RFStation] = None,
        use_regular_fft: bool = True,
    ):
        # Beam object in order to access the beam info
        self.beam = beam

        # Profile object in order to access the profile info
        self.profile = profile

        # Induced voltage from the sum of the wake sources in V
        self.induced_voltage: NDArray = np.zeros(
            int(profile.n_slices), dtype=bm.precision.real_t, order="C"
        )

        # Wake length in s (optional)
        self.wake_length_input = wake_length

        # Frequency resolution of the impedance (optional)
        self.frequency_resolution_input = frequency_resolution

        # Use regular numbers for fft (optional)
        self.use_regular_fft = use_regular_fft

        # RFStation object for turn counter and revolution period
        self.rf_params = rf_station  # todo

        # Multi-turn wake enable flag
        self.multi_turn_wake = multi_turn_wake

        # Multi-turn wake mode can be 'freq' or 'time' (default). If 'freq'
        # is used, each turn the induced voltage of previous turns is shifted
        # in the frequency domain. For 'time', a linear interpolation is used.
        self.mtw_mode = mtw_mode

        ###############
        # Previously only declared in process()
        self.n_fft: int | None = None
        self.wake_length: float | None = None
        self.n_induced_voltage: int | None = None
        self.n_mtw_memory: int | None = None
        self.front_wake_buffer: int | None = None
        self.buffer_size: int | None = None
        self.mtw_memory: NDArray | None = None
        self.total_impedance: NDArray | None = None
        self.induced_voltage_generation: Callable | None = None

        self.freq_mtw: NDArray | None = None
        self.omegaj_mtw: NDArray | None = None
        self.shift_trev: Callable | None = None
        self.time_mtw: NDArray | None = None
        ###############

        self._device: DeviceType = "CPU"
        self.process()

    @property
    def mtw_mode(self) -> Literal["freq", "time"]:
        """Multi-turn wake mode can be 'freq' or 'time' (default). If 'freq'
        is used, each turn the induced voltage of previous turns is shifted
        in the frequency domain. For 'time', a linear interpolation is used."""
        return self._mtw_mode

    @mtw_mode.setter
    def mtw_mode(self, mtw_mode: Literal["freq", "time"]):
        if mtw_mode not in ("freq", "time"):
            raise ValueError(
                f"{mtw_mode=} not valid, choose either 'freq' or 'time'"
            )
        self._mtw_mode = mtw_mode

    @property
    def RFParams(self):
        from warnings import warn

        warn(
            "RFParams is deprecated, use rf_params",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.rf_params

    @RFParams.setter
    def RFParams(self, val):
        from warnings import warn

        warn(
            "RFParams is deprecated, use rf_params",
            DeprecationWarning,
            stacklevel=2,
        )
        self.rf_params = val

    def process(self):
        """
        Reprocess the impedance contributions. To be run when profile changes
        """

        if (
            self.wake_length_input is not None
            and self.frequency_resolution_input is None
        ):
            # Number of points of the induced voltage array
            self.n_induced_voltage = int(
                np.ceil(self.wake_length_input / self.profile.bin_size)
            )
            if self.n_induced_voltage < self.profile.n_slices:
                # WakeLengthError
                raise RuntimeError(
                    "Error: too short wake length. "
                    + "Increase it above {0:1.2e} s.".format(
                        self.profile.n_slices * self.profile.bin_size
                    )
                )
            # Wake length in s, rounded up to the next multiple of bin size
            self.wake_length = self.n_induced_voltage * self.profile.bin_size
        elif (
            self.frequency_resolution_input is not None
            and self.wake_length_input is None
        ):
            self.n_induced_voltage = int(
                np.ceil(
                    1
                    / (self.profile.bin_size * self.frequency_resolution_input)
                )
            )
            if self.n_induced_voltage < self.profile.n_slices:
                # FrequencyResolutionError
                raise RuntimeError(
                    "Error: too large frequency_resolution. "
                    + "Reduce it below {0:1.2e} Hz.".format(
                        1 / (self.profile.cut_right - self.profile.cut_left)
                    )
                )
            self.wake_length = self.n_induced_voltage * self.profile.bin_size
            # Frequency resolution in Hz
        elif (
            self.wake_length_input is None
            and self.frequency_resolution_input is None
        ):
            # By default, the wake_length is the slicing frame length
            self.wake_length = self.profile.cut_right - self.profile.cut_left
            self.n_induced_voltage = self.profile.n_slices
        else:
            raise RuntimeError(
                "Error: only one of wake_length or "
                + "frequency_resolution can be specified."
            )

        if self.multi_turn_wake:
            # Number of points of the memory array for multi-turn wake
            self.n_mtw_memory = self.n_induced_voltage
            self.front_wake_buffer = 0

            if self.mtw_mode == "freq":
                # In frequency domain, an extra buffer for a revolution turn is
                # needed due to the circular time shift in frequency domain
                self.buffer_size = np.ceil(
                    np.max(self.rf_params.t_rev) / self.profile.bin_size
                )
                # Extending the buffer to reduce the effect of the front wake
                # FIXME buffer_extra not declared
                self.buffer_size += np.ceil(
                    np.max(self.buffer_extra) / self.profile.bin_size
                )
                self.n_mtw_memory += int(self.buffer_size)
                # Using next regular for FFTs speedup
                if self.use_regular_fft:
                    self.n_mtw_fft = next_regular(self.n_mtw_memory)
                else:
                    self.n_mtw_fft = self.n_mtw_memory
                # Frequency and omega arrays
                self.freq_mtw = bm.rfftfreq(
                    self.n_mtw_fft, d=self.profile.bin_size
                )
                self.omegaj_mtw = 2.0j * np.pi * self.freq_mtw
                # Selecting time-shift method
                self.shift_trev = self.shift_trev_freq
            elif self.mtw_mode == "time":
                # Selecting time-shift method
                self.shift_trev = self.shift_trev_time
                # Time array
                self.time_mtw = np.linspace(
                    0,
                    self.wake_length,
                    self.n_mtw_memory,
                    endpoint=False,
                    dtype=bm.precision.real_t,
                )
            else:
                raise RuntimeError(f"Invalid value for {self.mtw_mode=}")
            # Array to add and shift in time the multi-turn wake over the turns
            self.mtw_memory = np.zeros(
                self.n_mtw_memory, dtype=bm.precision.real_t, order="C"
            )

            # Select induced voltage generation method to be used
            self.induced_voltage_generation = self.induced_voltage_mtw
        else:
            self.induced_voltage_generation = self.induced_voltage_1turn

    def induced_voltage_1turn(
        self, beam_spectrum_dict: Optional[dict] = None
    ):  # todo improve type hint for dict
        """
        Method to calculate the induced voltage at the current turn. DFTs are
        used for calculations in time and frequency domain (see classes below)
        """
        if beam_spectrum_dict is None:
            beam_spectrum_dict = dict()

        if self.n_fft not in beam_spectrum_dict:
            self.profile.beam_spectrum_generation(self.n_fft)
            beam_spectrum_dict[self.n_fft] = self.profile.beam_spectrum

        # self.profile.beam_spectrum_generation(self.n_fft)
        beam_spectrum = beam_spectrum_dict[self.n_fft]

        # FIXME total_impedance might be not None
        induced_voltage = -(
            self.beam.particle.charge
            * e
            * self.beam.ratio
            * bm.irfft(
                self.total_impedance.astype(
                    dtype=bm.precision.complex_t, order="C", copy=False
                )
                * beam_spectrum
            )
        )

        self.induced_voltage = induced_voltage[
            : self.n_induced_voltage
        ].astype(dtype=bm.precision.real_t, order="C", copy=False)

    def induced_voltage_mtw(
        self, beam_spectrum_dict: Optional[dict] = None
    ):  # todo improve type hint for dict
        """
        Method to calculate the induced voltage taking into account the effect
        from previous passages (multi-turn wake)
        """
        if beam_spectrum_dict is None:
            beam_spectrum_dict = dict()
        # Shift of the memory wake field by the current revolution period
        self.shift_trev()

        # Induced voltage of the current turn calculation
        self.induced_voltage_1turn(beam_spectrum_dict)

        # Setting to zero to the last part to remove the contribution from the
        # front wake
        self.induced_voltage[
            self.n_induced_voltage - self.front_wake_buffer :
        ] = 0

        # Add the induced voltage of the current turn to the memory from
        # previous turns
        self.mtw_memory[: self.n_induced_voltage] += self.induced_voltage

        self.induced_voltage = self.mtw_memory[: self.n_induced_voltage]

    def shift_trev_freq(self):
        """
        Method to shift the induced voltage by a revolution period in the
        frequency domain
        """

        t_rev = self.rf_params.t_rev[self.rf_params.counter[0]]
        # Shift in frequency domain
        induced_voltage_f = bm.rfft(self.mtw_memory, self.n_mtw_fft)
        induced_voltage_f *= bm.exp(self.omegaj_mtw * t_rev)
        self.mtw_memory = bm.irfft(induced_voltage_f)[: self.n_mtw_memory]
        # Setting to zero to the last part to remove the contribution from the
        # circular convolution
        self.mtw_memory[-int(self.buffer_size) :] = 0

    def shift_trev_time(self):
        """
        Method to shift the induced voltage by a revolution period in the
        time domain (linear interpolation). The interpolation is necessary to allow
        for a time shift, which is not an integer multiple of the delta_t of the
        mtw_memory array (necessary due to shifting t_rev during acceleration).
        The values, which are outside of the interpolation range are filled with 0s.
        """

        t_rev = self.rf_params.t_rev[self.rf_params.counter[0]]

        self.mtw_memory = bm.interp(
            self.time_mtw + t_rev,
            self.time_mtw,
            self.mtw_memory,
            left=0,
            right=0,
        )

    def _track(self):
        """
        Tracking method
        """

        self.induced_voltage_generation()

        bm.linear_interp_kick(
            dt=self.beam.dt,
            dE=self.beam.dE,
            voltage=self.induced_voltage,
            bin_centers=self.profile.bin_centers,
            charge=self.beam.particle.charge,
            acceleration_kick=0.0,
        )

    def to_gpu(self, recursive=True):
        raise NotImplementedError()

    def to_cpu(self, recursive=True):
        raise NotImplementedError()


class InducedVoltageTime(_InducedVoltage):
    r"""
    Induced voltage derived from the sum of several wake fields (time domain)

    Parameters
    ----------
    beam: Beam
        Beam object
    profile : Profile
        Profile object
    wake_source_list : list
        Wake sources list (e.g. list of Resonator objects)
    wake_length : float, optional
        Wake length [s]
    multi_turn_wake : boolean, optional
        Multi-turn wake enable flag
    rf_station : RFStation, optional
        RFStation object for turn counter and revolution period
    mtw_mode : boolean, optional
        Multi-turn wake mode can be 'freq' or 'time' (default)
    use_regular_fft : boolean
        use the next_regular function to ensure regular number for FFT
        calculations (default is True for efficient calculations, for
        better control of the sampling frequency False is preferred)

    Attributes
    ----------
    wake_source_list : list
        Wake sources list (e.g. list of Resonator objects)
    total_wake : float array
        Total wake array of all sources in :math:`\Omega / s`
    use_regular_fft : boolean
        User set value to use (default) or not regular numbers for FFTs
    """

    @handle_legacy_kwargs
    def __init__(
        self,
        beam: Beam,
        profile: Profile,
        wake_source_list: list[_ImpedanceObject],
        wake_length: Optional[float] = None,
        multi_turn_wake: bool = False,
        rf_station: Optional[RFStation] = None,
        mtw_mode: Optional[MtwModeTypes] = "time",
        use_regular_fft: bool = True,
    ):
        # Wake sources list (e.g. list of Resonator objects)
        self.wake_source_list: list[_ImpedanceObject] = wake_source_list

        # Total wake array of all sources in :math:`\Omega / s`
        self.total_wake: NDArray | int = (
            0  # todo better handling of initialization
        )

        ###################################
        # previously only defined in process
        # fixme frequency_resolution vs frequency_resolution_input of parent class
        self.frequency_resolution: Optional[float] = None
        self.time: NDArray | None = None
        ####################################

        # Call the __init__ method of the parent class [calls process()]
        _InducedVoltage.__init__(
            self,
            beam,
            profile,
            frequency_resolution=None,
            wake_length=wake_length,
            multi_turn_wake=multi_turn_wake,
            rf_station=rf_station,
            mtw_mode=mtw_mode,
            use_regular_fft=use_regular_fft,
        )

    def process(self):
        """
        Reprocess the impedance contributions. To be run when profile changes
        """

        _InducedVoltage.process(self)

        # Number of points for the FFT, equal to the length of the induced
        # voltage array + number of profile -1 to calculate a linear convolution
        # in the frequency domain. The next regular number is used for speed,
        # therefore the frequency resolution is always equal or finer than
        # the input value
        if self.use_regular_fft:
            self.n_fft = next_regular(
                int(self.n_induced_voltage) + int(self.profile.n_slices) - 1
            )
        else:
            self.n_fft = (
                int(self.n_induced_voltage) + int(self.profile.n_slices) - 1
            )

        # Frequency resolution in Hz
        self.frequency_resolution = 1 / (self.n_fft * self.profile.bin_size)

        # Time array of the wake in s
        self.time = np.arange(
            0,
            self.wake_length,
            self.wake_length / self.n_induced_voltage,
            dtype=bm.precision.real_t,
        )

        # Processing the wakes
        self.sum_wakes(self.time)

    def sum_wakes(self, time_array: NDArray):
        """
        Summing all the wake contributions in one total wake.
        """

        self.total_wake = np.zeros(time_array.shape)
        for wake_object in self.wake_source_list:
            wake_object.wake_calc(time_array)
            self.total_wake += wake_object.wake

        # Pseudo-impedance used to calculate linear convolution in the
        # frequency domain (padding zeros)
        self.total_impedance = bm.rfft(self.total_wake, self.n_fft)

    def to_gpu(self, recursive: bool = True):
        """
        Transfer all necessary arrays to the GPU
        """
        # Check if to_gpu has been invoked already
        if self._device == "GPU":
            return

        import cupy as cp

        self.induced_voltage = cp.array(self.induced_voltage)
        self.time = cp.array(self.time)
        self.total_wake = cp.array(self.total_wake)
        self.total_impedance = cp.array(self.total_impedance)
        if self.mtw_memory is not None:
            self.mtw_memory = cp.array(self.mtw_memory)
        if self.time_mtw is not None:
            self.time_mtw = cp.array(self.time_mtw)
        if self.omegaj_mtw is not None:
            self.omegaj_mtw = cp.array(self.omegaj_mtw)
        if self.freq_mtw is not None:
            self.freq_mtw = cp.array(self.freq_mtw)
        if self.total_wake is not None:
            self.total_wake = cp.array(self.total_wake)
        if self.time is not None:
            self.time = cp.array(self.time)

        # to make sure it will not be called again
        self._device: DeviceType = "GPU"

    def to_cpu(self, recursive=True):
        """
        Transfer all necessary arrays back to the CPU
        """
        # Check if to_cpu has been invoked already
        if self._device == "CPU":
            return

        import cupy as cp

        self.induced_voltage = cp.asnumpy(self.induced_voltage)
        self.time = cp.asnumpy(self.time)
        self.total_wake = cp.asnumpy(self.total_wake)
        self.total_impedance = cp.asnumpy(self.total_impedance)
        if self.mtw_memory is not None:
            self.mtw_memory = cp.asnumpy(self.mtw_memory)
        if self.time_mtw is not None:
            self.time_mtw = cp.asnumpy(self.time_mtw)
        if self.omegaj_mtw is not None:
            self.omegaj_mtw = cp.asnumpy(self.omegaj_mtw)
        if self.freq_mtw is not None:
            self.freq_mtw = cp.asnumpy(self.freq_mtw)
        if self.total_wake is not None:
            self.total_wake = cp.asnumpy(self.total_wake)
        if self.time is not None:
            self.time = cp.asnumpy(self.time)

        # to make sure it will not be called again
        self._device: DeviceType = "CPU"


class InducedVoltageFreq(_InducedVoltage):
    r"""
    Induced voltage derived from the sum of several impedances

    Parameters
    ----------
    beam: Beam
        Beam object
    profile : Profile
        Profile object
    impedance_source_list : list
        Impedance sources list (e.g. list of Resonator objects)
    frequency_resolution : float
        Frequency resolution of the impedance [Hz]
    multi_turn_wake : boolean
        Multi-turn wake enable flag
    front_wake_length : float
        Lenght [s] of the front wake (if any) for multi-turn wake mode
    rf_station : RFStation, optional
        RFStation object for turn counter and revolution period
    mtw_mode : boolean, optional
        Multi-turn wake mode can be 'freq' or 'time' (default)
    use_regular_fft : boolean
        use the next_regular function to ensure regular number for FFT
        calculations (default is True for efficient calculations, for
        better control of the sampling frequency False is preferred)

    Attributes
    ----------
    impedance_source_list : list
        Impedance sources list (e.g. list of Resonator objects)
    total_impedance : float array
        Total impedance array of all sources in* :math:`\Omega`
    front_wake_length : float
        Lenght [s] of the front wake (if any) for multi-turn wake mode
    use_regular_fft : boolean
        User set value to use (default) or not regular numbers for FFTs
    """

    @handle_legacy_kwargs
    def __init__(
        self,
        beam: Beam,
        profile: Profile,
        impedance_source_list: list[_ImpedanceObject],
        frequency_resolution: Optional[float] = None,
        multi_turn_wake: bool = False,
        front_wake_length: float = 0,
        rf_station: Optional[RFStation] = None,
        mtw_mode: Optional[MtwModeTypes] = "time",
        use_regular_fft: bool = True,
    ):
        # Impedance sources list (e.g. list of Resonator objects)
        self.impedance_source_list: list[_ImpedanceObject] = (
            impedance_source_list
        )

        # Total impedance array of all sources in* :math:`\Omega`
        self.total_impedance: NDArray | int = 0

        # Lenght in s of the front wake (if any) for multi-turn wake mode.
        # If the impedance calculation is performed in frequency domain, an
        # artificial front wake may appear. With this option, it is possible to
        # set to zero a portion at the end of the induced voltage array.*
        self.front_wake_length: float = front_wake_length

        ###############
        # Previously only declared in process()
        self.freq: NDArray | None = None
        self.frequency_resolution: Optional[float] = None

        ###############

        # Call the __init__ method of the parent class
        _InducedVoltage.__init__(
            self,
            beam,
            profile,
            wake_length=None,
            frequency_resolution=frequency_resolution,
            multi_turn_wake=multi_turn_wake,
            rf_station=rf_station,
            mtw_mode=mtw_mode,
            use_regular_fft=use_regular_fft,
        )

    def process(self):
        """
        Reprocess the impedance contributions. To be run when profile change
        """

        _InducedVoltage.process(self)

        # Number of points for the FFT. The next regular number is used for
        # speed, therefore the frequency resolution is always equal or finer
        # than the input value
        if self.use_regular_fft:
            self.n_fft = next_regular(self.n_induced_voltage)
        else:
            self.n_fft = self.n_induced_voltage

        self.profile.beam_spectrum_freq_generation(self.n_fft)

        # Frequency array and resolution of the impedance in Hz
        self.freq = self.profile.beam_spectrum_freq
        self.frequency_resolution = 1 / (self.n_fft * self.profile.bin_size)

        # Length of the front wake in frequency domain calculations
        if self.front_wake_length:
            self.front_wake_buffer = int(
                np.ceil(np.max(self.front_wake_length) / self.profile.bin_size)
            )

        # Processing the impedances
        self.sum_impedances(self.freq)

    def sum_impedances(self, freq: NDArray):
        """
        Summing all the wake contributions in one total impedance.
        """

        self.total_impedance = np.zeros(
            freq.shape, dtype=bm.precision.complex_t, order="C"
        )

        for impedance_source in self.impedance_source_list:
            impedance_source.imped_calc(freq)
            self.total_impedance += impedance_source.impedance

        # Factor relating Fourier transform and DFT
        self.total_impedance /= self.profile.bin_size

    def to_gpu(self, recursive: bool = True):
        """
        Transfer all necessary arrays to the GPU
        """
        # Check if to_gpu has been invoked already
        if self._device == "GPU":
            return

        import cupy as cp

        self.induced_voltage = cp.array(self.induced_voltage)
        self.freq = cp.array(self.freq)
        self.total_impedance = cp.array(self.total_impedance)
        if self.mtw_memory is not None:
            self.mtw_memory = cp.array(self.mtw_memory)
        if self.time_mtw is not None:
            self.time_mtw = cp.array(self.time_mtw)
        if self.freq_mtw is not None:
            self.freq_mtw = cp.array(self.freq_mtw)
        if self.omegaj_mtw is not None:
            self.omegaj_mtw = cp.array(self.omegaj_mtw)

        # to make sure it will not be called again
        self._device: DeviceType = "GPU"

    def to_cpu(self, recursive=True):
        """
        Transfer all necessary arrays back to the CPU
        """
        # Check if to_cpu has been invoked already
        if self._device and self._device == "CPU":
            return

        import cupy as cp

        self.induced_voltage = cp.asnumpy(self.induced_voltage)
        self.freq = cp.asnumpy(self.freq)
        self.total_impedance = cp.asnumpy(self.total_impedance)
        if self.mtw_memory is not None:
            self.mtw_memory = cp.asnumpy(self.mtw_memory)
        if self.time_mtw is not None:
            self.time_mtw = cp.asnumpy(self.time_mtw)
        if self.freq_mtw is not None:
            self.freq_mtw = cp.asnumpy(self.freq_mtw)
        if self.omegaj_mtw is not None:
            self.omegaj_mtw = cp.asnumpy(self.omegaj_mtw)

        # to make sure it will not be called again
        self._device: DeviceType = "CPU"


class InductiveImpedance(_InducedVoltage):
    r"""
    Constant imaginary Z/n impedance

    Parameters
    ----------
    beam: Beam
        Beam object
    profile : Profile
        Profile object
    Z_over_n : float array
        Constant imaginary Z/n program in* :math:`\Omega`.
    rf_station : RFStation
        RFStation object for turn counter and revolution period
    deriv_mode : string, optional
        Derivation method to compute induced voltage

    Attributes
    ----------
    Z_over_n : float array
        Constant imaginary Z/n program in* :math:`\Omega`.
    deriv_mode : string, optional
        Derivation method to compute induced voltage
    """

    @handle_legacy_kwargs
    def __init__(
        self,
        beam: Beam,
        profile: Profile,
        Z_over_n: float,
        rf_station: RFStation,
        deriv_mode: BeamProfileDerivativeModes = "gradient",
    ):
        # Constant imaginary Z/n program in* :math:`\Omega`.
        self.Z_over_n = Z_over_n

        # Derivation method to compute induced voltage
        self.deriv_mode = deriv_mode

        # Call the __init__ method of the parent class
        _InducedVoltage.__init__(self, beam, profile, rf_station=rf_station)

    def induced_voltage_1turn(
        self, beam_spectrum_dict: Dict[int, NumpyArray] = {}
    ):
        """
        Method to calculate the induced voltage through the derivative of the
        profile. The impedance must be a constant Z/n.
        """

        index = self.rf_params.counter[0]

        induced_voltage = -(
            self.beam.particle.charge
            * e
            / (2 * np.pi)
            * self.beam.ratio
            * self.Z_over_n[index]
            * self.rf_params.t_rev[index]
            / self.profile.bin_size
            * self.profile.beam_profile_derivative(self.deriv_mode)[1]
        )

        self.induced_voltage = (
            induced_voltage[: self.n_induced_voltage]
        ).astype(dtype=bm.precision.real_t, order="C", copy=False)

    def to_gpu(self, recursive=True):
        """
        Transfer all necessary arrays to the GPU
        """
        # Check if to_gpu has been invoked already
        if hasattr(self, "_device") and self._device == "GPU":
            return

        import cupy as cp

        self.induced_voltage = cp.array(self.induced_voltage)
        self.Z_over_n = cp.array(self.Z_over_n)
        # to make sure it will not be called again
        self._device: DeviceType = "GPU"

    def to_cpu(self, recursive=True):
        """
        Transfer all necessary arrays back to the CPU
        """
        # Check if to_cpu has been invoked already
        if hasattr(self, "_device") and self._device == "CPU":
            return

        import cupy as cp

        self.induced_voltage = cp.asnumpy(self.induced_voltage)
        self.Z_over_n = cp.asnumpy(self.Z_over_n)
        # to make sure it will not be called again
        self._device: DeviceType = "CPU"


class InducedVoltageResonator(_InducedVoltage):
    r"""
    *Calculates the induced voltage of several resonators for arbitrary
    line density. It does so by linearly interpolating the line density and
    solving the convolution integral with the resonator impedance analytically.
    The line density need NOT be sampled at equidistant points. The times when
    the induced voltage is calculated need to be the same where the line
    density is sampled. If no time_array is passed, the induced voltage is
    evaluated at the points of the line density. This is necessary for
    compatibility with other functions that calculate the induced voltage.
    From the longest decay constant of the given modes, the function determines
    where to compute the induced voltages for the following turns in the
    multi-turn-wake case.
    Currently, the function requires the all quality factors :math:`Q>0.5`.*

    Parameters
    ----------
    beam: Beam
        Beam object
    profile : Profile
        Profile object
    resonators : Resonators
        Resonators object
    time_array : float array, optional
        Array of time values where the induced voltage is calculated.
        If left out, the induced voltage is calculated at the times of the line
        density.
    frequency_resolution : float, optional
        Frequency resolution of the impedance [Hz]. This is ignored in the context
        of this subclass
    wake_length : float, optional
        This is ignored in the context of this subclass
        , as the wake_length will be controlled by the setting of the
        decay percentage
    multi_turn_wake : boolean, optional
        Multi-turn wake enable flag
    mtw_mode : str
        Multi-turn wake mode can be 'freq' or 'time' (default). 'freq' is ignored in the
        context of this class.
    rf_station : RFStation, optional
        RFStation object for turn counter and revolution period
    use_regular_fft : boolean
        As FFTs are not used, the parameter will not change anything
    resonators: list of Resonators
        This input is necessary for the function to not throw an error and should include all
        resonators meant to be modeled by this class
    time_decay_factor: float, between 1 and 0.
        Fraction of the resonator with smallest decay constant $\tau$ to be decayed over a
        number of turns which is calculated. Default 0.01. (1%).


    Attributes
    ----------
    beam: Beam
        Copy of the Beam object in order to access the beam info.
    profile : Profile
        Copy of the Profile object in order to access the line density.
    time_array : float array
        Array of time values where the induced voltage is calculated.
        If left out, the induced voltage is calculated at the times of the
        line density
    R, omega_r, Q : lists of float
        Resonators parameters
    n_resonators : int
        Number of resonators
    induced_voltage : float array
        Computed induced voltage [V]
    """

    @handle_legacy_kwargs
    def __init__(
        self,
        beam: Beam,
        profile: Profile,
        resonators: Resonators,
        frequency_resolution: Optional[float] = None,
        wake_length: Optional[float] = None,
        multi_turn_wake: bool = False,
        mtw_mode: Optional[MtwModeTypes] = "time",
        rf_station: Optional[RFStation] = None,
        use_regular_fft: bool = True,
        time_decay_factor: Optional[float] = 0.01,
        time_array: Optional[float] = None,
    ):
        # Test if one or more quality factors is smaller than 0.5.
        if sum(resonators.Q < 0.5) > 0:
            # ResonatorError
            raise RuntimeError("All quality factors Q must be larger than 0.5")
        if mtw_mode != "time":
            warnings.warn(
                "InducedVoltageResonator only allows for 'time' mtw_mode, 'freq' will be ignored"
            )
        if wake_length is not None:
            warnings.warn(
                "InducedVoltageResonator ignores the setting of wake_length"
            )
        if frequency_resolution is not None:
            warnings.warn(
                "InducedVoltageResonator ignores the setting of frequency_resolution"
            )
        if not use_regular_fft:
            warnings.warn(
                "use_regular_fft is not supported and will be ignored by InducedVoltageResonator",
                UserWarning,
            )

        # Copy of the Beam object in order to access the beam info.
        self.beam = beam
        # Copy of the Profile object in order to access the line density.
        self.profile = profile

        # Make the time array necessary for wake calculation.
        # If the time decay is longer than n_turns of simulation, the induced voltage is calculated for n_turns.
        # Length of the time decay dictated by the time decay factor.

        if multi_turn_wake:
            if time_array is not None:
                warnings.warn(
                    "InducedVoltageResonator ignores time_array when multi_turn_wake=True"
                )

            # take the maximum of the resonators
            decay_time = 2 * np.max(resonators.Q / resonators.omega_R)
            decay_turns = np.ceil(
                np.log(time_decay_factor)
                / np.log(e)
                * decay_time
                / np.min(rf_station.t_rev)
            )

            n_turns_calculation = min(int(decay_turns), rf_station.n_turns)
            potential_min_cav = rf_station.phi_s[0] / rf_station.omega_rf[0, 0]
            min_index = np.abs(
                profile.bin_centers[0] - potential_min_cav
            ).argmin()

            self.time_array = np.array([])

            for turn_ind in range(n_turns_calculation):
                self.time_array = np.append(
                    self.time_array,
                    rf_station.t_rev[turn_ind] * turn_ind
                    + np.linspace(
                        profile.bin_centers[0],
                        profile.bin_centers[-1]
                        + 2
                        * (
                            profile.bin_centers[min_index]
                            - profile.bin_centers[0]
                        ),
                        profile.n_slices + 2 * min_index,
                    ),
                )
            self.atLineDensityTimes = False

        else:
            # Optional array of time values where the induced voltage is calculated.
            # If left out, the induced voltage is calculated at the times of the
            # line density.
            if time_array is None:
                self.time_array = self.profile.bin_centers
                self.atLineDensityTimes = True
            else:
                self.time_array = time_array
                self.atLineDensityTimes = False

        self.array_length = len(self.profile.bin_centers)
        self.n_time = len(self.time_array)
        # Copy of the shunt impedances of the Resonators in* :math:`\Omega`
        self.R = resonators.R_S
        # Copy of the resonant frequencies of the Resonators in 1/s
        self.omega_r = resonators.omega_R  # resonant frequencies [1/s]
        # Copy of the quality factors of the Resonators
        self.Q = resonators.Q
        # Number of resonators
        self.n_resonators = len(self.R)
        # For internal use
        self._Qtilde = self.Q * np.sqrt(1.0 - 1.0 / (4.0 * self.Q**2.0))
        self._reOmegaP = self.omega_r * self._Qtilde / self.Q
        self._imOmegaP = self.omega_r / (2.0 * self.Q)

        # Each the 'n_resonator' rows of the matrix holds the induced voltage
        # at the 'n_time' time-values of one cavity. For internal use.
        self._tmp_matrix = np.ones(
            (self.n_resonators, self.n_time),
            dtype=bm.precision.real_t,
            order="C",
        )

        # Slopes of the line segments. For internal use.
        self._kappa1 = np.zeros(
            int(self.profile.n_slices - 1),
            dtype=bm.precision.real_t,
            order="C",
        )

        # Matrix to hold n_times many time_array[t]-bin_centers arrays.
        self._deltaT = np.zeros(
            (self.n_time, self.profile.n_slices),
            dtype=bm.precision.real_t,
            order="C",
        )

        self.induced_voltage = np.zeros(
            self.n_time, dtype=bm.precision.real_t, order="C"
        )

        wake_length = len(self.time_array) * self.profile.bin_size

        # Call the __init__ method of the parent class [calls process()]
        super().__init__(
            beam=beam,
            profile=profile,
            frequency_resolution=None,
            wake_length=wake_length,
            multi_turn_wake=multi_turn_wake,
            rf_station=rf_station,
            mtw_mode="time",
            use_regular_fft=True,
        )

    def process(self):
        r"""
        Reprocess the impedance contributions. To be run when slicing changes
        """

        _InducedVoltage.process(self)

        # Since profile object changed, need to assign the proper dimensions to
        # _kappa1 and _deltaT
        self._kappa1 = np.zeros(
            int(self.profile.n_slices - 1),
            dtype=bm.precision.real_t,
            order="C",
        )
        self._deltaT = np.zeros(
            (self.n_time, self.profile.n_slices),
            dtype=bm.precision.real_t,
            order="C",
        )
        self.induced_voltage = np.zeros(
            self.n_time, dtype=bm.precision.real_t, order="C"
        )

    def induced_voltage_1turn(self, beam_spectrum_dict: Dict[Any, Any] = {}):
        r"""
        Method to calculate the induced voltage through linearly
        interpolating the line density and applying the analytic equation
        to the result.
        """
        self.induced_voltage, self._deltaT = (
            bm.resonator_induced_voltage_1_turn(
                self._kappa1,
                self.profile.n_macroparticles,
                self.profile.bin_centers,
                self.profile.bin_size,
                self.n_time,
                self._deltaT,
                self.time_array,
                self._reOmegaP,
                self._imOmegaP,
                self._Qtilde,
                self.n_resonators,
                self.omega_r,
                self.Q,
                self._tmp_matrix,
                self.beam.particle.charge,
                self.beam.n_macroparticles,
                self.beam.ratio,
                self.R,
                self.induced_voltage,
                bm.precision.real_t,
            )
        )

    def induced_voltage_mtw(self, beam_spectrum_dict={}):
        r"""
        Induced voltage method for InducedVoltageResonator.
        mtw_memory is shifted by one turn, setting the values to 0.
        The current turn's induced voltage is added to the memory of the previous turn.
        Implementation by F. Batsch.
        """
        # Shift the entries in array by 1 t_rev and set to 0
        self.mtw_memory = np.append(
            self.mtw_memory, np.zeros(self.array_length)
        )
        # Remove one turn length of memory
        self.mtw_memory = self.mtw_memory[self.array_length :]
        # Induced voltage of the current turn
        self.induced_voltage_1turn(beam_spectrum_dict)
        # Add induced voltage of the current turn, up to array length n_time, to the previous turn
        self.mtw_memory[: int(self.n_time)] += self.induced_voltage
        # Save the total induced voltage up to array length n_time
        self.induced_voltage = self.mtw_memory[: self.n_time]

    def to_gpu(self, recursive: bool = True):
        """
        Transfer all necessary arrays to the GPU
        """
        # Check if to_gpu has been invoked already
        if hasattr(self, "_device") and self._device == "GPU":
            return
        # todo for mtw
        import cupy as cp

        self.induced_voltage = cp.array(self.induced_voltage)
        if self.mtw_memory is not None:
            self.mtw_memory = cp.array(self.mtw_memory)
        self._kappa1 = cp.array(self._kappa1)
        self._deltaT = cp.array(self._deltaT)
        self.time_array = cp.array(self.time_array)
        self._tmp_matrix = cp.array(self._tmp_matrix)
        # to make sure it will not be called again
        self._device: DeviceType = "GPU"

    def to_cpu(self, recursive=True):
        """
        Transfer all necessary arrays back to the CPU
        """
        # Check if to_cpu has been invoked already
        if hasattr(self, "_device") and self._device == "CPU":
            return

        import cupy as cp

        # todo for mtw
        self.induced_voltage = cp.asnumpy(self.induced_voltage)
        if self.mtw_memory is not None:
            self.mtw_memory = cp.asnumpy(self.mtw_memory)
        self._kappa1 = cp.asnumpy(self._kappa1)
        self._deltaT = cp.asnumpy(self._deltaT)
        self.time_array = cp.asnumpy(self.time_array)
        self._tmp_matrix = cp.asnumpy(self._tmp_matrix)

        # to make sure it will not be called again
        self._device: DeviceType = "CPU"
