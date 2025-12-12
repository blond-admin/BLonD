# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Solvers to calculate the wake potential from impedance sources.

Authors
-------
Alexandre Lasheen
Danilo Quartullo,
Juan F. Esteban Mueller
Leonard Thiele
Markus Schwarz
Simon Lauber
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from scipy.constants import elementary_charge as e
from scipy.fft import next_fast_len

from blond.core.backends.backend import backend
from blond.core.base import DynamicParameter
from blond.core.beam.base import BeamBaseClass
from blond.core.ring.helpers import requires
from blond.core.simulation.simulation import Simulation
from blond.generals.warnings_ import NotTestedWarning
from blond.physics.impedances.base import (
    FreqDomain,
    TimeDomain,
    WakeField,
    WakeFieldSolver,
)
from blond.physics.impedances.sources import InductiveImpedance, Resonators
from blond.physics.profiles import (
    DynamicProfileConstCutoff,
    DynamicProfileConstNBins,
    StaticProfile,
)

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray


class InductiveImpedanceSolver(WakeFieldSolver):
    """Wakefield solver specialized for :class:`blond.physics.impedances.sources.InductiveImpedance`."""

    def __init__(self):
        super().__init__()
        self._beam: BeamBaseClass | None = None
        self._Z_over_n: float | None = None
        self._turn_i: DynamicParameter | None = None
        self._parent_wakefield: WakeField | None = None
        self._simulation: Simulation | None = None

    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ):
        """
        Lateinit method when WakeField is late-initialized.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        parent_wakefield
            Wakefield that this solver affiliated to.
        """
        self._parent_wakefield = parent_wakefield
        assert all(
            isinstance(o, InductiveImpedance) for o in parent_wakefield.sources
        )
        impedances: tuple[InductiveImpedance, ...] = parent_wakefield.sources
        self._Z_over_n = backend.float(
            np.sum(np.array([o.Z_over_n for o in impedances]))
        )
        self._turn_i = simulation.turn_i
        self._simulation = simulation

    def calc_induced_voltage(
        self, beam: BeamBaseClass
    ) -> NumpyArray | CupyArray:
        """
        Calculate the induced voltage based on the beam profile and beam parameters.

        Parameters
        ----------
        beam
            Simulation object of a particle beam.

        Returns
        -------
        induced_voltage
            Induced voltage, in [V].
        """
        _factor = self._hist_y_to_intensity_factor(
            beam=beam,
            profile=self._parent_wakefield.profile,
        )
        factor = backend.float(
            (_factor / (2 * np.pi))
            * (self._simulation.ring.circumference / beam.reference_velocity)
            / self._parent_wakefield.profile.hist_step
        )
        diff = self._parent_wakefield.profile.gradient_hist_y
        return factor * diff * self._Z_over_n


class PeriodicFreqSolver(WakeFieldSolver):
    """
    General wakefield solver to calculate wake-fields via frequency domain.

    Parameters
    ----------
    t_periodicity
        Periodicity that is assumed for fast fourier transform, in [s].
    allow_next_fast_len
        Allow to slightly change `t_periodicity` for
        faster execution of fft via `scipy.fft.next_fast_len`.

    Attributes
    ----------
    allow_next_fast_len
        Allow to slightly change `t_periodicity` for
        faster execution of fft via `scipy.fft.next_fast_len`.
    expect_profile_change
        If true, reloads internal data on each
        `calc_induced_voltage` for proper updating with
        dynamic parameters.

    Notes
    -----
    This solver is intended for the case, that the
    length of the beam profile is in the order of time as one revolution
    around the synchrotron takes ( long profiles).
    """

    def __init__(
        self,
        t_periodicity: float | None = None,
        allow_next_fast_len: bool = False,
    ):
        super().__init__()
        self.allow_next_fast_len = allow_next_fast_len
        self.expect_profile_change: bool = False
        self.expect_impedance_change = False

        self._t_periodicity = t_periodicity
        self._parent_wakefield: WakeField | None = None
        self._n_time: int | None = None
        self._n_freq: int | None = None
        self._freq_x: NumpyArray | None = None
        self._freq_y: NumpyArray | None = None

        self._simulation: Simulation | None = None

        self._freq_y_needs_update = True  # at least one update

        self._induced_voltage_buffer = {}

    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ):
        """
        Lateinit method when WakeField is late-initialized.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        parent_wakefield
            Wakefield that this solver affiliated to.
        """
        if self._t_periodicity is None:
            self._t_periodicity = simulation.magnetic_cycle.get_t_rev_init(
                circumference=simulation.ring.circumference,
                turn_i_init=0,
                t_init=0,
                particle_type=simulation.magnetic_cycle.reference_particle,
            )
            print(f"Set t_periodicity={self._t_periodicity} for {self}")
        self._simulation = simulation
        if parent_wakefield.profile is not None:
            is_static = isinstance(parent_wakefield.profile, StaticProfile)
            is_dynamic = isinstance(
                parent_wakefield.profile,
                DynamicProfileConstCutoff | DynamicProfileConstNBins,
            )
            self._parent_wakefield = parent_wakefield
            self._update_internal_data()

            if is_dynamic:
                if self.expect_profile_change is False:
                    warn(
                        f"Because you are using"
                        f" a `{type(parent_wakefield.profile)}`,"
                        f" the variable `update_on_calc` is set to"
                        f" True, which might impact performance."
                        f" Set True by yourself to deactivate this warning.",
                        stacklevel=2,
                    )
                    self.expect_profile_change = True
            elif is_static:
                pass
            else:
                raise NotImplementedError(
                    f"Unrecognized type(profile) = {type(parent_wakefield.profile)}"
                )
        else:
            raise Exception(f"{parent_wakefield.profile=}")

        for source in self._parent_wakefield.sources:
            if source.is_dynamic:
                if self.expect_impedance_change is False:
                    warn(
                        f"Because `{source}` is dynamic,"
                        f" the variable `expect_impedance_change` is set to"
                        f" True, which might impact performance."
                        f" Set True by yourself to deactivate this warning.",
                        stacklevel=2,
                    )
                    self.expect_impedance_change = True
                break

    @property
    def t_periodicity(self) -> float:
        """
        Periodicity that is assumed for fast fourier transform in  [s].

        Returns
        -------
        t_periodicity
            Periodicity for FFT, in [s].
        """
        return self._t_periodicity

    @t_periodicity.setter
    def t_periodicity(self, t_periodicity: float):
        self._t_periodicity = t_periodicity
        self._update_internal_data()
        self._freq_y_needs_update = True

    def _update_internal_data(self):
        """Rebuild internal data model."""
        assert self._parent_wakefield.profile is not None
        self._n_time = int(
            round(
                self._t_periodicity / self._parent_wakefield.profile.hist_step,
                0,
            )
        )
        assert self._n_time >= self._parent_wakefield.profile.n_bins, (
            f"Increase `t_periodicity` so that it is at least"
            f" as long as the beam profile or decrease the profile size."
            f" {self._n_time=}, but {self._parent_wakefield.profile.n_bins=}."
        )
        if self.allow_next_fast_len:
            self._n_time = next_fast_len(
                self._n_time,
                real=True,
            )

        self._freq_x = np.fft.rfftfreq(
            self._n_time, d=self._parent_wakefield.profile.hist_step
        ).astype(backend.float)
        self._n_freq = len(self._freq_x)

        self._freq_y_needs_update = True

    def _update_impedance_sources(self, beam: BeamBaseClass) -> None:
        """
        Update `_freq_y` array if `self._freq_y_needs_update=True`.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        # be lazy
        if not self._freq_y_needs_update:
            return

        if (self._freq_y is None) or (
            self._freq_x.shape != self._freq_y.shape
        ):
            self._freq_y = backend.zeros_like(
                self._freq_x, dtype=backend.complex
            )
        else:
            self._freq_y[:] = 0 + 0j
        for source in (
            self._parent_wakefield.sources
        ):  # todo update only dynamic sources
            if isinstance(source, FreqDomain):
                freq_y = source.get_impedance(
                    freq_x=self._freq_x,
                    simulation=self._simulation,
                    beam=beam,  # FIXME
                )
                assert not np.any(np.isnan(freq_y)), f"{type(source).__name__}"

                self._freq_y += backend.array(freq_y, dtype=backend.complex)
                # potentially on gpu
            else:
                raise Exception(
                    "Can only accept impedance that support `FreqDomain`"
                )
        # Factor relating Fourier transform and DFT
        self._freq_y /= self._parent_wakefield.profile.hist_step

        self._freq_y_needs_update = False  # after update, set lazy flag

    def calc_induced_voltage(
        self, beam: BeamBaseClass
    ) -> NumpyArray | CupyArray:
        """
        Calculate the induced voltage based on the beam profile and beam parameters.

        Parameters
        ----------
        beam
            Simulation object of a particle beam.

        Returns
        -------
        induced_voltage
            Induced voltage, in [V].
        """
        if self.expect_profile_change:  # dynamic profiles
            self._update_internal_data()  # might cause performance issues :(
        elif self.expect_impedance_change:
            # always trigger update
            self._freq_y_needs_update = True

        self._update_impedance_sources(beam=beam)

        _factor = self._hist_y_to_intensity_factor(
            beam=beam,
            profile=self._parent_wakefield.profile,
        )

        key = len(self._freq_y)  # todo
        if key in self._induced_voltage_buffer:
            # use `out` variable of fft to avoid array creation
            if backend.is_gpu:
                # At the time of writing (2025), out is not a keyword argument
                # of cp.fft.rfft, but might be in future.
                out = backend.fft.irfft(
                    self._freq_y
                    * self._parent_wakefield.profile.beam_spectrum(
                        n_fft=self._n_time
                    ),
                )
            else:
                out = self._induced_voltage_buffer[key]
                backend.fft.irfft(
                    self._freq_y
                    * self._parent_wakefield.profile.beam_spectrum(
                        n_fft=self._n_time
                    ),
                    out=out,
                )

            out *= _factor
            self._induced_voltage_buffer[key] = out
        else:
            # create array and safe it to buffer
            self._induced_voltage_buffer[key] = _factor * np.fft.irfft(
                self._freq_y
                * self._parent_wakefield.profile.beam_spectrum(
                    n_fft=self._n_time
                ),
            )
        # calculation in frequency domain must be with full periodicity.
        # The profile and corresponding induced voltage is only a part of
        # the full periodicity and must be thus truncated
        return self._induced_voltage_buffer[key][
            : self._parent_wakefield.profile.n_bins
        ]


class TimeDomainFftSolver(WakeFieldSolver):
    """
    Solver to calculate induced voltage using fftconvolve(wake,profile).

    Parameters
    ----------
    allow_next_fast_len : bool, optional
        If True, use scipy's next_fast_len to optimize FFT length.
        Default is True.

    Notes
    -----
    This method is intended for beam profiles that are only a fraction of
    the synchrotron revolution time (short profiles).
    """

    def __init__(
        self,
        allow_next_fast_len: bool = True,
    ):
        super().__init__()
        self.expect_impedance_change = False
        self._allow_next_fast_len = allow_next_fast_len

        self._parent_wakefield: WakeField | None = None
        self._wake_imp_y: NumpyArray | None = None
        self._simulation: Simulation | None = None

        self._wake_imp_y_needs_update = True  # update at least once

    @requires(["MagneticCycleBase"])  # because InductiveImpedance.get_
    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ) -> None:
        """
        Lateinit method when WakeField is late-initialized.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        parent_wakefield
            Wakefield that this solver affiliated to.
        """
        self._simulation = simulation
        if parent_wakefield.profile is not None:
            is_dynamic = isinstance(
                parent_wakefield.profile,
                DynamicProfileConstCutoff | DynamicProfileConstNBins,
            )
            self._parent_wakefield = parent_wakefield
            self._wake_imp_y_needs_update = True

            if is_dynamic and self.expect_impedance_change is False:
                warn(
                    f"Because you are using"
                    f" a `{type(parent_wakefield.profile)}`,"
                    f" the variable `update_on_calc` is set to"
                    f" True, which might impact performance."
                    f" Set True by yourself to deactivate this warning.",
                    stacklevel=2,
                )
                self.expect_impedance_change = True
            elif isinstance(parent_wakefield.profile, StaticProfile):
                pass
            else:
                raise NotImplementedError(
                    f"Unrecognized type(profile) = {type(parent_wakefield.profile)}"
                )
        else:
            raise Exception(f"{parent_wakefield.profile=}")
        for source in self._parent_wakefield.sources:
            if source.is_dynamic:
                if self.expect_impedance_change is False:
                    warn(
                        f"Because `{source}` is dynamic,"
                        f" the variable `expect_impedance_change` is set to"
                        f" True, which might impact performance."
                        f" Set True by yourself to deactivate this warning.",
                        stacklevel=2,
                    )
                    self.expect_impedance_change = True
                break

    def _update_impedance_sources(self, beam: BeamBaseClass) -> None:
        """
        Update `_wake_imp_y` array if `self.__wake_imp_y_needs_update=True`.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        if not self._wake_imp_y_needs_update:
            return
        _wake_x = self._parent_wakefield.profile.hist_x
        _wake_x = _wake_x - _wake_x.min()

        n_fft = 2 * len(_wake_x)
        if self._allow_next_fast_len:
            n_fft = next_fast_len(n_fft)

        n_t = (n_fft // 2) + 1

        if (self._wake_imp_y is None) or (
            (n_t,) != self._wake_imp_y.shape  # tuple vs shape-tuple
        ):
            self._wake_imp_y = backend.zeros(n_t, dtype=backend.complex)
        else:
            self._wake_imp_y[:] = 0 + 0j

        for source in self._parent_wakefield.sources:
            if isinstance(source, TimeDomain):
                # get the wake functions in time
                # but store already fft(wake) for convolution later
                wake_imp_y_tmp = source.get_wake_impedance(
                    time=_wake_x,
                    simulation=self._simulation,
                    beam=beam,
                    n_fft=n_fft,
                )
                assert not np.any(np.isnan(wake_imp_y_tmp)), (
                    f"{type(source).__name__}"
                )
                self._wake_imp_y += backend.array(
                    wake_imp_y_tmp,
                    dtype=backend.complex,
                )
            else:
                raise Exception(
                    "Can only accept impedance that support `TimeDomain`"
                )

        self._wake_imp_y_needs_update = False

    def calc_induced_voltage(
        self, beam: BeamBaseClass
    ) -> NumpyArray | CupyArray:
        """
        Calculate the induced voltage based on the beam profile and beam parameters.

        Parameters
        ----------
        beam
            Simulation object of a particle beam.

        Returns
        -------
        induced_voltage
            Induced voltage, in [V].
        """
        if self.expect_impedance_change:
            self._wake_imp_y_needs_update = True
        self._update_impedance_sources(beam=beam)

        _factor = self._hist_y_to_intensity_factor(
            beam=beam,
            profile=self._parent_wakefield.profile,
        )
        # Calculate the convolution of the wake and the beam
        # Usually this would be np.convolve(wake, beam).
        # This can be also done via fftconvolve.
        # Using ifft(fft(wake) * fft(beam)).
        # fft(wake)  is already precalculated in the memory.
        n_fft = 2 * len(self._parent_wakefield.profile.hist_x)
        if self._allow_next_fast_len:
            n_fft = next_fast_len(n_fft)
        induced_voltage = _factor * np.fft.irfft(
            self._wake_imp_y
            * self._parent_wakefield.profile.beam_spectrum(n_fft=n_fft)
        )

        # calculation in frequency domain must be with full periodicity.
        # The profile and corresponding induced voltage is only a part of
        # the full periodicity and must be thus truncated
        return induced_voltage


class SingleTurnResonatorConvolutionSolver(WakeFieldSolver):
    """
    Calculates the induced voltage by convolution of a wake with the bunch.

    Notes
    -----
    This solver is only compatible with
    :class:`~blond.physics.impedances.sources.Resonators` sources.
    """

    def __init__(self):
        warn("Untested code", NotTestedWarning, stacklevel=1)
        super().__init__()
        self._wake_function_vals: NumpyArray | None = None
        self._wake_function_time: NumpyArray | None = None
        self._wake_function_vals_needs_update = True  # initialization

        self._simulation: Simulation | None = None
        self._parent_wakefield: WakeField | None = None

    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ) -> None:
        """
        Lateinit method when WakeField is late-initialized.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        parent_wakefield
            Wakefield that this solver affiliated to.
        """
        self._simulation = simulation
        if parent_wakefield.profile is None:
            raise ValueError("Parent wakefield needs to have a profile.")
        self._parent_wakefield = parent_wakefield
        self._wake_function_vals_needs_update = True

        if not isinstance(parent_wakefield.profile, StaticProfile):
            raise RuntimeError(
                f"Expected `StaticProfile` but got {type(parent_wakefield.profile)=}."
            )

        for source in self._parent_wakefield.sources:
            if source.is_dynamic or not isinstance(source, Resonators):
                raise RuntimeError(
                    f"Expected `Resonators` and not source.is_dynamic, but got {type(source)=}."
                )

    def _update_potential_sources(self, zero_pinning: bool = False) -> None:
        """
        Update the internal wake kernels.

        Update `_wake_function_time`  and `_wake_function_vals` arrays
        if `self._wake_function_vals_needs_update=True`

        The time axis is chosen based on the profile in `_parent_wakefield.profile`

        Parameters
        ----------
        zero_pinning
            Causes values <= self._parent_wakefield.profile.hist_step * np.finfo(float).eps * len(self._wake_function_time)
            to be pinned to exactly zero. This prevents issues with the heaviside function around the 0 timestamp.
        """
        if not self._wake_function_vals_needs_update:
            return
        hist_step = self._parent_wakefield.profile.hist_step
        arr_len = len(self._parent_wakefield.profile.hist_x)
        self._wake_function_time = np.linspace(
            -(arr_len - 1) * hist_step,
            arr_len * hist_step,
            int(2 * arr_len - 1),
            endpoint=False,
        )  # necessary for boundary effects
        if zero_pinning:
            self._wake_function_time[
                np.abs(self._wake_function_time)
                <= self._parent_wakefield.profile.hist_step
                * np.finfo(float).eps
                * len(self._wake_function_time)
            ] = 0.0
        self._wake_function_vals = np.zeros_like(self._wake_function_time)
        for source in self._parent_wakefield.sources:
            self._wake_function_vals += source.get_wake(
                self._wake_function_time
            )

        self._wake_function_vals_needs_update = False  # avoid repeated update

    def calc_induced_voltage(
        self, beam: BeamBaseClass
    ) -> NumpyArray | CupyArray:
        """
        Calculate the induced voltage with the beam profile and parameters.

        Parameters
        ----------
        beam
            Simulation object of a particle beam.

        Returns
        -------
        induced_voltage
            Induced voltage in [V].
        """
        if self._wake_function_vals_needs_update:
            self._update_potential_sources()

        _charge_per_macroparticle = (-1 * beam.particle_type.charge * e) * (
            beam.intensity
            * self._parent_wakefield.profile.hist_y_to_density_factor
        )

        return _charge_per_macroparticle * np.convolve(
            self._wake_function_vals,
            self._parent_wakefield.profile.hist_y,
            mode="valid",
        )


class MultiPassResonatorSolver(WakeFieldSolver):
    """
    Calculates the multi-turn wake function.

    Solver, which saves the profiles of past passes and sums the
    wake fields of all previous and the current pass together.

    Parameters
    ----------
    decay_fraction_threshold
        Until which fraction of the decay will the profile
        still be considered for multi-pass wake calculation.

    Attributes
    ----------
    _wake_function_vals: deque
        List of wake function values: 0th entry being from the current pass,
        subsequent entries from previous passes.
    _wake_function_time: deque
        time axes corresponding to _wake_function_time.

    _past_profiles: deque
        List of previously passed profiles: 0th entry being from the current pass,
        subsequent entries from previous passes.
    _past_profile_times: deque
        time axes corresponding to _past_profiles.
    """

    def __init__(self, decay_fraction_threshold: float = 0.001):
        warn("Untested code", NotTestedWarning, stacklevel=1)
        super().__init__()

        self._last_reference_time: float | None = None

        self._maximum_storage_time: float | None = None
        self._decay_fraction_threshold = decay_fraction_threshold

        self._simulation: Simulation | None = None
        self._parent_wakefield: WakeField | None = None

        # define wake function values and corresponding time axis
        self._past_profiles: deque[NumpyArray] = deque()
        self._past_profile_times: deque[NumpyArray] = deque()
        self._past_charge_per_macroparticle: deque[float] = deque()
        self._past_profiles_counter_rotation_flag: deque[bool] = deque()

        self._wake_function_vals: deque[NumpyArray] = deque()
        self._wake_function_time: deque[NumpyArray] = deque()

    def _determine_storage_time(self):
        """
        Determine the maximum storage time, in [s].

        Sum up the contributions of all resonators and
        determine how long they should be stored in time.
        """
        if self._parent_wakefield is None:
            raise RuntimeError(
                "Parent wakefield must be present before this function can be called."
            )
        for source in self._parent_wakefield.sources:
            # Guarding against non-resonator sources is done in on_wakefield_init_simulation
            time_axis, envelope = source.calculate_envelope()

            storage_time = time_axis[
                np.abs(envelope - self._decay_fraction_threshold).argmin()
            ]
            self._maximum_storage_time = max(
                self._maximum_storage_time, storage_time
            )

    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ) -> None:
        """
        Lateinit method when WakeField is late-initialized.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        parent_wakefield
            Wakefield that this solver affiliated to.
        """
        self._simulation = simulation
        if parent_wakefield.profile is None:
            raise ValueError("Parent wakefield needs to have a profile.")
        self._parent_wakefield = parent_wakefield

        self._maximum_storage_time = 0
        self._last_reference_time = -np.finfo(float).eps

        if not isinstance(parent_wakefield.profile, StaticProfile):
            raise RuntimeError(
                f"Expected `StaticProfile` but got {type(parent_wakefield.profile)=}."
            )

        for source in self._parent_wakefield.sources:
            if source.is_dynamic or not isinstance(source, Resonators):
                raise RuntimeError(
                    f"Expected `Resonators` and not source.is_dynamic, but got {type(source)=}."
                )

        self._determine_storage_time()

    def _remove_fully_decayed_wake_profiles(
        self, indexes_to_check: int = 2
    ) -> None:
        """
        Go through wake functions and remove all arrays above storage time.

        Goes through _wake_function_time from the back (oldest profile)
        and removes all arrays from it, which are beyond
        ``self._maximum_storage_time``. only the last
        ``indexes_to_check`` entries are checked.

        Parameters
        ----------
        indexes_to_check
            Number of indices in the list, that get checked. This is the number of indices, which starts from the back.
        """
        if len(self._past_profiles) == 0:
            return
        for _ in range(indexes_to_check):
            if len(self._past_profiles) == 0:
                return
            if (
                np.min(self._past_profile_times[-1])
                > self._maximum_storage_time
            ):  # this assumes, that the time is increasing
                # throughout the array, which is ensured, when
                # the time is shifted
                self._past_profile_times.pop()
                self._past_profiles.pop()
                self._past_profiles_counter_rotation_flag.pop()
                self._wake_function_time.pop()
                self._wake_function_vals.pop()
            else:
                return

    def _update_past_profile_times_wake_times(self, current_time):
        """
        Advance the times in the past profile arrays.

        Advance the times in the past profile arrays by delta_t = current_time - self._last_reference_time and
        set self._last_reference_time to current_time afterwards.

        Parameters
        ----------
        current_time
            Simulation time at the moment of calling, has to be > self._last_reference_time.
        """
        delta_t = current_time - self._last_reference_time
        assert delta_t > 0  # TODO: performance = ?
        for prof_ind, profile_time in enumerate(self._past_profile_times):
            profile_time += delta_t  # NOQA # TODO test PLW2901 `for` loop variable `profile_time` overwritten by assignment target
            self._wake_function_time[prof_ind] += delta_t

        self._last_reference_time = current_time

    def _update_past_profile_wake_functions(self, zero_pinning: bool = False):
        """
        Update the wake functions according to the new timestamps.

        The arrays are expected to be cleaned before, such that they don't
        include arrays past self._maximum_storage_time.

        Parameters
        ----------
        zero_pinning
            Clips small values to zero.
            Causes ``values <= self._parent_wakefield.profile.hist_step * np.finfo(float).eps * len(self._wake_function_time)``
            to be pinned to exactly zero.
            This prevents issues with the heaviside function
            around the 0 timestamp.
        """
        for prof_ind in range(len(self._past_profiles)):
            if (
                prof_ind == 0
            ):  # current profile does not yet have arrays initialized
                hist_step = self._parent_wakefield.profile.hist_step
                arr_len = len(self._parent_wakefield.profile.hist_x)
                self._wake_function_time.appendleft(
                    np.linspace(
                        -(arr_len - 1) * hist_step,
                        arr_len * hist_step,
                        int(2 * arr_len - 1),
                        endpoint=False,
                    )
                )  # necessary for boundary effects
                if zero_pinning:
                    near_floating_point_precision_mask = np.abs(
                        self._wake_function_time[prof_ind]
                    ) <= hist_step * np.finfo(float).eps * len(
                        self._wake_function_time[prof_ind]
                    )
                    self._wake_function_time[prof_ind][
                        near_floating_point_precision_mask
                    ] = 0.0
                self._wake_function_vals.appendleft(
                    np.zeros_like(self._wake_function_time[prof_ind])
                )
            else:
                self._wake_function_vals[prof_ind] = np.zeros_like(
                    self._wake_function_vals[prof_ind]
                )
            # now that everything is initialized, same operation for all arrays
            for source in self._parent_wakefield.sources:  # TODO: do we ever need multiple resonstors objects in here --> probably not, resonators are defined in the Sources
                self._wake_function_vals[prof_ind] += (
                    source.get_wake_counter_rotation(
                        self._wake_function_time[prof_ind]
                    )
                    if (
                        self._past_profiles_counter_rotation_flag[prof_ind]
                        ^ self._past_profiles_counter_rotation_flag[0]
                    )
                    else source.get_wake(self._wake_function_time[prof_ind])
                )
                # exclusive OR, only if directionality of current profile and past profile differ,
                # its actually counter-rotating
                # first one is always corotating, as it's the one of the current turn

    def _update_potential_sources(self, beam: BeamBaseClass) -> None:
        """
        Update `_wake_function_time`  and `_wake_function_vals` arrays.

        The time axis is chosen based on
        the profile in `_parent_wakefield.profile`

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        self._update_past_profile_times_wake_times(beam.reference_time)
        self._remove_fully_decayed_wake_profiles()

        if len(self._past_profiles) != 0:  # ensure same time axis for profiles
            past_hist_step = (
                self._past_profile_times[-1][1]
                - self._past_profile_times[-1][0]
            )
            # TODO: big time jumps lead to problematic casting --> do we care about this?
            new_hist_step = (
                self._parent_wakefield.profile.hist_x[1]
                - self._parent_wakefield.profile.hist_x[0]
            )
            assert np.isclose(new_hist_step, past_hist_step, atol=0), (
                "Profile bin size needs to be constant: hist_step might be too small with casting to delta_t precision."
            )
        self._past_profile_times.appendleft(
            np.copy(self._parent_wakefield.profile.hist_x)
        )
        self._past_profiles.appendleft(
            np.copy(self._parent_wakefield.profile.hist_y)
        )
        self._past_profiles_counter_rotation_flag.appendleft(
            beam.is_counter_rotating
        )

        self._update_past_profile_wake_functions()

    def calc_induced_voltage(
        self, beam: BeamBaseClass
    ) -> NumpyArray | CupyArray:
        """
        Calculate the voltage induced by the beam profile.

        The function will call :func:`_update_potential_sources` and
        then compute the induced voltage based on all profiles
        which are in the `_past_profiles` array.

        Parameters
        ----------
        beam
            Instance on which the induced voltage is to be calculated, important for reference time.

        Returns
        -------
        induced_voltage
            Induced voltage in [V].
        """
        self._update_potential_sources(beam)

        _charge_per_macroparticle = (-1 * beam.particle_type.charge * e) * (
            beam.intensity
            * self._parent_wakefield.profile.hist_y_to_density_factor
        )
        self._past_charge_per_macroparticle.appendleft(
            _charge_per_macroparticle
        )

        wake_sum = np.zeros_like(self._past_profiles[0])
        for prof_ind in range(
            len(self._past_profiles)
        ):  # TODO: speedgain through circular shifting with numpy arrays instead of dequeue --> deque not usable with numba
            wake_sum += self._past_charge_per_macroparticle[
                prof_ind
            ] * np.convolve(
                self._wake_function_vals[prof_ind],
                self._past_profiles[prof_ind],
                mode="valid",
            )
        return wake_sum


class ContinuousMultiTurnTimeDomainSolver(WakeFieldSolver):
    """
    A solver for multi-turn wakefields, where the profile spans the entire revolution time.

    This class calculates the wakefield in cases where the
    profile spans the entire revolution time of the ring.
    The profiles from multiple turns are considered in a
    continuous manner, with neighboring profiles connected
    seamlessly across turns.

    Parameters
    ----------
    n_turns
        Number of turns that the multi-turn wake consists of.

    Notes
    -----
    Expects the parent wakefield to use a `StaticProfile`.
    This also means, that the length of a single turn should change only
    insignificantly during the runtime of the simulation, because it is
    assumed that the profile of the first turn is also a valid full turn
    representation of the last turn.
    """

    def __init__(self, n_turns: int) -> None:
        self._n_wakes_full_turn = n_turns

        self._parent_wakefield: WakeField | None = None
        self._wake_kernel: NumpyArray | CupyArray | None = None
        self._simulation: Simulation | None = None

        self._previous_wakes = deque(maxlen=n_turns)

    def _check_source_ducktypes(self):
        """Check that the sources implement ```get_wake```."""
        for source in self._parent_wakefield.sources:
            source: TimeDomain  # type hint what the we expect
            if not hasattr(source, "get_wake"):
                raise AttributeError(
                    f"The {source=} should implement `TimeDomain.get_wake`."
                )

    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ) -> None:
        """
        Lateinit method when WakeField is late-initialized.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        parent_wakefield
            `WakeField` that this solver affiliated to.
        """
        self._parent_wakefield = parent_wakefield
        self._simulation = simulation
        self._assert_profile_length_correct()
        self._check_source_ducktypes()

    def _update_wake_kernel(self) -> None:
        """
        Update the wakefield kernel that is used for convolution with the beam profile.

        Notes
        -----
        Expects the parent wakefield to use a `StaticProfile`.
        """
        # The assumptions below work only with static profiles.
        # This could be rewritten if necessary.
        width = (
            self._parent_wakefield.profile.cut_right
            - self._parent_wakefield.profile.cut_left
        )

        t_max = self._n_wakes_full_turn * width
        total_bins = (
            self._n_wakes_full_turn * self._parent_wakefield.profile.n_bins
        )
        time_axis = np.linspace(0, t_max, total_bins + 1)

        wake_kernel = None  # This needs to be derived
        for source in self._parent_wakefield.sources:
            source: TimeDomain  # type hint what the we expect
            wake_kernel_tmp = source.get_wake(time_axis)

            if wake_kernel is None:
                wake_kernel = wake_kernel_tmp
            else:
                wake_kernel += wake_kernel_tmp

        self._wake_kernel = wake_kernel

    def _assert_profile_length_correct(self):
        """
        Check that the length of the profile corresponds to one full turn.

        Raises
        ------
        AssertionError
            If the profile length does not correspond to one turn.
        """
        if not isinstance(self._parent_wakefield.profile, StaticProfile):
            warnings.warn(
                f"Expected StaticProfile, but"
                f" got {type(self._parent_wakefield.profile)=}",
                UserWarning,
                stacklevel=2,
            )
        profile = self._parent_wakefield.profile

        profile_width = profile.cut_right - profile.cut_left
        # todo check that the time of n_revolutions matches n * length_profile
        t_rev = self._simulation.magnetic_cycle.get_t_rev_init(
            circumference=self._simulation.ring.circumference,
            turn_i_init=0,
            t_init=0,
            particle_type=self._simulation.magnetic_cycle.reference_particle,
        )
        if isinstance(t_rev, float):
            assert abs(profile_width - t_rev) < profile.hist_step, (
                f"Expected profile length of {t_rev} s, but got "
                f"{profile_width} s.",
            )

    def calc_induced_voltage(
        self, beam: BeamBaseClass
    ) -> NumpyArray | CupyArray:
        """
        Calculate the induced voltage in this turn based on the last profiles.

        Parameters
        ----------
        beam
            Simulation object of a particle beam.

        Returns
        -------
        induced_voltage
            The induced voltage, in [V].
        """
        if self._wake_kernel is None:
            self._update_wake_kernel()

        _factor = self._hist_y_to_intensity_factor(
            beam=beam, profile=self._parent_wakefield.profile
        )

        # The speed of this function can be improved.
        # But: first correct, then fast!
        # Could use buffered fft convolution for example.

        induced_voltage_this_turn = _factor * backend.fftconvolve(
            self._parent_wakefield.profile.hist_y,
            self._wake_kernel,
            mode="full",
        )
        self._previous_wakes.appendleft(induced_voltage_this_turn)
        bins_per_profile = self._parent_wakefield.profile.n_bins

        def sel_current_profile(i: int) -> slice:
            return slice(
                i * bins_per_profile,  # start
                (i + 1) * bins_per_profile,  # stop
            )

        i = 0
        induced_voltage = self._previous_wakes[i][sel_current_profile(i)]

        for i in range(1, len(self._previous_wakes)):
            induced_voltage += self._previous_wakes[i][sel_current_profile(i)]

        return induced_voltage
