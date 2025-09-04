from __future__ import annotations

import math
import warnings
from collections import deque
from typing import TYPE_CHECKING
from typing import Optional
from typing import Optional as LateInit
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import elementary_charge as e
from scipy.fft import next_fast_len

from ..._core.backends.backend import backend
from ..._core.base import DynamicParameter
from ..._core.beam.base import BeamBaseClass
from ..._core.ring.helpers import requires
from ..._core.simulation.simulation import Simulation
from ..profiles import (
    DynamicProfile,
    DynamicProfileConstCutoff,
    DynamicProfileConstNBins,
    StaticProfile,
)
from .base import FreqDomain, TimeDomain, WakeField, WakeFieldSolver
from .sources import InductiveImpedance, Resonators

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray


class InductiveImpedanceSolver(WakeFieldSolver):
    def __init__(self):
        """Wakefield solver specialized for InductiveImpedance"""
        super().__init__()
        self._beam: LateInit[BeamBaseClass] = None
        self._Z_over_n: LateInit[float] = None
        self._turn_i: LateInit[DynamicParameter] = None
        self._parent_wakefield: LateInit[WakeField] = None
        self._simulation: LateInit[Simulation] = None

    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ):
        """Lateinit method when WakeField is late-initialized

        Parameters
        ----------
        simulation
            Simulation context manager
        parent_wakefield
            Wakefield that this solver affiliated to
        """
        self._parent_wakefield = parent_wakefield
        assert all(
            [
                isinstance(o, InductiveImpedance)
                for o in parent_wakefield.sources
            ]
        )
        impedances: Tuple[InductiveImpedance, ...] = parent_wakefield.sources
        self._Z_over_n = np.sum(np.array([o.Z_over_n for o in impedances]))
        self._turn_i = simulation.turn_i
        self._simulation = simulation

    def calc_induced_voltage(
        self, beam: BeamBaseClass
    ) -> NumpyArray | CupyArray:
        """
        Calculates the induced voltage based on the beam profile and beam parameters

        Parameters
        ----------
        beam
            Simulation object of a particle beam

        Returns
        -------
        induced_voltage
            Induced voltage, in [V]
        """
        ratio = beam.n_particles / beam.n_macroparticles_partial()
        factor = -(
            (beam.particle_type.charge * e)
            / (2 * np.pi)
            * ratio
            * (self._simulation.ring.circumference / beam.reference_velocity)
            / self._parent_wakefield.profile.hist_step
        )
        diff = self._parent_wakefield.profile.diff_hist_y
        return factor * diff * self._Z_over_n


class PeriodicFreqSolver(WakeFieldSolver):
    """General wakefield solver to calculate wake-fields via frequency domain

    Parameters
    ----------
    t_periodicity
        Periodicity that is assumed for fast fourier transform, in [s]
    allow_next_fast_len
        Allow to slightly change `t_periodicity` for
        faster execution of fft via `scipy.fft.next_fast_len`

    Attributes
    ----------
    allow_next_fast_len
        Allow to slightly change `t_periodicity` for
        faster execution of fft via `scipy.fft.next_fast_len`
    expect_profile_change
        If true, reloads internal data on each
        `calc_induced_voltage` for proper updating with
        dynamic parameters
    """

    def __init__(
        self,
        t_periodicity: Optional[float] = None,
        allow_next_fast_len: bool = False,
    ):
        """General wakefield solver to calculate wake-fields via frequency domain

        Parameters
        ----------
        t_periodicity
            Periodicity that is assumed for fast fourier transform, in [s]

            If None, it will be automatically set during `on_init_simulation`
            to the revolution time of the initial turn of the magnetic cycle
            with respect to the reference particle.
        allow_next_fast_len
            Allow to slightly change `t_periodicity` for
            faster execution of fft via `scipy.fft.next_fast_len`
        """

        super().__init__()
        self.allow_next_fast_len = allow_next_fast_len
        self.expect_profile_change: bool = False
        self.expect_impedance_change = False

        self._t_periodicity = t_periodicity
        self._parent_wakefield: LateInit[WakeField] = None
        self._n_time: LateInit[int] = None
        self._n_freq: LateInit[int] = None
        self._freq_x: LateInit[NumpyArray] = None
        self._freq_y: LateInit[NumpyArray] = None

        self._simulation: LateInit[Simulation] = None

        self._freq_y_needs_update = True  # at least one update

        self._induced_voltage_buffer = {}

    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ):
        """Lateinit method when WakeField is late-initialized

        Parameters
        ----------
        simulation
            Simulation context manager
        parent_wakefield
            Wakefield that this solver affiliated to
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
                parent_wakefield.profile, DynamicProfileConstCutoff
            ) or isinstance(parent_wakefield.profile, DynamicProfileConstNBins)
            self._parent_wakefield = parent_wakefield
            self._update_internal_data()

            if is_dynamic:
                if self.expect_profile_change is False:
                    warnings.warn(
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
                    warnings.warn(
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
        """Periodicity that is assumed for fast fourier transform in  [s]"""
        return self._t_periodicity

    @t_periodicity.setter
    def t_periodicity(self, t_periodicity: float):
        self._t_periodicity = t_periodicity
        self._update_internal_data()
        self._freq_y_needs_update = True

    def _update_internal_data(self):
        """Rebuild internal data model"""
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
        Updates `_freq_y` array if `self._freq_y_needs_update=True`

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        # be lazy
        if not self._freq_y_needs_update:
            return

        if (self._freq_y is None) or (
            self._freq_x.shape != self._freq_y.shape
        ):
            self._freq_y = np.zeros_like(self._freq_x, dtype=backend.complex)
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
                self._freq_y += freq_y
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
        Calculates the induced voltage based on the beam profile and beam parameters

        Parameters
        ----------
        beam
            Simulation object of a particle beam

        Returns
        -------
        induced_voltage
            Induced voltage, in [V]
        """
        if self.expect_profile_change:
            # always trigger update
            self._update_internal_data()  # might cause performance issues :(
        elif self.expect_impedance_change:
            # always trigger update
            self._freq_y_needs_update = True

        self._update_impedance_sources(beam=beam)

        _factor = (-1 * beam.particle_type.charge * e) * (
            # TODO this might be a problem with MPI
            beam.n_particles / beam.n_macroparticles_partial()
        )

        key = len(self._freq_y)  # todo
        if key in self._induced_voltage_buffer:
            # use `out` variable of fft to avoid array creation
            out = self._induced_voltage_buffer[key]
            np.fft.irfft(
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


class TimeDomainSolver(WakeFieldSolver):
    def __init__(self):
        """
        Solver to calculate induced voltage using fftconvolve(wake,profile)
        """
        super().__init__()
        self.expect_impedance_change = False

        self._parent_wakefield: LateInit[WakeField] = None
        self._wake_imp_y: LateInit[NumpyArray] = None
        self._simulation: LateInit[Simulation] = None

        self._wake_imp_y_needs_update = True  # update at least once

    @requires(["EnergyCycleBase"])  # because InductiveImpedance.get_
    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ) -> None:
        """Lateinit method when WakeField is late-initialized

        Parameters
        ----------
        simulation
            Simulation context manager
        parent_wakefield
            Wakefield that this solver affiliated to
        """
        self._simulation = simulation
        if parent_wakefield.profile is not None:
            is_dynamic = isinstance(
                parent_wakefield.profile, DynamicProfileConstCutoff
            ) or isinstance(parent_wakefield.profile, DynamicProfileConstNBins)
            self._parent_wakefield = parent_wakefield
            self._wake_imp_y_needs_update = True

            if is_dynamic and self.expect_impedance_change is False:
                warnings.warn(
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
                    warnings.warn(
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
        Updates `_wake_imp_y` array if `self.__wake_imp_y_needs_update=True`

        Parameters
        ----------
        beam
            Beam class to interact with this element

        """
        if not self._wake_imp_y_needs_update:
            return
        _wake_x = self._parent_wakefield.profile.hist_x
        _wake_x = _wake_x - _wake_x.min()

        n_fft = next_fast_len(2 * len(_wake_x))
        n_t = (n_fft // 2) + 1

        if (self._wake_imp_y is None) or (
            _wake_x.shape != self._wake_imp_y.shape
        ):
            self._wake_imp_y = np.zeros(n_t, dtype=backend.complex)
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
                self._wake_imp_y += wake_imp_y_tmp
            else:
                raise Exception(
                    "Can only accept impedance that support `TimeDomain`"
                )

        self._wake_imp_y_needs_update = False

    def calc_induced_voltage(
        self, beam: BeamBaseClass
    ) -> NumpyArray | CupyArray:
        """
        Calculates the induced voltage based on the beam profile and beam parameters

        Parameters
        ----------
        beam
            Simulation object of a particle beam

        Returns
        -------
        induced_voltage
            Induced voltage, in [V]
        """
        if self.expect_impedance_change:
            self._wake_imp_y_needs_update = True
        self._update_impedance_sources(beam=beam)

        _factor = (-1 * beam.particle_type.charge * e) * (
            # TODO this might be a problem with MPI
            beam.n_particles / beam.n_macroparticles_partial()
        )
        # Calculate the convolution of the wake and the beam
        # Usually this would be np.convolve(wake, beam).
        # This can be also done via fftconvolve.
        # Using ifft(fft(wake) * fft(beam)).
        # fft(wake)  is already precalculated in the memory.
        induced_voltage = _factor * np.fft.irfft(
            self._wake_imp_y
            * self._parent_wakefield.profile.beam_spectrum(
                n_fft=next_fast_len(
                    2 * len(self._parent_wakefield.profile.hist_x)
                )
            ),
        )

        # calculation in frequency domain must be with full periodicity.
        # The profile and corresponding induced voltage is only a part of
        # the full periodicity and must be thus truncated
        return induced_voltage


class AnalyticalSingleTurnResonatorSolver(WakeFieldSolver):
    """
    TODO: integration of BLond2 Induced voltage resonator algorithm
    """

    def __init__(self):
        raise NotImplementedError()


class SingleTurnResonatorConvolutionSolver(WakeFieldSolver):
    def __init__(self):
        """
        Solver to calculate induced voltage from convolution of a Resonator wake potential with bunch.
        """
        super().__init__()
        self._wake_pot_vals: LateInit[NumpyArray] = None
        self._wake_pot_time: LateInit[NumpyArray] = None
        self._wake_pot_vals_needs_update = True  # initialization

        self._simulation: LateInit[Simulation] = None
        self._parent_wakefield: LateInit[WakeField] = None

    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ) -> None:
        """Lateinit method when WakeField is late-initialized

        Parameters
        ----------
        simulation
            Simulation context manager
        parent_wakefield
            Wakefield that this solver affiliated to
        """
        self._simulation = simulation
        if parent_wakefield.profile is None:
            raise ValueError(f"Parent wakefield needs to have a profile.")
        self._parent_wakefield = parent_wakefield
        self._wake_pot_vals_needs_update = True

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
        Updates `_wake_pot_time`  and `_wake_pot_vals` arrays if `self._wake_pot_vals_needs_update=True`

        The time axis is chosen based on the profile in `_parent_wakefield.profile`

        Parameters
        ----------
        zero_pinning: boolean
            causes values <= self._parent_wakefield.profile.bin_size * np.finfo(float).eps * len(self._wake_pot_time)
            to be pinned to exactly zero. This prevents issues with the heaviside function around the 0 timestamp
        """
        if not self._wake_pot_vals_needs_update:
            return
        bin_size = self._parent_wakefield.profile.bin_size
        left_extend = (
            len(self._parent_wakefield.profile.hist_x)
            + self._parent_wakefield.profile.hist_x[0] / bin_size
            - 1
        )
        # time shift for alignment with wakepotential definition of 0
        right_extend = (
            len(self._parent_wakefield.profile.hist_x) - left_extend - 1
        )  # total length has to be 2*hist_x

        self._wake_pot_time = np.linspace(
            self._parent_wakefield.profile.hist_x[0] - left_extend * bin_size,
            self._parent_wakefield.profile.hist_x[-1]
            + right_extend * bin_size,
            int(
                len(self._parent_wakefield.profile.hist_x)
                + left_extend
                + right_extend
            ),
            endpoint=True,
        )  # necessary for boundary effects
        if zero_pinning:
            self._wake_pot_time[
                np.abs(self._wake_pot_time)
                <= self._parent_wakefield.profile.bin_size
                * np.finfo(float).eps
                * len(self._wake_pot_time)
            ] = 0.0
        self._wake_pot_vals = np.zeros_like(self._wake_pot_time)
        for source in self._parent_wakefield.sources:  # TODO: do we ever need multiple resonstors objects in here --> probably not, resonators are defined in the Sources
            self._wake_pot_vals += source.get_wake(self._wake_pot_time)

        self._wake_pot_vals_needs_update = False  # avoid repeated update

    def calc_induced_voltage(
        self, beam: BeamBaseClass
    ) -> NumpyArray | CupyArray:
        """
        Calculates the induced voltage based on the beam profile and beam parameters

        Parameters
        ----------
        beam
            Simulation object of a particle beam

        Returns
        -------
        induced_voltage
            Induced voltage in [V]
        """
        if self._wake_pot_vals_needs_update:
            self._update_potential_sources()

        _charge_per_macroparticle = (-1 * beam.particle_type.charge * e) * (
            beam.n_particles / beam.n_macroparticles_partial()
        )

        return _charge_per_macroparticle * np.convolve(
            self._wake_pot_vals,
            self._parent_wakefield.profile.hist_y,  # inverse for time-indexing
            mode="valid",
        )  # output is one element too long with valid


class MultiPassResonatorSolver(WakeFieldSolver):
    """
    Solver, which saves the profiles of past passes and sums the
    wakefields of all previous and the current pass together.

    Attributes
    -------
    _wake_pot_vals: deque
        List of wake potential values: 0th entry being from the current pass,
        subsequent entries from previous passes.
    _wake_pot_time: deque
        time axes corresponding to _wake_pot_vals.

    _past_profiles: deque
        List of previously passed profiles: 0th entry being from the current pass,
        subsequent entries from previous passes.
    _past_profile_times: deque
        time axes corresponding to _past_profiles.
    """

    def __init__(self, decay_fraction_threshold: float = 0.001):  # TODO
        """
        Parameters
        ----------
        decay_fraction_threshold: float
            Until which fraction of the decay will the profile
            still be considered for multi-pass wake calculation.
        """
        super().__init__()
        # define wake potential values and corresponding time axis
        self._wake_pot_vals: LateInit[deque[NumpyArray]] = None
        self._wake_pot_time: LateInit[deque[NumpyArray]] = None
        self._wake_pot_vals_needs_update = True  # initialization

        self._past_profiles: LateInit[deque[NumpyArray]] = None
        self._past_profile_times: LateInit[deque[NumpyArray]] = None
        self._last_reference_time: LateInit[float] = None
        self._past_charge_per_macroparticle: LateInit[deque[float]] = None

        self._maximum_storage_time: LateInit[float] = None
        self._decay_fraction_threshold = decay_fraction_threshold

        self._simulation: LateInit[Simulation] = None
        self._parent_wakefield: LateInit[WakeField] = None

    def _determine_storage_time(self):
        """
        Sum up the contributions of all resonators and
        determine how long they should be stored in time.
        """
        if self._parent_wakefield is None:
            raise RuntimeError(
                "Parent wakefield must be present before this function can be called."
            )
        for source in self._parent_wakefield.sources:
            envelope, time_axis = source.calculate_envelope()

            storage_time = time_axis[
                np.abs(envelope - self._decay_fraction_threshold).argmin()
            ]
            if storage_time > self._maximum_storage_time:
                self._maximum_storage_time = storage_time

    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ) -> None:
        """Lateinit method when WakeField is late-initialized

        Parameters
        ----------
        simulation
            Simulation context manager
        parent_wakefield
            Wakefield that this solver affiliated to
        """
        self._simulation = simulation
        if parent_wakefield.profile is None:
            raise ValueError(f"Parent wakefield needs to have a profile.")
        self._parent_wakefield = parent_wakefield
        self._wake_pot_vals_needs_update = True

        self._past_profiles = deque()
        self._past_profile_times = deque()
        self._past_charge_per_macroparticle = deque()

        self._wake_pot_vals = deque()
        self._wake_pot_time = deque()

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
        Goes through _wake_pot_time from the back (oldest profile) and removes all arrays from it, which are beyond
        self._maximum_storage_time. only the last indexes_to_check entries are checked.
        """
        if len(self._past_profiles) == 0:
            return
        for _ in range(indexes_to_check):
            if len(self._past_profiles) == 0:
                return
            if (
                np.min(self._past_profile_times[-1])
                > self._maximum_storage_time
            ):
                self._past_profile_times.pop()
                self._past_profiles.pop()
                self._wake_pot_time.pop()
                self._wake_pot_vals.pop()

    def _update_past_profile_times_wake_times(self, current_time):
        """
        Advances the times in the past profile arrays by delta_t = current_time - self._last_reference_time and
        sets self._last_reference_time to current_time afterwards.
        """
        delta_t = current_time - self._last_reference_time
        assert delta_t > 0  # TODO: performance = ?
        for prof_ind, profile_time in enumerate(self._past_profile_times):
            profile_time += delta_t
            self._wake_pot_time[prof_ind] += delta_t

        self._last_reference_time = current_time

    def _update_past_profile_potentials(self, zero_pinning: bool = False):
        """
        Updates the wake potentials according to the new timestamps.
        the arrays are expected to be cleaned before, such that they don't
        include arrays past self._maximum_storage_time.

        Parameters
        ----------
        zero_pinning: bool
            causes values <= self._parent_wakefield.profile.bin_size * np.finfo(float).eps * len(self._wake_pot_time)
            to be pinned to exactly zero. This prevents issues with the heaviside function around the 0 timestamp

        """
        for prof_ind in range(len(self._past_profiles)):
            if (
                prof_ind == 0
            ):  # current profile does not yet have arrays initialized
                profile_bin_size = (
                    self._past_profile_times[prof_ind][1]
                    - self._past_profile_times[prof_ind][0]
                )
                left_extend = (
                    len(self._past_profile_times[prof_ind])
                    + self._past_profile_times[prof_ind][0] / profile_bin_size
                    - 1
                )
                # time shift for alignment with wakepotential definition of 0
                right_extend = (
                    len(self._past_profile_times[prof_ind]) - left_extend - 1
                )  # total length has to be 2*hist_x
                self._wake_pot_time.appendleft(
                    np.linspace(
                        self._past_profile_times[prof_ind][0]
                        - left_extend * profile_bin_size,
                        self._past_profile_times[prof_ind][-1]
                        + right_extend * profile_bin_size,
                        int(
                            len(self._past_profile_times[prof_ind])
                            + left_extend
                            + right_extend
                        ),
                        endpoint=True,
                    )
                )  # necessary for boundary effects
                if zero_pinning:
                    self._wake_pot_time[prof_ind][
                        np.abs(self._wake_pot_time[prof_ind])
                        <= profile_bin_size
                        * np.finfo(float).eps
                        * len(self._wake_pot_time[prof_ind])
                    ] = 0.0

                self._wake_pot_vals.appendleft(
                    np.zeros_like(self._wake_pot_time[prof_ind])
                )
            else:
                self._wake_pot_vals[prof_ind] = np.zeros_like(
                    self._wake_pot_vals[prof_ind]
                )
            # now that everything is initialized, same operation for all arrays
            for source in self._parent_wakefield.sources:  # TODO: do we ever need multiple resonstors objects in here --> probably not, resonators are defined in the Sources
                self._wake_pot_vals[prof_ind] += source.get_wake(
                    self._wake_pot_time[prof_ind]
                )

    def _update_potential_sources(self, current_time: float = 0) -> None:
        """
        Updates `_wake_pot_time`  and `_wake_pot_vals` arrays if `self._wake_pot_vals_needs_update=True`

        The time axis is chosen based on the profile in `_parent_wakefield.profile`

        """
        if (
            not self._wake_pot_vals_needs_update
        ):  # TODO: how do we set this automagically?
            return

        self._update_past_profile_times_wake_times(current_time)
        self._remove_fully_decayed_wake_profiles()

        if len(self._past_profiles) != 0:  # ensure same time axis for profiles
            past_bin_size = (
                self._past_profile_times[-1][1]
                - self._past_profile_times[-1][0]
            )
            # TODO: big time jumps lead to problematic casting --> do we care about this?
            new_bin_size = (
                self._parent_wakefield.profile.hist_x[1]
                - self._parent_wakefield.profile.hist_x[0]
            )
            assert np.isclose(new_bin_size, past_bin_size, atol=0), (
                "Profile bin size needs to be constant: bin_size might be too small with casting to delta_t precision."
            )
        self._past_profile_times.appendleft(
            np.copy(self._parent_wakefield.profile.hist_x)
        )
        self._past_profiles.appendleft(
            np.copy(self._parent_wakefield.profile.hist_y)
        )

        self._update_past_profile_potentials()

        self._wake_pot_vals_needs_update = False  # avoid repeated update

    def calc_induced_voltage(
        self, beam: BeamBaseClass
    ) -> NumpyArray | CupyArray:
        if self._wake_pot_vals_needs_update:
            self._update_potential_sources(beam.reference_time)

        _charge_per_macroparticle = (-1 * beam.particle_type.charge * e) * (
            beam.n_particles / beam.n_macroparticles_partial()
        )
        self._past_charge_per_macroparticle.appendleft(
            _charge_per_macroparticle
        )

        wake_sum = 0
        for prof_ind in range(
            len(self._past_profiles)
        ):  # TODO: speedgain through circular shifting with numpy arrays instead of dequeue --> deque not usable with numba
            wake_sum += self._past_charge_per_macroparticle[
                prof_ind
            ] * np.convolve(
                self._wake_pot_vals[prof_ind],
                self._past_profiles[prof_ind],
                mode="valid",
            )  # inverse for time-indexing
        return wake_sum
