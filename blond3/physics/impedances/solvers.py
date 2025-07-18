from __future__ import annotations

import math
import warnings
from typing import Optional as LateInit, TYPE_CHECKING
from typing import (
    Tuple,
)

import numpy as np
from scipy.constants import elementary_charge as e
from scipy.fft import next_fast_len

from .base import WakeFieldSolver, WakeField, FreqDomain, TimeDomain
from .sources import InductiveImpedance
from ..profiles import (
    StaticProfile,
    DynamicProfileConstCutoff,
    DynamicProfileConstNBins,
)
from ..._core.backends.backend import backend
from ..._core.base import DynamicParameter
from ..._core.beam.base import BeamBaseClass
from ..._core.ring.helpers import requires
from ..._core.simulation.simulation import Simulation

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray
    from cupy.typing import NDArray as CupyArray


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
            [isinstance(o, InductiveImpedance) for o in parent_wakefield.sources]
        )
        impedances: Tuple[InductiveImpedance, ...] = parent_wakefield.sources
        self._Z_over_n = np.sum(np.array([o.Z_over_n for o in impedances]))
        self._turn_i = simulation.turn_i
        self._simulation = simulation

    def calc_induced_voltage(self, beam: BeamBaseClass) -> NumpyArray | CupyArray:
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
            * (self._simulation.ring.closed_orbit_length / beam.reference_velocity)
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

    def __init__(self, t_periodicity: float, allow_next_fast_len: bool = False):
        """General wakefield solver to calculate wake-fields via frequency domain

        Parameters
        ----------
        t_periodicity
            Periodicity that is assumed for fast fourier transform, in [s]
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
            math.ceil(self._t_periodicity / self._parent_wakefield.profile.hist_step)
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

        if (self._freq_y is None) or (self._freq_x.shape != self._freq_y.shape):
            self._freq_y = np.zeros_like(self._freq_x, dtype=backend.complex)
        else:
            self._freq_y[:] = 0 + 0j
        for (
            source
        ) in self._parent_wakefield.sources:  # todo update only dynamic sources
            if isinstance(source, FreqDomain):
                freq_y = source.get_impedance(
                    freq_x=self._freq_x,
                    simulation=self._simulation,
                    beam=beam,  # FIXME
                )
                assert not np.any(np.isnan(freq_y)), f"{type(source).__name__}"
                self._freq_y += freq_y
            else:
                raise Exception("Can only accept impedance that support `FreqDomain`")
        # Factor relating Fourier transform and DFT
        self._freq_y /= self._parent_wakefield.profile.hist_step

        self._freq_y_needs_update = False  # after update, set lazy flag

    def calc_induced_voltage(self, beam: BeamBaseClass) -> NumpyArray | CupyArray:
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
                * self._parent_wakefield.profile.beam_spectrum(n_fft=self._n_time),
                out=out,
            )
            out *= _factor
            self._induced_voltage_buffer[key] = out
        else:
            # create array and safe it to buffer
            self._induced_voltage_buffer[key] = _factor * np.fft.irfft(
                self._freq_y
                * self._parent_wakefield.profile.beam_spectrum(n_fft=self._n_time),
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
    ):
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
        if (self._wake_imp_y is None) or (_wake_x.shape != self._wake_imp_y.shape):
            self._wake_imp_y = np.zeros(
                len(np.fft.rfftfreq(len(_wake_x))), dtype=backend.complex
            )
        else:
            self._wake_imp_y[:] = 0 + 0j

        for source in self._parent_wakefield.sources:
            if isinstance(source, TimeDomain):
                wake_imp_y_tmp = source.get_wake_impedance(
                    time=_wake_x,
                    simulation=self._simulation,
                    beam=beam,
                )
                assert not np.any(np.isnan(wake_imp_y_tmp)), f"{type(source).__name__}"
                self._wake_imp_y += wake_imp_y_tmp
            else:
                raise Exception("Can only accept impedance that support `TimeDomain`")

        self._wake_imp_y_needs_update = False

    def calc_induced_voltage(self, beam: BeamBaseClass) -> NumpyArray | CupyArray:
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
        induced_voltage = _factor * np.fft.irfft(
            self._wake_imp_y
            * self._parent_wakefield.profile.beam_spectrum(
                n_fft=len(self._parent_wakefield.profile.hist_x)
            ),
        )
        # calculation in frequency domain must be with full periodicity.
        # The profile and corresponding induced voltage is only a part of
        # the full periodicity and must be thus truncated
        return induced_voltage


class AnalyticSingleTurnResonatorSolver(WakeFieldSolver):
    def __init__(self):  # TODO
        raise NotImplementedError()


class MutliTurnResonatorSolver(WakeFieldSolver):
    def __init__(self):  # TODO
        raise NotImplementedError()
