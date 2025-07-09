from __future__ import annotations

import math
import warnings
from typing import Optional as LateInit, TYPE_CHECKING
from typing import (
    Tuple,
)

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import elementary_charge as e
from scipy.fft import next_fast_len

from .base import WakeFieldSolver, WakeField, FreqDomain
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

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray
    from cupy.typing import NDArray as CupyArray


class InductiveImpedanceSolver(WakeFieldSolver):
    def __init__(self):
        super().__init__()
        self._beam: LateInit[BeamBaseClass] = None
        self._Z_over_n: LateInit[NumpyArray] = None
        self._turn_i: LateInit[DynamicParameter] = None
        self._parent_wakefield: LateInit[WakeField] = None
        self._simulation: LateInit[Simulation] = None

    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ):
        self._parent_wakefield = parent_wakefield
        assert all(
            [isinstance(o, InductiveImpedance) for o in parent_wakefield.sources]
        )
        impedances: Tuple[InductiveImpedance, ...] = parent_wakefield.sources
        self._Z_over_n = np.sum(np.array([o.Z_over_n for o in impedances]))
        self._turn_i = simulation.turn_i
        self._simulation = simulation

    def calc_induced_voltage(self, beam: BeamBaseClass) -> NumpyArray | CupyArray:
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
    def __init__(self, t_periodicity: float, allow_next_fast_len: bool = False):
        super().__init__()
        self.allow_next_fast_len = allow_next_fast_len

        self._t_periodicity = t_periodicity
        self._parent_wakefield: LateInit[WakeField] = None
        self._update_on_calc: LateInit[bool] = None
        self._n_time: LateInit[int] = None
        self._n_freq: LateInit[int] = None
        self._freq_x: LateInit[NumpyArray] = None
        self._freq_y: LateInit[NumpyArray] = None
        self._simulation: LateInit[Simulation] = None
        self._factor: LateInit[Simulation] = None

    @property
    def t_periodicity(self) -> float:
        return self._t_periodicity

    @t_periodicity.setter
    def t_periodicity(self, t_periodicity: float):
        self._t_periodicity = t_periodicity
        self._update_internal_data()

    def _update_internal_data(self):
        self._n_time = int(
            math.ceil(self._t_periodicity / self._parent_wakefield.profile.hist_step)
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
        self._freq_y = np.zeros_like(self._freq_x, dtype=backend.complex)
        for source in self._parent_wakefield.sources:
            if isinstance(source, FreqDomain):
                freq_y = source.get_impedance(freq_x=self._freq_x, sim=self._simulation)
                assert not np.any(np.isnan(freq_y)), f"{type(source).__name__}"
                self._freq_y += freq_y
            else:
                raise Exception("Can only accept impedance that support `FreqDomain`")
        # Factor relating Fourier transform and DFT
        self._freq_y /= self._parent_wakefield.profile.hist_step
        pass

    def _warning_callback(self, t_rev_new: float):  # TODO activate this again
        tolerance = 0.1 / 100
        deviation = abs(1 - t_rev_new / self._t_periodicity)
        if deviation > tolerance:
            warnings.warn(
                f"The PeriodicFreqSolver was configured for "
                f"{self._t_periodicity=:.2e} s, but the actual Ring "
                f"periodicity is {t_rev_new:.2e} s, a deviation of {deviation} %."
            )
    @requires(["EnergyCycleBase"]) # because InductiveImpedance.get_
    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ):
        self._factor = (
            -1
            * (
                simulation.beams[0].n_particles
                / simulation.beams[0].n_macroparticles_partial()
            )  # TODO this might be a problem with MPI
            * simulation.beams[0].particle_type.charge
            * e
        )
        self._simulation = simulation
        if parent_wakefield.profile is not None:
            is_static = isinstance(parent_wakefield.profile, StaticProfile)
            is_dynamic = isinstance(
                parent_wakefield.profile, DynamicProfileConstCutoff
            ) or isinstance(parent_wakefield.profile, DynamicProfileConstNBins)
            self._parent_wakefield = parent_wakefield
            self._update_internal_data()
            if is_static:
                self._update_on_calc = False
            elif is_dynamic:
                self._update_on_calc = True
            else:
                raise NotImplementedError(
                    f"Unrecognized type(profile) = {type(parent_wakefield.profile)}"
                )
        else:
            raise Exception(f"{parent_wakefield.profile=}")

    def calc_induced_voltage(self, beam: BeamBaseClass) -> NumpyArray | CupyArray:
        if self._update_on_calc:
            self._update_internal_data()  # might cause performance issues :(
        induced_voltage = self._factor * np.fft.irfft(
            self._freq_y
            * self._parent_wakefield.profile.beam_spectrum(n_fft=self._n_time),
        )
        # calculation in frequency domain must be with full periodicity.
        # The profile and corresponding induced voltage is only a part of
        # the full periodicity and must be thus truncated
        return induced_voltage[: self._parent_wakefield.profile.n_bins]


class AnalyticSingleTurnResonatorSolver(WakeFieldSolver):
    pass


class MutliTurnResonatorSolver(WakeFieldSolver):
    pass
