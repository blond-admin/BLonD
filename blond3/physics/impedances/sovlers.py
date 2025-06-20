from __future__ import annotations

import math
import warnings
from typing import Optional as LateInit
from typing import (
    Tuple,
)

import numpy as np
from numpy.typing import NDArray as NumpyArray

from .base import WakeFieldSolver, WakeField, FreqDomain
from .sources import InductiveImpedance
from ..profiles import (
    StaticProfile,
    DynamicProfileConstCutoff,
    DynamicProfileConstNBins,
)
from ...core.base import DynamicParameter
from ...core.beam.base import BeamBaseClass
from ...core.ring.helpers import requires
from ...core.simulation.simulation import Simulation


class InductiveImpedanceSolver(WakeFieldSolver):
    def __init__(self):
        super().__init__()
        self._beam: LateInit[BeamBaseClass] = None
        self._Z: LateInit[NumpyArray] = None
        self._T_rev_dynamic: LateInit[DynamicParameter] = None

    def on_wakefield_on_init_simulation(self, simulation: Simulation, parent_wakefield: WakeField):
        self._parent_wakefield = parent_wakefield
        assert all([isinstance(o, InductiveImpedance) for o in parent_wakefield.sources])
        impedances: Tuple[InductiveImpedance, ...] = parent_wakefield.sources
        self._Z = np.array([o.Z_over_n for o in impedances])
        self._T_rev_dynamic = simulation.ring.t_rev

    def calc_induced_voltage(self) -> NumpyArray | CupyArray:
        diff = self._parent_wakefield.profile.diff()
        return diff * self._Z * self._T_rev_dynamic.value


class PeriodicFreqSolver(WakeFieldSolver):
    def __init__(self, t_periodicity: float):
        super().__init__()
        self._t_periodicity = t_periodicity

        self._parent_wakefield: LateInit[WakeField] = None
        self._update_on_calc: LateInit[bool] = None
        self._n: LateInit[int] = None
        self._freq_x: LateInit[NumpyArray] = None
        self._freq_y: LateInit[NumpyArray] = None

    def _update_internal_data(self):
        self._n = int(
            math.ceil(self._t_periodicity / self._parent_wakefield.profile.dx)
        )
        self._freq_x = np.fft.rfftfreq(self._n, d=self._parent_wakefield.profile.dx)
        self._freq_y = np.zeros_like(self._freq_x)
        for source in self._parent_wakefield.sources:
            if isinstance(source, FreqDomain):
                self._freq_y += source.get_freq_y(freq_x=self._freq_x)
            else:
                raise Exception("Can only accept impedance that support `FreqDomain`")

    def _warning_callback(self, t_rev_new: float):
        tolerance = 0.1 / 100
        deviation = abs(1 - t_rev_new / self._t_periodicity)
        if deviation > tolerance:
            warnings.warn(
                f"The PeriodicFreqSolver was configured for "
                f"{self._t_periodicity=:.2e} s, but the actual Ring "
                f"periodicity is {t_rev_new:.2e} s, a deviation of {deviation} %."
            )

    def on_wakefield_on_init_simulation(self, simulation: Simulation, parent_wakefield: WakeField):
        simulation.ring.t_rev.on_change(self._warning_callback)

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
                    f"Unrecognized type(profile) = "
                    f"{type(parent_wakefield.profile)}"
                )
        else:
            raise Exception(f"{parent_wakefield.profile=}")

    def calc_induced_voltage(self) -> NumpyArray | CupyArray:
        if self._update_on_calc:
            self._update_internal_data()  # might cause performance issues :(

        induced_voltage = np.fft.irfft(
            self._freq_y * self._parent_wakefield.profile.beam_spectrum(self._n)
        )
        return induced_voltage




class AnalyticSingleTurnResonatorSolver(WakeFieldSolver):
    pass


class MutliTurnResonatorSolver(WakeFieldSolver):
    pass
