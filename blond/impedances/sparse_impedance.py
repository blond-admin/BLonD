# coding: utf8
# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Multi-pass (multi-turn) resonator induced voltage on sparse profiles**

Port of the time-domain multi-pass wake solver of the BLonD GitHub
development branch ``blonder_feature/sparse_profiles``
(``blond/physics/impedances/solvers.py::MultiPassResonatorSolver``) onto
the SparseBatch / SparseBucket profiles of this tree.

Past profiles are remembered together with their time axes; on every
call the stored axes recede by the elapsed revolution period, the
analytic resonator wake (`Resonators.wake_calc`) is evaluated at the
shifted offsets and convolved with each remembered profile, and the
contributions are summed. Because the wake is evaluated at arbitrary
time offsets, the profile windows can sit anywhere in the turn — which
is exactly what a sparse profile needs, with no zero-charge gap bins
ever computed. A profile older than the source's decay time
(`Resonators.get_decay_time`) is dropped.

:Authors: **Lina Valle**
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import e

from ..beam.sparse_profiles import SparseProfileBaseClass
from ..utils import bmath as bm

if TYPE_CHECKING:  # pragma: no cover
    from typing import Iterable, Optional

    from numpy.typing import NDArray as NumpyArray

    from ..beam.beam import Beam
    from ..input_parameters.rf_parameters import RFStation
    from .impedance_sources import Resonators


class InducedVoltageSparseMultiPass:
    """Induced voltage of resonators on a sparse profile, summed over
    the current and all remembered passes.

    Parameters
    ----------
    beam
        Beam object.
    profile
        SparseBatch / SparseBucket (any SparseProfileBaseClass), or a
        standard Profile-like object exposing ``bin_centers`` and
        ``n_macroparticles`` on a contiguous grid.
    resonators
        The Resonators impedance source(s); every source must implement
        ``wake_calc`` and ``get_decay_time``.
    rf_station
        RFStation used to advance the remembered passes by one
        revolution period per call. Without it, only the current pass
        contributes (single-turn wake).
    decay_fraction_threshold
        A remembered pass is dropped once the wake envelope has decayed
        below this fraction of its initial amplitude
        (`Resonators.get_decay_time`).

    Attributes
    ----------
    induced_voltage : float array
        Induced voltage per memory bin, in [V], on the time-ordered
        concatenated grid of the sparse profile.
    """

    def __init__(
        self,
        beam: Beam,
        profile,
        resonators: Resonators | Iterable[Resonators],
        rf_station: Optional[RFStation] = None,
        decay_fraction_threshold: float = 1e-3,
    ) -> None:
        self.beam = beam
        self.profile = profile
        self.rf_station = rf_station
        self.decay_fraction_threshold = decay_fraction_threshold

        try:
            self.resonators = list(resonators)
        except TypeError:
            self.resonators = [resonators]
        for source in self.resonators:
            for method in ("wake_calc", "get_decay_time"):
                if not hasattr(source, method):
                    raise AttributeError(
                        f"The source {source!r} should implement `{method}`."
                    )

        self.maximum_storage_time = max(
            source.get_decay_time(decay_fraction_threshold)
            for source in self.resonators
        )

        self.process()

    @property
    def _is_sparse(self) -> bool:
        return isinstance(self.profile, SparseProfileBaseClass)

    def _windows(self) -> list[tuple[NumpyArray, NumpyArray]]:
        """Current (bin_centers, histogram) per window, in time order."""
        if self._is_sparse:
            return [
                (
                    self.profile.profiles_list[p].bin_centers,
                    self.profile.profiles_list[p].n_macroparticles,
                )
                for p in self.profile.memory_time_order
            ]
        return [(self.profile.bin_centers, self.profile.n_macroparticles)]

    def process(self) -> None:
        """(Re-)initialise the wake memory, e.g. after the profile
        changed. Discards the remembered passes."""
        # Each entry: (bin_centers shifted to the current turn's frame,
        # histogram, charge per macroparticle). Newest first.
        self._past_windows: deque[tuple[NumpyArray, NumpyArray, float]] = (
            deque()
        )
        n_memory = sum(len(hist) for _, hist in self._windows())
        self.induced_voltage = np.zeros(n_memory, dtype=bm.precision.real_t)

    def _wake(self, time_array: NumpyArray) -> NumpyArray:
        wake = np.zeros(len(time_array))
        for source in self.resonators:
            source.wake_calc(time_array)
            wake += source.wake
        return wake

    def _drop_decayed_windows(self, now_reference: float) -> None:
        while self._past_windows:
            centers, _, _ = self._past_windows[-1]
            # age of the youngest bin of the oldest stored window
            if now_reference - centers[-1] > self.maximum_storage_time:
                self._past_windows.pop()
            else:
                return

    def induced_voltage_generation(self) -> None:
        """Compute the induced voltage of the current profile plus the
        wake of all remembered passes."""
        windows = self._windows()
        n_memory = sum(len(hist) for _, hist in windows)
        if len(self.induced_voltage) != n_memory:
            # multi-turn injection changed the number of windows
            self.induced_voltage = np.zeros(
                n_memory, dtype=bm.precision.real_t
            )

        if self.rf_station is not None and self._past_windows:
            # The remembered passes recede by one revolution period
            turn = self.rf_station.counter[0]
            t_rev = self.rf_station.t_rev[turn]
            self._past_windows = deque(
                (centers - t_rev, hist, factor)
                for centers, hist, factor in self._past_windows
            )
            self._drop_decayed_windows(windows[0][0][0])

        charge_per_mp = -self.beam.particle.charge * e * self.beam.ratio

        # All contributing (source) windows: the current turn's windows
        # plus the remembered ones, each with its charge factor
        sources = [
            (centers, hist, charge_per_mp) for centers, hist in windows
        ] + list(self._past_windows)

        self.induced_voltage[:] = 0
        offset = 0
        for centers_i, hist_i in windows:
            n_i = len(hist_i)
            d = centers_i[1] - centers_i[0]
            for centers_j, hist_j, factor_j in sources:
                n_j = len(hist_j)
                start_gap = centers_i[0] - centers_j[0]
                # Source window entirely after the target window: no
                # causal contribution (wake is zero for negative times)
                if centers_j[0] - centers_i[-1] > 0:
                    continue
                # wake at every offset t_i[m] - t_j[k]
                wake_times = start_gap + d * np.arange(-(n_j - 1), n_i)
                wake_vals = self._wake(wake_times)
                contribution = np.convolve(wake_vals, hist_j, mode="valid")
                self.induced_voltage[offset : offset + n_i] += (
                    factor_j * contribution
                )
            offset += n_i

        # Remember the current pass (newest first)
        if self.rf_station is not None:
            for centers, hist in windows:
                self._past_windows.appendleft(
                    (centers.copy(), hist.copy(), charge_per_mp)
                )

    def track(self) -> None:
        """Compute the induced voltage and apply the kick to the beam."""
        self.induced_voltage_generation()

        charge = self.beam.particle.charge
        offset = 0
        for centers, hist in self._windows():
            n_i = len(hist)
            bm.linear_interp_kick(
                dt=self.beam.dt,
                dE=self.beam.dE,
                voltage=np.ascontiguousarray(
                    self.induced_voltage[offset : offset + n_i]
                ),
                bin_centers=centers,
                charge=charge,
                acceleration_kick=0.0,
            )
            offset += n_i
