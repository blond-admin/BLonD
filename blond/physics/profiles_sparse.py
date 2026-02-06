# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Collection of implementations to calculate the beam profile."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np

from blond import StaticProfile, backend
from blond.core.base import BeamPhysicsRelevant
from blond.core.ring.helpers import requires

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


class MultiProfile(BeamPhysicsRelevant, ABC):
    def __init__(
        self, section_index: int = 0, name: str | None = None, **kwargs
    ) -> None:
        super().__init__(section_index, name)

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs,
    ) -> None:
        pass


class EquidistantMultiProfile(MultiProfile):
    def __init__(
        self,
        n_profiles: int,
        width_per_profile: float,
        bins_per_profile: int,
        offset: float = 0.0,
        section_index: int = 0,
        name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(section_index, name, **kwargs)
        self._n_profiles = n_profiles
        self._offset = offset
        self._width_per_profile = width_per_profile
        self._bins_per_profile = bins_per_profile
        self.profiles: tuple[StaticProfile] | None = None

        self._continuous_memory_hist_x = None
        self._continuous_memory_hist_y = None
        self._continuous_memory_mask = None

    @property
    def hist_x(self):
        return self._continuous_memory_hist_x[
            self._continuous_memory_mask_prof
        ]

    @property
    def hist_y(self):
        return self._continuous_memory_hist_y[
            self._continuous_memory_mask_prof
        ]

    @property
    def n_bins(self):
        return self._n_profiles * self._bins_per_profile

    def plot(self):
        for profile in self.profiles:
            profile.plot()

    @requires(["RFStationBaseClass"])  # for `get_t_rev_init`
    def on_init_simulation(self, simulation: Simulation) -> None:
        t_rev = simulation.get_t_rev_init()
        half_width = float(self._width_per_profile / 2)

        # Turn     |-----------|
        # Slots    |---|---|---| # 3 + 1
        # Used     ^   ^   ^   x
        centers = np.linspace(
            0,
            t_rev,
            self._n_profiles + 1,
            endpoint=True,
        )
        centers = centers[:-1]
        centers += self._offset

        self.profiles = tuple(
            StaticProfile(
                cut_left=float(center - half_width),
                cut_right=float(center + half_width),
                n_bins=self._bins_per_profile,
                name=f"{self.name}_{i}",
            )
            for i, center in enumerate(centers)
        )
        self._make_memory_continuous()

    def _make_memory_continuous(self):
        n = self._bins_per_profile
        total = 2 * self.n_bins

        self._continuous_memory_hist_x = backend.zeros(
            total,
            dtype=self.profiles[0]._hist_x.dtype,
        )
        self._continuous_memory_hist_y = backend.zeros_like(
            self._continuous_memory_hist_x
        )
        self._continuous_memory_mask = backend.zeros(total, dtype=bool)
        self._continuous_memory_mask_prof = backend.zeros(total, dtype=bool)
        for i, profile in enumerate(self.profiles):
            start = 2 * i * n
            stop = start + n
            sel = slice(start, stop)

            # core region
            self._continuous_memory_mask_prof[sel] = True
            self._continuous_memory_hist_x[sel] = profile._hist_x
            self._continuous_memory_hist_y[sel] = profile._hist_y

            # overwrite profile storage with views
            profile._hist_x = self._continuous_memory_hist_x[sel]
            profile._hist_y = self._continuous_memory_hist_y[sel]

            # ---- TODO FIX 1: extend mask ----
            # desired total width: 2*n - 1 centered on the profile
            center = start + n // 2
            half_width = n - 1
            width = n

            ext_start = max(start - n, 0)
            ext_stop = min(stop, total)

            self._continuous_memory_mask[ext_start:ext_stop] = True

            # ---- TODO FIX 2: fill hist_x in extended region ----
            dx = profile._hist_x[1] - profile._hist_x[0]

            # left extension
            if ext_start < start:
                k = start - ext_start
                self._continuous_memory_hist_x[ext_start:start] = (
                    profile._hist_x[0] - dx * backend.arange(k, 0, -1)
                )

            # right extension
            if stop < ext_stop:
                k = ext_stop - stop
                self._continuous_memory_hist_x[stop:ext_stop] = (
                    profile._hist_x[-1] + dx * backend.arange(1, k + 1)
                )

    def _track(self, beam: BeamBaseClass) -> None:
        for profile in self.profiles:
            profile.track(beam=beam)
