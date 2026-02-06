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
        return self._continuous_memory_hist_x[self._continuous_memory_mask]

    @property
    def hist_y(self):
        return self._continuous_memory_hist_y[self._continuous_memory_mask]

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
        self._continuous_memory_hist_x = backend.zeros(
            2
            * self.n_bins,  # to leave one profile space in between each profile
            # assume all dtypes are equal
            dtype=self.profiles[0]._hist_x.dtype,
        )
        self._continuous_memory_hist_y = backend.zeros_like(
            self._continuous_memory_hist_x
        )
        self._continuous_memory_mask = backend.zeros_like(
            self._continuous_memory_hist_x,
            dtype=bool,
        )
        for i, profile in enumerate(self.profiles):
            start = 2 * i * self._bins_per_profile
            stop = start + self._bins_per_profile
            # must be slice to have pointer access in numpy
            sel = slice(start, stop)

            self._continuous_memory_mask[sel] = True
            self._continuous_memory_hist_x[sel] = profile._hist_x
            self._continuous_memory_hist_y[sel] = profile._hist_y
            # intentionally overwrite internal memory
            profile._hist_x = self._continuous_memory_hist_x[sel]
            profile._hist_y = self._continuous_memory_hist_y[sel]

    def _track(self, beam: BeamBaseClass) -> None:
        for profile in self.profiles:
            profile.track(beam=beam)
