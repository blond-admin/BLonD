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
from unittest.mock import Mock

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

    def plot(self, **kwargs_plot):
        for profile in self.profiles:
            profile.plot(**kwargs_plot)

    @staticmethod
    def headless(
        t_rev: float,
        n_profiles: int,
        width_per_profile: float,
        bins_per_profile: int,
        offset: float = 0.0,
        section_index: int = 0,
        name: str | None = None,
    ) -> EquidistantMultiProfile:
        from blond.core.base import DynamicParameter

        d = EquidistantMultiProfile(
            n_profiles=n_profiles,
            width_per_profile=width_per_profile,
            bins_per_profile=bins_per_profile,
            offset=offset,
            section_index=section_index,
            name=name,
        )
        from blond.core.beam.base import BeamBaseClass
        from blond.core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        simulation.turn_i = Mock(DynamicParameter)
        simulation.turn_i.value = 0
        simulation.get_t_rev_init.return_value = t_rev
        d.on_init_simulation(simulation=simulation)
        d.on_run_simulation(
            simulation=simulation,
            n_turns=1,
            beam=Mock(BeamBaseClass),
        )
        return d

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
        dx = self.profiles[0]._hist_x[1] - self.profiles[0]._hist_x[0]

        for i, profile in enumerate(self.profiles):
            start = 2 * i * n
            stop = start + n
            sel = slice(start, stop)

            # core region
            self._continuous_memory_mask_prof[sel] = True
            self._continuous_memory_hist_x[sel] = profile._hist_x
            self._continuous_memory_hist_y[sel] = profile._hist_y

            ext_start = max(start - n, 0)
            ext_stop = min(stop, total)

            self._continuous_memory_mask[ext_start:ext_stop] = True

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
            profile._hist_x = self._continuous_memory_hist_x[sel]
            profile._hist_y = self._continuous_memory_hist_y[sel]

    def fix_deepcopy(self):
        for i, profile in enumerate(self.profiles):
            start = 2 * i * self._bins_per_profile
            stop = start + self._bins_per_profile
            sel = slice(start, stop)

            self.profiles[i]._hist_x = self._continuous_memory_hist_x[sel]
            self.profiles[i]._hist_y = self._continuous_memory_hist_y[sel]

    def _get_cut_arrays_and_bunch_indexes(self):
        """
        Build cut_left_array, cut_right_array, and bunch_indexes for sparse histogram.

        Returns
        -------
        tuple
            (cut_left_array, cut_right_array, bunch_indexes)
        """
        n_profiles = len(self.profiles)

        # Extract cut edges from profiles
        cut_left_array = backend.array(
            [profile.cut_left for profile in self.profiles],
            dtype=backend.float,
        )
        cut_right_array = backend.array(
            [profile.cut_right for profile in self.profiles],
            dtype=backend.float,
        )

        # Build bunch_indexes: maps bucket index to profile index
        # For equidistant profiles, each profile occupies one bucket
        # bucket i -> profile i (all buckets are filled)
        bunch_indexes = backend.arange(n_profiles, dtype=backend.float)

        return cut_left_array, cut_right_array, bunch_indexes

    def _track(self, beam: BeamBaseClass) -> None:
        """Track beam particles and fill histograms using optimized C++ function."""
        if len(beam._dt.array_local) == 0:
            # No particles to track
            return

        # Use optimized sparse_histogram_strided for single-call tracking
        stride = 2 * self._bins_per_profile

        # Build input arrays for C++ function
        cut_left_array, cut_right_array, bunch_indexes = (
            self._get_cut_arrays_and_bunch_indexes()
        )

        # Call optimized C++ function
        backend.specials.sparse_histogram_strided(
            input_array=beam._dt.array_local,
            output_array=self._continuous_memory_hist_y,
            cut_left_array=cut_left_array,
            cut_right_array=cut_right_array,
            bunch_indexes=bunch_indexes,
            n_slices_bucket=self._bins_per_profile,
            n_filled_buckets=self._n_profiles,
            stride=stride,
        )

        self.fix_deepcopy()
