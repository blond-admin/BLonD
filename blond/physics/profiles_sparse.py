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

from blond import StaticProfile
from blond.core.base import BeamPhysicsRelevant

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
        section_index: int = 0,
        name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(section_index, name, **kwargs)
        self._n_profiles = n_profiles
        self._width_per_profile = width_per_profile
        self._bins_per_profile = bins_per_profile
        self.profiles: tuple[StaticProfile] | None = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        t_rev = simulation.get_t_rev_init()
        half_width = float(self._width_per_profile / 2)
        start = half_width
        stop = t_rev - half_width
        centers = np.linspace(start, stop, self._n_profiles, endpoint=True)
        self.profiles = tuple(
            StaticProfile(
                cut_left=float(center - half_width),
                cut_right=float(center + half_width),
                n_bins=self._bins_per_profile,
                name=f"{self.name}_{i}",
            )
            for i, center in enumerate(centers)
        )

    def _track(self, beam: BeamBaseClass) -> None:
        for profile in self.profiles:
            profile.track(beam=beam)
