# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Collection of implementations to calculate the multi-profiles."""

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
    """
    Base class to implement a profile that represents several profiles.

    Parameters
    ----------
    section_index
            Identifier grouping elements that belong to the same section of the ring.
            Defaults to 0.
    name
        Human-readable name for the element. If not provided, a unique name is
        automatically generated.
    **kwargs
        Additional keyword arguments passed to the parent.
    """

    def __init__(
        self, section_index: int = 0, name: str | None = None, **kwargs
    ) -> None:
        super().__init__(section_index, name)

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Initialize the ring when a simulation is created.

        This method is automatically called during simulation initialization to
        validate the ring configuration. It checks that RF stations are properly
        configured and section indices are correctly ordered.

        Parameters
        ----------
        simulation
            The `Simulation` context manager that owns this ring.
        """
        pass

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs,
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation `Beam` object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        pass


class EquidistantMultiProfile(MultiProfile):
    """
    Holds many profiles, that have an even distance to each other and the same size.

    Parameters
    ----------
    n_profiles
        Number of profiles to use internally.
    width_per_profile
        The width of each profile,
        corresponding to ``cut_right - cut_left``.
    bins_per_profile
        Number of bins per profile.
    offset
        Offset all profiles by this number.
    section_index
        Identifier grouping elements that belong to the same section of the ring.
        Defaults to 0.
    name
        Human-readable name for the element. If not provided, a unique name is
        automatically generated.
    **kwargs
        Additional keyword arguments passed to the parent.
    """

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
        self.profiles: tuple[StaticProfile, ...] | None = None

        self._continuous_memory_hist_x = None
        self._continuous_memory_hist_y = None
        self._continuous_memory_mask = None

    @property
    def hist_x(self):
        """
        One array with all profile.hist_x concatenated.

        Returns
        -------
        hist_x
            One array with all profile.hist_x concatenated.
        """
        return self._continuous_memory_hist_x[
            self._continuous_memory_mask_prof
        ]

    @property
    def hist_y(self):
        """
        One array with all profile.hist_y concatenated.

        Returns
        -------
        hist_x
            One array with all profile.hist_y concatenated.
        """
        return self._continuous_memory_hist_y[
            self._continuous_memory_mask_prof
        ]

    @property
    def n_bins(self):
        """
        Total number of bins among all profiles.

        Returns
        -------
        n_bins
            Total number of bins among all profiles.
        """
        return self._n_profiles * self._bins_per_profile

    def plot(self, **kwargs_plot):
        """
        Plot each profile.

        Parameters
        ----------
        **kwargs_plot
            Additional keyword arguments passed to ``matplotlib.pyplot.plot()``
            for customizing the plot appearance (e.g., ``color='red', linewidth=2``).
        """
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
        """
        Make a instance of this class that does not rely on `Simulation`.

        Parameters
        ----------
        t_rev
            Revolution period.
        n_profiles
            Number of profiles to use internally.
        width_per_profile
            The width of each profile,
            corresponding to ``cut_right - cut_left``.
        bins_per_profile
            Number of bins per profile.
        offset
            Offset all profiles by this number.
        section_index
            Identifier grouping elements that belong to the same section of the ring.
            Defaults to 0.
        name
            Human-readable name for the element. If not provided, a unique name is
            automatically generated.

        Returns
        -------
        equidistant_profile
            The fully initialized ``EquidistantMultiProfile``.
        """
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
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            Simulation context manager.
        """
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
        """
        Fuse all profiles together in one array.

        This method fuses all profile arrays into one big array.
        In between each histogram there is one histogram space,
        so that no side effects appear when applying convolution
        on the full array.
        """
        bins_per_profile = self._bins_per_profile

        # Keep one profile space in between each profile
        # to make convolution on `_continuous_memory_hist_y` possible.
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
            start = 2 * i * bins_per_profile
            stop = start + bins_per_profile
            sel = slice(start, stop)

            # core region
            self._continuous_memory_mask_prof[sel] = True
            self._continuous_memory_hist_x[sel] = profile._hist_x
            self._continuous_memory_hist_y[sel] = profile._hist_y

            # Extend the coordinates (for convolution),
            # so that the profile time coordinates start already
            # before the profile.
            ext_start = max(start - bins_per_profile, 0)
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

        self._bind_profiles()

    def _bind_profiles(self):  # TODO
        """Bind the memory of all ``self.profiles`` to the contigous memory."""
        for i, _profile in enumerate(self.profiles):
            start = 2 * i * self._bins_per_profile
            stop = start + self._bins_per_profile
            sel = slice(start, stop)

            self.profiles[i]._hist_x = self._continuous_memory_hist_x[sel]
            self.profiles[i]._hist_y = self._continuous_memory_hist_y[sel]

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        if len(beam._dt.array_local) == 0:
            # No particles to track
            return

        backend.specials.sparse_histogram_strided(
            x=beam._dt.array_local,
            out=self._continuous_memory_hist_y,
            first_left_cut=self.profiles[0].cut_left,
            left_cut_distance=(
                self.profiles[1].cut_left - self.profiles[0].cut_left
            ),
            bins_per_profile=self.profiles[0].n_bins,
            cut_width=(self.profiles[0].cut_right - self.profiles[0].cut_left),
            n_profiles=self._n_profiles,
            stride=(2 * self._bins_per_profile),
        )
