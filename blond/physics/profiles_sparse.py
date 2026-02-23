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
    from numpy._typing import NDArray as NumpyArray

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
        pass  # pragma: no cover


def _gen_array_bucket_index_to_memory_index(
    filling_pattern: NumpyArray, bins_per_profile: int
):
    """
    Generate the indexing to convert between positional index and memory index.

    Create a linear mapping from the bunch index
    to the sparse profile memory.
    e.g. ``[8, 8, 8, 16]`` with ``bins_per_profile = 8``,
    with the ``filling_pattern = [1, 0, 0, 1]``.

    Parameters
    ----------
    filling_pattern
        Filling pattern as a boolean array
        where ``True`` means filled bucket.
        For example ``filling_pattern = [1, 0, 0, 1]``,
        meaning that only the first and last profile are in active use.
    bins_per_profile
        Number of bins per profile.

    Returns
    -------
    bucket_index_to_memory_index
        Mapping the linear bucket index ``idx = (pos - start) / step`` to
        the memory position, skipping sparse profiles.
    """
    # create a linear mapping from the bunch index
    # to the sparse profile memory.
    # e.g. [8, 8, 8, 16] with `bins_per_profile = 8`
    # used  ^  x  x   ^
    bucket_index_to_memory_index = (
        backend.cumulative_sum(backend.array(filling_pattern), dtype=np.int32)
        - 1
        # minus one so that first
        # bucket is at index 0
    ) * np.int32(bins_per_profile)
    return bucket_index_to_memory_index


class EquidistantMultiProfile(MultiProfile):
    """
    Holds many profiles, that have an even distance to each other and the same size.

    Parameters
    ----------
    filling_pattern
        Filling pattern as a boolean array
        where ``True`` means filled bucket.
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
        filling_pattern: NumpyArray,
        bins_per_profile: int,
        offset: float = 0.0,
        section_index: int = 0,
        name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(section_index, name, **kwargs)

        self._bins_per_profile = int(bins_per_profile)

        # e.g. [1, 0, 0, 1]
        self._filling_pattern = backend.array(filling_pattern, dtype=bool)

        # create a linear mapping from the bunch index
        # to the sparse profile memory.
        # e.g. [8, 8, 8, 16] with `bins_per_profile = 8`
        # used  ^  x  x   ^
        self._bucket_index_to_memory_index = (
            _gen_array_bucket_index_to_memory_index(
                filling_pattern=filling_pattern,
                bins_per_profile=bins_per_profile,
            )
        )

        self._offset = offset

        self._left_cut_distance: float | None = None
        self._first_left_cut: float | None = None
        self.profiles: tuple[StaticProfile, ...] | None = None

        self._continuous_memory_hist_x = None
        self._continuous_memory_hist_y = None

    @property
    def hist_x(self):
        """
        One array with all profile.hist_x concatenated.

        Returns
        -------
        hist_x
            One array with all profile.hist_x concatenated.
        """
        return self._continuous_memory_hist_x

    @property
    def hist_y(self):
        """
        One array with all profile.hist_y concatenated.

        Returns
        -------
        hist_x
            One array with all profile.hist_y concatenated.
        """
        return self._continuous_memory_hist_y

    @property
    def n_bins(self):
        """
        Total number of bins among all profiles.

        Returns
        -------
        n_bins
            Total number of bins among all profiles.
        """
        return len(self._continuous_memory_hist_x)

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
        filling_pattern: NumpyArray,
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
        filling_pattern
            Filling pattern as a boolean array
            where ``True`` means filled bucket.
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
            filling_pattern=filling_pattern,
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
        n_slots = len(self._filling_pattern)

        # Turn     |-----------|
        # Starts   |---|---|---| # ``n_slots + 1``
        # Used     ^   ^   ^   x
        starts = (
            np.linspace(0, t_rev, n_slots + 1, endpoint=True)[:-1]
            + self._offset
        )
        self._first_left_cut = starts[0]
        self._left_cut_distance = (
            starts[1] - starts[0]
        )  # intentionally neglecting `_filling_pattern`

        profile_width = t_rev / n_slots
        assert np.isclose(
            starts[1] - starts[0], profile_width
        )  # just to be sure

        profiles = []
        for i in range(len(self._filling_pattern)):
            if self._filling_pattern[i]:
                profiles.append(
                    StaticProfile(
                        cut_left=float(starts[i]),
                        cut_right=float(starts[i] + profile_width),
                        n_bins=self._bins_per_profile,
                        name=f"{self.name}_{i}",
                    )
                )

        self.profiles = tuple(profiles)
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
        total = len(self.profiles) * bins_per_profile

        self._continuous_memory_hist_x = backend.zeros(
            total,
            dtype=self.profiles[0]._hist_x.dtype,
        )
        self._continuous_memory_hist_y = backend.zeros_like(
            self._continuous_memory_hist_x
        )

        for i, profile in enumerate(self.profiles):
            sel = self._get_slice_single_profile(i)

            # core region
            self._continuous_memory_hist_x[sel] = profile._hist_x
            self._continuous_memory_hist_y[sel] = profile._hist_y

        self._bind_profiles()

    def _bind_profiles(self):
        """Bind the memory of all ``self.profiles`` to the contigous memory."""
        for i, _profile in enumerate(self.profiles):
            sel = self._get_slice_single_profile(i)

            self.profiles[i]._hist_x = self._continuous_memory_hist_x[sel]
            self.profiles[i]._hist_y = self._continuous_memory_hist_y[sel]

    def _get_slice_single_profile(self, i: int):
        """
        Get slice indices to select the i-th active profile.

        Parameters
        ----------
        i
            Activate profile number.

        Returns
        -------
        sel
            ``slice`` selection of the profile within the memory.
        """
        bins_per_profile = self._bins_per_profile
        start = i * bins_per_profile
        stop = start + bins_per_profile
        sel = slice(start, stop)
        return sel

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
        assert self._bucket_index_to_memory_index[-1] + self.profiles[
            0
        ].n_bins <= len(self._continuous_memory_hist_y)
        beam._dt.histogram_sparse(
            out=self._continuous_memory_hist_y,
            first_left_cut=self._first_left_cut,
            left_cut_distance=self._left_cut_distance,
            bins_per_profile=self.profiles[
                0
            ].n_bins,  # assume all are the same
            cut_width=(self.profiles[0].cut_right - self.profiles[0].cut_left),
            n_active_profiles=len(self.profiles),
            filling_pattern=self._filling_pattern,
            bucket_index_to_memory_index=self._bucket_index_to_memory_index,
        )
