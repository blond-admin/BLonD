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
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np

from blond import StaticProfile, backend
from blond.core.base import BeamPhysicsRelevant

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

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


class StaticMultiProfile(MultiProfile):
    """
    Holds many profiles, that have an even distance to each other and the same size.

    Parameters
    ----------
    profiles
        A ;ist of profiles that should be considered by this class.
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
        profiles: Iterable[StaticProfile],
        section_index: int = 0,
        name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(section_index, name, **kwargs)
        profiles = tuple(sorted(profiles, key=lambda p: p.cut_left))
        for i in range(len(profiles) - 1):
            this_stop = profiles[i].cut_right
            next_start = profiles[i + 1].cut_left
            assert this_stop <= next_start, (
                "The profiles are not allowed to overlap."
                f"{this_stop=} {next_start=}"
            )
        self.profiles: tuple[StaticProfile, ...] = profiles
        self._continuous_memory_hist_x: NumpyArray | CupyArray | None = None
        self._continuous_memory_hist_y: NumpyArray | CupyArray | None = None
        self._left_cuts: NumpyArray | CupyArray | None = None
        self._right_cuts: NumpyArray | CupyArray | None = None
        self._bins_per_profile: NumpyArray | CupyArray | None = None

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
        artists = []
        for profile in self.profiles:
            artists.extend(profile.plot(**kwargs_plot))
        return artists

    @staticmethod
    def headless(
        profiles: Sequence[StaticProfile],
        section_index: int = 0,
        name: str | None = None,
    ) -> StaticMultiProfile:
        """
        Make a instance of this class that does not rely on `Simulation`.

        Parameters
        ----------
        profiles
            A ;ist of profiles that should be considered by this class.
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

        d = StaticMultiProfile(
            profiles=profiles,
            section_index=section_index,
            name=name,
        )
        from blond.core.beam.base import BeamBaseClass
        from blond.core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        simulation.turn_i = Mock(DynamicParameter)
        simulation.turn_i.value = 0
        d.on_init_simulation(simulation=simulation)
        d.on_run_simulation(
            simulation=simulation,
            n_turns=1,
            beam=Mock(BeamBaseClass),
        )
        return d

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            Simulation context manager.
        """
        # allow user modifications on single profiles before
        self._make_memory_continuous()

    def _make_memory_continuous(self):
        """
        Fuse all profiles together in one array.

        This method fuses all profile arrays into one big array.
        In between each histogram there is one histogram space,
        so that no side effects appear when applying convolution
        on the full array.
        """
        self._continuous_memory_hist_x = backend.concatenate(
            [p.hist_x for p in self.profiles], dtype=backend.float
        )
        self._continuous_memory_hist_y = backend.concatenate(
            [p.hist_y for p in self.profiles], dtype=backend.float
        )
        self._left_cuts = backend.array(
            [p.cut_left for p in self.profiles], dtype=backend.float
        )

        self._right_cuts = backend.array(
            [p.cut_right for p in self.profiles], dtype=backend.float
        )

        self._bins_per_profile = backend.array(
            [p.n_bins for p in self.profiles], dtype=np.int32
        )

        self._bind_profiles()

    def _bind_profiles(self):  # TODO
        """Bind the memory of all ``self.profiles`` to the contigous memory."""
        start = 0
        for i, _profile in enumerate(self.profiles):
            stop = start + self._bins_per_profile[i]
            sel = slice(
                start, stop
            )  # must be a slice to get the pointers to the original array

            # let `_hist_x` point to the continous memory
            self.profiles[i]._hist_x = self._continuous_memory_hist_x[sel]
            self.profiles[i]._hist_y = self._continuous_memory_hist_y[sel]
            start = stop  # next stat is current stop

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

        backend.specials.sparse_histogram(
            x=beam._dt.array_local,
            out=self._continuous_memory_hist_y,
            left_cuts=self._left_cuts,
            right_cuts=self._right_cuts,
            bins_per_profile=self._bins_per_profile,
            start_indices=backend.array(
                np.cumsum(self._bins_per_profile) - self._bins_per_profile,
                dtype=np.int32,
            ),
        )
