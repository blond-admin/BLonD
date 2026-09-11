# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Module holding all observables for the simulation."""

from __future__ import annotations

import logging
import math
import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.gridspec import GridSpec
from matplotlib.image import AxesImage
from numpy.typing import NDArray as NumpyArray

from blond import backend
from blond.core.base import MainLoopRelevant
from blond.core.ring.helpers import requires
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.generals.warnings_ import PerformanceWarning
from blond.handle_results.array_recorders import DenseArrayRecorder
from blond.physics.cavities import (
    SingleHarmonicRFStation,
)
from blond.physics.drifts import DriftSimple
from blond.physics.feedbacks.cavity_feedback import (
    IQCavityFeedbackBase,
)

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from blond import WakeField
    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation
    from blond.generals.typing_ import AnyArray
    from blond.physics.cavities import (
        SingleHarmonicRFStation,
    )
    from blond.physics.profiles import DynamicProfileConstNBins, StaticProfile

logger = logging.getLogger(__name__)


def _plot_profile_waterfall(
    hist_x: NumpyArray,
    hist_y: NumpyArray,
    turns: NumpyArray,
    ax: Axes | None,
    kwargs_pcolormesh: dict | None,
) -> QuadMesh:
    """
    Make a 2D waterfall plot of a beam profile's evolution over turns.

    The profile amplitude is color-coded, with time on the x-axis and
    turn number on the y-axis.

    Parameters
    ----------
    hist_x
        Histogram x-axis, either of shape ``(n_bins,)`` (shared across
        all turns) or ``(n_observations, n_bins)`` (one x-axis per turn).
    hist_y
        Histogram amplitude of shape ``(n_observations, n_bins)``.
    turns
        Turn number of each observation, of shape ``(n_observations,)``.
    ax
        `Axes` to plot into. The current axes are used if `None`.
    kwargs_pcolormesh
        Keyword arguments for `matplotlib.axes.Axes.pcolormesh`.

    Returns
    -------
    mesh
        The `QuadMesh` pyplot object holding the waterfall plot.
    """
    if kwargs_pcolormesh is None:
        kwargs_pcolormesh = {}
    if ax is None:
        ax = plt.gca()

    default_kwargs_pcolormesh = {"shading": "nearest", "cmap": "viridis"}
    for key, value in default_kwargs_pcolormesh.items():
        if key not in kwargs_pcolormesh:
            kwargs_pcolormesh[key] = value

    if hist_x.ndim == 1:
        mesh = ax.pcolormesh(hist_x, turns, hist_y, **kwargs_pcolormesh)
    else:
        y = np.broadcast_to(turns[:, np.newaxis], hist_x.shape)
        mesh = ax.pcolormesh(hist_x, y, hist_y, **kwargs_pcolormesh)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Turn")
    return mesh


# DEV NOTE
# The main reason to have so much boilerplate code
# is providing an interface that allows autocompletion
# and allow testing beforehand.


class ObservablesBaseClass(MainLoopRelevant):
    """
    Base class to define observations.

    Parameters
    ----------
    folder
        Target folder to save the data at.
        Use `rename` to change the destination.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(self, folder: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if len(folder) > 0:
            assert folder.endswith("/") or folder.endswith("\\")
        self.common_filepath = folder + "last"
        logger.info(f"Will save {self} to {self.common_filepath}_,,,")

    def get_recorders(self) -> list[tuple[str, DenseArrayRecorder]]:
        """
        Get all `DenseArrayRecorder` inside the current instance.

        Returns
        -------
        recorders
            List of ((attribute name, attribute), ...).
        """
        self.assert_lateinit()
        recorders = [
            (attribute, instance)
            for attribute, instance in self.__dict__.items()
            if isinstance(instance, DenseArrayRecorder)  # initialized
        ]
        return recorders

    def rename(self, new_common_filepath: str) -> None:
        """
        Change the common save name of all internal arrays.

        Parameters
        ----------
        new_common_filepath
            The new common name of all internal arrays.

        Notes
        -----
        This has no effect on files that are already saved to the disk.
        """
        old_common_filepath = self.common_filepath
        for _attribute_name, instance in self.get_recorders():
            if old_common_filepath not in instance.filepath:
                # it would not make sense to replace the old filepath
                raise NameError(
                    f"{instance.filepath} does not include"
                    f" {old_common_filepath} anymore. This might be caused"
                    f" by a manual override of the filename."
                )
            instance.filepath = instance.filepath.replace(
                old_common_filepath,
                new_common_filepath,
            )
        self.common_filepath = new_common_filepath
        logger.info(
            f"Changed save target of {self} to {self.common_filepath}."
        )

    def to_disk(self) -> None:
        """Save data to disk."""
        for _attribute_name, instance in self.get_recorders():
            array_recorder: DenseArrayRecorder = instance
            logger.info(f"Saved {array_recorder.filepath_array}")
            array_recorder.to_disk()

    def from_disk(self) -> None:
        """Load data from disk."""
        for attribute_name, instance in self.get_recorders():
            array_recorder: DenseArrayRecorder = instance
            logger.info(f"Loaded {array_recorder.filepath_array}")

            self.__setattr__(
                attribute_name,
                array_recorder.from_disk(
                    filepath=array_recorder.filepath,
                ),
            )

    def assert_lateinit(self):
        """Check that DenseArrays are already initialized."""
        for parameter, value in self.__dict__.items():
            if value is None:  # uninitialized
                assert value is not None, f"`{parameter}` was not initialized."


class ObservablesOncePerTurnBase(ObservablesBaseClass):
    """
    Observe attributes during simulation.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    folder
        Path to the target folder used for
        saving or loading files.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        each_turn_i: int,
        folder: str = "",
        **kwargs,
    ):
        super().__init__(folder=folder, **kwargs)
        self.each_turn_i = each_turn_i

        self._n_turns: int | None = None
        self._turns_array: NumpyArray | None = None

        self._last_turn_i_observed = (
            -1
        )  # to avoid double recordings with multiple drifts in one section
        self._last_section_i_observed = -1

        self._simulation: Simulation | None = None

    def _calc_n_entries(self, n_turns: int) -> int:
        """
        Calculate the number of entries considering `each_turn_i`.

        Parameters
        ----------
        n_turns
            Number of turns that the simulation is foreseen to run.

        Returns
        -------
        n_entries
            The number of observations during the simulation.
        """
        return int(math.ceil(n_turns / self.each_turn_i))

    @property  # as readonly attributes
    def turns_array(self) -> NumpyArray | None:
        """
        Helper method to get x-axis array with turn-number of shape ``(n_observations, )``.

        Helper method to get x-axis array with turn-number for which the
        observations are performed.

        Returns
        -------
        turns_array
            Array with turn numbers for observations.
        """
        return self._turns_array

    @abstractmethod  # pragma: no cover
    def _update(self) -> None:
        """Update memory with new values."""
        pass

    def update(self) -> None:
        """Update memory with new values."""
        if self._last_turn_i_observed != self._simulation.turn_counter.value:
            self._update()
            self._last_turn_i_observed = self._simulation.turn_counter.value
        else:
            raise RuntimeError(
                f"{self} already called update in this turn for turn {self._last_turn_i_observed}."
                f" Was this observation added twice?",
            )

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # this is not used in this context
        n_turns: int,
        **kwargs: dict[str, Any],
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
        self._n_turns = int(n_turns)

        self._turns_array = np.arange(0, n_turns, self.each_turn_i, dtype=int)
        assert len(self._turns_array) == self._calc_n_entries(n_turns=n_turns)

        self._simulation = simulation


class BeamHist2dOncePerTurn(ObservablesOncePerTurnBase):
    """
    Save a 2D histogram of the beam during the simulation.

    This is intended to save the beam coordinates in
    less memory intensive way than storing the ``dt`` and ``dE``
    coordinates directly.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        called each ``each_turn_i``-th turn.
    folder
        Path to the target folder used for
        saving or loading files.
    bins
        Resolution, i.e. number of bins to use for histogram calculation.
    range
        The time and energy limits of the histogram.
        If not given, each turn will directly use ``dt.min()`` and ``dt.max()``.

    See Also
    --------
    BeamObservationOncePerTurn : To save the entire ``dt`` and ``dE`` coordinates directly.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>> from blond import Simulation
    >>> from blond import BeamHist2dOncePerTurn
    >>>
    >>> sim = Simulation(...)
    >>> beam_observation = BeamHist2dOncePerTurn(
    ...     each_turn_i=2,
    ...     bins=128,
    ...     range=[[0, 2.5e-9], [-4e8, 4e8]],
    ... )
    >>>
    >>> sim.run_simulation(
    ...     beams=...,
    ...     observe=(beam_observation,),
    ... )
    >>> beam_observation.plot_fancy(result_idx=-1)
    """

    def __init__(
        self,
        each_turn_i: int,
        folder: str = "",
        bins: int | tuple[int, int] = 32,
        range: AnyArray | None = None,
    ):
        super().__init__(
            each_turn_i=each_turn_i,
            folder=folder,
        )
        self._beam: BeamBaseClass | None = None
        self._hist2d: DenseArrayRecorder | None = None
        self._xedges: DenseArrayRecorder | None = None
        self._yedges: DenseArrayRecorder | None = None
        self._reference_time: DenseArrayRecorder | None = None
        self._intensity: DenseArrayRecorder | None = None
        self._reference_total_energy: DenseArrayRecorder | None = None

        self._consider_intensity: bool | None = None

        if isinstance(bins, int):
            self._bins = (bins, bins)
        else:
            self._bins = bins
        self._range = range
        self._density = True

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # not used in this context
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation :class:`~blond.core.beam.beams.Beam` object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        from blond.generals.distributed.helpers import mpi_is_distributed

        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )
        if beam.is_distributed:
            raise NotImplementedError(
                "This needs to be implemented."
                " Contact the devs if you need it."
            )
        self._beam = beam
        self._consider_intensity = beam.intensity != 0

        n_entries = self._calc_n_entries(n_turns)
        if mpi_is_distributed():  # pragma: no cover
            warnings.warn(
                "Saving beam with `BeamHist2dOncePerTurn` only from "
                "MPI-rank 0.",
                UserWarning,
                stacklevel=2,
            )
        shape = (n_entries, self._bins[0], self._bins[1])

        self._hist2d = DenseArrayRecorder(
            f"{self.common_filepath}_hist2d",
            shape,
        )

        self._xedges = DenseArrayRecorder(
            f"{self.common_filepath}_xedges",
            (n_entries, self._bins[0] + 1),
        )

        self._yedges = DenseArrayRecorder(
            f"{self.common_filepath}_yedges",
            (n_entries, self._bins[1] + 1),
        )

        self._reference_time = DenseArrayRecorder(
            f"{self.common_filepath}_reference_time",
            (n_entries,),
        )

        self._intensity = DenseArrayRecorder(
            f"{self.common_filepath}_intensity",
            (n_entries,),
        )
        self._reference_total_energy = DenseArrayRecorder(
            f"{self.common_filepath}_reference_total_energy",
            (n_entries,),
        )

    def _update(self) -> None:
        """Update memory with new values."""
        assert self._hist2d is not None
        assert self._xedges is not None
        assert self._intensity is not None
        assert self._yedges is not None
        assert self._beam is not None
        assert self._beam._dt is not None
        assert self._reference_time is not None
        assert self._reference_total_energy is not None

        self._reference_time.write(self._beam.reference.time)
        self._reference_total_energy.write(self._beam.reference.total_energy)
        self._intensity.write(self._beam.intensity)
        H, xedges, yedges = backend.histogram2d(
            self._beam.dt.array_local,
            self._beam.dE.array_local,
            bins=self._bins,
            range=self._range,
            density=self._density,
        )
        self._hist2d.write(H)
        self._xedges.write(xedges)
        self._yedges.write(yedges)

    def plot(
        self, result_idx: int, kwargs_imshow: dict | None = None
    ) -> AxesImage:
        """
        Make a plot of the beam 2D histogram.

        Parameters
        ----------
        result_idx
            Index of the recorded result to show.
        kwargs_imshow
            Keyword arguments for `matplotlib.pyplot.imshow`.

        Returns
        -------
        image
            The `AxesImage` pyplot object.
        """
        if kwargs_imshow is None:
            kwargs_imshow = {}

        assert self._intensity is not None
        assert self._hist2d is not None
        assert self._xedges is not None
        assert self._yedges is not None
        assert result_idx < self._intensity._write_idx
        if result_idx < 0:
            result_idx = self._intensity._write_idx + result_idx
        ax = plt.gca()

        if self._consider_intensity:
            H = (
                self._hist2d._memory[result_idx, :, :]
                * self._intensity._memory[result_idx]
            )
        else:
            H = self._hist2d._memory[result_idx, :, :]

        xedges = self._xedges._memory[result_idx, :]
        yedges = self._yedges._memory[result_idx, :]

        default_kwargs_imshow = {
            "origin": "lower",
            "extent": (
                float(xedges[0]),
                float(xedges[-1]),
                float(yedges[0]),
                float(yedges[-1]),
            ),
            "aspect": "auto",
            "cmap": "viridis",
        }
        # prevent overriding user arguments
        for key, value in default_kwargs_imshow.items():
            if key not in kwargs_imshow:
                kwargs_imshow[key] = value

        im = ax.imshow(H.T, **kwargs_imshow)
        return im

    def plot_fancy(
        self,
        result_idx: int,
        kwargs_imshow: dict | None = None,
        kwargs_bar: dict | None = None,
    ) -> tuple[Axes, Axes, Axes]:
        """
        Make a fancy plot of the beam 2D histogram and the corresponding histograms of the dE and dt projections.

        Parameters
        ----------
        result_idx
            Index of the recorded result to show.
        kwargs_imshow
            Keyword arguments for `matplotlib.pyplot.imshow`.
        kwargs_bar
            Keyword arguments for `matplotlib.pyplot.bar` and `matplotlib.pyplot.barh`.

        Returns
        -------
        ax_main
            The  pyplot `Axes` of the main 2D plot.
        ax_xhist
            The  pyplot `Axes` of the x histogram.
        ax_yhist
            The  pyplot `Axes` of the y histogram.
        """
        if kwargs_imshow is None:
            kwargs_imshow = {}
        if kwargs_bar is None:
            kwargs_bar = {}

        assert self._intensity is not None
        assert self._hist2d is not None
        assert self._xedges is not None
        assert self._yedges is not None
        assert result_idx < self._intensity._write_idx
        if result_idx < 0:
            result_idx = self._intensity._write_idx + result_idx

        if self._consider_intensity:
            H = (
                self._hist2d._memory[result_idx, :, :]
                * self._intensity._memory[result_idx]
            )
        else:
            H = self._hist2d._memory[result_idx, :, :]

        xedges = self._xedges._memory[result_idx, :]
        yedges = self._yedges._memory[result_idx, :]

        # Create figure with GridSpec
        fig = plt.figure()
        gs = GridSpec(4, 4, figure=fig)

        ax_main = fig.add_subplot(gs[1:, :-1])  # main 2D histogram
        ax_xhist = fig.add_subplot(gs[0, :-1])  # top X histogram
        ax_yhist = fig.add_subplot(gs[1:, -1])  # right Y histogram

        default_kwargs_imshow = {
            "origin": "lower",
            "extent": [xedges[0], xedges[-1], yedges[0], yedges[-1]],
            "aspect": "auto",
            "cmap": "viridis",
        }
        # prevent overriding user arguments
        for key, value in default_kwargs_imshow.items():
            if key not in kwargs_imshow:
                kwargs_imshow[key] = value

        default_kwargs_bar = {
            "align": "center",
            "color": "gray",
        }
        # prevent overriding user arguments
        for key, value in default_kwargs_bar.items():
            if key not in kwargs_bar:
                kwargs_bar[key] = value

        # Main 2D histogram
        ax_main.imshow(H.T, **kwargs_imshow)

        # X histogram (sum over Y)
        x_counts = H.sum(axis=1)
        ax_xhist.bar(
            (xedges[:-1] + xedges[1:]) / 2,
            x_counts,
            width=np.diff(xedges),
            **kwargs_bar,
        )

        ax_xhist.set_xticks([], [])
        ax_xhist.set_xlim(ax_main.get_xlim())
        max1 = ax_xhist.get_ylim()

        # Y histogram (sum over X)
        y_counts = H.sum(axis=0)
        ax_yhist.barh(
            (yedges[:-1] + yedges[1:]) / 2,
            y_counts,
            height=np.diff(yedges),
            **kwargs_bar,
        )

        ax_yhist.set_yticks([], [])
        ax_yhist.set_ylim(ax_main.get_ylim())
        max2 = ax_yhist.get_xlim()

        ax_xhist.set_ylim(max(max1, max2))
        ax_yhist.set_xlim(max(max1, max2))

        return (
            ax_main,
            ax_xhist,
            ax_yhist,
        )


class BeamObservationOncePerTurn(ObservablesOncePerTurnBase):
    """
    Observe the bunch coordinates during simulation execution after a drift element.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    folder
        Path to the target folder used for
        saving or loading files.
    warn
        If ``True``, emits a warning about the performance impact.

    See Also
    --------
    BeamHist2dOncePerTurn : To save a 2D histogram of ``dt`` and ``dE``.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>> from blond import Simulation
    >>> from blond import BeamObservationOncePerTurn
    >>>
    >>> sim = Simulation(...)
    >>> bunch_observation = BeamObservationOncePerTurn(each_turn_i=2)
    >>>
    >>> sim.run_simulation(
    ...     beams=...,
    ...     observe=(bunch_observation,),
    ... )
    >>> turn_0 = 0 # first turn
    >>> turn_2 = 1  # after 2 turns, because `each_turn_i = 2`
    >>> for index in (turn_0, turn_2):
    ...     plt.hist2d(
    ...         bunch_observation.dts[index, :],
    ...         bunch_observation.dEs[index, :],
    ...         bins=256,
    ...         range=[[0, 2.5e-9], [-4e8, 4e8]],
    ...     )
    """

    def __init__(
        self,
        each_turn_i: int,
        folder: str = "",
        warn: bool = True,
    ):
        if warn:
            warnings.warn(
                "`BeamObservationOncePerTurn` will significantly"
                " degrade your performance, use with caution."
                " To deactivate this message,"
                " set ``BeamObservationOncePerTurn(..., warn=False)``.",
                PerformanceWarning,
                stacklevel=2,
            )
        super().__init__(
            each_turn_i=each_turn_i,
            folder=folder,
        )
        self._beam: BeamBaseClass | None = None
        self._dts: DenseArrayRecorder | None = None
        self._dEs: DenseArrayRecorder | None = None
        self._flags: DenseArrayRecorder | None = None
        self._reference_time: DenseArrayRecorder | None = None
        self._reference_total_energy: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # not used in this context
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation :class:`~blond.core.beam.beams.Beam` object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        from blond.generals.distributed.helpers import mpi_is_distributed

        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )
        if beam.is_distributed:
            raise NotImplementedError(
                "This needs to be implemented."
                " Contact the devs if you need it."
            )
        self._beam = beam
        n_entries = self._calc_n_entries(n_turns)
        n_macroparticles = int(beam._dt.local_size)
        if mpi_is_distributed():
            warnings.warn(
                "Saving beam with `BeamObservationOncePerTurn` only from "
                "MPI-rank 0.",
                UserWarning,
                stacklevel=2,
            )
        shape = (n_entries, n_macroparticles)

        self._dts = DenseArrayRecorder(
            f"{self.common_filepath}_dts",
            shape,
        )
        self._dEs = DenseArrayRecorder(
            f"{self.common_filepath}_dEs",
            shape,
        )
        self._flags = DenseArrayRecorder(
            f"{self.common_filepath}_flags",
            shape,
        )

        self._reference_time = DenseArrayRecorder(
            f"{self.common_filepath}_reference_time",
            (n_entries,),
        )
        self._reference_total_energy = DenseArrayRecorder(
            f"{self.common_filepath}_reference_total_energy",
            (n_entries,),
        )

    def _update(self) -> None:
        """Update memory with new values."""
        # TODO allow several bunches

        self._reference_time.write(self._beam.reference.time)
        self._reference_total_energy.write(self._beam.reference.total_energy)

        if self._beam._dt.local_size < self._dts._memory.shape[1]:
            mask = backend.zeros(self._dts._memory.shape[1], dtype=bool)
            mask[self._beam.read_partial_ids()] = True
        else:
            mask = None

        self._dts.write(self._beam.read_partial_dt(), mask=mask)
        self._dEs.write(self._beam.read_partial_dE(), mask=mask)
        self._flags.write(self._beam.read_partial_flags(), mask=mask)

    @property  # as readonly attributes
    def reference_time(self):
        """
        Return reference time of shape ``(n_observations, n_bins)``.

        Returns
        -------
        reference_time
            Reference time array.
        """
        return self._reference_time.get_valid_entries()

    @property  # as readonly attributes
    def reference_total_energy(self):
        """
        Return total energy of shape ``(n_observations, n_bins)``.

        Returns
        -------
        reference_total_energy
            Total energy array.
        """
        return self._reference_total_energy.get_valid_entries()

    @property  # as readonly attributes
    def dts(self):
        """
        Return array of dts of shape ``(n_observations, n_macroparticles)``.

        Returns
        -------
        dts
            Time coordinate array.
        """
        return self._dts.get_valid_entries()

    @property  # as readonly attributes
    def dEs(self):
        """
        Return array of dEs of shape ``(n_observations, n_macroparticles)``.

        Returns
        -------
        dEs
            Energy coordinate array.
        """
        return self._dEs.get_valid_entries()

    @property  # as readonly attributes
    def flags(self):
        """
        Return flags of particles, eg if lost or not of shape ``(n_observations, n_macroparticles)``.

        Returns
        -------
        flags
            Particle flags array.
        """
        return self._flags.get_valid_entries()


class BeamStatisticsOncePerTurn(ObservablesOncePerTurnBase):
    """
    Observe the beam statistics during simulation execution after a drift element.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    folder
        Path to the target folder used for
        saving or loading files.

    Examples
    --------
    >>> bunch_statistics = BeamStatisticsOncePerTurn(each_turn_i=2, beam=...)
    >>>
    >>> sim.run_simulation(
    ...     beams=...,
    ...     observe=(bunch_statistics,),
    ... )
    >>> turn_0 = 0  # first turn
    >>> turn_2 = 1  # after 2 turns, because `each_turn_i = 2`
    >>> for index in (turn_0, turn_2)
    ...     plt.plot(
    ...         bunch_statistics.bunch_position()[index, :],
    ...     )
    """

    def __init__(
        self,
        each_turn_i: int,
        folder: str = "",
    ):
        super().__init__(
            each_turn_i=each_turn_i,
            folder=folder,
        )
        self._beam: BeamBaseClass | None = None
        self._bunch_position: DenseArrayRecorder | None = None
        self._energy_spread: DenseArrayRecorder | None = None
        self._bunch_length: DenseArrayRecorder | None = None
        self._reference_time: DenseArrayRecorder | None = None
        self._reference_total_energy: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # not used in this context
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation :class:`~blond.core.beam.beams.Beam` object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )
        self._beam = beam
        n_entries = self._calc_n_entries(n_turns)

        self._bunch_position = DenseArrayRecorder(
            f"{self.common_filepath}_bunch_position",
            n_entries,
        )
        self._energy_spread = DenseArrayRecorder(
            f"{self.common_filepath}_energy_spread",
            n_entries,
        )
        self._bunch_length = DenseArrayRecorder(
            f"{self.common_filepath}_bunch_length",
            n_entries,
        )
        self._reference_time = DenseArrayRecorder(
            f"{self.common_filepath}_reference_time",
            n_entries,
        )
        self._reference_total_energy = DenseArrayRecorder(
            f"{self.common_filepath}_reference_total_energy",
            n_entries,
        )

    def _update(self) -> None:
        """Update memory with new values."""
        # TODO allow several bunches

        # MPI capable
        self._bunch_position.write(self._beam._dt.mean())
        self._energy_spread.write(self._beam._dE.std())
        self._bunch_length.write(self._beam._dt.std())

        self._reference_time.write(self._beam.reference.time)
        self._reference_total_energy.write(self._beam.reference.total_energy)

    @property  # as readonly attributes
    def bunch_position(self):
        """
        Return array of bunch_position of shape (n_observations,).

        Returns
        -------
        bunch_position
            Bunch position array.
        """
        return self._bunch_position.get_valid_entries()

    @property  # as readonly attributes
    def energy_spread(self):
        """
        Return array of energy spread of shape (n_observations,).

        Returns
        -------
        energy_spread
            Energy spread array.
        """
        return self._energy_spread.get_valid_entries()

    @property  # as readonly attributes
    def bunch_length(self):
        """
        Return array of bunch_length of shape (n_observations,).

        Returns
        -------
        bunch_length
            Bunch length array.
        """
        return self._bunch_length.get_valid_entries()

    @property  # as readonly attributes
    def reference_time(self):
        """
        Return reference time of shape (n_observations,).

        Returns
        -------
        reference_time
            Reference time array.
        """
        return self._reference_time.get_valid_entries()

    @property  # as readonly attributes
    def reference_total_energy(self):
        """
        Return reference total energy of shape (n_observations,).

        Returns
        -------
        reference_total_energy
            Total energy array.
        """
        return self._reference_total_energy.get_valid_entries()


class RFStationPhaseObservation(ObservablesOncePerTurnBase):
    """
    Observe the RF station parameters during the execution of the simulation.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    rf_station
        Class that implements beam-RF interactions in a synchrotron.
    folder
        Path to the target folder used for
        saving or loading files.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>> from blond import Simulation
    >>> sim = Simulation( ... )
    >>> rf_station_observation = RFStationPhaseObservation(each_turn_i=2, rf_station=...)
    >>> sim.run_simulation(
    ...     beams=...,
    ...     observe=(rf_station_observation,),
    ... )
    >>> turn_0 = 0  # first turn
    >>> turn_2 = 1  # after 2 turns, because `each_turn_i = 2`
    >>> plt.scatter(
    ...     rf_station_observation.turns_array[[turn_0, turn_2]],
    ...     rf_station_observation.phases[[turn_0, turn_2]],
    ... )
    >>> plt.plot(
    ...     rf_station_observation.turns_array[:], rf_station_observation.phases[:]
    ... )
    """

    def __init__(
        self,
        each_turn_i: int,
        rf_station: SingleHarmonicRFStation,
        folder: str = "",
    ):
        super().__init__(each_turn_i=each_turn_i, folder=folder)
        self._rf_station = rf_station
        self._phases: DenseArrayRecorder | None = None
        self._omegas: DenseArrayRecorder | None = None
        self._voltages: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # not used in this context
        n_turns: int,
        **kwargs: dict[str, Any],
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
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )

        n_entries = self._calc_n_entries(n_turns)
        n_harmonics = int(self._rf_station.n_rf)
        shape = (n_entries, n_harmonics)
        self._phases = DenseArrayRecorder(
            f"{self.common_filepath}_phases",
            shape,
        )
        self._omegas = DenseArrayRecorder(
            f"{self.common_filepath}_omegas",
            shape,
        )
        self._voltages = DenseArrayRecorder(
            f"{self.common_filepath}_voltages",
            shape,
        )

    def _update(self) -> None:
        """Update memory with new values."""
        self._phases.write(self._rf_station.phi_rf)
        self._omegas.write(self._rf_station.omega_rf)
        self._voltages.write(self._rf_station.voltage)

    @property  # as readonly attributes
    def phases(self) -> NumpyArray:
        """
        RF station's effective phase of shape ``(n_observations, )``, in [rad].

        Returns
        -------
        phases
            Array of RF phases.
        """
        return self._phases.get_valid_entries()

    @property  # as readonly attributes
    def omegas(self) -> NumpyArray:
        """
        RF station's angular frequency of shape ``(n_observations, )``, in [Hz].

        Returns
        -------
        omegas
            Array of RF angular frequencies.
        """
        return self._omegas.get_valid_entries()

    @property  # as readonly attributes
    def voltages(self) -> NumpyArray:
        """
        RF station's effective voltage of shape ``(n_observations, )``, in [V].

        Returns
        -------
        voltages
            Array of RF voltages.
        """
        return self._voltages.get_valid_entries()


class IQCavityFeedbackObservation(ObservablesOncePerTurnBase):
    """
    Observe the RF station parameters during the execution of the simulation.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    feedback
        Class that implements beam-RF interactions in a synchrotron.
    folder
        Path to the target folder used for
        saving or loading files.

    Notes
    -----
    Layout of the coarse matrices (``v_ant_coarse``, ``i_gen_coarse``,
    ``i_beam_coarse``): every column addresses the same per-turn
    coarse-grid cell in all three matrices. Rows are padded to
    ``len_coarse_max`` columns; cells the turn's grid does not reach are
    ``NaN``. The antenna voltage and generator current span the whole
    per-turn grid (backfill reconstruction segments followed by the
    forward segment), so their valid columns are ``[0, n_grid)``. The
    beam current exists only on the forward (real passage) segment --
    the last ``rf_centers_lengths[-1]`` cells of the grid -- so its
    valid columns are ``[n_grid - n_forward, n_grid)``; all other
    columns are ``NaN``. Plotting the same column of the three matrices
    therefore always refers to one and the same coarse-grid cell.

    Examples
    --------
    TODO:
    """

    def __init__(
        self,
        each_turn_i: int,
        feedback: IQCavityFeedbackBase,
        folder: str = "",
    ):
        super().__init__(each_turn_i=each_turn_i, folder=folder)
        self._feedback = feedback

        self._v_ant_fine: DenseArrayRecorder | None = None
        self._i_beam_fine: DenseArrayRecorder | None = None
        self._i_gen_fine: DenseArrayRecorder | None = None
        self._kick_voltage_fine: DenseArrayRecorder | None = None

        self._v_ant_coarse: DenseArrayRecorder | None = None
        self._i_beam_coarse: DenseArrayRecorder | None = None
        self._i_gen_coarse: DenseArrayRecorder | None = None

        self._v_corr: DenseArrayRecorder | None = None
        self._phi_corr: DenseArrayRecorder | None = None

        self._n_samples_fine: float | None = None
        self._len_coarse_max: int | None = None

    @requires(["IQCavityFeedbackBase"])
    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # not used in this context
        n_turns: int,
        **kwargs: dict[str, Any],
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
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )
        self._n_samples_fine = self._feedback.profile.n_bins
        n_entries = n_turns // self.each_turn_i + 2

        # One turn of coarse grid plus one section of overshoot: in the first
        # turn the last station's grid also covers a partial span from the
        # next turn. The station count comes from the feedback, which counts
        # RFStationBaseClass -- deriving it here by filtering the ring for
        # SingleHarmonicRFStation divided by zero on a ring whose stations are
        # all multi-harmonic (the two are siblings, not parent and child).
        #
        # The analytic term is only an estimate: the real grid is produced
        # by np.arange walks over up to n_rf_stations_in_ring + 1 segments
        # (one per station passage plus the overshoot segment), and each
        # walk can yield one cell more than the analytic fraction whenever
        # harmonic * segment_fraction / n_rf_periods_per_coarse_grid is not
        # an integer (sub-stepping, non-divisor coarse steps, ramping
        # design t_rf). Measured: 51801 cells against a 51800 prediction
        # (2 stations, n_rf_periods_per_coarse_grid = 0.75, station at the
        # section end, where the overshoot segment spans a full section and
        # consumes the analytic 1/n_stations term exactly). Hence the
        # margin of one extra cell per possible segment. Over-allocation is
        # harmless -- unwritten columns stay NaN-masked -- while an
        # under-allocation would abort the run in `_update`.
        n_stations = self._feedback.n_rf_stations_in_ring
        self._len_coarse_max = (
            int(
                np.ceil(
                    (1 + 1 / n_stations)
                    * self._feedback.harmonic
                    / self._feedback.n_rf_periods_per_coarse_grid
                )
            )
            + n_stations
            + 1
        )

        shape_coarse = (n_entries, self._len_coarse_max)
        shape_fine = (n_entries, self._n_samples_fine)

        self._v_ant_fine = DenseArrayRecorder(
            f"{self.common_filepath}_v_ant_fine", shape_fine, dtype=complex
        )
        self._i_beam_fine = DenseArrayRecorder(
            f"{self.common_filepath}_i_beam_fine", shape_fine, dtype=complex
        )
        self._i_gen_fine = DenseArrayRecorder(
            f"{self.common_filepath}_i_gen_fine", shape_fine, dtype=complex
        )
        self._kick_voltage_fine = DenseArrayRecorder(
            f"{self.common_filepath}_kick_voltage_fine",
            shape_fine,
            dtype=complex,
        )

        self._v_ant_coarse = DenseArrayRecorder(
            f"{self.common_filepath}_v_ant_coarse", shape_coarse, dtype=complex
        )
        self._i_beam_coarse = DenseArrayRecorder(
            f"{self.common_filepath}_i_beam_coarse",
            shape_coarse,
            dtype=complex,
        )
        self._i_gen_coarse = DenseArrayRecorder(
            f"{self.common_filepath}_i_gen_coarse",
            shape_coarse,
            dtype=complex,
        )

        self._v_corr = DenseArrayRecorder(
            f"{self.common_filepath}_v_corr", shape_fine
        )
        self._phi_corr = DenseArrayRecorder(
            f"{self.common_filepath}_phi_corr", shape_fine
        )

    def _update(
        self,
    ) -> None:
        """Update memory with new values."""
        self._v_ant_fine.write(
            self._feedback.antenna_voltage_fine_grid
        )  # TODO: redo without capitalization
        self._i_beam_fine.write(self._feedback.beam_current_fine_grid)
        self._i_gen_fine.write(self._feedback.generator_current_fine_grid)
        self._kick_voltage_fine.write(
            self._feedback._parent_rf_station.calc_gap_voltage_with_feedbacks()
        )

        n_grid = len(self._feedback.antenna_voltage_coarse_grid)
        # The beam current is forward-segment-local (see the class Notes);
        # its columns are shifted by the forward offset so that they line
        # up with the whole-grid antenna voltage / generator current.
        forward_offset = int(self._feedback.forward_offset)
        n_forward = len(self._feedback.beam_current_forward_coarse_grid)
        n_needed = max(n_grid, forward_offset + n_forward)
        if n_needed > self._len_coarse_max:
            raise RuntimeError(
                f"IQCavityFeedbackObservation of {self._feedback}: the "
                f"coarse grid of turn {self._simulation.turn_counter.value} "
                f"has {n_needed} cells, but only {self._len_coarse_max} "
                f"columns were allocated (analytic prediction plus one "
                f"cell per segment). The np.arange grid walk produced "
                f"more cells than that upper bound -- increase the "
                f"per-segment margin added to `_len_coarse_max` in "
                f"`IQCavityFeedbackObservation.on_run_simulation`."
            )

        coarse_mask = np.zeros(self._len_coarse_max, dtype=bool)
        coarse_mask[:n_grid] = True

        self._v_ant_coarse.write(
            self._feedback.antenna_voltage_coarse_grid,
            mask=coarse_mask,
        )
        self._i_gen_coarse.write(
            self._feedback.generator_current_coarse_grid,
            mask=coarse_mask,
        )

        coarse_mask = np.zeros(self._len_coarse_max, dtype=bool)
        coarse_mask[forward_offset : forward_offset + n_forward] = True

        self._i_beam_coarse.write(
            self._feedback.beam_current_forward_coarse_grid,
            mask=coarse_mask,
        )

        self._v_corr.write(self._feedback.relative_voltage_correction)
        self._phi_corr.write(self._feedback.phase_correction)

    @property  # as readonly attributes
    def len_coarse_max(self) -> int | None:
        """
        Allocated column count of the coarse matrices, in [1].

        The per-turn coarse grid is padded to this width (see the Notes
        of this class); ``on_run_simulation`` derives it once from the
        feedback's harmonic, coarse step and station count, plus a margin
        of one cell per possible segment. Like the recorders it sizes, it
        is ``None`` until then.

        Read-only: every coarse recorder is allocated with this width and
        ``_update`` masks its rows against it, so a later write would
        disagree with the buffers already in place.

        Returns
        -------
        len_coarse_max
            Number of columns allocated per coarse-matrix row, or
            ``None`` before ``on_run_simulation``.
        """
        return self._len_coarse_max

    @property  # as readonly attributes
    def v_corr(self) -> NumpyArray:
        """
        Relative voltage correction of the feedback ``(n_observations, n_fine)``, in [1].

        Returns
        -------
        v_corr
            Array of voltage corrections on the fine grid.
        """
        return self._v_corr.get_valid_entries()

    @property  # as readonly attributes
    def phi_corr(self) -> NumpyArray:
        """
        Phase correction of the feedback``(n_observations, n_fine)``, in [rad].

        Returns
        -------
        phi_corr
            Array of phase corrections on the fine grid.
        """
        return self._phi_corr.get_valid_entries()

    @property  # as readonly attributes
    def i_beam_fine(self) -> NumpyArray:
        """
        Beam current on the fine grid as observed by the feedback ``(n_observations, n_fine)``, in [A].

        Returns
        -------
        i_beam_fine
            Array of beam currents on the fine grid.
        """
        return self._i_beam_fine.get_valid_entries()

    @property  # as readonly attributes
    def i_gen_fine(self) -> NumpyArray:
        """
        Generator current on the fine grid as observed by the feedback ``(n_observations, n_fine)``, in [A].

        Returns
        -------
        i_gen_fine
            Array of generator currents on the fine grid.
        """
        return self._i_gen_fine.get_valid_entries()

    @property  # as readonly attributes
    def v_ant_fine(self) -> NumpyArray:
        """
        Antenna Voltage in the feedback ``(n_observations, n_fine)``, in [V].

        Returns
        -------
        v_ant_fine
            Array of antenna voltages on the fine grid.
        """
        return self._v_ant_fine.get_valid_entries()

    @property  # as readonly attributes
    def kick_voltage_fine(self) -> NumpyArray:
        """
        Kick voltage in the feedback ``(n_observations, n_fine)``, in [V].

        Returns
        -------
        kick_voltage_fine
            Array of kick voltages on the fine grid.
        """
        return self._kick_voltage_fine.get_valid_entries()

    @property  # as readonly attributes
    def i_beam_coarse(self) -> NumpyArray:
        """
        Beam current on the coarse grid as observed by the feedback ``(n_observations, n_coarse)``, in [A].

        Columns are aligned with ``v_ant_coarse`` / ``i_gen_coarse``;
        entries outside the forward segment are ``NaN`` (see the class
        Notes for the layout).

        Returns
        -------
        i_beam_coarse
            Array of beam currents on the coarse grid.
        """
        return self._i_beam_coarse.get_valid_entries()

    @property  # as readonly attributes
    def i_gen_coarse(self) -> NumpyArray:
        """
        Generator current on the coarse grid as observed by the feedback ``(n_observations, n_coarse)``, in [A].

        Returns
        -------
        i_gen_coarse
            Array of generator currents on the coarse grid.
        """
        return self._i_gen_coarse.get_valid_entries()

    @property  # as readonly attributes
    def v_ant_coarse(self) -> NumpyArray:
        """
        Antenna Voltage in the feedback ``(n_observations, n_coarse)``, in [V].

        Each row records the feedback's ``antenna_voltage_coarse_grid``,
        the demodulation-frame sum of the beam- and generator-sourced
        envelope components. Only the generator-sourced part carries the
        frame rotation, so under an accumulated RF phase slip it is that
        part alone which appears rotated by minus the slip: a beam-free
        driven run therefore shows a pure rotation at constant magnitude,
        while with beam loading present the sum's magnitude moves too. A
        driven readout must be compared in the kick frame, not naively
        against the complex setpoint.

        Returns
        -------
        v_ant_coarse
            Array of antenna voltages on the coarse grid.
        """
        return self._v_ant_coarse.get_valid_entries()


class StaticProfileObservation(ObservablesOncePerTurnBase):
    """
    Observation of a static beam profile.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    profile
        Class for the calculation of beam profile
        that doesn't change its parameters.
    folder
        Path to the target folder used for
        saving or loading files.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>> from blond import Simulation
    >>> sim = Simulation(...)
    >>> profile_obs = StaticProfileObservation(each_turn_i=2, profile=...)
    >>> sim.run_simulation(
    ...     beams=...,
    ...     observe=(profile_obs,),
    ... )
    >>> turn_0 = 0  # first turn
    >>> turn_2 = 1  # after 2 turns, because `each_turn_i = 2`
    >>> for index in (turn_0, turn_2):
    ...     plt.plot(
    ...         profile_obs.hist_x, profile_obs.hist_y[index, :]
    ...     )
    """

    def __init__(
        self,
        each_turn_i: int,
        profile: StaticProfile,
        folder: str = "",
    ):
        super().__init__(
            each_turn_i=each_turn_i,
            folder=folder,
        )
        self._profile = profile
        self._hist_y: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # not used in this context
        n_turns: int,
        **kwargs: dict[str, Any],
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
        super().on_run_simulation(
            simulation=simulation,
            n_turns=n_turns,
            beam=beam,
        )
        n_entries = self._calc_n_entries(n_turns)
        n_bins = int(self._profile.n_bins)
        self._hist_y = DenseArrayRecorder(
            f"{self.common_filepath}_hist_y",
            (n_entries, n_bins),
        )

    def _update(self) -> None:
        """Update memory with new values."""
        self._hist_y.write(
            copy_to_cpu(self._profile.hist_y),
        )

    @property  # as readonly attributes
    def hist_x(self) -> NumpyArray:
        """
        Histogram x axis, always the same.

        Returns
        -------
        hist_x
            Histogram x-axis array.
        """
        return copy_to_cpu(self._profile.hist_x)

    @property  # as readonly attributes
    def hist_y(self) -> NumpyArray:
        """
        Histogram amplitude for each observed turn.

        Returns
        -------
        hist_y
            Histogram amplitude array.
        """
        return self._hist_y.get_valid_entries()

    def plot_waterfall(
        self,
        ax: Axes | None = None,
        kwargs_pcolormesh: dict | None = None,
    ) -> QuadMesh:
        """
        Make a 2D waterfall plot of the profile evolution over turns.

        The profile amplitude is color-coded, with time on the x-axis
        and turn number on the y-axis.

        Parameters
        ----------
        ax
            `Axes` to plot into. The current axes are used if `None`.
        kwargs_pcolormesh
            Keyword arguments for `matplotlib.axes.Axes.pcolormesh`.

        Returns
        -------
        mesh
            The `QuadMesh` pyplot object holding the waterfall plot.
        """
        hist_y = self.hist_y
        turns = self.turns_array[: hist_y.shape[0]]
        return _plot_profile_waterfall(
            hist_x=self.hist_x,
            hist_y=hist_y,
            turns=turns,
            ax=ax,
            kwargs_pcolormesh=kwargs_pcolormesh,
        )


class StaticMultiProfileObservation(ObservablesOncePerTurnBase):
    """
    Observation of multiple profiles in one observation object. The profiles need to have the same n_bins.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    profiles
        List of class for the calculation of beam profile
        that doesn't change its parameters.
    folder
        Path to the target folder used for
        saving or loading files.
    sort_profiles_by_section
        Whether to sort profiles by section index.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>> from blond import Simulation
    >>> sim = Simulation(...)
    >>> profile_obs = StaticMultiProfileObservation(each_turn_i=2, profiles=...)
    >>> sim.run_simulation(
    ...     beams=...,
    ...     observe=(profile_obs,),
    ... )
    >>> # This example assumes that two profiles are in `profile_obs`
    >>> turn_0_profile0 = 0  # turn_0 simulation
    >>> turn_0_profile1 = 1  # turn_0 simulation
    >>> turn_2_profile0 = 2  # after 2 turns, because `each_turn_i = 2`
    >>> turn_2_profile1 = 3  # after 2 turns, because `each_turn_i = 2`
    >>> plt.plot(
    >>>     profile_obs.hist_x[0], profile_obs.hist_y[0][0]
    >>> )
    >>> plt.plot(
    >>>     profile_obs.hist_x[1], profile_obs.hist_y[0][1]
    >>> )
    >>> plt.plot(
    >>>     profile_obs.hist_x[0], profile_obs.hist_y[1][0]
    >>> )
    >>> plt.plot(
    >>>     profile_obs.hist_x[1], profile_obs.hist_y[1][1]
    >>> )
    """

    def __init__(
        self,
        each_turn_i: int,
        profiles: list[StaticProfile],
        folder: str = "",
        sort_profiles_by_section=True,
    ):
        super().__init__(each_turn_i=each_turn_i, folder=folder)

        if sort_profiles_by_section:
            profiles = sorted(profiles, key=lambda prof: prof.section_index)
        self._profiles = profiles

        assert all(
            prof.n_bins == self._profiles[0].n_bins for prof in self._profiles
        ), "n_bins should be equal for all given profiles"

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # this is not used in this context
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
            Simulation beam object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )
        n_turns_observation = int(len(self._turns_array) // self.each_turn_i)
        n_bins = self._profiles[0].n_bins
        shape = (n_turns_observation, len(self._profiles), n_bins)
        self._hist_y = DenseArrayRecorder(
            f"{self.common_filepath}_hist_y",
            shape,
        )

    def _update(self) -> None:
        """Update the data."""
        self._hist_y.write(
            [copy_to_cpu(prof.hist_y) for prof in self._profiles]
        )

    @property  # as readonly attributes
    def hist_x(self) -> list[NumpyArray]:
        """
        Histogram x axis, always the same of shape ``((n_bins, ), ..)``.

        Returns
        -------
        hist_x
            List of histogram x-axis arrays.
        """
        return [
            copy_to_cpu(self._profiles[i].hist_x)
            for i in range(len(self._profiles))
        ]

    @property  # as readonly attributes
    def hist_y(self) -> NumpyArray:
        """
        Histogram of given profiles of shape ``(n_observations, n_bins)``.

        Returns
        -------
        hist_y
            Histogram amplitude array.
        """
        return self._hist_y.get_valid_entries()


class WakeFieldObservation(ObservablesOncePerTurnBase):
    """
    Observe the calculation of wake-fields.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    wakefield
        Manager class to calculate wake-fields.
    folder
        Path to the target folder used for
        saving or loading files.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>> from blond import Simulation
    >>> sim = Simulation(...)
    >>> wake_obs = WakeFieldObservation(wakefield=..., each_turn_i=2)
    >>> sim.run_simulation(
    ...     beams=...,
    ...     observe=(wake_obs,),
    ... )
    >>> turn_0 = 0  # first turn
    >>> turn_2 = 1  # after 2 turns, because `each_turn_i = 2`
    >>> for index in (turn_0, turn_2):
    ...     plt.plot(wake_obs.induced_voltage[index, :])
    """

    def __init__(
        self,
        each_turn_i: int,
        wakefield: WakeField,
        folder: str = "",
    ):
        super().__init__(
            each_turn_i=each_turn_i,
            folder=folder,
        )
        self._wakefield = wakefield
        self._induced_voltage: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # not used in this context
        n_turns: int,
        **kwargs: dict[str, Any],
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
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )

        n_entries = self._calc_n_entries(n_turns)
        n_bins = int(self._wakefield._profile.n_bins)
        self._induced_voltage = DenseArrayRecorder(
            f"{self.common_filepath}_induced_voltage",
            (n_entries, n_bins),
        )

    def _update(self) -> None:
        """Update memory with new values."""
        try:
            self._induced_voltage.write(
                self._wakefield.induced_voltage,
            )
        except AttributeError:
            self._induced_voltage.write(
                np.zeros(self._wakefield._profile.n_bins)
            )

    @property  # as readonly attributes
    def induced_voltage(self) -> NumpyArray:
        """
        Induced voltage, in [V] from given beam profile and sources  of shape ``(n_observations, n_bins)``.

        Returns
        -------
        induced_voltage
            Array of induced voltages.
        """
        return self._induced_voltage.get_valid_entries()


class DynamicProfileConstNBinsObservation(ObservablesOncePerTurnBase):
    """
    Observation of a dynamic beam profile with changing width, while keeping a constant bin number.

    Parameters
    ----------
     each_turn_i
        Value to control that the element is
        callable each n-th turn.
    profile
        Class for the calculation of beam profile
        with a change in width, but a constant bin number.
    folder
        Path to the target folder used for
        saving or loading files.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>> from blond import Simulation
    >>> sim = Simulation(...)
    >>> profile_obs = DynamicProfileConstNBinsObservation(each_turn_i=2, profile=...)
    >>> sim.run_simulation(
    ...     beams=...,
    ...     observe=(profile_obs,),
    ... )
    >>> turn_0 = 0  # first turn
    >>> turn_2 = 1  # after 2 turns, because `each_turn_i = 2`
    >>> for index in (turn_0, turn_2):
    ...     plt.plot(
    ...         profile_obs.hist_x[index, :], profile_obs.hist_y[index, :]
    ...     )
    """

    def __init__(
        self,
        each_turn_i: int,
        profile: DynamicProfileConstNBins,
        folder: str = "",
    ):
        super().__init__(each_turn_i=each_turn_i, folder=folder)
        self._profile = profile
        self._hist_y: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when :meth:`blond.core.simulation.simulation.Simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation beam object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )

        n_entries = self._calc_n_entries(n_turns)
        n_bins = int(self._profile.n_bins)
        shape = (n_entries, n_bins)
        self._hist_y = DenseArrayRecorder(
            f"{self.common_filepath}_hist_y",
            shape,
        )
        self._hist_x = DenseArrayRecorder(
            f"{self.common_filepath}_hist_x",
            shape,
        )

    def _update(self) -> None:
        """Update memory with new values."""
        self._hist_y.write(self._profile.hist_y)
        self._hist_x.write(self._profile.hist_x)

    @property  # as readonly attributes
    def hist_y(self) -> NumpyArray:
        """
        Histogram amplitude of shape ``(n_observations, n_bins)``.

        Returns
        -------
        hist_y
            Histogram amplitude array.
        """
        return self._hist_y.get_valid_entries()

    @property  # as readonly attributes
    def hist_x(self) -> NumpyArray:
        """
        Get x-axis of histogram, in [s], i.e. `bin_centers` of shape ``(n_observations, n_bins)``.

        Returns
        -------
        hist_x
            Histogram x-axis array.
        """
        return self._hist_x.get_valid_entries()

    def plot_waterfall(
        self,
        ax: Axes | None = None,
        kwargs_pcolormesh: dict | None = None,
    ) -> QuadMesh:
        """
        Make a 2D waterfall plot of the profile evolution over turns.

        The profile amplitude is color-coded, with time on the x-axis
        and turn number on the y-axis.

        Parameters
        ----------
        ax
            `Axes` to plot into. The current axes are used if `None`.
        kwargs_pcolormesh
            Keyword arguments for `matplotlib.axes.Axes.pcolormesh`.

        Returns
        -------
        mesh
            The `QuadMesh` pyplot object holding the waterfall plot.
        """
        hist_y = self.hist_y
        turns = self.turns_array[: hist_y.shape[0]]
        return _plot_profile_waterfall(
            hist_x=self.hist_x,
            hist_y=hist_y,
            turns=turns,
            ax=ax,
            kwargs_pcolormesh=kwargs_pcolormesh,
        )


class SimulationObservation(ObservablesOncePerTurnBase):
    """
    Observation of the `Simulation` object itself.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    folder
        Path to the target folder used for
        saving or loading files.
    separatrix_points
        Number of points to observe the separatrix with.
    separatrix_lim
        If not provided, the separatrix is recorded within
        ``beam.dt_min`` and ``beam.dt_max``.
    """

    def __init__(
        self,
        each_turn_i: int,
        folder: str = "",
        separatrix_points: int = 256,
        separatrix_lim: tuple[float, float] | None = None,
    ):
        super().__init__(each_turn_i=each_turn_i, folder=folder)
        self._simulation: Simulation | None = None

        self._t_revs: DenseArrayRecorder | None = None
        self._separatrices: DenseArrayRecorder | None = None
        self._separatrix_points = separatrix_points
        self._separatrix_lim = separatrix_lim
        self._beam = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when :meth:`blond.core.simulation.simulation.Simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation beam object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )

        n_entries = self._calc_n_entries(n_turns=n_turns)
        shape = n_entries
        self._t_revs = DenseArrayRecorder(
            f"{self.common_filepath}_t_revs",
            shape,
        )
        self._separatrices = DenseArrayRecorder(
            f"{self.common_filepath}_separatrices",
            (n_entries, 2, self._separatrix_points),
        )
        self._simulation = simulation
        self._beam = beam

    def _update(
        self,
    ) -> None:
        """Update memory with new values."""
        assert self._beam is not None
        assert self._simulation is not None
        assert self._t_revs is not None
        assert self._separatrices is not None

        self._t_revs.write(self._simulation.current_t_rev)

        # Separatrix
        sep = self._simulation._get_separatrix_helper()
        separatrix_lim = (
            (self._beam.dt_min, self._beam.dt_max)
            if self._separatrix_lim is None
            else self._separatrix_lim
        )
        ts = np.linspace(*separatrix_lim, self._separatrix_points)
        self._separatrices.write(sep.get_separatrix(beam=self._beam, dt=ts))

    @property  # as readonly attributes
    def t_revs(self) -> NumpyArray:
        """
        Revolution time, in [s] of shape ``(n_observations)``.

        Returns
        -------
        t_rev
            Revolution time, in [s] of shape ``(n_observations)``.
        """
        return self._t_revs.get_valid_entries()


class DriftObservation(ObservablesOncePerTurnBase):
    """
    Observation of `eta_0` of the `DriftSimple` object.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    drift
        `DriftSimple` object.
    folder
        Path to the target folder used for
        saving or loading files.
    """

    def __init__(
        self,
        each_turn_i: int,
        drift: DriftSimple,
        folder: str = "",
    ):
        super().__init__(each_turn_i=each_turn_i, folder=folder)
        self._drift: DriftSimple = drift

        self._eta_0s: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when :meth:`blond.core.simulation.simulation.Simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation beam object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )

        self._eta_0s = DenseArrayRecorder(
            f"{self.common_filepath}_eta_0s",
            (self._calc_n_entries(n_turns=n_turns)),
        )

    def _update(
        self,
    ) -> None:
        """Update memory with new values."""
        self._eta_0s.write(float(self._drift._last_eta_0))

    @property  # as readonly attributes
    def eta_0s(self) -> NumpyArray:
        """
        Drift in arc parameter eta of shape ``(n_observations)``.

        Returns
        -------
        eta_0
            Drift in arc parameter eta of shape ``(n_observations)``.
        """
        return self._eta_0s.get_valid_entries()


class _CounterRotatingPassage(ObservablesOncePerTurnBase):
    """
    Record the counter-rotating half of a :class:`FullTurnCavityObservation`.

    :meth:`~blond.core.base.SimulationElementBase.add_observable` keys its
    dict on ``id(beam)``, and
    :meth:`ObservablesOncePerTurnBase.update`
    refuses a second call within one turn, so one instance cannot serve
    both beams of a counter-rotating run.  The parent therefore takes the
    co-rotating slot itself and hands this arm the counter-rotating one;
    the arm owns no storage and only forwards the passage to the parent.

    Parameters
    ----------
    each_turn_i
        Record every ``each_turn_i``-th turn.
    parent
        The observation that owns the recorders.
    folder
        Target folder for :meth:`~blond.core.simulation.simulation.Simulation.save_results`.

    Notes
    -----
    Found and late-initialised by ``Simulation._exec_all_in_tree``, which
    walks the whole attribute tree, so this arm needs no entry in
    ``run_simulation(observe=...)`` -- and must not have one, or it would
    be updated twice per turn and raise.
    """

    def __init__(
        self,
        each_turn_i: int,
        parent: FullTurnCavityObservation,
        folder: str = "",
    ) -> None:
        super().__init__(each_turn_i=each_turn_i, folder=folder)
        self._parent = parent

    def _update(self) -> None:
        """Record the counter-rotating passage into the parent."""
        self._parent.record_passage(1)


class FullTurnCavityObservation(ObservablesOncePerTurnBase):
    """
    Record one cavity feedback over a whole turn: both passages, both grids.

    :class:`IQCavityFeedbackObservation`
    records the feedback once per turn, at the end, and therefore keeps
    only the coarse grid standing at that moment.  That grid spans the
    interval between the two counter-rotating passages at the station --
    ``|n - 2 i - 1| / n`` of a revolution -- so it is **not** a full turn:
    0.9375 turn at RCS1's station 0, but only 0.4375 turn at station 4,
    and the missing part is exactly where the other beam passes.

    This observation instead records at **every** passage of **both**
    beams.  The feedback rebuilds its grid at each passage, running from
    the other beam's previous passage up to this one, so the two records
    of a turn abut without gap or overlap and their union is the whole
    revolution.  Measured on RCS1 station 0: the co-rotating passage
    covers ``[0.0312, 0.0937]`` turn and the counter-rotating one
    ``[0.0937, 1.0312]``.

    Both fine grids are kept separately, one per beam, because each beam
    sees its own passage: the fine arrays are rebuilt per passage and the
    end-of-turn record only ever holds the last beam's.

    Parameters
    ----------
    each_turn_i
        Record every ``each_turn_i``-th turn.
    feedback
        The cavity feedback to watch.
    beams
        ``(co_rotating, counter_rotating)``, in that order.  Held to read
        each passage's arrival time from ``beam.reference.time``.
    section_index
        Index of the RF station the feedback belongs to (bookkeeping only).
    folder
        Target folder for :meth:`~blond.core.simulation.simulation.Simulation.save_results`.  Must end in a
        path separator when non-empty.

    Attributes
    ----------
    counter_arm
        The observable that takes the counter-rotating slot.  Attach it
        alongside this object; both must reach the feedback through
        ``add_observable``.

    Notes
    -----
    Coarse voltages are **per cavity** [V] and fine-grid voltages carry the
    station total, following ``IQCavityFeedbackTimingClass``.

    Rows are NaN-padded to a fixed width, sized as in
    ``IQCavityFeedbackObservation``: one turn of coarse grid plus one
    section of overshoot, plus one cell per possible segment.  Both
    passages share that width even though one is usually much shorter, so
    a row of either can be read with the same mask.

    Memory is the price of the whole turn: two passages of coarse grid per
    turn instead of one, i.e. about twice ``IQCavityFeedbackObservation``.
    """

    def __init__(
        self,
        each_turn_i: int,
        feedback: IQCavityFeedbackBase,
        beams: tuple[BeamBaseClass, BeamBaseClass],
        section_index: int = 0,
        folder: str = "",
    ) -> None:
        super().__init__(each_turn_i=each_turn_i, folder=folder)
        self._feedback = feedback
        self._beams = tuple(beams)
        self.section_index = int(section_index)
        self.counter_arm = _CounterRotatingPassage(
            each_turn_i=each_turn_i, parent=self, folder=folder
        )

        self._len_coarse_max: int | None = None
        self._n_samples_fine: int | None = None
        self._v_ant_coarse: list[DenseArrayRecorder] = []
        self._i_gen_coarse: list[DenseArrayRecorder] = []
        self._i_beam_coarse: list[DenseArrayRecorder] = []
        self._i_refl_coarse: list[DenseArrayRecorder] = []
        self._v_ant_fine: list[DenseArrayRecorder] = []
        self._i_gen_fine: list[DenseArrayRecorder] = []
        self._i_beam_fine: list[DenseArrayRecorder] = []
        self._v_corr: list[DenseArrayRecorder] = []
        self._phi_corr: list[DenseArrayRecorder] = []
        self._arrival_time: list[DenseArrayRecorder] = []
        self._forward_offset: list[DenseArrayRecorder] = []

    @requires(["IQCavityFeedbackBase"])
    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # always beams[0]; unused here
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Allocate one set of recorders per passage.

        Parameters
        ----------
        simulation
            The running :class:`~blond.core.simulation.simulation.Simulation`.
        beam
            Ignored -- this observable watches a feedback, not a beam.
        n_turns
            Number of turns the simulation will run.
        **kwargs
            Unused, kept for interface compatibility.
        """
        super().on_run_simulation(
            simulation=simulation, beam=beam, n_turns=n_turns
        )
        if self._v_ant_coarse:  # both arms reach this; allocate once
            return
        self._n_samples_fine = int(self._feedback.profile.n_bins)
        n_entries = self._calc_n_entries(n_turns=n_turns) + 2
        # Same bound as IQCavityFeedbackObservation: one turn of coarse
        # grid plus one section of overshoot, plus a margin of one cell
        # per possible segment, because each np.arange segment walk can
        # yield one cell more than the analytic fraction.
        n_stations = int(self._feedback.n_rf_stations_in_ring)
        self._len_coarse_max = (
            int(
                np.ceil(
                    (1 + 1 / n_stations)
                    * self._feedback.harmonic
                    / self._feedback.n_rf_periods_per_coarse_grid
                )
            )
            + n_stations
            + 1
        )
        shape_coarse = (n_entries, self._len_coarse_max)
        shape_fine = (n_entries, self._n_samples_fine)

        prefix = f"{self.common_filepath}_fullturn_s{self.section_index}"
        for name in ("co", "counter"):
            stem = f"{prefix}_{name}"
            self._v_ant_coarse.append(
                DenseArrayRecorder(
                    f"{stem}_v_ant_coarse", shape_coarse, dtype=complex
                )
            )
            self._i_gen_coarse.append(
                DenseArrayRecorder(
                    f"{stem}_i_gen_coarse", shape_coarse, dtype=complex
                )
            )
            self._i_beam_coarse.append(
                DenseArrayRecorder(
                    f"{stem}_i_beam_coarse", shape_coarse, dtype=complex
                )
            )
            self._i_refl_coarse.append(
                DenseArrayRecorder(
                    f"{stem}_i_refl_coarse", shape_coarse, dtype=complex
                )
            )
            self._v_ant_fine.append(
                DenseArrayRecorder(
                    f"{stem}_v_ant_fine", shape_fine, dtype=complex
                )
            )
            self._i_gen_fine.append(
                DenseArrayRecorder(
                    f"{stem}_i_gen_fine", shape_fine, dtype=complex
                )
            )
            self._i_beam_fine.append(
                DenseArrayRecorder(
                    f"{stem}_i_beam_fine", shape_fine, dtype=complex
                )
            )
            self._v_corr.append(
                DenseArrayRecorder(f"{stem}_v_corr", shape_fine)
            )
            self._phi_corr.append(
                DenseArrayRecorder(f"{stem}_phi_corr", shape_fine)
            )
            self._arrival_time.append(
                DenseArrayRecorder(f"{stem}_arrival_time", n_entries)
            )
            self._forward_offset.append(
                DenseArrayRecorder(f"{stem}_forward_offset", n_entries)
            )

    def _update(self) -> None:
        """Record the co-rotating passage; the arm records the other."""
        self.record_passage(0)

    def record_passage(self, index: int) -> None:
        """
        Store the feedback state at one beam's passage.

        Parameters
        ----------
        index
            ``0`` for the co-rotating beam, ``1`` for the counter-rotating
            one.

        Raises
        ------
        RuntimeError
            If the coarse grid of this passage is wider than the allocated
            row, which would otherwise truncate the record silently.
        """
        feedback = self._feedback
        n_grid = len(feedback.antenna_voltage_coarse_grid)
        forward_offset = int(feedback.forward_offset)
        n_forward = len(feedback.beam_current_forward_coarse_grid)
        n_needed = max(n_grid, forward_offset + n_forward)
        if n_needed > self._len_coarse_max:
            raise RuntimeError(
                f"FullTurnCavityObservation of section {self.section_index}: "
                f"the coarse grid of this passage has {n_needed} cells, but "
                f"only {self._len_coarse_max} columns were allocated. "
                "Increase the per-segment margin in `on_run_simulation`."
            )

        whole_mask = np.zeros(self._len_coarse_max, dtype=bool)
        whole_mask[:n_grid] = True
        self._v_ant_coarse[index].write(
            feedback.antenna_voltage_coarse_grid, mask=whole_mask
        )
        self._i_gen_coarse[index].write(
            feedback.generator_current_coarse_grid, mask=whole_mask
        )
        # Computed here, at the passage, rather than derived at plot time:
        # it is a property of the field and the drive as they stood during
        # this passage, and costs one complex subtract per cell.
        self._i_refl_coarse[index].write(
            feedback.reflected_current(), mask=whole_mask
        )

        # The beam current is forward-segment-local: shift its columns by
        # the forward offset so that they index the same cells as the
        # whole-grid antenna voltage and generator current.
        forward_mask = np.zeros(self._len_coarse_max, dtype=bool)
        forward_mask[forward_offset : forward_offset + n_forward] = True
        self._i_beam_coarse[index].write(
            feedback.beam_current_forward_coarse_grid, mask=forward_mask
        )

        self._v_ant_fine[index].write(feedback.antenna_voltage_fine_grid)
        self._i_gen_fine[index].write(feedback.generator_current_fine_grid)
        self._i_beam_fine[index].write(feedback.beam_current_fine_grid)
        self._v_corr[index].write(feedback.relative_voltage_correction)
        self._phi_corr[index].write(feedback.phase_correction)

        self._arrival_time[index].write(
            float(self._beams[index].reference.time)
        )
        self._forward_offset[index].write(float(forward_offset))

    @property
    def antenna_voltage_coarse(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Antenna voltage on the coarse grid per passage, in [V] per cavity.

        Returns
        -------
        antenna_voltage_coarse
            ``(co_rotating, counter_rotating)``, each of shape
            ``(n_records, len_coarse_max)`` and NaN-padded to the
            right.
        """
        return tuple(rec.get_valid_entries() for rec in self._v_ant_coarse)

    @property
    def generator_current_coarse(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generator current on the coarse grid per passage, in [A] per cavity.

        Returns
        -------
        generator_current_coarse
            ``(co_rotating, counter_rotating)``, NaN-padded to the right.
        """
        return tuple(rec.get_valid_entries() for rec in self._i_gen_coarse)

    @property
    def beam_current_coarse(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Beam current on the coarse grid per passage, in [A].

        Returns
        -------
        beam_current_coarse
            ``(co_rotating, counter_rotating)``, shifted by
            :attr:`forward_offset` so that its columns index the
            same cells as :attr:`antenna_voltage_coarse`; cells
            outside the forward segment are ``NaN``.
        """
        return tuple(rec.get_valid_entries() for rec in self._i_beam_coarse)

    @property
    def reflected_current_coarse(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Reflected current on the coarse grid per passage, in [A] per cavity.

        Returns
        -------
        reflected_current_coarse
            ``(co_rotating, counter_rotating)``. Evaluated at the passage
            as ``V_ant / ((R/Q) Q_L) - r_gen I_gen``, the generator current
            rotated into the frame each cell was composed in. Zero only
            when the beam absorbs the whole forward wave; with no beam a
            superconducting cavity reflects all of it -- see
            :meth:`~blond.physics.feedbacks.generator_regulation.GeneratorRegulationMixin.reflected_current`.
        """
        return tuple(rec.get_valid_entries() for rec in self._i_refl_coarse)

    @property
    def antenna_voltage_fine(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Antenna voltage on the fine grid per passage, in [V].

        Returns
        -------
        antenna_voltage_fine
            ``(co_rotating, counter_rotating)``, carrying the station
            total rather than the per-cavity value.
        """
        return tuple(rec.get_valid_entries() for rec in self._v_ant_fine)

    @property
    def generator_current_fine(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generator current on the fine grid per passage, in [A].

        Returns
        -------
        generator_current_fine
            ``(co_rotating, counter_rotating)``.
        """
        return tuple(rec.get_valid_entries() for rec in self._i_gen_fine)

    @property
    def beam_current_fine(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Beam current on the fine grid per passage, in [A].

        Returns
        -------
        beam_current_fine
            ``(co_rotating, counter_rotating)``.
        """
        return tuple(rec.get_valid_entries() for rec in self._i_beam_fine)

    @property
    def relative_voltage_correction(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Amplitude correction handed to the beam per passage, in [1].

        Returns
        -------
        relative_voltage_correction
            ``(co_rotating, counter_rotating)``.
        """
        return tuple(rec.get_valid_entries() for rec in self._v_corr)

    @property
    def phase_correction(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Phase correction handed to the beam per passage, in [rad].

        Returns
        -------
        phase_correction
            ``(co_rotating, counter_rotating)``.
        """
        return tuple(rec.get_valid_entries() for rec in self._phi_corr)

    @property
    def arrival_time(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Absolute time of each passage, in [s].

        Returns
        -------
        arrival_time
            ``(co_rotating, counter_rotating)``. ``beam.reference.time``
            when the feedback tracked that beam, which is what
            places the two passages of a turn on one axis.
        """
        return tuple(rec.get_valid_entries() for rec in self._arrival_time)

    @property
    def forward_offset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Backfill cells preceding the forward segment per passage, in [1].

        Returns
        -------
        forward_offset
            ``(co_rotating, counter_rotating)``. The forward segment is
            the passage itself, so cell ``forward_offset`` of a row
            is the one the bunch arrives in. This is what turns a
            column index into an absolute time -- see
            :meth:`cell_times`.
        """
        return tuple(rec.get_valid_entries() for rec in self._forward_offset)

    def cell_times(self, index: int, record: int) -> np.ndarray:
        """
        Give the absolute times of one recorded passage's coarse cells.

        Parameters
        ----------
        index
            ``0`` for the co-rotating beam, ``1`` for the counter-rotating
            one.
        record
            Which recorded passage, i.e. which row of the arrays.

        Returns
        -------
        times
            Cell centre times [s], NaN where the row is padded.

        Notes
        -----
        Cell ``k`` sits at ``arrival + (k - forward_offset) * dt_cell``,
        because the forward segment is the passage itself and so its first
        cell is the arrival.  ``rf_centers`` cannot be used directly here:
        its entries are segment-local, and the array is not globally
        monotonic.

        Concatenating both passages of a turn under this rule tiles the
        revolution exactly, which is the point of the class.
        """
        voltage = self._v_ant_coarse[index].get_valid_entries()[record]
        cell_duration = self._feedback.n_rf_periods_per_coarse_grid * (
            2.0 * np.pi / self._feedback.omega_rf_design
        )
        offset = float(self._forward_offset[index].get_valid_entries()[record])
        arrival = float(self._arrival_time[index].get_valid_entries()[record])
        cells = np.arange(voltage.size, dtype=float)
        times = arrival + (cells - offset) * cell_duration
        return np.where(np.isfinite(voltage), times, np.nan)


class CavityEnvelopeSummary(ObservablesOncePerTurnBase):
    """
    Per-turn scalar summary of a cavity-voltage envelope.

    Reads ``feedback.antenna_voltage_coarse_grid`` at the end of every turn
    -- i.e. after *both* counter-rotating beams have passed the station --
    and stores a handful of scalars instead of the full grid.

    Parameters
    ----------
    each_turn_i
        Record every ``each_turn_i``-th turn.
    feedback
        The cavity feedback to watch.
    section_index
        Index of the RF station the feedback belongs to (bookkeeping only).
    folder
        Target folder for :meth:`~blond.core.simulation.simulation.Simulation.save_results`.  Must end in a
        path separator when non-empty.

    Notes
    -----
    All voltages are **per cavity** [V]: the coarse grid of
    ``IQCavityFeedbackTimingClass`` is normalised per cavity, while the fine
    grid carries the station total.
    """

    def __init__(
        self,
        each_turn_i: int,
        feedback: IQCavityFeedbackBase,
        section_index: int = 0,
        folder: str = "",
    ) -> None:
        super().__init__(each_turn_i=each_turn_i, folder=folder)
        self._feedback = feedback
        self.section_index = int(section_index)

        self._magnitude_min: DenseArrayRecorder | None = None
        self._magnitude_mean: DenseArrayRecorder | None = None
        self._magnitude_max: DenseArrayRecorder | None = None
        self._magnitude_end: DenseArrayRecorder | None = None
        self._phase_end: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # always beams[0]; unused here
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Allocate the recorders when the simulation starts.

        Parameters
        ----------
        simulation
            The running :class:`~blond.core.simulation.simulation.Simulation`.
        beam
            Ignored -- this observable watches a feedback, not a beam.
        n_turns
            Number of turns the simulation will run.
        **kwargs
            Unused, kept for interface compatibility.
        """
        super().on_run_simulation(
            simulation=simulation, beam=beam, n_turns=n_turns
        )
        n_entries = self._calc_n_entries(n_turns=n_turns)
        prefix = f"{self.common_filepath}_envelope_s{self.section_index}"
        self._magnitude_min = DenseArrayRecorder(f"{prefix}_min", n_entries)
        self._magnitude_mean = DenseArrayRecorder(f"{prefix}_mean", n_entries)
        self._magnitude_max = DenseArrayRecorder(f"{prefix}_max", n_entries)
        self._magnitude_end = DenseArrayRecorder(f"{prefix}_end", n_entries)
        self._phase_end = DenseArrayRecorder(f"{prefix}_phase", n_entries)

    def _update(self) -> None:
        """Record the current envelope statistics."""
        envelope = self._feedback.antenna_voltage_coarse_grid
        magnitude = np.abs(envelope)
        self._magnitude_min.write(float(np.nanmin(magnitude)))
        self._magnitude_mean.write(float(np.nanmean(magnitude)))
        self._magnitude_max.write(float(np.nanmax(magnitude)))
        self._magnitude_end.write(float(magnitude[-1]))
        self._phase_end.write(float(np.angle(envelope[-1])))

    @property
    def magnitude_min(self) -> np.ndarray:
        """
        Smallest ``|V_ant|`` over the recorded window [V per cavity].

        NOT a per-turn minimum, for the same reason as
        :attr:`magnitude_mean`: the window is the coarse grid standing at
        the end of the turn, which spans ``|n - 2 i - 1| / n`` of a
        revolution and is therefore station-dependent -- 0.9375 turn at
        stations 0 and 15 but 0.0625 turn at stations 7 and 8 for RCS1's
        16 sections.  A station whose window happens to exclude the sag
        reports a shallower minimum than one whose window contains it, so
        the per-station curves are not directly comparable even though
        they share an axis.

        Returns
        -------
        magnitude_min
            Smallest ``|V_ant|`` over the recorded window [V per cavity].
        """
        return self._magnitude_min.get_valid_entries()

    @property
    def magnitude_mean(self) -> np.ndarray:
        """
        Mean ``|V_ant|`` over the recorded window [V per cavity].

        NOT a turn average: the feedback rebuilds its coarse grid at every
        passage, so the grid standing at the end of a turn spans
        ``|n - 2 i - 1| / n`` of a revolution -- 0.9375 turn at stations 0
        and 15 but 0.0625 turn at stations 7 and 8 for RCS1's 16 sections.
        The window is therefore station-dependent and this mean is not
        comparable across stations.

        Returns
        -------
        magnitude_mean
            Mean ``|V_ant|`` over the recorded window [V per cavity].
        """
        return self._magnitude_mean.get_valid_entries()

    @property
    def magnitude_max(self) -> np.ndarray:
        """
        Largest ``|V_ant|`` seen during each turn [V per cavity].

        Returns
        -------
        magnitude_max
            Largest ``|V_ant|`` seen during each turn [V per cavity].
        """
        return self._magnitude_max.get_valid_entries()

    @property
    def magnitude_end(self) -> np.ndarray:
        """
        ``|V_ant|`` at the end of each turn [V per cavity].

        Returns
        -------
        magnitude_end
            ``|V_ant|`` at the end of each turn [V per cavity].
        """
        return self._magnitude_end.get_valid_entries()

    @property
    def phase_end(self) -> np.ndarray:
        """
        ``arg(V_ant)`` at the end of each turn [rad].

        Returns
        -------
        phase_end
            ``arg(V_ant)`` at the end of each turn [rad].
        """
        return self._phase_end.get_valid_entries()


class ControllerCorrectionSummary(ObservablesOncePerTurnBase):
    """
    Per-turn summary of the correction the loop hands to the beam.

    The cavity feedback's output to the RF station is two fine-grid
    arrays: ``relative_voltage_correction`` (amplitude, in units of the
    station voltage) and ``phase_correction`` [rad].  The station applies
    entry ``j`` to whichever particles fall in profile bin ``j``, so the
    single number that describes what the *bunch* received is the
    **charge-weighted** mean over the profile -- not the plain mean over
    the window, in which the bunch occupies only a few percent of the bins
    and empty bins would count equally.  Both are recorded, because their
    difference is itself diagnostic: they agree only when the correction
    is flat across the window.

    Six scalars per turn, so this can be attached to every station where
    :class:`IQCavityFeedbackObservation` (which keeps both full grids) can
    only be afforded on one.

    Parameters
    ----------
    each_turn_i
        Record every ``each_turn_i``-th turn.
    feedback
        The cavity feedback whose readout is summarised.
    profile
        The station's live profile, supplying the charge weights.  It must
        be the profile the feedback itself reads, or the weights do not
        line up with the correction arrays.
    section_index
        Index of the RF station (bookkeeping only).
    beam_label
        Name of the beam whose passage this instance samples, or ``""``
        (default) for the once-per-turn instance that is not tied to a
        beam.  Bookkeeping: it keeps the recorder filenames unique and
        labels the plot.
    folder
        Target folder for :meth:`~blond.core.simulation.simulation.Simulation.save_results`.

    Notes
    -----
    Passed to ``run_simulation(observe=...)`` this reads the state
    standing at the **end of the turn**, like :class:`CavityEnvelopeSummary`
    -- and the feedback overwrites both arrays at *every* passage, so an
    end-of-turn read keeps only the later of the turn's two
    counter-rotating passages, and which beam that is differs between the
    first and second half of the element list.  Attached to a station with
    :meth:`~blond.core.base.SimulationElementBase.add_observable` instead,
    it fires at that beam's own passage and both are recorded separately.

    ``phase_correction`` is the cheapest no-op check of a feedback: a
    driven cavity sitting on its setpoint with no beam must hand the
    station a phase of exactly zero, so a run at negligible intensity
    that shows anything else has a feedback that is not phase-neutral.
    """

    def __init__(  # noqa: PLR0913 - mirrors CavityEnvelopeSummary's shape
        self,
        each_turn_i: int,
        feedback: IQCavityFeedbackBase,
        profile: StaticProfile,
        section_index: int = 0,
        beam_label: str = "",
        folder: str = "",
    ) -> None:
        super().__init__(each_turn_i=each_turn_i, folder=folder)
        self._feedback = feedback
        self._profile = profile
        self.section_index = int(section_index)
        self.beam_label = beam_label

        self._v_corr_bunch: DenseArrayRecorder | None = None
        self._v_corr_window: DenseArrayRecorder | None = None
        self._v_corr_spread: DenseArrayRecorder | None = None
        self._phi_corr_bunch: DenseArrayRecorder | None = None
        self._phi_corr_window: DenseArrayRecorder | None = None
        self._phi_corr_spread: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # always beams[0]; unused here
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Allocate the recorders when the simulation starts.

        Parameters
        ----------
        simulation
            The running :class:`~blond.core.simulation.simulation.Simulation`.
        beam
            Ignored -- this observable watches a feedback, not a beam.
        n_turns
            Number of turns the simulation will run.
        **kwargs
            Unused, kept for interface compatibility.
        """
        super().on_run_simulation(
            simulation=simulation, beam=beam, n_turns=n_turns
        )
        n_entries = self._calc_n_entries(n_turns=n_turns)
        prefix = f"{self.common_filepath}_correction_s{self.section_index}"
        if self.beam_label:
            prefix = f"{prefix}_{self.beam_label}"
        self._v_corr_bunch = DenseArrayRecorder(f"{prefix}_v_bunch", n_entries)
        self._v_corr_window = DenseArrayRecorder(
            f"{prefix}_v_window", n_entries
        )
        self._v_corr_spread = DenseArrayRecorder(
            f"{prefix}_v_spread", n_entries
        )
        self._phi_corr_bunch = DenseArrayRecorder(
            f"{prefix}_phi_bunch", n_entries
        )
        self._phi_corr_window = DenseArrayRecorder(
            f"{prefix}_phi_window", n_entries
        )
        self._phi_corr_spread = DenseArrayRecorder(
            f"{prefix}_phi_spread", n_entries
        )

    def _summarise(
        self, values: np.ndarray, weights: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Reduce one fine-grid correction array to three scalars.

        Parameters
        ----------
        values
            The correction on the fine grid.
        weights
            Charge per bin, from the live profile.

        Returns
        -------
        bunch, window, spread
            Charge-weighted mean, plain window mean, and the peak-to-peak
            spread over the bins that actually hold charge.  All three are
            ``NaN`` when the window holds no charge at all, which is a
            real configuration (``--intensity-scale 0``) and not an error.
        """
        finite = np.isfinite(values)
        window = float(np.mean(values[finite])) if finite.any() else np.nan
        occupied = finite & (weights > 0.0)
        if not occupied.any():
            return np.nan, window, np.nan
        total = float(np.sum(weights[occupied]))
        bunch = float(np.sum(values[occupied] * weights[occupied]) / total)
        return bunch, window, float(np.ptp(values[occupied]))

    def _update(self) -> None:
        """Record the current correction summary."""
        weights = np.asarray(copy_to_cpu(self._profile.hist_y), dtype=float)
        v_corr = np.asarray(
            copy_to_cpu(self._feedback.relative_voltage_correction),
            dtype=float,
        )
        phi_corr = np.asarray(
            copy_to_cpu(self._feedback.phase_correction), dtype=float
        )

        bunch, window, spread = self._summarise(v_corr, weights)
        self._v_corr_bunch.write(bunch)
        self._v_corr_window.write(window)
        self._v_corr_spread.write(spread)

        bunch, window, spread = self._summarise(phi_corr, weights)
        self._phi_corr_bunch.write(bunch)
        self._phi_corr_window.write(window)
        self._phi_corr_spread.write(spread)

    @property
    def v_corr_bunch(self) -> np.ndarray:
        """
        Charge-weighted amplitude correction per turn [1].

        ``1.0`` means the bunch saw exactly the station's nominal voltage.

        Returns
        -------
        v_corr_bunch
            Charge-weighted amplitude correction per turn [1].
        """
        return self._v_corr_bunch.get_valid_entries()

    @property
    def v_corr_window(self) -> np.ndarray:
        """
        Plain window mean of the amplitude correction per turn [1].

        Kept beside :attr:`v_corr_bunch` because the gap between the two
        is the signature of a correction that varies across the profile
        window rather than being the rigid offset it is often assumed to
        be.

        Returns
        -------
        v_corr_window
            Plain window mean of the amplitude correction per turn [1].
        """
        return self._v_corr_window.get_valid_entries()

    @property
    def v_corr_spread(self) -> np.ndarray:
        """
        Peak-to-peak amplitude correction across the bunch [1].

        Returns
        -------
        v_corr_spread
            Peak-to-peak amplitude correction across the bunch [1].
        """
        return self._v_corr_spread.get_valid_entries()

    @property
    def phi_corr_bunch(self) -> np.ndarray:
        """
        Charge-weighted phase correction per turn [rad].

        Returns
        -------
        phi_corr_bunch
            Charge-weighted phase correction per turn [rad].
        """
        return self._phi_corr_bunch.get_valid_entries()

    @property
    def phi_corr_window(self) -> np.ndarray:
        """
        Plain window mean of the phase correction per turn [rad].

        Returns
        -------
        phi_corr_window
            Plain window mean of the phase correction per turn [rad].
        """
        return self._phi_corr_window.get_valid_entries()

    @property
    def phi_corr_spread(self) -> np.ndarray:
        """
        Peak-to-peak phase correction across the bunch [rad].

        Returns
        -------
        phi_corr_spread
            Peak-to-peak phase correction across the bunch [rad].
        """
        return self._phi_corr_spread.get_valid_entries()
