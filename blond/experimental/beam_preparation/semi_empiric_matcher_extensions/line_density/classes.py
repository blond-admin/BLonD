# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from blond.acc_math.empiric.potential_well import PotentialWellHelper
from blond.experimental.beam_preparation.semi_empiric_matcher_extensions.line_density.callables import (
    occupation_per_equipotential_to_density,
    occupation_per_equipotential_to_histogram,
)

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


class SemiEmpiricMatcherAddon(ABC):
    """Abstract class to define addons for the `SemiEmpiricMatcher`."""

    @abstractmethod  # pragma: no cover
    def hamilton_to_density_function(
        self,
        time_grid: NumpyArray | CupyArray,
        deltaE_grid: NumpyArray | CupyArray,
        hamilton_2D: NumpyArray | CupyArray,
    ) -> NumpyArray | CupyArray:
        """
        This function is an endpoint for the `SemiEmpiricMatcher`.

        Parameters
        ----`------
        deltaE_grid
            The time coordinates corresponding to `hamilton_2D`, in [eV].
        time_grid
            The time coordinates corresponding to `hamilton_2D`, in [s].
        hamilton_2D
            A 2D array representing the spatial Hamilton field.

        Returns
        -------
        density : NumpyArray or CupyArray
            A 2D array of the same shape as `hamilton_2D`, representing the
            computed density distribution. Values are scaled between 0 and 1.
        """
        pass


class ProfileMatcherAddon(SemiEmpiricMatcherAddon):
    """
    Helper class to match a beam profile to a target simulation histogram.

    Parameters
    ----------
    hist_x : array-like
        Time coordinates of the histogram [s].
    hist_y : array-like
        Histogram amplitude [arbitrary units].

    Attributes
    ----------
    smoothness : float
        Controls how the internal state is smoothed to produce a stable
        distribution that is less sensitive to noise.
        From 0 to 1.
        - 0: Inactive.
        - 1: Gaussian smoothing with sigma = full witdth of `hist_x`.
    maxiter : int
        Maximum number of iterations allowed when matching the density
        distribution to the target beam profile.
    atol : float
        Absolute tolerance for convergence. The solver compares the last
        two matched profiles, and stops early if changes are smaller
        than this threshold.
    recenter : bool
        If True, recenters the mean of the profile to the minimum of the
        potential. Only applicable for single-bucket simulations.
    animate_fitting : bool
        If True, displays an animation showing how the beam profile is
        matched to the histogram (`hist_y`). Useful for debugging and
        tuning `maxiter`, `smoothness`, and `atol`.
    plot_result : bool
        If True, generates a plot after the matching process is complete.
    plot_result_blocking : bool
        If True, the plot will be displayed immediately by calling
        `plt.show()`.

    Examples
    --------
    >>> from blond.experimental.beam_preparation.semi_empiric_matcher import (
    ...     SemiEmpiricMatcher,
    ... )
    >>> matcher_addon = ProfileMatcherAddon(hist_x=..., hist_y=...)
    >>> # Set attributes to change the behaviour
    >>> matcher_addon.smoothness = 0.05

    >>> simulation.prepare_beam(
    ...     beam=...,
    ...     preparation_routine=SemiEmpiricMatcher(
    ...         time_limit=(0, 2.5e-9),
    ...         n_macroparticles=1e6,
    ...         seed=0,
    ...         maxiter_intensity_effects=0,
    ...         hamilton_to_density_function=matcher_addon.hamilton_to_density_function,
    ...         hamilton_to_density_kwargs={},
    ...         animate=True,
    ...     ),
    ... )
    """

    def __init__(
        self,
        hist_x: NumpyArray | CupyArray,
        hist_y: NumpyArray | CupyArray,
    ):
        self._hist_x = hist_x
        self._hist_y = hist_y
        self.recenter = False
        self.animate_fitting = False
        self.plot_result = False
        self.plot_result_blocking = False
        self.maxiter = 100
        self.smoothness = 0.05  # from 0 to 1
        self.atol = 1e-3
        self._animation_fignumber = None
        self._animation_pause = 1e-3
        self._result_fignumber = None

    def hamilton_to_density_function(
        self,
        time_grid: NumpyArray | CupyArray,
        deltaE_grid: NumpyArray | CupyArray,
        hamilton_2D: NumpyArray | CupyArray,
    ) -> NumpyArray | CupyArray:
        """
        This function is an endpoint for the `SemiEmpiricMatcher`.

        Parameters
        ----`------
        deltaE_grid
            The time coordinates corresponding to `hamilton_2D`, in [eV].
        time_grid
            The time coordinates corresponding to `hamilton_2D`, in [s].
        hamilton_2D
            A 2D array representing the spatial Hamilton field.

        Returns
        -------
        density : NumpyArray or CupyArray
            A 2D array of the same shape as `hamilton_2D`, representing the
            computed density distribution. Values are scaled between 0 and 1.
        """
        if self.recenter:
            mid = time_grid.shape[1] // 2

            import numpy as np
            from scipy.signal import find_peaks

            x = time_grid[:, mid]
            y = hamilton_2D[:, mid]

            # find local minima by finding peaks in -y
            min_indices, _ = find_peaks(-y)

            # index of the lowest local minimum
            lowest_min_index = min_indices[np.argmin(y[min_indices])]

            # x-coordinate of the lowest local minimum
            x_lowest_min = x[lowest_min_index]

            center_ham = x_lowest_min
            center_prof = np.average(self._hist_x, weights=self._hist_y)
            correction = center_ham - center_prof

        else:
            correction = 0.0
        hist_x_interp = time_grid[:, 0]
        hist_y_interp = np.interp(
            hist_x_interp,
            self._hist_x + correction,  # todo if recenter
            self._hist_y,
            left=0,
            right=0,
        )

        density = self._solve_for_density(
            hamilton_2D=hamilton_2D,
            histogram_desired=hist_y_interp,
        )

        if self.plot_result:
            self._plot_result(
                time_grid=time_grid,
                deltaE_grid=deltaE_grid,
                hamilton_2D=hamilton_2D,
                density=density,
                hist_x_interp=hist_x_interp,
                hist_y_interp=hist_y_interp,
            )
            if self.plot_result_blocking:
                plt.show()
            else:
                plt.draw()
                plt.pause(0.1)

        return density

    def _solve_for_density(
        self,
        hamilton_2D: NumpyArray,
        histogram_desired: NumpyArray,
    ) -> NumpyArray:
        density = np.zeros(hamilton_2D.shape, float)

        potential_well_helper = PotentialWellHelper(
            np.arange(hamilton_2D.shape[0]),
            hamilton_2D[:, hamilton_2D.shape[1] // 2],
        )
        mask = potential_well_helper.get_in_bucket_mask()
        diff_mask = np.diff(mask.astype(int))
        starts = np.where(diff_mask == 1)[0]
        stops = (
            np.where(diff_mask == -1)[0] + 1
        )  # adjust stops for inclusive behavior

        if mask[0]:
            # handle start if mask starts with 1
            starts = np.concatenate(([0], starts))
        if mask[-1]:
            # append len(mask) to handle end if mask ends with 1
            stops = np.append(stops, len(mask))

        for bucket_i in range(min(len(starts), len(stops))):
            sel = slice(  # slicing required for inplace operation
                int(starts[bucket_i]),
                int(stops[bucket_i]),
            )

            self._solve_for_density_single_bucket(
                hamilton_2D=hamilton_2D[sel, :],
                histogram_desired=histogram_desired[sel],
                density_write=density[sel, :],
            )
        return density

    def _solve_for_density_single_bucket(
        self,
        hamilton_2D: NumpyArray,
        histogram_desired: NumpyArray,
        density_write: NumpyArray,
    ) -> None:
        """Try to derive a density distribution according to the Hamiltonian and histogram.

        Parameters
        ----------
        hamilton_2D
            A 2D array representing the spatial Hamilton field.
        histogram_desired
            Histogram that represents the target value of what the density
            distribution should represent.

        Returns
        -------
        density : NumpyArray
            A 2D array of the same shape as `hamilton_2D`, representing the
            computed density distribution. Values are scaled between 0 and 1.

        """

        histogram_desired = histogram_desired.copy()
        histogram_desired_normalized = (
            histogram_desired / histogram_desired.mean()
        )

        occupation_per_equipotential = (
            histogram_desired.copy()
        )  # initial guess
        histogram = occupation_per_equipotential_to_histogram(
            occupation_per_equipotential=occupation_per_equipotential,
            potential_2D=hamilton_2D,
        )
        scale = histogram_desired.mean() / histogram.mean()
        occupation_per_equipotential *= scale

        histogram = occupation_per_equipotential_to_histogram(
            occupation_per_equipotential=occupation_per_equipotential,
            potential_2D=hamilton_2D,
        )
        histogram_normalized = histogram / histogram.mean()

        previous_histogram_normalized = histogram_normalized
        assert self.maxiter > 0, (
            f"`maxiter` must be bigger 0, but is {self.maxiter=}."
        )
        for i in tqdm(range(self.maxiter), "ProfileMatcherAddon"):
            residual = histogram_desired - histogram
            update_occupation_per_equipotential_to_density = scale * residual

            if self.smoothness > 0:
                # smooth 2nd derivative
                force_smoothness = np.gradient(
                    np.gradient(occupation_per_equipotential, edge_order=1),
                    edge_order=1,
                )
                update_occupation_per_equipotential_to_density += (
                    force_smoothness
                )

            occupation_per_equipotential += (
                update_occupation_per_equipotential_to_density
            )
            occupation_per_equipotential[occupation_per_equipotential < 0] = (
                0  # negative entries are unphysical
            )

            if self.smoothness > 0:
                occupation_per_equipotential_to_density_smooth = (
                    gaussian_filter1d(
                        occupation_per_equipotential,
                        sigma=int(
                            self.smoothness * len(occupation_per_equipotential)
                        ),
                    )
                )
            else:
                occupation_per_equipotential_to_density_smooth = (
                    occupation_per_equipotential
                )
            histogram = occupation_per_equipotential_to_histogram(
                occupation_per_equipotential=occupation_per_equipotential_to_density_smooth,
                potential_2D=hamilton_2D,
            )
            histogram_normalized = histogram / histogram.mean()
            max_change = np.abs(
                histogram_normalized - previous_histogram_normalized
            ).max()
            if max_change < self.atol:
                break

            if self.animate_fitting:
                self._draw_animation(
                    histogram_desired_normalized=histogram_desired_normalized,
                    histogram_normalized=histogram_normalized,
                    i=i,
                    max_change=max_change,
                    previous_histogram_normalized=previous_histogram_normalized,
                    occupation_per_equipotential_to_density=occupation_per_equipotential,
                    occupation_per_equipotential_to_density_smooth=occupation_per_equipotential_to_density_smooth,
                )

            previous_histogram_normalized = histogram_normalized

        occupation_per_equipotential_to_density(
            occupation_per_equipotential=occupation_per_equipotential_to_density_smooth,
            potential_2D=hamilton_2D,
            density_write=density_write,
        )
        # normalize
        density_write /= np.sum(density_write)

    def _draw_animation(
        self,
        histogram_desired_normalized: NumpyArray,
        histogram_normalized: NumpyArray,
        i: int,
        max_change: float,
        previous_histogram_normalized: NumpyArray,
        occupation_per_equipotential_to_density: NumpyArray,
        occupation_per_equipotential_to_density_smooth: NumpyArray,
    ):
        if self._animation_fignumber is None:
            fig = plt.figure()
            self._animation_fignumber = fig.number
        else:
            plt.figure(self._animation_fignumber)

        plt.clf()
        ax = plt.subplot(2, 1, 1)
        plt.title(
            f"Iteration: {i}/{self.maxiter} |"
            f" atol : {max_change:.1e}/{self.atol:.1e}"
        )
        plt.plot(histogram_desired_normalized)
        plt.plot(previous_histogram_normalized, "-", color="grey")
        plt.plot(histogram_normalized, "--")
        plt.xlabel("Time [s]")
        plt.ylabel("Density [arb. unit]")
        plt.subplot(2, 1, 2, sharex=ax)
        plt.plot(occupation_per_equipotential_to_density, label="raw")
        plt.plot(
            occupation_per_equipotential_to_density_smooth, label="smoothed"
        )
        plt.legend(loc="upper right")
        plt.xlabel("State ID")
        plt.ylabel("Amplitude")
        plt.draw()
        plt.pause(self._animation_pause)

    def _plot_result(
        self,
        time_grid,
        deltaE_grid,
        hamilton_2D,
        density,
        hist_x_interp,
        hist_y_interp,
    ):
        if self._result_fignumber is None:
            fig = plt.figure()
            self._result_fignumber = fig.number
        else:
            plt.figure(self._result_fignumber)
        ax1 = plt.subplot(3, 1, 1)
        plt.title("Hamilton")
        plt.pcolor(
            time_grid, deltaE_grid, hamilton_2D, cmap="viridis", shading="auto"
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Energy [eV]")

        ax2 = plt.subplot(3, 1, 2, sharex=ax1, sharey=ax1)
        plt.title("Density Distribution")

        plt.pcolor(
            time_grid,
            deltaE_grid,
            density,
            cmap="viridis",
            shading="auto",
            vmin=1e-24,
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Energy [eV]")

        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        plt.title("Line Density (normalized)")
        plt.plot(
            hist_x_interp, hist_y_interp / hist_y_interp.sum(), label="Target"
        )
        plt.plot(
            hist_x_interp,
            density.sum(axis=1) / density.sum(),
            linestyle="--",
            label="Fitted",
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Density [arb. unit]")

        plt.legend()
