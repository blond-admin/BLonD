# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
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
from scipy.signal import find_peaks
from tqdm import tqdm

from blond.acc_math.empiric.potential_well import PotentialWellHelper
from blond.experimental.beam_preparation.semi_empiric_matcher_extensions.line_density.callables import (
    occupation_per_equipotential_to_density,
    occupation_per_equipotential_to_histogram,
)
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.generals.plotting_ import DEFAULT_CMAP

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


class SemiEmpiricMatcherAddon(ABC):
    """Abstract class to define addons for the `SemiEmpiricMatcher`."""

    @abstractmethod  # pragma: no cover
    def hamilton_to_density_function(
        self,
        time_grid: NumpyArray,
        deltaE_grid: NumpyArray,
        hamilton_2D: NumpyArray,
    ) -> NumpyArray:
        """
        Density on the grid, to pass as ``hamilton_to_density_function``.

        Parameters
        ----------
        time_grid
            Time coordinates of the grid, in [s].
        deltaE_grid
            Energy coordinates of the grid, in [eV].
        hamilton_2D
            Hamiltonian on the grid, in [eV].

        Returns
        -------
        density
            Density on the grid, of the same shape as ``hamilton_2D``,
            scaled between 0 and 1.
        """
        pass


class ProfileMatcherAddon(SemiEmpiricMatcherAddon):
    """
    Match the beam to a measured line density (bunch profile).

    Its :meth:`hamilton_to_density_function` is passed to
    :class:`~blond.experimental.beam_preparation.semi_empiric_matcher.SemiEmpiricMatcher`.
    It solves for the occupation of each contour of the Hamiltonian, such
    that the line density of the bunch reproduces the profile, in every
    iteration of the intensity effects.

    With windows (one ``time_limit`` per bucket), the profile can be given

    - once for all buckets, here or as one dict in
      ``hamilton_to_density_kwargs``, with ``recenter = True``, so that it is
      placed at the centre of each bucket, or
    - per bucket, as one ``dict(hist_x=..., hist_y=...)`` per window in
      ``hamilton_to_density_kwargs``, either at the measured times or, with
      ``recenter = True``, at the centre of its bucket.

    Parameters
    ----------
    hist_x : array-like, optional
        Time coordinates of the profile, in [s]. The profile for all
        buckets, unless given per bucket in ``hamilton_to_density_kwargs``.
    hist_y : array-like, optional
        Profile amplitude, in arbitrary units, at ``hist_x``.

    Attributes
    ----------
    smoothness : float
        Width of the Gaussian smoothing of the occupation per contour, as a
        fraction of the number of time points of the grid (with windows:
        of the bucket), from 0 (no smoothing) to 1. Smoothing makes the
        result less sensitive to noise in the profile, but broadens bunches
        that are not much longer than ``smoothness`` times the bucket.
        Default is 0.05.
    maxiter : int
        Maximum number of iterations allowed when matching the density
        distribution to the target beam profile.
    atol : float
        Absolute tolerance for convergence. The solver compares the last
        two matched profiles, and stops early if changes are smaller
        than this threshold.
    recenter : bool
        If True, the profile is shifted so that its mean lies at the deepest
        minimum of the potential well, in every call: with windows, at the
        bottom of each bucket. If False (default), the profile stays at its
        measured times. With a single ``time_limit`` spanning several
        buckets, only the deepest one is recentred on.
    animate_fitting : bool
        If True, displays an animation showing how the beam profile is
        matched to the histogram (`hist_y`). Useful for debugging and
        tuning `maxiter`, `smoothness`, and `atol`.
    animate_fitting_pause_time : float
        Time to pause after each frame of the fitting animation, in [s].
    plot_result : bool
        If True, generates a plot after the matching process is complete.
    plot_result_blocking : bool
        If True, the plot will be displayed immediately by calling
        `plt.show()`.
    plot_result_pause_time : float
        Time to pause after drawing the non-blocking result plot, in [s].

    Raises
    ------
    ValueError
        If only one of ``hist_x`` and ``hist_y`` is given, or the profile
        has no positive value or a NaN.

    Examples
    --------
    One profile for a single bucket:

    >>> from blond.experimental import (
    ...     SemiEmpiricMatcher,
    ...     ProfileMatcherAddon,
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

    One profile per bucket, each at the centre of its bucket:

    >>> matcher_addon = ProfileMatcherAddon()
    >>> matcher_addon.recenter = True
    >>> SemiEmpiricMatcher(
    ...     # one window of +- t_rf around each bucket to fill
    ...     time_limit=[((k - 1) * t_rf, (k + 1) * t_rf) for k in buckets],
    ...     n_macroparticles=1e6,
    ...     hamilton_to_density_function=matcher_addon.hamilton_to_density_function,
    ...     hamilton_to_density_kwargs=[
    ...         dict(hist_x=hist_x, hist_y=hist_y)
    ...         for hist_x, hist_y in profiles  # one per bucket
    ...     ],
    ... )
    """

    def __init__(
        self,
        hist_x: NumpyArray | CupyArray | None = None,
        hist_y: NumpyArray | CupyArray | None = None,
    ):
        # profile for all buckets, or None if given per call
        self._profile: tuple[NumpyArray, NumpyArray] | None = (
            None
            if hist_x is None and hist_y is None
            else _checked_profile(hist_x, hist_y)
        )
        self.recenter = False
        self.animate_fitting = False
        self.plot_result = False
        self.plot_result_blocking = False
        self.maxiter = 100
        self.smoothness = 0.05  # from 0 to 1
        self.atol = 1e-3
        self._animation_fignumber = None
        self.animate_fitting_pause_time = 1e-3
        self.plot_result_pause_time = 0.1
        self._result_fignumber = None

    def hamilton_to_density_function(
        self,
        time_grid: NumpyArray,
        deltaE_grid: NumpyArray,
        hamilton_2D: NumpyArray,
        hist_x: NumpyArray | CupyArray | None = None,
        hist_y: NumpyArray | CupyArray | None = None,
    ) -> NumpyArray:
        """
        Density in H whose line density matches the profile.

        Pass it as ``hamilton_to_density_function`` to
        :class:`~blond.experimental.beam_preparation.semi_empiric_matcher.SemiEmpiricMatcher`,
        with ``hamilton_to_density_kwargs={}`` for the profile given to the
        addon, or with the keys ``hist_x`` and ``hist_y``: one dict for all
        buckets, or one per window.

        Parameters
        ----------
        time_grid
            Time coordinates of the grid, in [s].
        deltaE_grid
            Energy coordinates of the grid, in [eV].
        hamilton_2D
            Hamiltonian on the grid, in [eV].
        hist_x
            Time coordinates of the profile, in [s]. Default is the profile
            given to the addon.
        hist_y
            Profile amplitude, in arbitrary units, at ``hist_x``.

        Returns
        -------
        density : NumpyArray
            Density on the grid, of the same shape as ``hamilton_2D``,
            scaled between 0 and 1.

        Raises
        ------
        ValueError
            If there is no profile, here or given to the addon, or if the
            profile does not overlap the time range of the grid, e.g. a
            bucket of another window without ``recenter``.

        See Also
        --------
        blond.experimental.beam_preparation.bucket_filler_functions.hamilton_to_density_by_bunch_length :
            Fill to a target bunch length.
        blond.experimental.beam_preparation.bucket_filler_functions.hamilton_to_density_by_emittance :
            Fill to a target emittance.
        blond.experimental.beam_preparation.bucket_filler_functions.hamilton_to_density_by_metric :
            Fill to a target of any metric.
        blond.experimental.beam_preparation.bucket_filler_functions.hamilton_to_density_by_max :
            Fill up to a given Hamiltonian.
        """
        hist_x, hist_y = self._profile_of_call(hist_x, hist_y)
        if self.recenter:
            correction = self._calculate_recentering_correction(
                hamilton_2D, time_grid, hist_x, hist_y
            )
        else:
            correction = 0.0

        hist_x_interp, hist_y_interp = self._interpolate_hist(
            correction, time_grid, hist_x, hist_y
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
                plt.pause(self.plot_result_pause_time)

        return density

    def _profile_of_call(
        self,
        hist_x: NumpyArray | CupyArray | None,
        hist_y: NumpyArray | CupyArray | None,
    ) -> tuple[NumpyArray, NumpyArray]:
        """The profile given to the call, else the one of the addon.

        Parameters
        ----------
        hist_x
            Time coordinates of the profile, in [s], or None.
        hist_y
            Profile amplitude, or None.

        Returns
        -------
        hist_x, hist_y
            The checked profile, on the host.

        Raises
        ------
        ValueError
            If there is no profile.
        """
        if hist_x is not None or hist_y is not None:
            return _checked_profile(hist_x, hist_y)
        if self._profile is None:
            raise ValueError(
                "ProfileMatcherAddon has no profile: give `hist_x` and"
                " `hist_y` to the addon, or in `hamilton_to_density_kwargs`"
                " (one dict for all buckets, or one per window)."
            )
        return self._profile

    def _interpolate_hist(
        self,
        correction: float,
        time_grid: NumpyArray,
        hist_x: NumpyArray,
        hist_y: NumpyArray,
    ) -> tuple[NumpyArray, NumpyArray]:
        """The profile, shifted by ``correction``, on the time axis of the grid.

        Parameters
        ----------
        correction
            Shift of the profile in time, in [s].
        time_grid
            Time coordinates of the grid, in [s].
        hist_x
            Time coordinates of the profile, in [s].
        hist_y
            Profile amplitude.

        Returns
        -------
        hist_x_interp, hist_y_interp
            Time axis of the grid, and the profile on it.

        Raises
        ------
        ValueError
            If the profile does not overlap the time axis of the grid.
        """
        hist_x_interp = time_grid[:, 0]
        hist_y_interp = np.interp(
            hist_x_interp,
            hist_x + correction,
            hist_y,
            left=0,
            right=0,
        )
        if not np.any(hist_y_interp > 0):
            raise ValueError(
                "The profile, at"
                f" [{hist_x[0] + correction:.6e}, {hist_x[-1] + correction:.6e}]"
                " s, does not overlap the time range"
                f" [{hist_x_interp[0]:.6e}, {hist_x_interp[-1]:.6e}] s of the"
                " bucket. Set `recenter = True` to place it at the centre of"
                " each bucket, or give one profile per window, at the time"
                " of its bucket."
            )
        return hist_x_interp, hist_y_interp

    def _calculate_recentering_correction(
        self,
        hamilton_2D: NumpyArray,
        time_grid: NumpyArray,
        hist_x: NumpyArray,
        hist_y: NumpyArray,
    ) -> float:
        """
        Shift that moves the mean of the profile to the potential minimum.

        Parameters
        ----------
        hamilton_2D
            Hamiltonian on the grid, in [eV].
        time_grid
            Time coordinates of the grid, in [s].
        hist_x
            Time coordinates of the profile, in [s].
        hist_y
            Profile amplitude.

        Returns
        -------
        correction
            Time of the deepest minimum of the potential well minus the mean
            of the profile, in [s].
        """
        mid = time_grid.shape[1] // 2

        x = time_grid[:, mid]
        y = hamilton_2D[:, mid]
        # find local minima by finding peaks in -y
        min_indices, _ = find_peaks(-y)
        # index of the lowest local minimum
        lowest_min_index = min_indices[np.argmin(y[min_indices])]
        # x-coordinate of the lowest local minimum
        center_ham = x[lowest_min_index]
        center_prof = np.average(hist_x, weights=hist_y)
        correction = float(center_ham - center_prof)
        return correction

    def _solve_for_density(
        self,
        hamilton_2D: NumpyArray,
        histogram_desired: NumpyArray,
    ) -> NumpyArray:
        """
        Derive the density function given a Hamiltonian and a beam profile.

        Parameters
        ----------
        hamilton_2D
            A 2D array representing the spatial Hamilton field.
        histogram_desired
            Histogram amplitude [arbitrary units].

        Returns
        -------
        density
            The density distribution according to the Hamiltonian,
            that should approximate the desired histogram.
        """
        density = np.zeros(hamilton_2D.shape, float)

        potential_well_helper = PotentialWellHelper(
            np.arange(hamilton_2D.shape[0]),
            hamilton_2D[:, hamilton_2D.shape[1] // 2],
        )

        # solve for each bucket independently
        n_bins_smoothing = int(self.smoothness * len(histogram_desired))

        for sel in potential_well_helper.get_principal_bucket_slices():
            selected_desired_histogram = histogram_desired[sel]
            is_empty_bucket = np.all(selected_desired_histogram == 0)
            if is_empty_bucket:
                continue
            self._solve_for_density_single_bucket(
                hamilton_2D=hamilton_2D[sel, :],
                histogram_desired=selected_desired_histogram,
                density_write=density[sel, :],
                n_bins_smoothing=n_bins_smoothing,
            )
        return density

    def _solve_for_density_single_bucket(
        self,
        hamilton_2D: NumpyArray,
        histogram_desired: NumpyArray,
        density_write: NumpyArray,
        n_bins_smoothing: int,
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

            occupation_per_equipotential += (
                update_occupation_per_equipotential_to_density
            )
            occupation_per_equipotential[occupation_per_equipotential < 0] = (
                0  # negative occupations are unphysical
            )

            if self.smoothness > 0 and n_bins_smoothing > 0:
                occupation_per_equipotential_to_density_smooth = (
                    gaussian_filter1d(
                        occupation_per_equipotential,
                        sigma=n_bins_smoothing,
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
    ) -> None:
        """
        Draw the current state of fitting.

        Parameters
        ----------
        histogram_desired_normalized
            Desired histogram amplitude [arbitrary units].
        histogram_normalized
            Current histogram amplitude [arbitrary units].
        i
            Iteration step.
        max_change
            Convergence parameter.
        previous_histogram_normalized
            Histogram amplitude from the previous
             iteration step [arbitrary units].
        occupation_per_equipotential_to_density
            The internal state that is optimized.
        occupation_per_equipotential_to_density_smooth
            The internal state that is optimized (smoothed).
        """
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
        plt.ylabel("Occupation per equipotential line")
        plt.tight_layout()
        plt.draw()
        plt.pause(self.animate_fitting_pause_time)

    def _plot_result(
        self,
        time_grid,
        deltaE_grid,
        hamilton_2D,
        density,
        hist_x_interp,
        hist_y_interp,
    ) -> None:
        """
        Plot the final `density` together with all other constraining varibles.

        Parameters
        ----------
        time_grid
            The time coordinates corresponding to `hamilton_2D`, in [s].
        deltaE_grid
            The time coordinates corresponding to `hamilton_2D`, in [eV].
        hamilton_2D
            A 2D array representing the spatial Hamilton field.
        density
            A 2D array of the same shape as `hamilton_2D`, representing the
            computed density distribution. Values are scaled between 0 and 1.
        hist_x_interp
            Time coordinates of the histogram [s].
        hist_y_interp
            Histogram amplitude [arbitrary units].
        """
        if self._result_fignumber is None:
            fig = plt.figure()
            self._result_fignumber = fig.number
        else:
            plt.figure(self._result_fignumber)
        ax1 = plt.subplot(3, 1, 1)
        plt.title("Hamilton")
        plt.pcolor(
            time_grid,
            deltaE_grid,
            hamilton_2D,
            cmap=DEFAULT_CMAP,
            shading="auto",
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Energy [eV]")

        ax2 = plt.subplot(3, 1, 2, sharex=ax1, sharey=ax1)
        plt.title("Density Distribution")

        plt.pcolor(
            time_grid,
            deltaE_grid,
            density,
            cmap=DEFAULT_CMAP,
            shading="auto",
            vmin=1e-24,
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Density [arb. unit]")

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
        plt.tight_layout()


def _checked_profile(
    hist_x: NumpyArray | CupyArray | None,
    hist_y: NumpyArray | CupyArray | None,
) -> tuple[NumpyArray, NumpyArray]:
    """Validate a profile and copy it to the host.

    Parameters
    ----------
    hist_x
        Time coordinates of the profile, in [s].
    hist_y
        Profile amplitude, at ``hist_x``.

    Returns
    -------
    hist_x, hist_y
        The profile, as float NumPy arrays.

    Raises
    ------
    ValueError
        If only one of them is given, their shapes differ, or the profile
        has no positive value or a NaN.
    """
    if hist_x is None or hist_y is None:
        raise ValueError("Give both `hist_x` and `hist_y`, or neither.")
    hist_x = np.asarray(copy_to_cpu(hist_x), dtype=float)
    hist_y = np.asarray(copy_to_cpu(hist_y), dtype=float)
    if (
        hist_x.shape != hist_y.shape
        or np.any(np.isnan(hist_x))
        or np.any(np.isnan(hist_y))
        or not np.any(hist_y > 0)
    ):
        raise ValueError(
            "The profile needs `hist_x` and `hist_y` of the same shape,"
            " without NaN, and with a positive value."
        )
    return hist_x, hist_y
