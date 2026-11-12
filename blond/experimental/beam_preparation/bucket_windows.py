# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""One window per bucket to fill, for the `SemiEmpiricMatcher`.

The bucket geometry (:func:`select_bucket`, :func:`check_bucket_continuity`)
and :class:`BucketWindows`, the collaborator through which
:class:`~blond.experimental.beam_preparation.semi_empiric_matcher.SemiEmpiricMatcher`
fills one bucket per window when ``time_limit`` is a list of windows.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from blond.acc_math.empiric.potential_well import PotentialWellHelper
from blond.beam_preparation.helpers import generate_particle_coordinates
from blond.core.backends.backend import backend
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.generals.exceptions_ import SemiEmpiricMatcherError

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Sequence
    from typing import Any

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.experimental.beam_preparation.semi_empiric_matcher import (
        DebuggingEndpoints,
    )

# Density removed at the separatrix above which `BucketWindows` warns
_WARN_ABOVE_FRACTION_OUTSIDE = 0.01


def select_bucket(
    ts: NumpyArray,
    potential_well: NumpyArray,
    window_centre: float,
    bucket_length: float,
) -> slice:
    """Index slice of the bucket to fill in a window around a bucket.

    The window should span :math:`\\pm` ``bucket_length`` around the nominal
    bucket centre. Among the buckets of
    :class:`~blond.acc_math.empiric.potential_well.PotentialWellHelper`
    that do not touch the window edges (partial neighbours) and are not
    nested inside another one (inner wells of e.g. a double-harmonic
    potential), the one whose centre is nearest to ``window_centre`` is
    selected. It spans from the lower of its two barriers to where the
    potential reaches that height again, i.e. the separatrix at
    :math:`\\Delta E = 0`.

    Parameters
    ----------
    ts
        Time coordinates of ``potential_well``, uniformly spaced, in [s].
    potential_well
        Potential well on the host, in [eV].
    window_centre
        Nominal centre of the bucket, in [s].
    bucket_length
        Nominal length of a bucket, half the window, in [s].

    Returns
    -------
    selection
        Slice of ``ts`` from the first to the last point of the bucket.

    Raises
    ------
    SemiEmpiricMatcherError
        If there is no complete bucket in the window, or if the nearest one
        is more than half a bucket away from ``window_centre``.
    """
    step = ts[1] - ts[0]
    buckets = np.asarray(
        PotentialWellHelper(ts, potential_well).bucket_list, dtype=float
    ).reshape(-1, 2)
    complete = [
        bucket
        for bucket in buckets
        if bucket[0] > ts[0] + 0.5 * step and bucket[1] < ts[-1] - 0.5 * step
    ]
    if not complete:
        raise SemiEmpiricMatcherError(
            "No complete bucket in the window around"
            f" {window_centre=:.6e} s. The window is too narrow, or there"
            " is no stationary bucket, e.g. because the beam loading exceeds"
            " the restoring RF voltage."
        )
    outermost = [
        bucket
        for bucket in complete
        if not any(
            other[0] <= bucket[0]
            and bucket[1] <= other[1]
            and not np.array_equal(other, bucket)
            for other in complete
        )
    ]
    centres = np.array([0.5 * (bucket[0] + bucket[1]) for bucket in outermost])
    nearest = outermost[int(np.argmin(np.abs(centres - window_centre)))]
    offset = 0.5 * (nearest[0] + nearest[1]) - window_centre
    if abs(offset) > 0.5 * bucket_length:
        raise SemiEmpiricMatcherError(
            f"The bucket nearest to {window_centre=:.6e} s is"
            f" {offset / bucket_length:+.2f} bucket lengths away, more than"
            " half a bucket: its identity is lost. Typically, there is no"
            " static equilibrium with intensity effects."
        )
    i_start = int(np.searchsorted(ts, nearest[0] - 0.5 * step))
    i_stop = int(np.searchsorted(ts, nearest[1] + 0.5 * step))
    return slice(i_start, i_stop)


def check_bucket_continuity(
    previous_centres: Sequence[float],
    centres: Sequence[float],
    bucket_lengths: Sequence[float],
) -> None:
    """Check that no selected bucket jumped between two iterations.

    A bucket shifted by more than half a bucket is indistinguishable from
    its neighbour in a single potential well (a shift of :math:`+0.55` looks
    like :math:`-0.45`), so :func:`select_bucket` alone cannot detect it.
    Buckets may drift between iterations, but not jump.

    Parameters
    ----------
    previous_centres
        Centres of the selected buckets in the previous iteration, in [s].
    centres
        Centres of the selected buckets now, in [s].
    bucket_lengths
        Nominal bucket length of each window, in [s].

    Raises
    ------
    SemiEmpiricMatcherError
        If a bucket moved by more than half a bucket length.
    """
    for i, (previous, centre, bucket_length) in enumerate(
        zip(previous_centres, centres, bucket_lengths, strict=True)
    ):
        jump = (centre - previous) / bucket_length
        if abs(jump) > 0.5:  # noqa: PLR2004
            raise SemiEmpiricMatcherError(
                f"The selected bucket {i} jumped by {jump:+.2f} bucket"
                " lengths between two iterations: it drifted out of its"
                " window. Typically, there is no static equilibrium with"
                " intensity effects."
            )


def split_macroparticles(
    bucket_weights: Sequence[float],
    n_macroparticles: int,
) -> NumpyArray:
    """Number of macroparticles per bucket, proportional to its weight.

    Parameters
    ----------
    bucket_weights
        Relative intensity of each bucket, non-negative, not all zero.
    n_macroparticles
        Total number of macroparticles.

    Returns
    -------
    n_per_bucket
        Integer count per bucket, summing to ``n_macroparticles``. The
        rounding remainder goes to the heaviest bucket.
    """
    weights = np.asarray(bucket_weights, dtype=float)
    n_per_bucket = np.floor(weights / weights.sum() * n_macroparticles).astype(
        int
    )
    n_per_bucket[np.argmax(weights)] += n_macroparticles - n_per_bucket.sum()
    return n_per_bucket


def cut_at_separatrix(
    density: NumpyArray,
    hamilton_2D: NumpyArray,
) -> tuple[NumpyArray, float]:
    """Remove the density outside the separatrix of one bucket.

    ``hamilton_2D`` must be the grid of one bucket, cut out between its
    barriers as by :func:`select_bucket`: then :math:`H(t, \\Delta E = 0)`
    is the potential well, and the lower of its two edges is the
    separatrix.

    Parameters
    ----------
    density
        Density on the grid of ``hamilton_2D``; not modified.
    hamilton_2D
        Hamiltonian of the bucket, in [eV].

    Returns
    -------
    density_cut
        Copy of ``density``, zero where :math:`H` is above the separatrix.
    fraction_outside
        Share of the density that was removed.
    """
    potential_well = hamilton_2D.min(axis=1)
    hamilton_separatrix = min(potential_well[0], potential_well[-1])
    outside = hamilton_2D > hamilton_separatrix
    total = density.sum()
    fraction_outside = (
        float(density[outside].sum() / total) if total > 0 else 0.0
    )
    density_cut = np.array(density, dtype=float)
    density_cut[outside] = 0.0
    return density_cut, fraction_outside


def _stitch_potential_wells(
    windows_ts: Sequence[NumpyArray],
    wells: Sequence[NumpyArray],
) -> list[NumpyArray]:
    """Shift the potential wells of overlapping windows onto one another.

    Each window is integrated and minimum-subtracted on its own, so its
    offset is arbitrary. Where two windows overlap, they integrate the same
    kicks and differ by a constant only: shifting each window by its mean
    difference to the window before it (in time) restores one continuous
    well. Windows without overlap start a separate group. Each group is
    finally minimum-subtracted again. For plotting only; the matching does
    not depend on these offsets.

    Parameters
    ----------
    windows_ts
        Time coordinates of each window, increasing, in [s].
    wells
        Potential well of each window, on the host, in [eV].

    Returns
    -------
    stitched
        The shifted wells, in the order of ``wells``.
    """
    order = np.argsort([ts[0] for ts in windows_ts])
    stitched: list[NumpyArray] = [np.asarray(well) for well in wells]
    groups: list[list[int]] = []
    previous = None
    for i in order:
        ts = windows_ts[i]
        overlap = (
            np.zeros(len(ts), dtype=bool)
            if previous is None
            else ts <= windows_ts[previous][-1]
        )
        if np.count_nonzero(overlap) >= 2:  # noqa: PLR2004
            previous_well = np.interp(
                ts[overlap], windows_ts[previous], stitched[previous]
            )
            stitched[i] = stitched[i] + np.mean(
                previous_well - stitched[i][overlap]
            )
            groups[-1].append(i)
        else:
            groups.append([i])
        previous = i
    for group in groups:
        minimum = min(stitched[i].min() for i in group)
        for i in group:
            stitched[i] = stitched[i] - minimum
    return stitched


def _is_window_list(time_limit: Any) -> bool:
    """Whether ``time_limit`` is a list of windows, not one range.

    Parameters
    ----------
    time_limit
        ``(start, stop)``, or a sequence of such pairs.

    Returns
    -------
    is_window_list
        True for a sequence of pairs.
    """
    return len(time_limit) > 0 and np.ndim(time_limit[0]) == 1


class BucketWindows:
    """One window per bucket to fill, see `SemiEmpiricMatcher`.

    Holds the windows and their per-bucket settings, selects the bucket in
    each window and populates a beam bucket by bucket. Only the particle
    coordinates of each bucket are kept, so that the memory does not grow
    with the number of windows (except with ``debug_helper``).

    Parameters
    ----------
    windows
        ``(start, stop)`` per bucket to fill, in [s], each centred on the
        nominal bucket centre and spanning plus/minus one bucket length.
    bucket_weights
        Relative intensity per window, non-negative, not all zero. ``None``
        gives equal intensities.
    density_kwargs
        Keyword arguments of the density function, one dict for all
        windows or one per window.
    n_time
        Number of points of the potential well in each window.

    Attributes
    ----------
    windows
        ``(start, stop)`` per window, in [s].
    bucket_weights
        Relative intensity per window.
    density_kwargs
        Keyword arguments of the density function per window.
    windows_ts
        Time coordinates of the potential well per window, in [s].
    span
        Start of the first and end of the last window, in [s].
    selected_buckets
        ``(start, stop)`` of the bucket selected in each window by the last
        call of :meth:`populate_beam`, in [s], or ``None`` before.

    Raises
    ------
    ValueError
        On windows with other than two bounds or ``stop <= start``, and on
        per-window arguments that do not match the number of windows.
    """

    def __init__(
        self,
        windows: Sequence[Sequence[float]],
        bucket_weights: Sequence[float] | None,
        density_kwargs: dict[str, Any] | Sequence[dict[str, Any]],
        n_time: int,
    ) -> None:
        if any(len(window) != 2 for window in windows):  # noqa: PLR2004
            raise ValueError(
                f"Every window must be (start, stop), but got {windows}."
            )
        self.windows = [(float(start), float(stop)) for start, stop in windows]
        if any(stop <= start for start, stop in self.windows):
            raise ValueError(
                f"Every window must have start < stop: {self.windows}"
            )
        n_windows = len(self.windows)
        self.bucket_weights = (
            np.ones(n_windows)
            if bucket_weights is None
            else np.array(bucket_weights, dtype=float)
        )
        if (
            self.bucket_weights.shape != (n_windows,)
            or np.any(self.bucket_weights < 0)
            or not self.bucket_weights.sum() > 0
        ):
            raise ValueError(
                f"`bucket_weights` needs {n_windows} non-negative values, one"
                f" per window and not all zero, but got {bucket_weights}."
            )
        if isinstance(density_kwargs, dict):
            density_kwargs = [density_kwargs] * n_windows
        if len(density_kwargs) != n_windows:
            raise ValueError(
                "`hamilton_to_density_kwargs` needs one dict for all windows"
                f" or one per window, {n_windows}, but got"
                f" {len(density_kwargs)}."
            )
        self.density_kwargs = list(density_kwargs)
        self.windows_ts = [
            np.linspace(start, stop, n_time) for start, stop in self.windows
        ]
        self.span = (
            min(start for start, _ in self.windows),
            max(stop for _, stop in self.windows),
        )
        self.selected_buckets: list[tuple[float, float]] | None = None

    @classmethod
    def from_time_limit(
        cls,
        time_limit: Any,
        hamilton_to_density_kwargs: dict[str, Any] | Sequence[dict[str, Any]],
        bucket_weights: Sequence[float] | None,
        n_time: int,
    ) -> BucketWindows | None:
        """Windows from the arguments of `SemiEmpiricMatcher`.

        Parameters
        ----------
        time_limit
            One ``(start, stop)``, or one per bucket to fill, in [s].
        hamilton_to_density_kwargs
            One dict, or one per window.
        bucket_weights
            Relative intensity per window, or ``None``.
        n_time
            Number of points of the potential well in each window.

        Returns
        -------
        bucket_windows
            ``None`` for a single ``(start, stop)``.

        Raises
        ------
        ValueError
            On invalid windows, or on per-window arguments without windows.
        """
        if not _is_window_list(time_limit):
            if bucket_weights is not None or not isinstance(
                hamilton_to_density_kwargs, dict
            ):
                raise ValueError(
                    "`bucket_weights` and one `hamilton_to_density_kwargs`"
                    " per window require one window per bucket in"
                    " `time_limit`."
                )
            return None
        return cls(
            windows=time_limit,
            bucket_weights=bucket_weights,
            density_kwargs=hamilton_to_density_kwargs,
            n_time=n_time,
        )

    def split_potential_wells(
        self, potential_well: NumpyArray | CupyArray
    ) -> list[NumpyArray]:
        """Split the concatenated potential wells of all windows.

        Parameters
        ----------
        potential_well
            Potential wells of all windows, concatenated, in [eV].

        Returns
        -------
        wells
            One potential well per window, on the host, in [eV].
        """
        potential_well = copy_to_cpu(potential_well)
        bounds = np.cumsum([0] + [len(ts) for ts in self.windows_ts])
        return [
            potential_well[start:stop]
            for start, stop in zip(bounds, bounds[1:], strict=False)
        ]

    def select_buckets(self, wells: Sequence[NumpyArray]) -> list[slice]:
        """Select the bucket in each window, see :func:`select_bucket`.

        Sets `selected_buckets`, after checking that no bucket jumped since
        the previous selection and that no two windows selected the same
        bucket.

        Parameters
        ----------
        wells
            Potential well of each window, on the host, in [eV].

        Returns
        -------
        selections
            Slice of the selected bucket in each window.

        Raises
        ------
        SemiEmpiricMatcherError
            See :func:`select_bucket` and :func:`check_bucket_continuity`,
            and if two windows selected overlapping buckets.
        """
        bucket_lengths = [0.5 * (stop - start) for start, stop in self.windows]
        selections = [
            select_bucket(
                ts=ts,
                potential_well=well,
                window_centre=0.5 * (start + stop),
                bucket_length=bucket_length,
            )
            for ts, well, (start, stop), bucket_length in zip(
                self.windows_ts,
                wells,
                self.windows,
                bucket_lengths,
                strict=True,
            )
        ]
        selected_buckets = [
            (float(ts[selection][0]), float(ts[selection][-1]))
            for ts, selection in zip(self.windows_ts, selections, strict=True)
        ]
        if self.selected_buckets is not None:
            check_bucket_continuity(
                previous_centres=[
                    0.5 * (start + stop)
                    for start, stop in self.selected_buckets
                ],
                centres=[
                    0.5 * (start + stop) for start, stop in selected_buckets
                ],
                bucket_lengths=bucket_lengths,
            )
        self._check_no_overlap(selected_buckets)
        self.selected_buckets = selected_buckets
        return selections

    def _check_no_overlap(
        self, selected_buckets: Sequence[tuple[float, float]]
    ) -> None:
        """Raise if two windows selected overlapping buckets.

        Neighbouring buckets share their barrier, up to one grid step.

        Parameters
        ----------
        selected_buckets
            ``(start, stop)`` of the selected bucket per window, in [s].

        Raises
        ------
        SemiEmpiricMatcherError
            If two selected buckets overlap by more than one grid step.
        """
        grid_step = max(ts[1] - ts[0] for ts in self.windows_ts)
        in_time_order = sorted(selected_buckets)
        for (start, stop), (next_start, next_stop) in zip(
            in_time_order, in_time_order[1:], strict=False
        ):
            if next_start < stop - grid_step:
                raise SemiEmpiricMatcherError(
                    "Two windows selected overlapping buckets,"
                    f" [{start:.6e}, {stop:.6e}] and"
                    f" [{next_start:.6e}, {next_stop:.6e}] s."
                )

    def populate_beam(
        self,
        beam: BeamBaseClass,
        potential_well: NumpyArray | CupyArray,
        *,
        eta: float,
        n_energy: int,
        hamilton_to_density_function: Callable[..., NumpyArray],
        n_macroparticles: int,
        seed: int | None,
        backend_dependent_random_numbers: bool,
        debug_helper: DebuggingEndpoints | None = None,
    ) -> None:
        """Populate ``beam``, bucket by bucket, from the potential wells.

        In each window, the selected bucket gets its share of
        ``n_macroparticles`` from ``hamilton_to_density_function`` on the
        Hamiltonian of the bucket, zero at its bottom, with the density cut
        at the separatrix. The grids of one bucket are dropped before the
        next one is built.

        Parameters
        ----------
        beam
            Beam to populate; its coordinates are overwritten.
        potential_well
            Potential wells of all windows, concatenated, in [eV].
        eta
            Slippage factor of the analytic drift term.
        n_energy
            Number of energy points of the Hamiltonian grid.
        hamilton_to_density_function
            See `SemiEmpiricMatcher`.
        n_macroparticles
            Total number of macroparticles.
        seed
            Random seed; window ``i`` uses ``seed + i``.
        backend_dependent_random_numbers
            See `SemiEmpiricMatcher`.
        debug_helper
            If given, keeps the density and Hamiltonian of every bucket.

        Warns
        -----
        UserWarning
            If more than 1 % of a bucket's density lies outside the
            separatrix.
        """
        # prevent circular import
        from blond.experimental.beam_preparation.semi_empiric_matcher import (
            get_hamilton_semi_analytic,
        )

        wells = self.split_potential_wells(potential_well)
        selections = self.select_buckets(wells)
        n_per_bucket = split_macroparticles(
            self.bucket_weights, n_macroparticles
        )
        densities, hamiltons_2D = [], []
        dts, dEs = [], []
        for i_window, (ts, well, selection) in enumerate(
            zip(self.windows_ts, wells, selections, strict=True)
        ):
            deltaE_grid, time_grid, hamilton_2D = get_hamilton_semi_analytic(
                ts=ts[selection],
                potential_well=well[selection] - well[selection].min(),
                reference_total_energy=beam.reference.total_energy,
                beta=beam.reference.beta,
                eta=eta,
                shape=(len(ts[selection]), n_energy),
            )
            density, fraction_outside = cut_at_separatrix(
                density=hamilton_to_density_function(
                    time_grid=time_grid,
                    deltaE_grid=deltaE_grid,
                    hamilton_2D=hamilton_2D,
                    **self.density_kwargs[i_window],
                ),
                hamilton_2D=hamilton_2D,
            )
            if fraction_outside > _WARN_ABOVE_FRACTION_OUTSIDE:
                warnings.warn(
                    f"SemiEmpiricMatcher: {fraction_outside:.1%} of the"
                    f" density in window {i_window} lies outside the"
                    " separatrix and is removed.",
                    stacklevel=2,
                )
            if debug_helper is not None:
                densities.append(density)
                hamiltons_2D.append(hamilton_2D)
            if n_per_bucket[i_window] == 0:
                continue
            dt, dE = generate_particle_coordinates(
                time_grid=time_grid.T,
                deltaE_grid=deltaE_grid.T,
                density_grid=density.T,
                n_macroparticles=int(n_per_bucket[i_window]),
                seed=None if seed is None else seed + i_window,
                backend_dependent_random_numbers=(
                    backend_dependent_random_numbers
                ),
            )
            dts.append(dt)
            dEs.append(dE)

        if debug_helper is not None:
            debug_helper.last_density = densities
            debug_helper.last_hamilton_2D = hamiltons_2D
        # fresh arrays, a copy would double the memory
        beam.setup_beam(
            dt=backend.concatenate(dts),
            dE=backend.concatenate(dEs),
            mpi_mode="all-ranks",
            copy=False,
        )

    def plot_potential_well(
        self, potential_well: NumpyArray | CupyArray, color: str
    ) -> None:
        """Plot the potential wells of all windows, stitched together.

        Parameters
        ----------
        potential_well
            Potential wells of all windows, concatenated, in [eV].
        color
            Matplotlib colour of all windows.
        """
        stitched = _stitch_potential_wells(
            self.windows_ts, self.split_potential_wells(potential_well)
        )
        for ts, well in zip(self.windows_ts, stitched, strict=True):
            plt.plot(ts, well, color=color)
