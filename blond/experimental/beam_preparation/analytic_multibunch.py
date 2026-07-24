# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Sequential multi-bunch matching (BLonD 2 iterative method).

Ports the BLonD 2 ``matched_from_distribution_density_multibunch`` /
``matched_from_line_density_multibunch`` workflow: bunches are matched
**one by one** in ascending bucket order, each seeing the RF potential
plus the induced voltage of the previously generated bunches (propagated
forward through an ``extra_voltage`` on the single-bunch matcher). One
class covers both input types — and mixed trains — because the per-bunch
specification *is* a single-bunch matcher instance
(:class:`~blond.experimental.beam_preparation.analytic_matcher.AnalyticDistributionMatcher`
for matching from a distribution function,
:class:`~blond.experimental.beam_preparation.analytic_matcher.LineDensityMatcher`
for matching from a line density, e.g. a measured profile).

This is the *sequential* method: each bunch is self-consistent with its
own wake and with its predecessors', but earlier bunches never see the
wake of later ones. The fully self-consistent all-bunches-at-once port
of the BLonD 2 ``match_beam_from_distribution`` is a separate, future
matcher.

The filling pattern is expressed in its reduced form — ``bucket_indices``
plus per-bunch sequences aligned to it — deliberately matching the
representation of the (not yet merged) FillingPattern proposal, so a
future ``from_filling_pattern`` adapter needs no change to this core.
"""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING

import numpy as np

from blond.beam_preparation.base import MatchingRoutine
from blond.experimental.beam_preparation.analytic_induced_potential import (
    clone_wakefields_on_smooth_profile,
    induced_voltage_from_line_density,
)
from blond.experimental.beam_preparation.analytic_matcher import (
    _AnalyticMatcherBase,
    _machine_parameters,
    _ring_has_wakefields,
)
from blond.generals.cupy.no_cupy_import import AllowPlotting, copy_to_cpu

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation

# Fraction of one bucket added on each side of the train frame, so the
# local single-bunch frames (legacy 40 % margin) stay inside it.
_TRAIN_FRAME_MARGIN_BUCKETS = 0.2


class SequentialMultiBunchMatcher(MatchingRoutine):
    r"""
    Multi-bunch beam matched bunch by bunch (BLonD 2 iterative method).

    Bunches are generated in ascending bucket order; after each bunch,
    the induced voltage of the accumulated train (smooth line
    densities, no macroparticles) is recomputed on a train-spanning
    grid and handed to the next bunch's matcher as an extra voltage —
    so every bunch is matched in the RF potential distorted by all its
    predecessors (and by its own wake, through the single-bunch
    intensity iteration).

    Parameters
    ----------
    bunch_matchers
        Per-bunch specifications, as single-bunch matcher instances:
        a sequence with one matcher per bucket of ``bucket_indices``
        (mixed :class:`AnalyticDistributionMatcher` /
        :class:`LineDensityMatcher` types allowed), or a single
        template instance cloned for every bunch (with per-bunch seeds
        derived as ``seed + bunch index`` — unlike BLonD 2, which
        reused the same seed and generated identical sampling noise in
        every bunch). Instances are deep-copied: the originals are
        never modified; per-bunch diagnostics live on
        ``self.bunch_matchers`` after the run.
    bucket_indices
        Occupied buckets of the main RF harmonic, ascending (e.g.
        ``[0, 10, 20]``; gaps and batches free). Mutually exclusive
        with ``n_bunches``/``bunch_spacing_buckets``.
    n_bunches
        Constant-spacing convenience (the BLonD 2 interface): number of
        bunches, with ``bunch_spacing_buckets`` between them.
    bunch_spacing_buckets
        Spacing between consecutive bunches, in buckets of the main
        harmonic (with ``n_bunches``).
    bunch_intensities
        Bunch intensities, in particles per bunch: a scalar (same for
        every bunch) or a sequence aligned with the bunches. ``None``
        splits ``beam.intensity`` equally. If the sum differs from
        ``beam.intensity``, the beam intensity is overwritten with a
        warning (BLonD 2 behaviour).
    n_points_per_bucket_induced
        Resolution of the train-spanning grid used for the accumulated
        induced voltage, in points per bucket.
    verbose
        If True, print per-bunch progress (per-matcher diagnostics are
        controlled by each bunch matcher's own ``verbose``).
    plot
        If True, draw the accumulated train line density and its
        induced voltage after the last bunch.

    Attributes
    ----------
    bunch_matchers
        The per-bunch matcher instances actually run (deep copies of
        the specs): per-bunch diagnostics (``matched_bunch_length``,
        ``profile_reconstruction_error``, ...) are read from here.
    bucket_indices
        The resolved occupied-bucket indices (after run).
    bunch_intensities
        The resolved per-bunch intensities, in particles (after run).

    Examples
    --------
    >>> template = AnalyticDistributionMatcher(
    ...     n_macroparticles=1e5,
    ...     distribution_type="parabolic_amplitude",
    ...     bunch_length=1.2e-9,
    ...     seed=0,
    ... )
    >>> simulation.prepare_beam(
    ...     beam=beam,
    ...     preparation_routine=SequentialMultiBunchMatcher(
    ...         bunch_matchers=template,
    ...         n_bunches=12,
    ...         bunch_spacing_buckets=10,
    ...         bunch_intensities=1.15e11,
    ...     ),
    ... )
    """

    def __init__(
        self,
        bunch_matchers: Sequence[_AnalyticMatcherBase] | _AnalyticMatcherBase,
        bucket_indices: Sequence[int] | None = None,
        n_bunches: int | None = None,
        bunch_spacing_buckets: int | None = None,
        bunch_intensities: float | Sequence[float] | None = None,
        n_points_per_bucket_induced: int = 128,
        verbose: bool = False,
        plot: bool = False,
    ) -> None:
        super().__init__()
        if (bucket_indices is None) == (n_bunches is None):
            raise ValueError(
                "Specify exactly one filling input: `bucket_indices`, "
                "or `n_bunches` with `bunch_spacing_buckets`."
            )
        if bucket_indices is not None:
            resolved_indices = np.asarray(bucket_indices, dtype=int)
            if len(resolved_indices) == 0:
                raise ValueError("`bucket_indices` is empty.")
            if not np.all(np.diff(resolved_indices) > 0):
                raise ValueError(
                    "`bucket_indices` must be strictly increasing "
                    "(the sequential method propagates the wake "
                    "forward in time)."
                )
            if resolved_indices[0] < 0:
                raise ValueError("`bucket_indices` must be >= 0.")
        else:
            if bunch_spacing_buckets is None:
                raise ValueError(
                    "`n_bunches` requires `bunch_spacing_buckets`."
                )
            resolved_indices = np.arange(int(n_bunches), dtype=int) * int(
                bunch_spacing_buckets
            )
        self.bucket_indices: NumpyArray = resolved_indices

        n_bunches_resolved = len(resolved_indices)
        if isinstance(bunch_matchers, _AnalyticMatcherBase):
            # Template mode: one clone per bunch, seeds derived so the
            # bunches carry independent sampling noise.
            template_seed = bunch_matchers._constructor_kwargs["seed"]
            self.bunch_matchers = [
                bunch_matchers.clone(
                    seed=None
                    if template_seed is None
                    else template_seed + bunch_i
                )
                for bunch_i in range(n_bunches_resolved)
            ]
        else:
            bunch_matchers = list(bunch_matchers)
            if len(bunch_matchers) != n_bunches_resolved:
                raise ValueError(
                    f"Got {len(bunch_matchers)} bunch matchers for "
                    f"{n_bunches_resolved} occupied buckets."
                )
            for matcher in bunch_matchers:
                if not isinstance(matcher, _AnalyticMatcherBase):
                    raise TypeError(
                        "Each bunch matcher must be an analytic "
                        "single-bunch matcher instance "
                        "(AnalyticDistributionMatcher or "
                        f"LineDensityMatcher), got {type(matcher)}."
                    )
            self.bunch_matchers = [
                copy.deepcopy(matcher) for matcher in bunch_matchers
            ]

        self._bunch_intensities_input = bunch_intensities
        self.bunch_intensities: NumpyArray | None = None
        self._n_points_per_bucket_induced = int(n_points_per_bucket_induced)
        self._verbose = verbose
        self._plot = plot

    def _resolve_intensities(self, beam: BeamBaseClass) -> NumpyArray:
        """Per-bunch intensities; warn/overwrite the beam total (BLonD 2)."""
        n_bunches = len(self.bucket_indices)
        if self._bunch_intensities_input is None:
            intensities = np.full(n_bunches, beam.intensity / n_bunches)
        elif np.ndim(self._bunch_intensities_input) == 0:
            intensities = np.full(
                n_bunches, float(self._bunch_intensities_input)
            )
        else:
            intensities = np.asarray(
                self._bunch_intensities_input, dtype=float
            )
            if len(intensities) != n_bunches:
                raise ValueError(
                    f"Got {len(intensities)} bunch intensities for "
                    f"{n_bunches} occupied buckets."
                )
        if not np.isclose(np.sum(intensities), beam.intensity):
            warnings.warn(
                "The summed bunch intensities "
                f"({np.sum(intensities):.4e}) do not match "
                f"beam.intensity ({beam.intensity:.4e}); the beam "
                "intensity is overwritten with the sum.",
                UserWarning,
                stacklevel=3,
            )
            beam.intensity = float(np.sum(intensities))
        return intensities

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """
        Populate the `Beam` object with the matched bunch train.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation :class:`~blond.core.beam.beams.Beam` object.
        """
        from blond.core.beam.beams import Beam

        super().prepare_beam(simulation=simulation, beam=beam)

        params = _machine_parameters(simulation, beam)
        bucket_size = 2.0 * np.pi / params["omega_rf"]
        intensities = self._resolve_intensities(beam)
        self.bunch_intensities = intensities

        # Train-spanning grid for the accumulated induced voltage: the
        # ring's wakefields are cloned onto it ONCE; the accumulated
        # smooth line density drives the voltage seen by later bunches.
        with_wakefields = _ring_has_wakefields(simulation)
        if with_wakefields:
            n_buckets_span = (
                float(self.bucket_indices[-1])
                + 1.0
                + 2.0 * _TRAIN_FRAME_MARGIN_BUCKETS
            )
            train_time = np.linspace(
                -_TRAIN_FRAME_MARGIN_BUCKETS * bucket_size,
                (
                    float(self.bucket_indices[-1])
                    + 1.0
                    + _TRAIN_FRAME_MARGIN_BUCKETS
                )
                * bucket_size,
                int(round(n_buckets_span * self._n_points_per_bucket_induced)),
            )
            wakefield_clones, smooth_profile = (
                clone_wakefields_on_smooth_profile(simulation, train_time)
            )
        else:
            train_time = None
            wakefield_clones, smooth_profile = [], None

        train_line_density = (
            np.zeros_like(train_time) if with_wakefields else None
        )
        train_induced_voltage = None
        all_dt = []
        all_dE = []

        for bunch_i, bucket_index in enumerate(self.bucket_indices):
            matcher = self.bunch_matchers[bunch_i]
            bucket_offset = float(bucket_index) * bucket_size
            if self._verbose:
                print(
                    "[SequentialMultiBunchMatcher] bunch "
                    f"{bunch_i + 1}/{len(self.bucket_indices)} in bucket "
                    f"{bucket_index} ({type(matcher).__name__})"
                )

            # Predecessors' induced voltage, shifted into the local
            # single-bucket frame (extra_voltage holds edge values
            # outside its range; the train grid covers the local frame
            # for every bunch, so no edge extrapolation occurs).
            if train_induced_voltage is not None:
                predecessor_voltage = (
                    train_time - bucket_offset,
                    train_induced_voltage,
                )
                if matcher._extra_voltage is not None:
                    # Combine with a user-provided extra voltage.
                    user_time, user_values = matcher._extra_voltage
                    predecessor_voltage = (
                        predecessor_voltage[0],
                        predecessor_voltage[1]
                        + np.interp(
                            predecessor_voltage[0], user_time, user_values
                        ),
                    )
                matcher._extra_voltage = predecessor_voltage

            # The internal per-bunch beam carries this bunch's
            # intensity, so the single-bunch intensity machinery works
            # unchanged.
            bunch_beam = Beam(
                intensity=float(intensities[bunch_i]),
                particle_type=beam.particle_type,
            )
            matcher.prepare_beam(simulation=simulation, beam=bunch_beam)

            all_dt.append(
                copy_to_cpu(bunch_beam.read_partial_dt()) + bucket_offset
            )
            all_dE.append(copy_to_cpu(bunch_beam.read_partial_dE()))

            if with_wakefields:
                # Accumulate this bunch's matched smooth line density
                # on the train grid, weighted by its intensity, and
                # recompute the train's induced voltage for the next
                # bunch (BLonD 2 order: computed after each bunch).
                bunch_density_train = np.interp(
                    train_time,
                    matcher.matched_time_array + bucket_offset,
                    matcher.matched_line_density,
                    left=0.0,
                    right=0.0,
                )
                bunch_total = bunch_density_train.sum()
                assert bunch_total > 0.0, (
                    f"Matched line density of bunch {bunch_i} vanished "
                    "on the train grid."
                )
                train_line_density += (
                    bunch_density_train
                    / bunch_total
                    * float(intensities[bunch_i])
                )
                train_beam = Beam(
                    intensity=float(np.sum(intensities[: bunch_i + 1])),
                    particle_type=beam.particle_type,
                )
                train_induced_voltage = induced_voltage_from_line_density(
                    wakefield_clones,
                    smooth_profile,
                    train_line_density,
                    train_beam,
                )

        beam.setup_beam(dt=np.concatenate(all_dt), dE=np.concatenate(all_dE))

        if self._verbose:
            print(
                "[SequentialMultiBunchMatcher] populated "
                f"{len(self.bucket_indices)} bunches, "
                f"{sum(len(dt) for dt in all_dt)} macroparticles, total "
                f"intensity {float(np.sum(intensities)):.4e}"
            )

        if self._plot and with_wakefields:
            self._plot_train(
                train_time, train_line_density, train_induced_voltage
            )

    def _plot_train(
        self,
        train_time: NumpyArray,
        train_line_density: NumpyArray,
        train_induced_voltage: NumpyArray,
    ) -> None:
        """Accumulated train line density and its induced voltage."""
        import matplotlib.pyplot as plt

        with AllowPlotting():
            fig, (ax_density, ax_voltage) = plt.subplots(
                2, 1, num="SequentialMultiBunchMatcher", sharex=True
            )
            ax_density.plot(train_time * 1e9, train_line_density)
            ax_density.set_ylabel("Line density [particles/bin]")
            ax_density.set_title("Accumulated bunch train")
            ax_density.grid(alpha=0.3)
            ax_voltage.plot(
                train_time * 1e9, train_induced_voltage / 1e3, color="C3"
            )
            ax_voltage.set_xlabel("Time [ns]")
            ax_voltage.set_ylabel("Induced voltage [kV]")
            ax_voltage.grid(alpha=0.3)
            fig.tight_layout()
