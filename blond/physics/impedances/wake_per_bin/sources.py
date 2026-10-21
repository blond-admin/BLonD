# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Bin-averaged wakes of the sources in :mod:`blond.physics.impedances.sources`.

Each mix-in is listed first among its source's bases, so that it overrides
the generic stencil of
:class:`~blond.physics.impedances.wake_per_bin.time_domain.TimeDomainWakePerBin`.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

from blond.core.backends.backend import backend
from blond.physics.impedances.wake_per_bin.bin_average import (
    bspline_window_moments,
    triple_box_average_poles,
)

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


# Number of bins the bin-average axis is extended by on either side before
# the 5-point stencil is applied, so that every returned sample sees its real
# neighbours instead of edge-clamped ones, and so that an onset falling near
# the first requested bin still has a cell below it.
_BIN_AVERAGE_PAD = 3

# Fraction of the table's own first spacing within which a query time counts
# as sitting *on* the causal onset. Mirrors the default of
# `Resonators.heaviside_eps_at_0`, which guards the same comparison against
# a 0 that `cupy.linspace` may return as e.g. -1.6e-27.
_ONSET_TOLERANCE_EPS = 0.01


class ResonatorsWakePerBin:
    """
    Exact bin-averaged wake of :class:`~blond.physics.impedances.sources.Resonators`.

    At Q = 0.5 the pole pair degenerates and the bin average divides by
    zero, which the quality-factor check of ``Resonators`` excludes.
    """

    def get_wake_per_bin(
        self, time: NumpyArray | CupyArray
    ) -> NumpyArray | CupyArray:
        """
        Exact bin average of the resonator wake, in [V].

        Parameters
        ----------
        time
            Time array (bin centres, uniformly spaced) at which the wake is
            evaluated, in [s].

        Returns
        -------
        wake
            Bin-averaged wake, in [V].
        """
        return self._wake_per_bin(time, self._shunt_impedances)

    def get_wake_per_bin_counter_rotation(
        self, time: NumpyArray | CupyArray
    ) -> NumpyArray | CupyArray:
        """
        Exact bin average of the counter-rotating resonator wake, in [V].

        Parameters
        ----------
        time
            Time array (bin centres, uniformly spaced) at which the wake is
            evaluated, in [s].

        Returns
        -------
        wake
            Bin-averaged wake, in [V].
        """
        if self._shunt_impedances_counter_rotating is None:
            raise RuntimeError(
                "_shunt_impedances_counter_rotating needs to be set before"
                " calling this function."
            )
        return self._wake_per_bin(
            time, self._shunt_impedances_counter_rotating
        )

    def _wake_per_bin(
        self,
        time: NumpyArray | CupyArray,
        shunt_impedances: NumpyArray | CupyArray,
    ) -> NumpyArray | CupyArray:
        r"""
        Bin-averaged wake of the resonators as a sum of poles.

        Each resonator is one pole :math:`p = -\alpha + i \bar\omega` with
        residue :math:`\rho = R \alpha (1 + i \alpha / \bar\omega)`, so that
        its wake is :math:`2\,\mathrm{Re}[\rho e^{p t}]` for :math:`t > 0`
        (Step 38 of ``docs/models_new/pole-residue/
        bin_averaged_wake_derivation.py``).

        Parameters
        ----------
        time
            Time array (bin centres, uniformly spaced), in [s].
        shunt_impedances
            Shunt impedances to use, in [:math:`\Omega`].

        Returns
        -------
        wake
            Bin-averaged wake, in [V].
        """
        poles = -self._alpha + 1j * self._omega_bar
        residues = (
            shunt_impedances
            * self._alpha
            * (1.0 + 1j * self._alpha / self._omega_bar)
        )
        return triple_box_average_poles(
            time, poles, residues, time[1] - time[0]
        )


class ImpedanceTableTimeWakePerBin:
    """
    Bin-averaged wake of :class:`~blond.physics.impedances.sources.ImpedanceTableTime`.

    The ``first_sample_is_half_jump`` constructor argument of
    ``ImpedanceTableTime`` sets how ``wake_y[0]`` is read where the table
    starts with a causal jump, i.e. where the wake is zero below
    ``wake_x[0]`` and non-zero at it. The table itself records no such
    convention, so it cannot be inferred from the data -- but
    :meth:`get_wake_per_bin` integrates the jump exactly and therefore
    needs its size.

    ``True`` (default) -- ``wake_y[0]`` is *half* the jump, and the
    wake rises to ``2 * wake_y[0]`` immediately above ``wake_x[0]``.
    This is what a table sampled from a BLonD wake at its own onset
    holds:
    :meth:`~blond.physics.impedances.sources.Resonators.heaviside_eps_at_0`
    returns ``sign(t) + 1``, which is 2 above ``t = 0`` but 1 *at* it (the
    beam-loading theorem -- a particle sees half of its own wake), so
    point-sampling
    at the onset stores the midpoint of the jump. For a bin *integral*
    that midpoint is a measure-zero artefact of point sampling, and
    doubling recovers the true right limit.

    ``False`` -- ``wake_y[0]`` is read literally, as the right limit
    ``W(wake_x[0]^+)``. Use this for tables produced by an external
    wake solver (CST, ABCI, a measurement), which record the wake value
    itself; doubling those would overstate the onset by a factor of two.

    The setting is irrelevant whenever ``wake_y[0]`` is zero -- there is
    no jump to interpret then, which is the common case for external
    tables that start at or below ``t = 0``.
    """

    @property
    def _onset_jump(self) -> float:
        """
        Size of the causal jump at ``wake_x[0]``, in [V].

        Reads ``wake_y[0]`` under the convention selected by
        ``first_sample_is_half_jump`` (see the class docstring).

        Returns
        -------
        onset_jump
            Height the wake rises to immediately above ``wake_x[0]``,
            in [V].
        """
        first_sample = float(self._wake_y[0])
        if self._first_sample_is_half_jump:
            return 2.0 * first_sample
        return first_sample

    def _warn_if_outside_table(self, time: NumpyArray | CupyArray) -> None:
        """
        Warn when ``time`` reaches outside the tabulated range.

        ``TimeDomain.get_impedance_from_wake`` samples one bin below the axis
        it was handed, to pick up the bin-average kernel's non-causal tap
        (see :meth:`get_wake_per_bin`). That much undershoot is the kernel
        doing its job, not the table being too short, so only warn beyond it.

        Parameters
        ----------
        time
            Time array the wake was requested on, in [s].
        """
        bin_step = (time[1] - time[0]) if len(time) > 1 else 0.0
        if time.min() < (self._wake_x.min() - bin_step):
            warnings.warn(
                "Interpolation of wake outside boundaries",
                stacklevel=1,
            )
        if time.max() > self._wake_x.max():
            warnings.warn(
                "Interpolation of wake outside boundaries",
                stacklevel=1,
            )

    def _wake_with_resolved_onset(
        self, time: NumpyArray | CupyArray
    ) -> NumpyArray | CupyArray:
        r"""
        Tabulated wake with the causal jump at the table start resolved.

        A causal wake table is zero below its first time and non-zero at it,
        so ``wake_x[0]`` carries a jump. How big that jump is depends on how
        the table was produced, which the table does not record: the
        ``first_sample_is_half_jump`` constructor argument selects it (see
        the class docstring), and ``_onset_jump`` reads it out.

        By default BLonD's own convention applies: BLonD samples a jump at
        its own midpoint -- both :meth:`~blond.physics.impedances.sources.Resonators.get_wake`
        and :meth:`~blond.physics.impedances.sources.TravelingWaveCavity.wake_calc` build their wake from
        ``sign(t) + 1``, which is *half* the jump at ``t = 0`` (the
        beam-loading theorem: a particle sees half of its own wake). A table
        sampled that way therefore stores :math:`W(0^+) / 2` in
        ``wake_y[0]``, and the function it represents rises to
        :math:`2 \, \mathrm{wake\_y}[0]` immediately above ``wake_x[0]``.
        With ``first_sample_is_half_jump=False`` -- an external wake-solver
        table -- the onset is :math:`y_0` itself instead.

        This returns the piecewise-linear table with that first segment
        replaced by the line from :math:`(x_0,\, \mathrm{onset})` to
        :math:`(x_1,\, y_1)`; everywhere else it is plain interpolation.
        For a table whose first sample is zero -- a wake that starts
        continuously -- it changes nothing at all, and neither convention
        makes any difference.

        Parameters
        ----------
        time
            Time array at which the wake is evaluated, in [s].

        Returns
        -------
        wake
            Wake of the resolved model, in [V].
        """
        wake = backend.interp(time, self._wake_x, self._wake_y, left=0.0)
        onset_value = self._onset_jump
        first_time = float(self._wake_x[0])
        if len(self._wake_x) < 2 or onset_value == 0.0:  # NOQA PLR2004
            return wake
        second_time = float(self._wake_x[1])
        if second_time <= first_time:
            return wake
        first_segment = onset_value + (
            float(self._wake_y[1]) - onset_value
        ) * (time - first_time) / (second_time - first_time)
        inside_first_segment = (time >= first_time) & (time <= second_time)
        return backend.where(inside_first_segment, first_segment, wake)

    def get_wake(self, time: NumpyArray | CupyArray) -> NumpyArray | CupyArray:
        r"""
        Point-sampled tabulated wake, interpolated onto ``time``.

        Below the first tabulated time the wake is zero, not the clamped
        boundary value: a wake table is causal, so ``W`` vanishes before the
        table starts. Clamping there would fabricate a spurious term of order
        ``W(wake_x[0])`` whenever the caller samples below the table -- which
        :meth:`~blond.physics.impedances.base.TimeDomain.get_impedance_from_wake`
        always does, since it shifts the axis by one bin to pick up the
        bin-average kernel's non-causal tap. Above the last tabulated time the
        value is still clamped (see the warning below): where the wake
        continues is unknown once the table ends.

        Exactly at ``wake_x[0]`` the tabulated wake jumps, and a particle
        sitting on the discontinuity sees the midpoint of that jump -- the
        beam-loading theorem, the same statement
        :meth:`~blond.physics.impedances.sources.Resonators.heaviside_eps_at_0` makes by returning 1 rather
        than 2 at ``t = 0``. *Which* tabulated number that midpoint is
        depends on ``first_sample_is_half_jump`` (see the class docstring):

        * ``True`` (default) -- ``wake_y[0]`` already *is* the midpoint, so
          it is returned verbatim and this reduces to plain causal
          interpolation.
        * ``False`` -- ``wake_y[0]`` is the right limit
          :math:`W(\mathrm{wake\_x}[0]^+)`, so the onset sample returns
          ``0.5 * wake_y[0]`` instead. Strictly above the onset the
          interpolant is untouched, strictly below it the wake stays zero.

        "Exactly at the onset" is a comparison within
        ``_ONSET_TOLERANCE_EPS`` of the table's own first spacing, not a
        float equality. A table whose first sample is zero carries no jump,
        and both settings then agree. The bin-averaged version,
        :meth:`get_wake_per_bin`, integrates the *function* that sampling
        represents and so resolves the jump over a whole bin; see
        ``_wake_with_resolved_onset``.

        Parameters
        ----------
        time
            Time array at which the wake is evaluated, in [s].

        Returns
        -------
        wake
            Interpolated wake, in [V].
        """
        self._warn_if_outside_table(time)
        # `left=0.0` enforces causality below the table; the right side keeps
        # the interpolator's default clamp, guarded by the warning above.
        wake = backend.interp(time, self._wake_x, self._wake_y, left=0.0)
        if self._first_sample_is_half_jump:
            # `wake_y[0]` is the jump midpoint already, so the point sample
            # of the table is what a particle at the onset sees.
            return wake
        return self._halve_literal_onset_sample(time, wake)

    def _halve_literal_onset_sample(
        self,
        time: NumpyArray | CupyArray,
        wake: NumpyArray | CupyArray,
    ) -> NumpyArray | CupyArray:
        """
        Put the jump midpoint on the samples sitting at the causal onset.

        Only reached for ``first_sample_is_half_jump=False``, where
        ``wake_y[0]`` is the right limit rather than the midpoint of the
        jump (see :meth:`get_wake`). The tolerance comes from
        the table's first spacing rather than from ``time``, so it is
        defined for a single-sample or non-uniform query too. Everything
        stays on the active backend, so a table on the GPU needs no host
        transfer.

        Parameters
        ----------
        time
            Time array the wake was requested on, in [s].
        wake
            Causally interpolated wake on ``time``, in [V].

        Returns
        -------
        wake
            The same wake, with any sample at the onset replaced by half
            the jump, in [V].
        """
        if len(self._wake_x) < 2:  # NOQA PLR2004
            # A single-point table has no spacing to build a tolerance from
            # and no segment to interpolate along either.
            return wake
        onset_time = self._wake_x[0]
        tolerance = _ONSET_TOLERANCE_EPS * backend.abs(
            self._wake_x[1] - onset_time
        )
        at_onset = backend.abs(time - onset_time) <= tolerance
        return backend.where(at_onset, 0.5 * self._wake_y[0], wake)

    def get_wake_per_bin(
        self, time: NumpyArray | CupyArray
    ) -> NumpyArray | CupyArray:
        r"""
        Exact bin-average of the tabulated wake, jump included.

        Overrides
        :meth:`~blond.physics.impedances.base.TimeDomain.get_wake_per_bin`.
        The generic default there B-spline-averages the piecewise-linear
        interpolant through the *query* samples, which turns the causal jump
        at ``wake_x[0]`` into a one-bin ramp. The residual that leaves is of
        order :math:`(76 / 384)\, W(0)` at the samples straddling the onset,
        and -- unlike every other error of the stencil -- it does **not**
        shrink when the table is refined, because the stencil only ever sees
        the wake at the query points.

        What is integrated here instead is the model the table really
        represents: zero below ``wake_x[0]``, the causal jump at
        ``wake_x[0]`` (resolved as in ``_wake_with_resolved_onset``), and
        piecewise linear above it. Writing that model as the query-grid
        interpolant :math:`g` plus a difference supported on the single cell
        :math:`[t_{m-1},\, t_m]` that contains the onset -- with
        :math:`t_m` the first grid point at or above the onset,
        :math:`w = (t_m - \tau) / \Delta t \in [0, 1)` the width of the part
        of that cell above the onset :math:`\tau`, :math:`Y` the jump and
        :math:`W_m` the model at :math:`t_m` -- the difference is

        .. math::
            f - g = Y \, D_w + W_m \, (U_w - U_1) ,

        where :math:`U_w` is the ramp rising from 0 to 1 over the last
        :math:`w` bins before :math:`t_m` and :math:`D_w = \mathrm{box}_w
        - U_w` its descending mirror. Since :math:`D_w` and :math:`U_w`
        integrate against the B-spline in closed form (see
        :func:`~blond.physics.impedances.wake_per_bin.bin_average.bspline_window_moments`),
        the correction to the stencil at the
        grid point :math:`t_m + v \Delta t` is exactly

        .. math::
            Y \, I_0(v, w) + (W_m - Y) \frac{I_1(v, w)}{w}
            - W_m \, I_1(v, 1) ,

        which is non-zero only for :math:`v \in \{-2, ..., 2\}`, the support
        of the kernel. Everywhere else the stencil is untouched, so away from
        the onset the accuracy is unchanged: the only error left is the
        piecewise-linear representation of a smooth wake, which *is* second
        order in the bin width.

        Parameters
        ----------
        time
            Time array (bin centres) at which the wake is evaluated, in [s].

        Returns
        -------
        wake
            Bin-averaged wake, in [V].
        """
        self._warn_if_outside_table(time)
        bin_step = float(time[1] - time[0])
        n_bins = len(time)
        index = backend.arange(
            -_BIN_AVERAGE_PAD,
            n_bins + _BIN_AVERAGE_PAD,
            dtype=backend.float,
        )
        time_extended = float(time[0]) + index * bin_step
        wake = self._wake_with_resolved_onset(time_extended)
        # The (1, 76, 230, 76, 1) / 384 stencil of
        # `TimeDomain.get_wake_per_bin`, on the padded axis so that no
        # returned sample has to fall back on an edge-clamped neighbour.
        stencil = (
            wake[:-4]
            + 76.0 * wake[1:-3]
            + 230.0 * wake[2:-2]
            + 76.0 * wake[3:-1]
            + wake[4:]
        ) / 384.0
        binned = stencil[1 : 1 + n_bins]
        self._add_onset_correction(binned, wake, time_extended, bin_step)
        return binned

    def _add_onset_correction(
        self,
        binned: NumpyArray | CupyArray,
        wake: NumpyArray | CupyArray,
        time_extended: NumpyArray | CupyArray,
        bin_step: float,
    ) -> None:
        """
        Add the closed-form jump correction of :meth:`get_wake_per_bin`.

        Modifies ``binned`` in place at the at most five samples the
        B-spline reaches over the causal onset from.

        Parameters
        ----------
        binned
            Stencil result to correct, in [V].
        wake
            Wake of the resolved model on ``time_extended``, in [V].
        time_extended
            Padded, uniform time axis the stencil was evaluated on, in [s].
        bin_step
            Bin width, in [s].
        """
        onset_time = float(self._wake_x[0])
        first_above = math.ceil(
            (onset_time - float(time_extended[0])) / bin_step
        )
        if not 1 <= first_above <= len(time_extended) - 1:
            # The onset is outside the padded axis, so no returned sample
            # can see it.
            return
        width = (float(time_extended[first_above]) - onset_time) / bin_step
        width = min(max(width, 0.0), 1.0)
        jump = self._onset_jump
        node_value = float(wake[first_above])
        for index in range(first_above - 2, first_above + 3):
            out_index = index - _BIN_AVERAGE_PAD
            if not 0 <= out_index < len(binned):
                continue
            offset = float(index - first_above)
            box_average, ramp_moment = bspline_window_moments(offset, width)
            onset_ramp = ramp_moment / width if width > 0.0 else 0.0
            grid_ramp = bspline_window_moments(offset, 1.0)[1]
            binned[out_index] += (
                jump * box_average
                + (node_value - jump) * onset_ramp
                - node_value * grid_ramp
            )


class TravelingWaveCavityWakePerBin:
    """
    Point wake of :class:`~blond.physics.impedances.sources.TravelingWaveCavity`.

    Feeds the generic bin average of
    :class:`~blond.physics.impedances.wake_per_bin.time_domain.TimeDomainWakePerBin`.
    """

    def get_wake(self, time: NumpyArray | CupyArray) -> NumpyArray | CupyArray:
        """
        Point-charge wake at ``time``, see :meth:`~blond.physics.impedances.sources.TravelingWaveCavity.wake_calc`.

        Parameters
        ----------
        time
            Time array at which the wake is evaluated, in [s].

        Returns
        -------
        wake
            Wake, in [V].
        """
        return self.wake_calc(time=time)
