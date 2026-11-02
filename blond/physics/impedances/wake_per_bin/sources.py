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

import cmath
import math
import warnings
from typing import TYPE_CHECKING

from blond.core.backends.backend import backend
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.impedances.wake_per_bin.bin_average import (
    bspline_kink_kernel,
    bspline_step_kernel,
    triple_box_average_linear_pole,
    triple_box_average_poles,
)

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


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
    r"""
    Bin-averaged wake of :class:`~blond.physics.impedances.sources.ImpedanceTableTime`.

    ``wake_y[0]`` is the wake value just above ``wake_x[0]``, the right
    limit :math:`W(\mathrm{wake\_x}[0]^+)`, as an external wake solver
    (CST, ABCI, a measurement) records it. Below ``wake_x[0]`` the wake is
    zero, so a non-zero ``wake_y[0]`` is a causal jump of that height. A
    jump elsewhere is spelled as a repeated time, ``wake_x = [..., x, x,
    ...]``. A table point-sampled from a BLonD wake at ``t = 0`` holds
    half the jump there (see
    :meth:`~blond.physics.impedances.sources.Resonators.heaviside_eps_at_0`);
    store the right limit instead.
    """

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
        :meth:`~blond.physics.impedances.sources.Resonators.heaviside_eps_at_0`
        makes by returning 1 rather than 2 at ``t = 0``. So the onset sample
        returns ``0.5 * wake_y[0]``. Strictly above the onset the
        interpolant is untouched, strictly below it the wake stays zero.

        "Exactly at the onset" is a comparison within
        ``_ONSET_TOLERANCE_EPS`` of the table's own first spacing, not a
        float equality. The bin-averaged version,
        :meth:`get_wake_per_bin`, integrates the *function* that sampling
        represents and so resolves the jump over a whole bin.

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
        return self._halve_onset_sample(time, wake)

    def _halve_onset_sample(
        self,
        time: NumpyArray | CupyArray,
        wake: NumpyArray | CupyArray,
    ) -> NumpyArray | CupyArray:
        """
        Put the jump midpoint on the samples sitting at the causal onset.

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
        Exact bin average of the tabulated wake, at table resolution.

        Overrides
        :meth:`~blond.physics.impedances.base.TimeDomain.get_wake_per_bin`.
        The generic default there samples the wake at the bin width and
        B-spline-averages the interpolant through those samples, which
        throws away all of a table that is finer than the bins: an
        oscillation the bins do not resolve aliases instead of averaging
        out.

        What is integrated here is the model the table represents, at its
        own resolution: zero below ``wake_x[0]``, a causal jump there to
        ``wake_y[0]``, piecewise linear through
        the samples, a jump at any repeated ``wake_x`` and clamped past the
        end. Writing it as a continuous piecewise-linear part :math:`g`,
        steps :math:`J_k \Theta(t - x_k)` and kinks
        :math:`\sigma_k (t - x_k)_+` (:math:`\sigma_k` the change of
        slope at :math:`x_k`), its average with the quadratic B-spline is

        .. math::
            \overline W(t) = g(t) + \sum_k J_k \Theta(t - x_k)
              + \sum_k \left[J_k\, D(u_k)
              + \sigma_k \Delta t\, E(u_k)\right],
            \qquad u_k = (t - x_k) / \Delta t,

        with :math:`D` and :math:`E` from
        :func:`~blond.physics.impedances.wake_per_bin.bin_average.bspline_step_kernel`
        and
        :func:`~blond.physics.impedances.wake_per_bin.bin_average.bspline_kink_kernel`.
        Both vanish for :math:`|u_k| \geq 3/2`, so every sample reaches
        at most four bins and the cost is linear in the table length.
        The step is compared with the same :math:`t - x_k` throughout, so
        a bin centre a rounding error away from the onset needs no
        tolerance: the average is continuous in :math:`t`.

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
        self._warn_if_outside_table(time)
        bin_step = float(time[1] - time[0])
        first, stop = self._knot_window(time, bin_step)
        knots = self._wake_x[first:stop]
        values = self._wake_y[first:stop]
        knot_spacing = knots[1:] - knots[:-1]
        value_change = values[1:] - values[:-1]
        has_length = knot_spacing > 0.0
        slope = backend.where(
            has_length,
            value_change / backend.where(has_length, knot_spacing, 1.0),
            0.0,
        )
        zero = backend.zeros(1, dtype=backend.float)
        # Past either end of a window inside the table the slope is not
        # known, and taken as zero: those knots are out of reach anyway.
        slope_change = backend.concatenate((slope, zero)) - (
            backend.concatenate((zero, slope))
        )
        jump = backend.concatenate(
            (
                values[:1] if first == 0 else zero,
                backend.where(has_length, 0.0, value_change),
            )
        )
        cumulative_jump = self._jumps_up_to(first) + backend.cumulative_sum(
            jump
        )
        continuous_part = backend.interp(
            time, knots, values - cumulative_jump, left=0.0
        )
        steps_passed = backend.searchsorted(knots, time, side="right")
        steps = backend.where(
            steps_passed > 0,
            cumulative_jump[
                backend.where(steps_passed > 0, steps_passed - 1, 0)
            ],
            0.0,
        )
        return (
            continuous_part
            + steps
            + self._local_corrections(
                time, bin_step, knots, jump, slope_change
            )
        )

    def _knot_window(
        self, time: NumpyArray | CupyArray, bin_step: float
    ) -> tuple[int, int]:
        """
        Range of table samples that can reach a bin of ``time``.

        The step and kink kernels vanish more than 1.5 bins away, so only
        the knots within 1.5 bins of the query matter, plus two on either
        side: one to bracket the query for the interpolation and one more
        so that the first and last knot within reach see their real
        neighbouring segments.

        Parameters
        ----------
        time
            Bin centres, uniformly spaced, in [s].
        bin_step
            Bin width, in [s].

        Returns
        -------
        window
            ``(first, stop)``, a slice of ``wake_x``.
        """
        reach = 1.5 * bin_step
        first = int(
            backend.searchsorted(
                self._wake_x, float(time[0]) - reach, side="left"
            )
        )
        stop = int(
            backend.searchsorted(
                self._wake_x, float(time[-1]) + reach, side="right"
            )
        )
        return max(first - 2, 0), min(stop + 2, len(self._wake_x))

    def _jumps_up_to(self, last: int) -> float:
        """
        Sum of the jumps at ``wake_x[0]`` to ``wake_x[last]``, in [V].

        The causal onset (if ``last`` > 0; a window starting at the table
        start carries it itself) and every jump spelled as a repeated time.
        Repeated times are rare, so they are found with a boolean scan, not
        a full-length float temporary.

        Parameters
        ----------
        last
            Index of the last knot whose jump is summed.

        Returns
        -------
        jumps
            Sum of those jumps, in [V].
        """
        if last == 0:
            return 0.0
        repeated = (
            self._wake_x[1 : last + 1] == self._wake_x[:last]
        ).nonzero()[0] + 1
        jumps = backend.sum(
            self._wake_y[repeated] - self._wake_y[repeated - 1]
        )
        return float(self._wake_y[0]) + float(jumps)

    @staticmethod
    def _local_corrections(
        time: NumpyArray | CupyArray,
        bin_step: float,
        knots: NumpyArray | CupyArray,
        jump: NumpyArray | CupyArray,
        slope_change: NumpyArray | CupyArray,
    ) -> NumpyArray | CupyArray:
        """
        Scatter each knot's step and kink kernels onto its nearby bins.

        Parameters
        ----------
        time
            Bin centres, uniformly spaced, in [s].
        bin_step
            Bin width, in [s].
        knots
            Table times, in [s].
        jump
            Step at each knot, in [V].
        slope_change
            Change of slope at each knot, in [V/s].

        Returns
        -------
        corrections
            Sum of the kernels per bin, in [V].
        """
        n_bins = len(time)
        nearest_below = backend.floor(
            (knots - float(time[0])) / bin_step
        ).astype(int)
        correction = backend.zeros(n_bins, dtype=backend.float)
        # |u| < 3/2 reaches the bins from one below to two above the knot.
        for shift in range(-1, 3):
            index = nearest_below + shift
            inside = (index >= 0) & (index < n_bins)
            index = backend.where(inside, index, 0)
            distance = (time[index] - knots) / bin_step
            weight = jump * bspline_step_kernel(
                distance
            ) + slope_change * bin_step * bspline_kink_kernel(distance)
            correction += backend.bincount(
                index,
                weights=backend.where(inside, weight, 0.0),
                minlength=n_bins,
            )
        return correction


class TravelingWaveCavityWakePerBin:
    """
    Bin-averaged wake of :class:`~blond.physics.impedances.sources.TravelingWaveCavity`.

    The wake is a linear envelope times an undamped pole between
    ``t = 0`` and the filling time, so its bin average is evaluated in
    closed form, jumps included.
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

    def get_wake_per_bin(
        self, time: NumpyArray | CupyArray
    ) -> NumpyArray | CupyArray:
        r"""
        Exact bin average of the wake, in [V].

        Overrides
        :meth:`~blond.physics.impedances.base.TimeDomain.get_wake_per_bin`.
        Each mode, :math:`W = \frac{4R}{\tilde a}(1 - t / \tilde a)
        \cos(\omega_r t)` on :math:`0 < t \leq \tilde a`, is a linear
        envelope times the undamped pole :math:`p = i \omega_r`, switched
        on at :math:`t = 0` and off again at :math:`\tilde a`:

        .. math::
            W = \mathrm{Re}[(A - B t) e^{p t}]\,\Theta(t)
              + \mathrm{Re}[B e^{p \tilde a} (t - \tilde a)
                e^{p (t - \tilde a)}]\,\Theta(t - \tilde a),

        with :math:`A = 4R / \tilde a` and :math:`B = A / \tilde a`, since
        :math:`A - B \tilde a = 0`. Both terms are B-spline-averaged in
        closed form by
        :func:`~blond.physics.impedances.wake_per_bin.bin_average.triple_box_average_linear_pole`,
        so the result is exact at any frequency, including above the bin
        Nyquist frequency, where sampling the wake at the bin width would
        alias.

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
        bin_step = float(time[1] - time[0])
        wake = backend.zeros(time.shape, dtype=backend.float)
        for i in range(len(self.R_S)):
            a_tilde = float(self.a_factor[i]) / (2 * math.pi)
            pole = 2j * math.pi * float(self.frequency_R[i])
            constant = 4 * float(self.R_S[i]) / a_tilde
            slope = constant / a_tilde
            wake += triple_box_average_linear_pole(
                time, pole, constant, -slope, bin_step
            )
            wake += triple_box_average_linear_pole(
                time - a_tilde,
                pole,
                0.0,
                slope * cmath.exp(pole * a_tilde),
                bin_step,
            )
        return wake

    def _wake_parameters_salt(self) -> object:
        """
        Summary of ``R_S``, ``frequency_R`` and ``a_factor``.

        They are public and may be changed after construction, so the
        cached impedance has to follow them.

        Returns
        -------
        salt
            The parameters as bytes.
        """
        return tuple(
            copy_to_cpu(parameter).tobytes()
            for parameter in (self.R_S, self.frequency_R, self.a_factor)
        )
