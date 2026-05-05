# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Implementation of `SymbolicSeparatrixHelper`."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import sympy

from blond.core.base import HasSymbolicHamiltonian
from blond.utilities.separatrix.helpers import _get_omega_min

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.lines import Line2D
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CanonicalBucket:
    """
    One canonical RF bucket worth of geometry, exploited by periodicity.

    The full Hamiltonian decomposes as ``H = a*dE**2 + U(dt)`` and ``U(dt)``
    splits into a periodic part plus a linear acceleration tilt
    ``ref_E_change * dt``. ``U`` therefore repeats every ``period``, with
    each successive copy shifted up by ``shift_per_period``. One canonical
    UFP is enough to reconstruct every bucket by integer offsets:

    * UFP at period ``n``: ``ufp_dt + n*period`` with potential
      ``ufp_potential + n*shift_per_period``.
    * Bucket ``n`` is bounded by UFP ``n`` and UFP ``n+1``.
    """

    ufp_dt: float
    ufp_potential: float
    period: float
    shift_per_period: float


class SymbolicSeparatrixHelper:
    """
    A helper tool to derive the separatrix dynamically from a symbolic Hamiltonian.

    This captures only the instantaneous separatrix and does not
    automatically update, when the simulation variables are updated.

    Parameters
    ----------
    hamiltonian
        Sympy-based expression of the Hamiltonian.
    omega_min
        Minimum RF frequency in the `Ring`. The longest RF period
        ``2*pi/omega_min`` is used as the canonical scan window when
        locating unstable fixed points.

    See Also
    --------
    blond.physics.cavities.SingleHarmonicRFStation.get_hamilton_symbolic : Partial Hamiltonian definition for RFStation.
    blond.physics.cavities.MultiHarmonicRFStation.get_hamilton_symbolic : Partial Hamiltonian definition for RFStation.
    blond.physics.drifts.DriftSimple.get_hamilton_symbolic : Partial Hamiltonian definition for Drift.
    blond.physics.drifts.DriftExact.get_hamilton_symbolic : Partial Hamiltonian definition for Drift.
    """

    #: Number of grid points used when scanning one canonical period to
    #: locate UFPs of the potential.
    _CANONICAL_SCAN_RESOLUTION = 10_000

    #: Tolerance (in fractional bucket index) within which ``ratio`` is
    #: snapped to the nearest integer, so a UFP exactly on a bucket
    #: boundary is assigned to the bucket on its right despite float
    #: round-off.
    _BUCKET_BOUNDARY_TOLERANCE = 1e-9

    def __init__(self, hamiltonian: sympy.Expr, omega_min: float):
        self._hamiltonian = hamiltonian
        self._omega_min = omega_min

    @staticmethod
    def from_simulation(simulation: Simulation) -> SymbolicSeparatrixHelper:
        """
        Instantiate `SymbolicSeparatrixHelper` from a `Simulation`.

        Parameters
        ----------
        simulation
            `Simulation` object that the separatrix is derived from.

        Returns
        -------
        symbolic_separatrix_helper
            The initialized `SymbolicSeparatrixHelper`.
        """
        partials = [
            element.get_hamilton_symbolic()
            for element in simulation.ring.elements.get_elements(
                HasSymbolicHamiltonian
            )
        ]
        if not partials:
            raise ValueError(
                "No elements with `HasSymbolicHamiltonian` found."
            )
        hamiltonian = sympy.Add(*partials)
        return SymbolicSeparatrixHelper(
            hamiltonian=hamiltonian,
            omega_min=_get_omega_min(simulation.ring),
        )

    def get_separatrix(
        self,
        beam: BeamBaseClass,
        dt: NumpyArray,
    ) -> tuple[NumpyArray, NumpyArray]:
        """
        Compute the separatrix boundary in longitudinal phase space.

        Substitutes numerical beam values into the symbolic Hamiltonian,
        then solves ``H(dt, dE) = H_sep(dt)`` for ``dE`` at each ``dt``.
        Each RF bucket has its own ``H_sep``: the lower of the two bounding
        UFPs above transition (``a > 0``), the higher one below transition
        (``a < 0``). With non-zero ``reference_energy_change`` consecutive
        UFPs differ in height, so this per-dt treatment is required;
        without acceleration the result reduces to the familiar single
        ``H_sep`` answer.

        Parameters
        ----------
        beam
            Beam whose reference coordinates supply β, γ, E and charge.
        dt
            Time-deviation grid [s] over which to evaluate the separatrix.

        Returns
        -------
        separatrix
            Array of shape ``(2, len(dt))``: row 0 the upper branch
            (``dE ≥ 0``), row 1 the lower branch (``dE ≤ 0``), in [eV].
            Entries are ``NaN`` for ``dt`` lying outside any RF bucket
            (above the local barrier, or no bucket exists at all because
            the linear tilt eliminated all extrema).
        """
        a, potential = self._substitute_symbols(beam=beam)

        H_sep = self._H_sep_per_dt(dt, a=a, potential=potential)
        dE_sep = self._dE_sep_upper(dt, a=a, potential=potential, H_sep=H_sep)
        return np.stack([dE_sep, -dE_sep])

    def plot_separatrix(
        self,
        beam: BeamBaseClass,
        dt: NumpyArray,
        **kwargs_plot,
    ) -> list[Line2D]:
        """
        Plot the longitudinal phase-space separatrix.

        Calls :meth:`get_separatrix` and draws both branches on the current
        matplotlib axes. The label (if given) is applied only to the upper
        branch so the legend shows a single entry.

        Parameters
        ----------
        beam
            Beam whose reference coordinates supply β, γ, E and charge.
        dt
            Time-deviation grid [s] spanning at least the full RF bucket,
            including the unstable fixed point.
        **kwargs_plot
            Additional keyword arguments forwarded to ``matplotlib.pyplot.plot``
            (e.g. ``color``, ``linewidth``, ``linestyle``).

        Returns
        -------
        artists
            List of matplotlib objects.

        Notes
        -----
        This method does not call ``plt.show()``; call that separately.
        """
        kwargs_plot.setdefault("color", "red")
        kwargs_plot.setdefault("linestyle", "dashed")

        separatrix = self.get_separatrix(beam=beam, dt=dt)

        label = kwargs_plot.pop("label", None)
        artists = plt.plot(dt, separatrix[0], label=label, **kwargs_plot)
        kwargs_plot.pop("color", None)
        artists.extend(
            plt.plot(
                dt,
                separatrix[1],
                color=artists[0].get_color(),
                **kwargs_plot,
            )
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Energy offset [eV]")
        return artists

    def _substitute_symbols(
        self, beam: BeamBaseClass
    ) -> tuple[float, Callable[[NumpyArray], NumpyArray]]:
        """
        Substitute beam scalars into the Hamiltonian.

        Parameters
        ----------
        beam
            Beam whose reference coordinates supply β, γ, E and charge.

        Returns
        -------
        a
            Kinetic coefficient ``coeff(dE, 2)`` of the Hamiltonian.
        potential
            Numpy-vectorized callable for ``H(dt, dE=0)``.
        """
        dt_sym, dE_sym = sympy.symbols("dt dE", real=True)
        beta_sym, gamma_sym, E_sym, q_sym = sympy.symbols(
            "beta gamma E q", real=True
        )
        ham = self._hamiltonian.subs(
            {
                beta_sym: beam.reference.beta,
                gamma_sym: beam.reference.gamma,
                E_sym: beam.reference.total_energy,
                q_sym: float(beam.particle_type.charge),
            }
        )
        a = float(ham.coeff(dE_sym, 2))
        potential = sympy.lambdify(
            dt_sym, ham.subs(dE_sym, 0), modules="numpy"
        )
        return a, potential

    @staticmethod
    def _dE_sep_upper(
        dt: NumpyArray,
        a: float,
        potential: Callable[[NumpyArray], NumpyArray],
        H_sep: NumpyArray,
    ) -> NumpyArray:
        """
        Solve ``a*dE**2 + U(dt) = H_sep(dt)`` for the upper branch ``dE >= 0``.

        Parameters
        ----------
        dt
            Time-deviation grid [s].
        a
            Kinetic coefficient ``coeff(dE, 2)`` of the Hamiltonian.
        potential
            Numpy-vectorized callable for ``H(dt, dE=0)``.
        H_sep
            Per-dt Hamiltonian value on the bucket separatrix.

        Returns
        -------
        dE_sep
            Upper-branch separatrix energy at each ``dt``. ``NaN`` where
            ``H_sep`` is ``NaN`` (no bucket exists) or ``(H_sep - U)/a < 0``
            (particle above the local barrier).
        """
        f_values = np.asarray(potential(dt), dtype=float)
        with np.errstate(invalid="ignore"):
            inside = (H_sep - f_values) / a
        dE_sep = np.full(dt.shape, np.nan, dtype=float)
        in_bucket = np.isfinite(inside) & (inside >= 0)
        dE_sep[in_bucket] = np.sqrt(inside[in_bucket])
        return dE_sep

    def _H_sep_per_dt(
        self,
        dt: NumpyArray,
        a: float,
        potential: Callable[[NumpyArray], NumpyArray],
    ) -> NumpyArray:
        """
        Compute the per-``dt`` Hamiltonian value on the bucket separatrix.

        Locates the canonical UFP within one RF period and extrapolates
        bucket boundaries by integer offsets of ``period`` (in dt) and
        ``shift_per_period`` (in potential).

        Parameters
        ----------
        dt
            Time-deviation grid [s].
        a
            Kinetic coefficient ``coeff(dE, 2)`` of the Hamiltonian.
        potential
            Numpy-vectorized callable for ``H(dt, dE=0)``.

        Returns
        -------
        H_sep
            Per-dt separatrix Hamiltonian value. ``NaN`` when no interior
            extremum exists (the linear tilt is so strong that the
            potential is monotonic over a period).
        """
        if a == 0.0:
            return np.full(dt.shape, np.nan, dtype=float)

        bucket = self._find_canonical_bucket(
            period_start=float(np.min(dt)), a=a, potential=potential
        )
        if bucket is None:
            return np.full(dt.shape, np.nan, dtype=float)

        bucket_index = self._bucket_index(dt, bucket)
        potential_left = (
            bucket.ufp_potential + bucket_index * bucket.shift_per_period
        )
        potential_right = potential_left + bucket.shift_per_period
        if a > 0:
            return np.minimum(potential_left, potential_right)
        return np.maximum(potential_left, potential_right)

    def _find_canonical_bucket(
        self,
        period_start: float,
        a: float,
        potential: Callable[[NumpyArray], NumpyArray],
    ) -> _CanonicalBucket | None:
        """
        Locate one canonical UFP inside ``[period_start, period_start + period]``.

        For multi-harmonic potentials with several extrema per period, the
        canonical UFP is the highest local max (``a > 0``) or lowest local
        min (``a < 0``) — i.e., the outer-separatrix barrier; sub-bucket
        structure is not represented.

        Parameters
        ----------
        period_start
            Left edge of the canonical scan window [s].
        a
            Kinetic coefficient ``coeff(dE, 2)`` of the Hamiltonian.
        potential
            Numpy-vectorized callable for ``H(dt, dE=0)``.

        Returns
        -------
        bucket
            Canonical bucket geometry, or ``None`` when no interior
            extremum exists in the scan window.
        """
        period = 2.0 * np.pi / self._omega_min
        scan_dt = np.linspace(
            period_start,
            period_start + period,
            self._CANONICAL_SCAN_RESOLUTION + 1,
        )
        scan_potential = np.asarray(potential(scan_dt), dtype=float)
        shift_per_period = float(scan_potential[-1] - scan_potential[0])

        extremum_indices = self._interior_extrema(scan_potential, a=a)
        if extremum_indices.size == 0:
            return None

        if a > 0:
            ufp_index = extremum_indices[
                np.argmax(scan_potential[extremum_indices])
            ]
        else:
            ufp_index = extremum_indices[
                np.argmin(scan_potential[extremum_indices])
            ]
        return _CanonicalBucket(
            ufp_dt=float(scan_dt[ufp_index]),
            ufp_potential=float(scan_potential[ufp_index]),
            period=period,
            shift_per_period=shift_per_period,
        )

    @staticmethod
    def _interior_extrema(values: NumpyArray, a: float) -> NumpyArray:
        """
        Find indices of interior local maxima (``a > 0``) or minima (``a < 0``).

        Parameters
        ----------
        values
            1-D array sampled along the dt-axis.
        a
            Kinetic coefficient ``coeff(dE, 2)`` — its sign decides whether
            UFPs are local maxima or local minima.

        Returns
        -------
        indices
            Indices into ``values`` of interior extrema (boundaries
            excluded).
        """
        slope = np.diff(values)
        if a > 0:
            is_extremum = (slope[:-1] > 0) & (slope[1:] < 0)
        else:
            is_extremum = (slope[:-1] < 0) & (slope[1:] > 0)
        return np.flatnonzero(is_extremum) + 1

    @classmethod
    def _bucket_index(
        cls, dt: NumpyArray, bucket: _CanonicalBucket
    ) -> NumpyArray:
        """
        Compute the bucket index ``n`` containing each ``dt``.

        ``floor((dt - ufp_dt) / period)`` is ambiguous on the boundary due
        to float roundoff (``n - 1e-16`` floors to ``n - 1``), so we round
        to the nearest integer when within :attr:`_BUCKET_BOUNDARY_TOLERANCE`
        — that way a UFP is treated as the LEFT boundary of bucket ``n``
        (where the separatrix touches ``dE = 0``).

        Parameters
        ----------
        dt
            Time-deviation grid [s].
        bucket
            Canonical bucket geometry from :meth:`_find_canonical_bucket`.

        Returns
        -------
        bucket_index
            Integer-valued float array; ``dt`` lies in bucket
            ``[UFP_n, UFP_{n+1}]``.
        """
        ratio = (dt - bucket.ufp_dt) / bucket.period
        nearest = np.round(ratio)
        return np.where(
            np.abs(ratio - nearest) < cls._BUCKET_BOUNDARY_TOLERANCE,
            nearest,
            np.floor(ratio),
        )
