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
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import sympy

from blond.core.base import (
    HasSymbolicHamiltonian,
)
from blond.utilities.separatrix.helpers import _get_omega_min

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.lines import Line2D
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


logger = logging.getLogger(__name__)


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
        Minimum RF frequency in the `Ring`.
        This is internally used to derive the separatrix from ``H_max``
        within the according period.

    See Also
    --------
    blond.physics.cavities.SingleHarmonicRFStation.get_hamilton_symbolic : Partial Hamiltonian definition for RFStation.
    blond.physics.cavities.MultiHarmonicRFStation.get_hamilton_symbolic : Partial Hamiltonian definition for RFStation.
    blond.physics.drifts.DriftSimple.get_hamilton_symbolic : Partial Hamiltonian definition for Drift.
    blond.physics.drifts.DriftExact.get_hamilton_symbolic : Partial Hamiltonian definition for Drift.
    """

    def __init__(self, hamiltonian: sympy.Expr, omega_min: float):
        self._ham = hamiltonian

        # Settings for `_get_H_sep`
        self._find_Hmax_omega_min = omega_min
        self._find_Hmax_points = 10_000

    @staticmethod
    def from_simulation(simulation: Simulation):
        """
        Instantiate `SymbolicSeparatrixHelper` from a `Simulation`.

        Parameters
        ----------
        simulation
            `Simulation` object that the separatrix is derived from.

        Returns
        -------
        symbolic_separatrix_heler
            The initialized `SymbolicSeparatrixHelper`.
        """
        ham = None
        for element in simulation.ring.elements.get_elements(
            HasSymbolicHamiltonian
        ):
            partial = element.get_hamilton_symbolic()
            ham = partial if ham is None else ham + partial

        if ham is None:
            raise ValueError(
                "No elements with `HasSymbolicHamiltonian` found."
            )

        return SymbolicSeparatrixHelper(
            hamiltonian=ham,
            omega_min=_get_omega_min(simulation.ring),
        )

    def get_separatrix(
        self,
        beam: BeamBaseClass,
        dt: NumpyArray,
    ) -> tuple[NumpyArray, NumpyArray]:
        """
        Compute the separatrix boundary in longitudinal phase space.

        Accumulates the partial Hamiltonians from all elements that implement
        :class:`HasSymbolicHamiltonian`, substitutes numerical beam values,
        then solves ``H(dt, dE) = H_sep(dt)`` for ``dE`` at each point of
        ``dt``.

        Parameters
        ----------
        beam
            Beam whose reference coordinates supply β, γ, E and charge.
        dt
            Time-deviation grid [s] over which to evaluate the separatrix.

        Returns
        -------
        separatrix
            Array of shape ``(2, len(dt))`` where row 0 is the upper branch
            (``dE ≥ 0``) and row 1 is the lower branch (``dE ≤ 0``), in [eV].
            Entries are ``NaN`` for ``dt`` that lie outside any RF bucket
            (e.g. beyond the bracketing unstable fixed points, or in regions
            where the linear acceleration tilt has eliminated all extrema).

        Notes
        -----
        Each RF bucket has its own ``H_sep`` set by the lower of the two
        bounding unstable fixed points (UFPs) when above transition
        (``a > 0``), or the higher one when below transition (``a < 0``).
        Without acceleration the potential is periodic and all UFPs share
        the same height — the per-bucket result reduces to the old single
        ``H_sep`` answer. With non-zero ``reference_energy_change`` from an
        accelerating cycle the potential is a tilted cosine; consecutive
        UFPs differ in height, so this per-dt treatment is required for the
        result to be physical.
        """
        # Substitute numerical beam values
        dt_sym, dE_sym = sympy.symbols("dt dE", real=True)
        beta_sym, gamma_sym, E_sym, q_sym = sympy.symbols(
            "beta gamma E q", real=True
        )
        ham = self._ham.subs(
            {
                beta_sym: beam.reference.beta,
                gamma_sym: beam.reference.gamma,
                E_sym: beam.reference.total_energy,
                q_sym: float(beam.particle_type.charge),
            }
        )

        # Extract kinetic coefficient a (coeff of dE²) and potential f(dt)
        a = float(ham.coeff(dE_sym, 2))
        f_num = sympy.lambdify(dt_sym, ham.subs(dE_sym, 0), modules="numpy")

        dt_arr = np.asarray(dt, dtype=float)
        H_sep = self._H_sep_per_dt(dt_arr, a=a, f_num=f_num)

        f_values = np.asarray(f_num(dt_arr), dtype=float)
        with np.errstate(invalid="ignore"):
            inside = (H_sep - f_values) / a
        dE_sep = np.full(dt_arr.shape, np.nan, dtype=float)
        in_bucket = np.isfinite(inside) & (inside >= 0)
        dE_sep[in_bucket] = np.sqrt(inside[in_bucket])

        return dt, np.stack([dE_sep, -dE_sep])

    def _H_sep_per_dt(
        self,
        dt: NumpyArray,
        a: float,
        f_num: Callable,
    ) -> NumpyArray:
        pass
        """
        Compute the per-``dt`` Hamiltonian value on the bucket separatrix.

        ``f := H(dt, dE=0)`` decomposes as a periodic part plus the linear
        acceleration tilt ``ref_E_change * dt``, so it repeats every
        ``T_rf = 2π/omega_min`` with each period shifted up by exactly
        ``shift_per_period := f(dt₀ + T_rf) - f(dt₀) = ref_E_change * T_rf``.
        The bucket structure is therefore identical in every period and we
        only need to find one canonical UFP, then extrapolate by integer
        offsets:

        * UFP at period ``n``: ``ufp_dt + n*T_rf`` with potential
          ``ufp_potential + n*shift_per_period``.
        * Bucket ``n`` is bounded by UFP ``n`` and UFP ``n+1``.
        * ``H_sep`` is the lower (``a > 0``) or higher (``a < 0``) of the
          two bounding UFP potentials.

        Returns ``NaN`` when the linear tilt has eliminated all interior
        extrema of ``f`` over a period (no bucket exists anywhere).
        """
        if a == 0.0:
            return np.full(dt.shape, np.nan, dtype=float)

        period = 2.0 * np.pi / self._find_Hmax_omega_min

        # Scan ONE canonical period — endpoints differ by exactly
        # shift_per_period (= ref_E_change * period).
        period_start = float(np.min(dt))
        scan_dt = np.linspace(
            period_start,
            period_start + period,
            self._find_Hmax_points + 1,
        )
        scan_potential = np.asarray(f_num(scan_dt), dtype=float)
        shift_per_period = float(scan_potential[-1] - scan_potential[0])

        # Interior extrema of the potential within this canonical period.
        slope = np.diff(scan_potential)
        if a > 0:
            is_local_extremum = (slope[:-1] > 0) & (slope[1:] < 0)
        else:
            is_local_extremum = (slope[:-1] < 0) & (slope[1:] > 0)
        extremum_indices = np.flatnonzero(is_local_extremum) + 1

        if extremum_indices.size == 0:
            # Linear tilt overwhelms the cosine — f is monotonic, no buckets.
            return np.full(dt.shape, np.nan, dtype=float)

        # Pick the outer-separatrix UFP: highest local max (a > 0) or
        # lowest local min (a < 0) within the period. For single-harmonic
        # there is only one and this is trivially it; for multi-harmonic
        # this selects the outer barrier (sub-bucket structure inside is
        # not represented).
        if a > 0:
            ufp_index = extremum_indices[
                np.argmax(scan_potential[extremum_indices])
            ]
        else:
            ufp_index = extremum_indices[
                np.argmin(scan_potential[extremum_indices])
            ]
        ufp_dt = scan_dt[ufp_index]
        ufp_potential = scan_potential[ufp_index]

        # Bucket n is bounded by UFP_n at ufp_dt + n*period and UFP_{n+1}.
        bucket_index = np.floor((dt - ufp_dt) / period)
        potential_left_ufp = ufp_potential + bucket_index * shift_per_period
        potential_right_ufp = potential_left_ufp + shift_per_period
        if a > 0:
            return np.minimum(potential_left_ufp, potential_right_ufp)
        return np.maximum(potential_left_ufp, potential_right_ufp)

    def plot_separatrix(
        self,
        beam: BeamBaseClass,
        dt: NumpyArray,
        **kwargs_plot,
    ) -> list[Line2D]:
        """
        Plot the longitudinal phase-space separatrix.

        Calls :meth:`get_separatrix` and draws both branches on the current
        matplotlib axes.  The label (if given) is applied only to the upper
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

        See Also
        --------
        get_separatrix : Compute the separatrix boundary numerically.

        Notes
        -----
        This method does not call ``plt.show()``; call that separately.
        """
        default_kwargs = {
            "color": "red",
            "linestyle": "dashed",
        }
        for key, value in default_kwargs.items():
            if key not in kwargs_plot:
                kwargs_plot[key] = value
        dt, separatrix = self.get_separatrix(
            beam=beam,
            dt=dt,
        )
        label = kwargs_plot.pop("label", None)
        artists = plt.plot(
            dt,
            separatrix[0],
            label=label,
            **kwargs_plot,
        )
        if "color" in kwargs_plot:
            kwargs_plot.pop("color")
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
