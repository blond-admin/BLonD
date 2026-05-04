# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Implementation of `SymbolicSeparatrixHelper`."""

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
        self._find_Hmax_margin = 5 / 100
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
    ) -> NumpyArray:
        """
        Compute the separatrix boundary in longitudinal phase space.

        Accumulates the partial Hamiltonians from all elements that implement
        :class:`HasSymbolicHamiltonian`, substitutes numerical beam values,
        then solves ``H(dt, dE) = H_sep`` for ``dE`` at each point of ``dt``.

        Parameters
        ----------
        beam
            Beam whose reference coordinates supply β, γ, E and charge.
        dt
            Time-deviation grid [s] over which to evaluate the separatrix.
            Should span the full RF bucket including the unstable fixed point.

        Returns
        -------
        separatrix
            Array of shape ``(2, len(dt))`` where row 0 is the upper branch
            (``dE ≥ 0``) and row 1 is the lower branch (``dE ≤ 0``), in [eV·s].
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

        H_sep = self._get_H_sep(a, f_num)

        # Evaluate f on the user-supplied grid, then solve H = H_sep for dE
        f_values = np.asarray(f_num(dt), dtype=float)
        inside = (H_sep - f_values) / a
        dE_sep = np.sqrt(np.maximum(inside, 0.0))

        return np.stack([dE_sep, -dE_sep])

    def _get_H_sep(self, a: float, f_num: Callable) -> float:
        """
        Calculate the ``H`` value at the border of the separatrix.

        Find H_sep on a dense internal grid covering at least one full period
        of the lowest RF frequency — independent of the user-supplied dt which
        may be sparse or not reach the unstable fixed point.

        Parameters
        ----------
        a
            The kinetic coefficient a (coeff of dE²) and potential f(dt).
        f_num
            The numeric function that evaluates ``H(dt, dE=0)``.

        Returns
        -------
        H_sep
            Hamiltonian value on the separatrix.
        """
        # 1.1× the longest RF period gives a 10 % margin on each side
        T_rf = 2.0 * np.pi / self._find_Hmax_omega_min
        dt_scan = np.linspace(
            -self._find_Hmax_margin * T_rf,
            (1 + self._find_Hmax_margin) * T_rf,
            self._find_Hmax_points,
        )

        f_scan = np.asarray(f_num(dt_scan), dtype=float)
        H_sep = float(np.max(f_scan) if a > 0 else np.min(f_scan))
        return H_sep

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
        separatrix = self.get_separatrix(beam=beam, dt=dt)
        label = kwargs_plot.pop("label", None)
        artists = plt.plot(dt, separatrix[0], label=label, **kwargs_plot)
        if "color" in kwargs_plot:
            kwargs_plot.pop("color")
        artists.extend(
            plt.plot(
                dt, separatrix[1], color=artists[0].get_color(), **kwargs_plot
            )
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Energy offset [eV]")
        return artists
