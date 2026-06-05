# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Implementation of `SymbolicSeparatrixHelper`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import sympy

from blond.core.backends.backend import backend
from blond.core.base import HasSymbolicHamiltonian
from blond.core.beam.beams import Beam
from blond.utilities.separatrix.helpers import _get_omega_min

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from matplotlib.lines import Line2D
    from numpy.typing import NDArray as NumpyArray
    from sympy import Expr

    from blond.core.beam.base import BeamBaseClass
    from blond.core.beam.particle_types import ParticleType
    from blond.core.simulation.simulation import Simulation


@dataclass(frozen=True)
class _CanonicalBucket:
    """
    One canonical RF bucket worth of geometry, exploited by periodicity.

    The full Hamiltonian decomposes as ``H = a*dE**2 + U(dt)`` and ``U(dt)``
    splits into a periodic part plus a linear acceleration tilt
    ``ref_E_change * dt``. ``U`` therefore repeats every ``period``, with
    each successive copy shifted up by ``shift_per_period``. One canonical
    unstable fixed point (UFP) is enough to reconstruct every bucket by
    integer offsets:

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

    #: Maximum absolute imaginary part for a complex root returned by
    #: :func:`sympy.nroots` to be treated as a real root of the
    #: ``K(dE) = H_sep - U(dt)`` polynomial.
    _ROOT_IMAG_TOLERANCE = 1e-9

    #: Lower bound for ``Re(root)`` to be treated as a non-negative real
    #: root despite numerical round-off; values down to ``-_ROOT_NEG_TOLERANCE``
    #: are clamped to zero.
    _ROOT_NEG_TOLERANCE = 1e-9

    def __init__(self, hamiltonian: sympy.Expr, omega_min: float):
        self._hamiltonian = hamiltonian
        self._omega_min = omega_min

    @classmethod
    def from_simulation(
        cls, simulation: Simulation, omega_min: float | None = None
    ) -> SymbolicSeparatrixHelper:
        """
        Instantiate `SymbolicSeparatrixHelper` from a `Simulation`.

        Parameters
        ----------
        simulation
            `Simulation` object that the separatrix is derived from.
        omega_min
            Minimum RF frequency in the `Ring`. The longest RF period
            ``2*pi/omega_min`` is used as the canonical scan window when
            locating unstable fixed points.

        Returns
        -------
        symbolic_separatrix_helper
            The initialized `SymbolicSeparatrixHelper`.
        """
        # todo make this consider the Baker–Campbell–Hausdorff formula
        #  See Wikipedia https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula
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
        return cls(
            hamiltonian=hamiltonian,
            omega_min=_get_omega_min(simulation.ring)
            if omega_min is None
            else float(omega_min),
        )

    def is_in_separatrix(
        self,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        particle_type: ParticleType,
        total_energy: float,
        intensity: float,
    ):
        r"""
        Test which phase-space points lie inside the separatrix.

        Evaluates the symbolic Hamiltonian at each ``(dt, dE)`` coordinate
        and compares it against the separatrix level ``H_sep`` of the
        bucket. The direction of the comparison depends on the transition
        state: above transition (``kinetic_coeff > 0``) the interior
        satisfies ``H < H_sep``, while below transition
        (``kinetic_coeff < 0``) it satisfies ``H > H_sep``. Using a fixed
        ``<`` comparison would misclassify the synchronous particle below
        transition and make ``BiGaussian`` reinsertion loop forever.

        Parameters
        ----------
        dt
            Time-deviation coordinates [s] of the points to test.
        dE
            Energy-deviation coordinates [eV] of the points to test, paired
            element-wise with ``dt``.
        particle_type
            Particle species, used to build the reference beam for symbol
            substitution.
        total_energy
            Reference total energy :math:`E_0` [eV] of the beam.
        intensity
            Beam intensity [number of charges], used for symbol
            substitution.

        Returns
        -------
        mask
            Boolean array, ``True`` where the corresponding ``(dt, dE)``
            point lies inside the separatrix. Same shape and device
            (NumPy or CuPy) as the inputs.
        """
        # todo optimize performance of this script

        # Use beam as a shortcut for defining what is
        # needed for `_substitute_symbols`.
        beam = Beam(
            intensity=intensity,
            particle_type=particle_type,
        )
        beam.reference.total_energy = total_energy
        kinetic_coeffs, potential = self._substitute_symbols(beam=beam)
        callable_hamiltonian = self._get_callable_hamiltonian(beam=beam)
        H_sep_per_dt = self._H_sep_per_dt(
            dt=np.array(
                [float(dt.mean())]
            ),  # FIXME, relies on mean dt being in a bucket..
            kinetic_coeffs=kinetic_coeffs,
            potential=potential,
        )
        # Whether ``H`` is below or above the separatrix level inside the
        # bucket depends on the sign of the kinetic coefficient ``c_2``
        # (i.e. on the transition state), exactly as in ``get_separatrix`` /
        # ``_H_sep_per_dt``. Above transition (``c_2 > 0``) the bounding UFPs
        # are maxima of ``U`` and the interior satisfies ``H < H_sep``; below
        # transition (``c_2 < 0``) the stable fixed point is a maximum of
        # ``U`` while the UFPs are minima, so the interior satisfies
        # ``H > H_sep``. A fixed ``<`` comparison would wrongly classify the
        # synchronous particle itself as outside below transition, making
        # ``BiGaussian`` reinsertion loop forever.
        kinetic_coeff = self._dE_squared_coefficient(kinetic_coeffs)
        ham_values = callable_hamiltonian(dt, dE)
        if kinetic_coeff < 0:
            mask = ham_values > backend.array(H_sep_per_dt)
        else:
            mask = ham_values < backend.array(H_sep_per_dt)
        return mask

    def get_separatrix(
        self,
        beam: BeamBaseClass,
        dt: NumpyArray,
    ) -> tuple[NumpyArray, NumpyArray]:
        r"""
        Compute the separatrix boundary in longitudinal phase space.

        Substitutes numerical beam values into the symbolic Hamiltonian,
        then solves ``H(dt, dE) = H_sep(dt)`` for ``dE`` at each ``dt``.
        Each RF bucket has its own ``H_sep``: the lower of the two bounding
        UFPs above transition (``kinetic_coeff > 0``), the higher one below transition
        (``kinetic_coeff < 0``). With non-zero ``reference_energy_change`` consecutive
        UFPs differ in height, so this per-dt treatment is required;
        without acceleration the result reduces to the familiar single
        ``H_sep`` answer.

        Parameters
        ----------
        beam
            Beam whose reference coordinates supply :math:`\beta`,
            :math:`\gamma`, :math:`E` and charge.
        dt
            Time-deviation grid [s] over which to evaluate the separatrix.

        Returns
        -------
        separatrix
            Array of shape ``(2, len(dt))``: row 0 the upper branch
            (``dE >= 0``), row 1 the lower branch (``dE <= 0``), in [eV].
            Entries are ``NaN`` for ``dt`` lying outside any RF bucket
            (above the local barrier, or no bucket exists at all because
            the linear tilt eliminated all extrema).
        """
        kinetic_coeffs, potential = self._substitute_symbols(beam=beam)
        H_sep = self._H_sep_per_dt(
            dt, kinetic_coeffs=kinetic_coeffs, potential=potential
        )
        dE_upper, dE_lower = self._dE_sep_branches(
            dt,
            kinetic_coeffs=kinetic_coeffs,
            potential=potential,
            H_sep=H_sep,
        )
        return np.stack([dE_upper, dE_lower])

    def plot_separatrix(
        self,
        beam: BeamBaseClass,
        dt: NumpyArray | None = None,
        **kwargs_plot,
    ) -> list[Line2D]:
        r"""
        Plot the longitudinal phase-space separatrix.

        Calls :meth:`get_separatrix` and draws both branches on the current
        matplotlib axes. The label (if given) is applied only to the upper
        branch so the legend shows a single entry.

        Parameters
        ----------
        beam
            Beam whose reference coordinates supply :math:`\beta`,
            :math:`\gamma`, :math:`E` and charge.
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
        if dt is None:
            s0, s1 = beam.dt_min, beam.dt_max
            r = s1 - s0
            dt = np.linspace(
                s0 - r, s1 + r, self._CANONICAL_SCAN_RESOLUTION + 1
            )

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
    ) -> tuple[tuple[float, ...], Callable[[NumpyArray], NumpyArray]]:
        r"""
        Substitute beam scalars into the Hamiltonian.

        The Hamiltonian decomposes as ``H(dt, dE) = K(dE) + U(dt)``: the
        kinetic part ``K`` is a polynomial in ``dE`` with coefficients
        independent of ``dt`` (degree 2 for the simple drift, degree
        ``2 + len(higher_order_alpha)`` for the exact drift), and the
        potential part ``U`` is a function of ``dt`` only.

        Parameters
        ----------
        beam
            Beam whose reference coordinates supply :math:`\beta`,
            :math:`\gamma`, :math:`E` and charge.

        Returns
        -------
        kinetic_coeffs
            Coefficients of ``K(dE)`` in descending degree order
            (``c_n, c_{n-1}, ..., c_2, c_1, c_0``). For physical
            Hamiltonians ``c_0`` and ``c_1`` are zero.
        potential
            Numpy-vectorized callable for ``H(dt, dE=0)``.
        """
        ham = self._substitute_beam_reference(beam)
        dt_sym, dE_sym = sympy.symbols("dt dE", real=True)
        U_expr = ham.subs(dE_sym, 0)
        K_expr = sympy.expand(ham - U_expr)
        if K_expr == 0:
            kinetic_coeffs: tuple[float, ...] = (0.0,)
        else:
            K_poly = sympy.Poly(K_expr, dE_sym)
            kinetic_coeffs = tuple(float(c) for c in K_poly.all_coeffs())
        u_lambda = sympy.lambdify(dt_sym, U_expr, modules="numpy")

        def potential(dt: NumpyArray) -> NumpyArray:
            # ``sympy.lambdify`` collapses a constant ``U_expr`` (e.g.
            # ``voltage=0`` and no acceleration tilt) to a scalar-valued
            # callable. Broadcast so downstream code can rely on an
            # array of the same shape as ``dt``.
            return np.broadcast_to(
                np.asarray(u_lambda(dt), dtype=float),
                np.shape(np.asarray(dt)),
            )

        return kinetic_coeffs, potential

    def _get_callable_hamiltonian(
        self, beam: BeamBaseClass
    ) -> tuple[
        tuple[float, ...], Callable[[NumpyArray, NumpyArray], NumpyArray]
    ]:
        r"""
        Substitute beam scalars into the Hamiltonian.

        The Hamiltonian decomposes as ``H(dt, dE) = K(dE) + U(dt)``: the
        kinetic part ``K`` is a polynomial in ``dE`` with coefficients
        independent of ``dt`` (degree 2 for the simple drift, degree
        ``2 + len(higher_order_alpha)`` for the exact drift), and the
        potential part ``U`` is a function of ``dt`` only.

        Parameters
        ----------
        beam
            Beam whose reference coordinates supply :math:`\beta`,
            :math:`\gamma`, :math:`E` and charge.

        Returns
        -------
        kinetic_coeffs
            Coefficients of ``K(dE)`` in descending degree order
            (``c_n, c_{n-1}, ..., c_2, c_1, c_0``). For physical
            Hamiltonians ``c_0`` and ``c_1`` are zero.
        potential
            Numpy-vectorized callable for ``H(dt, dE=0)``.
        """
        ham = self._substitute_beam_reference(beam)

        dt_sym, dE_sym = sympy.symbols("dt dE", real=True)
        ham_lambda = sympy.lambdify((dt_sym, dE_sym), ham, modules="numpy")

        def ham_callable(
            dt: NumpyArray | CupyArray, dE: NumpyArray | CupyArray
        ) -> NumpyArray | CupyArray:
            # ``sympy.lambdify`` collapses a constant ``U_expr`` (e.g.
            # ``voltage=0`` and no acceleration tilt) to a scalar-valued
            # callable. Broadcast so downstream code can rely on an
            # array of the same shape as ``dt``. Dispatch on the input's
            # array module so the result stays on the same device (a plain
            # ``np.asarray`` would raise on a CuPy array).
            return backend.broadcast_to(
                backend.asarray(ham_lambda(dt, dE), dtype=float),
                backend.asarray(dt).shape,
            )

        return ham_callable

    def _substitute_beam_reference(self, beam: BeamBaseClass) -> Expr:
        beta_sym, gamma_sym, E_sym, q_sym = sympy.symbols(
            "beta gamma E q", real=True
        )
        ham = self._hamiltonian.subs(
            {
                beta_sym: float(beam.reference.beta),
                gamma_sym: float(beam.reference.gamma),
                E_sym: float(beam.reference.total_energy),
                q_sym: float(beam.particle_type.charge),
            }
        )
        return ham

    @classmethod
    def _dE_sep_branches(
        cls,
        dt: NumpyArray,
        kinetic_coeffs: tuple[float, ...],
        potential: Callable[[NumpyArray], NumpyArray],
        H_sep: NumpyArray,
    ) -> tuple[NumpyArray, NumpyArray]:
        """
        Solve ``K(dE) + U(dt) = H_sep(dt)`` for both separatrix branches.

        ``K(dE)`` is the full polynomial in ``dE`` -- degree 2 for the
        simple drift, ``2 + len(higher_order_alpha)`` for the exact drift.
        For each ``dt`` we adjust the constant term to ``-(H_sep - U(dt))``,
        call :func:`numpy.roots` (a LAPACK companion-matrix eigenvalue
        solver, far faster than :func:`sympy.nroots` while returning
        identical answers for purely numerical coefficients), and pick:

        * the **smallest non-negative** real root as the upper branch
          (``dE >= 0``);
        * the **largest non-positive** real root as the lower branch
          (``dE <= 0``).

        When ``K`` is symmetric (``DriftSimple`` or ``DriftExact`` with
        ``alpha_0 != 0`` and small higher-order alphas), the lower-branch
        root is just ``-dE_upper``. When ``K`` carries odd-degree terms
        -- e.g., ``DriftExact`` with ``alpha_0 = 0`` and non-trivial
        ``higher_order_alpha``, where a ``dE**3`` coefficient appears --
        the polynomial is asymmetric and the two branches must be
        computed independently. The earlier ``np.stack([dE, -dE])``
        mirror produced a visibly wrong lower branch in that regime.

        Parameters
        ----------
        dt
            Time-deviation grid [s].
        kinetic_coeffs
            Coefficients of ``K(dE)`` in descending degree order, as
            returned by :meth:`_substitute_symbols`.
        potential
            Numpy-vectorized callable for ``H(dt, dE=0)``.
        H_sep
            Per-dt Hamiltonian value on the bucket separatrix.

        Returns
        -------
        dE_upper
            Upper-branch separatrix energy at each ``dt``. ``NaN`` where
            ``H_sep`` is ``NaN`` or no non-negative real root exists at
            this ``dt`` (particle above the local barrier).
        dE_lower
            Lower-branch separatrix energy at each ``dt``. ``NaN`` under
            the analogous conditions on the non-positive side.
        """
        f_values = np.asarray(potential(dt), dtype=float)
        dE_upper = np.full(dt.shape, np.nan, dtype=float)
        dE_lower = np.full(dt.shape, np.nan, dtype=float)
        if len(kinetic_coeffs) < 2:  # NOQA PLR2004
            return dE_upper, dE_lower

        base_coeffs = np.asarray(kinetic_coeffs, dtype=float)
        rhs_values = H_sep - f_values
        for i in range(dt.shape[0]):
            rhs = rhs_values[i]
            if not np.isfinite(rhs):
                continue
            coeffs = base_coeffs.copy()
            coeffs[-1] -= rhs
            try:
                roots = np.roots(coeffs)
            except (np.linalg.LinAlgError, ValueError):
                continue
            real_parts = roots[
                np.abs(roots.imag) < cls._ROOT_IMAG_TOLERANCE
            ].real
            non_neg = real_parts[real_parts >= -cls._ROOT_NEG_TOLERANCE]
            non_pos = real_parts[real_parts <= cls._ROOT_NEG_TOLERANCE]
            if non_neg.size:
                dE_upper[i] = max(0.0, float(non_neg.min()))
            if non_pos.size:
                dE_lower[i] = min(0.0, float(non_pos.max()))
        return dE_upper, dE_lower

    def _H_sep_per_dt(
        self,
        dt: NumpyArray,
        kinetic_coeffs: tuple[float, ...],
        potential: Callable[[NumpyArray], NumpyArray],
    ) -> NumpyArray:
        """
        Compute the per-``dt`` Hamiltonian value on the bucket separatrix.

        Locates the canonical UFP within one RF period and extrapolates
        bucket boundaries by integer offsets of ``period`` (in dt) and
        ``shift_per_period`` (in potential). Only ``sign(c_2)`` is used
        from ``kinetic_coeffs`` -- see :meth:`_find_canonical_bucket` for
        why higher-order ``K(dE)`` terms cannot enter the UFP topology.

        Parameters
        ----------
        dt
            Time-deviation grid [s].
        kinetic_coeffs
            Coefficients of ``K(dE)`` in descending degree order, as
            returned by :meth:`_substitute_symbols`.
        potential
            Numpy-vectorized callable for ``H(dt, dE=0)``.

        Returns
        -------
        H_sep
            Per-dt separatrix Hamiltonian value. ``NaN`` when no interior
            extremum exists (the linear tilt is so strong that the
            potential is monotonic over a period) or when ``K(dE)`` lacks
            a ``dE**2`` term.
        """
        kinetic_coeff = self._dE_squared_coefficient(kinetic_coeffs)
        bucket = (
            None
            if kinetic_coeff == 0.0
            else self._find_canonical_bucket(
                period_start=float(np.min(dt)),
                kinetic_coeff=kinetic_coeff,
                potential=potential,
            )
        )
        if bucket is None:
            return np.full(dt.shape, np.nan, dtype=float)

        bucket_index = self._bucket_index(dt, bucket)
        potential_left = (
            bucket.ufp_potential + bucket_index * bucket.shift_per_period
        )
        potential_right = potential_left + bucket.shift_per_period
        # The two bounding UFPs of bucket ``n`` differ by ``shift_per_period``
        # under acceleration. The separatrix is the H-level set anchored at
        # the *lower* UFP for ``c_2 > 0`` (UFPs are local maxima of U) and
        # at the *higher* UFP for ``c_2 < 0`` (UFPs are local minima).
        bounding_potentials = np.minimum if kinetic_coeff > 0 else np.maximum
        return bounding_potentials(potential_left, potential_right)

    @staticmethod
    def _dE_squared_coefficient(kinetic_coeffs: tuple[float, ...]) -> float:
        """
        Return ``c_2`` from a descending-degree coefficient tuple.

        ``kinetic_coeffs`` is ordered ``(c_n, c_{n-1}, ..., c_2, c_1, c_0)``
        so the ``dE**2`` entry sits at index ``len-3``. Returns ``0.0``
        when the polynomial is degree < 2 (degenerate ``H = U(dt)``).

        Parameters
        ----------
        kinetic_coeffs
            Coefficients of ``K(dE)`` in descending degree order
            (``c_n, c_{n-1}, ..., c_2, c_1, c_0``). For physical
            Hamiltonians ``c_0`` and ``c_1`` are zero.

        Returns
        -------
        kinetic_coeff
            ``c_2`` from a descending-degree coefficient tuple.
        """
        return float(kinetic_coeffs[-3]) if len(kinetic_coeffs) >= 3 else 0.0  # NOQA PLR2004

    def _find_canonical_bucket(
        self,
        period_start: float,
        kinetic_coeff: float,
        potential: Callable[[NumpyArray], NumpyArray],
    ) -> _CanonicalBucket | None:
        """
        Locate one canonical UFP inside ``[period_start, period_start + period]``.

        For multi-harmonic potentials with several extrema per period, the
        canonical UFP is the highest local max (``a > 0``) or lowest local
        min (``a < 0``) -- i.e., the outer-separatrix barrier; sub-bucket
        structure is not represented.

        Why only the sign of ``kinetic_coeff`` matters

        ``H = K(dE) + U(dt)`` with ``K(dE) = c_2 dE**2 + c_3 dE**3 + ...``.
        A fixed point satisfies ``K'(dE) = 0`` and ``U'(dt) = 0``; since
        ``K`` starts at ``dE**2`` the first condition pins ``dE = 0``, so
        every fixed point sits on ``dE = 0`` at an extremum of ``U``. The
        Hessian at such a point is diagonal,

            d2H/dE2 = 2 c_2,         d2H/dE ddt = 0,
            d2H/dt2 = U''(dt_ext),

        and a saddle (UFP) requires the two diagonal entries to have
        opposite signs:

        ===============   ==========================   =====================
        ``sign(c_2)``     UFP location                 stable-phase location
        ===============   ==========================   =====================
        ``> 0``           local **max** of ``U(dt)``   local **min** of ``U``
        ``< 0``           local **min** of ``U(dt)``   local **max** of ``U``
        ===============   ==========================   =====================

        Every higher derivative of ``K`` carries an extra factor of ``dE``
        and so vanishes at the UFP -- ``c_3, c_4, ...`` shape the
        separatrix curve in :meth:`_dE_sep_branches` but cannot change
        which extremum of ``U`` is the UFP. One bit (``sign(c_2)``) is
        the irreducible minimum of ``K``-information needed here.

        Parameters
        ----------
        period_start
            Left edge of the canonical scan window [s].
        kinetic_coeff
            Kinetic coefficient ``coeff(dE, 2)`` of the Hamiltonian.
            Only its sign is used.
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

        extremum_indices = self._interior_extrema(
            scan_potential, kinetic_coeff=kinetic_coeff
        )
        if extremum_indices.size == 0:
            return None

        if kinetic_coeff > 0:
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
    def _interior_extrema(
        values: NumpyArray, kinetic_coeff: float
    ) -> NumpyArray:
        """
        Find indices of interior local maxima (``a > 0``) or minima (``a < 0``).

        Parameters
        ----------
        values
            1-D array sampled along the dt-axis.
        kinetic_coeff
            Kinetic coefficient ``coeff(dE, 2)`` -- its sign decides whether
            UFPs are local maxima or local minima.

        Returns
        -------
        indices
            Indices into ``values`` of interior extrema (boundaries
            excluded).
        """
        slope = np.diff(values)
        if kinetic_coeff > 0:
            is_extremum = (slope[:-1] > 0) & (slope[1:] < 0)
        else:
            is_extremum = (slope[:-1] < 0) & (slope[1:] > 0)
        return np.flatnonzero(is_extremum) + 1

    def _bucket_index(
        self, dt: NumpyArray, bucket: _CanonicalBucket
    ) -> NumpyArray:
        """
        Compute the bucket index ``n`` containing each ``dt``.

        Bucket ``n`` is the half-open interval ``[UFP_n, UFP_{n+1})``,
        so a UFP belongs to the bucket on its right. The ratio
        ``(dt - ufp_dt) / period`` is shifted by
        :attr:`_BUCKET_BOUNDARY_TOLERANCE` before flooring, so a sample
        sitting on a UFP up to roundoff is assigned to that right-hand
        bucket rather than to the one ending at it.

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
            ``[UFP_n, UFP_{n+1})``.
        """
        ratio = (dt - bucket.ufp_dt) / bucket.period
        return np.floor(ratio + self._BUCKET_BOUNDARY_TOLERANCE)
