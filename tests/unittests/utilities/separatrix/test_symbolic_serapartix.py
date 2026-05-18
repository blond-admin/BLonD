# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/
import unittest
from unittest.mock import Mock

import numpy as np
import pytest
import sympy
from matplotlib import pyplot as plt

from blond import (
    Beam,
    DriftSimple,
    MagneticCyclePerTurn,
    MultiHarmonicRFStation,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    proton,
)
from blond.handle_results.helpers import callers_relative_path
from blond.physics.drifts import DriftExact
from blond.testing.backend_testing import multi_backend_testcase
from blond.testing.helpers import allclose_tolerances
from blond.utilities.separatrix.symbolic_serapartix import (
    SymbolicSeparatrixHelper,
)


class TestSymbolicSeparatrixHelper:
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @multi_backend_testcase("Cupy64Bit", "Numpy64Bit")
    @pytest.mark.backend_mutation
    def test_integration(self):
        DEV_DRAW = False
        ring = Ring(26658.883)

        rf_station1 = MultiHarmonicRFStation(
            section_index=0, n_harmonics=2, main_harmonic_idx=0
        )
        base_harmonic = 35640
        rf_station1.harmonic = np.array([base_harmonic, 4 * base_harmonic])
        rf_station1.voltage = np.array([6e6, 6e6 / 2])
        rf_station1.phi_rf_design = np.array([0, 0])
        N_TURNS = int(1e3)

        energy_cycle = MagneticCyclePerTurn.init_from_linspace(
            values=np.linspace(450e9, 451e9, N_TURNS + 1),
            reference_particle=proton,
        )

        drift1 = DriftSimple(
            section_index=0,
            orbit_length=ring.circumference / 2,
        )
        drift1.momentum_compaction_factor = momentum_compaction_factor(
            transition_gamma=55.759505
        )

        drift2 = DriftExact(
            orbit_length=ring.circumference / 2,
            section_index=1,
            momentum_compaction_factor=drift1.momentum_compaction_factor,
            higher_order_alpha=np.array(
                [drift1.alpha_0 * 2, drift1.alpha_0 * (-3)]
            ),
        )

        rf_station2 = SingleHarmonicRFStation(
            section_index=1,
            harmonic=base_harmonic,
            voltage=6e6,
            phi_rf=np.deg2rad(20),
        )

        ring.add_elements(
            (drift1, rf_station1), deepcopy=True, section_index=0
        )
        ring.add_elements(
            (drift2, rf_station2), deepcopy=True, section_index=1
        )

        sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)

        t_rf = sim.get_t_rev_init() / base_harmonic

        beam1 = Beam.simple_gaussian(
            n_macroparticles=int(1e5),
            intensity=1e9,
            particle_type=proton,
            dt_scale=t_rf / 4,
            dE_scale=0.1e9 / 2,
            dt_offset=t_rf / 2,
            seed=1,
        )
        t0 = beam1.dt.min()
        t1 = beam1.dt.max()
        r = t1 - t0
        trange0_ = (t0 - 2 * r, t1 + 2 * r)
        plt.figure("Dynamic beam")
        plt.xlim(trange0_)

        def custom_action(simulation: Simulation, beam: Beam):
            plt.figure("Dynamic beam")

            dt = beam.read_partial_dt()
            beam.plot_scatter()
            separatrix_dE = SymbolicSeparatrixHelper.from_simulation(
                simulation=sim
            ).get_separatrix(
                beam=beam,
                dt=np.linspace(*trange0_, 1000),
            )
            if simulation.turn_i.value == 0:
                separatrix_dE_pinned = np.loadtxt(
                    callers_relative_path(
                        "resources/separatrix_dE_pinned.txt", stacklevel=1
                    ),
                )
                np.testing.assert_allclose(
                    separatrix_dE,
                    separatrix_dE_pinned,
                    **allclose_tolerances(separatrix_dE_pinned),
                )
            if DEV_DRAW:
                sim.plot_separatrix(
                    beam=beam,
                    dt=np.linspace(*trange0_, 1000),
                )
                plt.xlim(trange0_)
                plt.ylim(-2e9, 2e9)

                plt.draw()
                plt.pause(0.1)
                plt.cla()

        custom_action.each_turn_i = 10
        sim.run_simulation(
            beams=(beam1,),
            n_turns=N_TURNS if DEV_DRAW else 1,
            callbacks=custom_action,
        )


class TestSymbolicSeparatrixInternals(unittest.TestCase):
    """Cover edge-case branches of the private helpers."""

    OMEGA_MIN = 2.0 * np.pi  # canonical period of 1 s

    def _helper(self) -> SymbolicSeparatrixHelper:
        return SymbolicSeparatrixHelper(
            hamiltonian=sympy.Integer(0), omega_min=self.OMEGA_MIN
        )

    def test_interior_extrema_negative_a_finds_local_minima(self):
        values = np.array([1.0, 0.0, 1.0])
        idx = SymbolicSeparatrixHelper._interior_extrema(
            values, kinetic_coeff=-1.0
        )
        np.testing.assert_array_equal(idx, np.array([1]))

    def test_interior_extrema_negative_a_ignores_local_maxima(self):
        values = np.array([0.0, 1.0, 0.0])
        idx = SymbolicSeparatrixHelper._interior_extrema(
            values, kinetic_coeff=-1.0
        )
        self.assertTrue(idx.size == 0)

    def test_find_canonical_bucket_no_extremum_returns_none(self):
        helper = self._helper()
        bucket = helper._find_canonical_bucket(
            period_start=0.0,
            kinetic_coeff=1.0,
            potential=lambda dt: np.asarray(dt, dtype=float),
        )
        self.assertTrue(bucket is None)

    def test_find_canonical_bucket_negative_a_picks_local_minimum(self):
        helper = self._helper()

        def potential(dt):
            return np.sin(2 * np.pi * dt) - 0.1 * dt

        bucket = helper._find_canonical_bucket(
            period_start=0.0, kinetic_coeff=-1.0, potential=potential
        )
        self.assertTrue(bucket is not None)
        self.assertTrue(0.7 < bucket.ufp_dt < 0.8)
        np.testing.assert_allclose(bucket.ufp_potential, -1.075, atol=0.01)
        np.testing.assert_allclose(bucket.shift_per_period, -0.1, atol=1e-9)

    def test_H_sep_per_dt_zero_a_returns_all_nan(self):
        helper = self._helper()
        dt = np.linspace(0.0, 1.0, 11)
        H_sep = helper._H_sep_per_dt(
            dt,
            kinetic_coeffs=(0.0, 0.0, 0.0),
            potential=lambda x: np.cos(2 * np.pi * x),
        )
        self.assertTrue(H_sep.shape == dt.shape)
        self.assertTrue(np.all(np.isnan(H_sep)))

    def test_H_sep_per_dt_no_canonical_bucket_returns_all_nan(self):
        helper = self._helper()
        dt = np.linspace(0.0, 1.0, 11)
        H_sep = helper._H_sep_per_dt(
            dt,
            kinetic_coeffs=(1.0, 0.0, 0.0),
            potential=lambda x: np.asarray(x, dtype=float),
        )
        self.assertTrue(np.all(np.isnan(H_sep)))

    def test_H_sep_per_dt_negative_a_uses_maximum_branch(self):
        helper = self._helper()

        def potential(dt):
            return np.sin(2 * np.pi * dt) - 0.1 * dt

        dt = np.array([0.6, 0.9, 1.2])
        H_sep = helper._H_sep_per_dt(
            dt, kinetic_coeffs=(-1.0, 0.0, 0.0), potential=potential
        )

        bucket = helper._find_canonical_bucket(
            period_start=float(np.min(dt)),
            kinetic_coeff=-1.0,
            potential=potential,
        )
        self.assertTrue(bucket is not None)
        bucket_index = helper._bucket_index(dt, bucket)
        left = bucket.ufp_potential + bucket_index * bucket.shift_per_period
        right = left + bucket.shift_per_period
        # Sanity: shift_per_period != 0 so np.maximum and np.minimum diverge.
        self.assertTrue(not np.allclose(left, right))
        np.testing.assert_allclose(H_sep, np.maximum(left, right))


class TestSubstituteSymbols(unittest.TestCase):
    """Cover `SymbolicSeparatrixHelper._substitute_symbols`."""

    def _beam(self) -> Mock:
        beam = Mock()
        beam.reference.beta = 1.0
        beam.reference.gamma = 1.0
        beam.reference.total_energy = 1.0
        beam.particle_type.charge = 1.0
        return beam

    def test_returns_full_polynomial_coefficients(self):
        """`kinetic_coeffs` carries every ``dE**k`` term, not just dE**2.

        Prevents regressing to the old behavior, where higher-order
        ``dE**k`` contributions from ``DriftExact`` were silently
        discarded.
        """
        dt_sym, dE_sym = sympy.symbols("dt dE", real=True)
        # K(dE) has degree-4 with non-trivial dE**3 (asymmetric).
        hamiltonian = (
            3.0 * dE_sym**4
            + 5.0 * dE_sym**3
            + 7.0 * dE_sym**2
            + sympy.cos(dt_sym)
        )
        helper = SymbolicSeparatrixHelper(
            hamiltonian=hamiltonian,
            omega_min=2 * np.pi,
        )
        kinetic_coeffs, potential = helper._substitute_symbols(
            beam=self._beam(),
        )
        # Descending-degree, with c_1 = c_0 = 0 since U(dt) was split off.
        np.testing.assert_allclose(
            np.asarray(kinetic_coeffs),
            np.array([3.0, 5.0, 7.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(
            potential(np.array([0.0, np.pi])),
            np.array([1.0, -1.0]),
        )

    def test_pure_dE_squared_hamiltonian_keeps_trailing_zeros(self):
        """``DriftSimple``-style ``c * dE**2`` returns ``(c, 0, 0)``."""
        dt_sym, dE_sym = sympy.symbols("dt dE", real=True)
        hamiltonian = 2.5 * dE_sym**2 + sympy.sin(dt_sym)
        helper = SymbolicSeparatrixHelper(
            hamiltonian=hamiltonian,
            omega_min=2 * np.pi,
        )
        kinetic_coeffs, _ = helper._substitute_symbols(beam=self._beam())
        np.testing.assert_allclose(
            np.asarray(kinetic_coeffs),
            np.array([2.5, 0.0, 0.0]),
        )

    def test_no_kinetic_part_returns_single_zero(self):
        """Degenerate ``H = U(dt)`` only -- ``kinetic_coeffs = (0.0,)``."""
        dt_sym = sympy.symbols("dt", real=True)
        helper = SymbolicSeparatrixHelper(
            hamiltonian=sympy.cos(dt_sym),
            omega_min=2 * np.pi,
        )
        kinetic_coeffs, _ = helper._substitute_symbols(beam=self._beam())
        self.assertEqual(kinetic_coeffs, (0.0,))


class TestDESepBranches(unittest.TestCase):
    """
    Cover `SymbolicSeparatrixHelper._dE_sep_branches`.

    The upper and lower branches must be solved independently from the
    polynomial roots; mirroring ``-dE_upper`` is only correct when
    ``K(dE)`` is even in ``dE``. Asymmetric ``K`` arises e.g. from
    ``DriftExact`` with ``alpha_0 = 0`` and non-trivial
    ``higher_order_alpha``.
    """

    def _helper(self) -> SymbolicSeparatrixHelper:
        return SymbolicSeparatrixHelper(
            hamiltonian=sympy.Integer(0),
            omega_min=2.0 * np.pi,
        )

    def test_symmetric_K_gives_mirrored_branches(self):
        """``K = c2 dE**2`` -> ``dE_lower = -dE_upper`` to machine eps."""
        dt = np.array([0.0, 0.5])
        # K(dE) = 1.0 * dE**2; rhs = H_sep - U; here H_sep = 4, U = 0.
        kinetic_coeffs = (1.0, 0.0, 0.0)
        upper, lower = SymbolicSeparatrixHelper._dE_sep_branches(
            dt,
            kinetic_coeffs=kinetic_coeffs,
            potential=lambda x: np.zeros_like(x),
            H_sep=np.array([4.0, 9.0]),
        )
        np.testing.assert_allclose(upper, np.array([2.0, 3.0]))
        np.testing.assert_allclose(lower, np.array([-2.0, -3.0]))

    def test_asymmetric_K_gives_non_mirrored_branches(self):
        """
        ``K = dE**2 - 0.1 * dE**3`` is asymmetric.

        For ``rhs = 1`` the inner roots are ``dE_upper ≈ +1.06`` and
        ``dE_lower ≈ -0.95`` (they would coincide at ``±1`` if the cubic
        term were dropped) -- |upper| > |lower| precisely because the
        cubic shifts the asymmetric polynomial to the right.
        """
        kinetic_coeffs = (-0.1, 1.0, 0.0, 0.0)  # -0.1 dE**3 + 1.0 dE**2
        upper, lower = SymbolicSeparatrixHelper._dE_sep_branches(
            np.array([0.0]),
            kinetic_coeffs=kinetic_coeffs,
            potential=lambda x: np.zeros_like(x),
            H_sep=np.array([1.0]),
        )
        # Cross-check the upper root analytically with numpy.roots.
        roots = np.roots([-0.1, 1.0, 0.0, -1.0])
        real_roots = sorted(roots[np.abs(roots.imag) < 1e-9].real)
        # real_roots: [neg_root, pos_root_inner, pos_root_outer]
        self.assertEqual(len(real_roots), 3)
        np.testing.assert_allclose(upper[0], real_roots[1], rtol=1e-9)
        np.testing.assert_allclose(lower[0], real_roots[0], rtol=1e-9)
        self.assertGreater(abs(upper[0]), abs(lower[0]))

    def test_picks_smallest_non_negative_root_for_upper(self):
        """
        Two positive real roots -- the inner separatrix is the smaller one.

        Polynomial: ``(dE-1)(dE-3) = dE**2 - 4 dE + 3``. Roots at +1 and
        +3; the upper branch must be +1, not +3.
        """
        kinetic_coeffs = (1.0, -4.0, 0.0)  # already includes "const term"
        upper, lower = SymbolicSeparatrixHelper._dE_sep_branches(
            np.array([0.0]),
            kinetic_coeffs=kinetic_coeffs,
            potential=lambda x: np.zeros_like(x),
            H_sep=np.array([-3.0]),  # rhs = -3 -> coeffs[-1] becomes +3
        )
        np.testing.assert_allclose(upper[0], 1.0, atol=1e-10)
        # No non-positive real root for (dE-1)(dE-3) -> NaN.
        self.assertTrue(np.isnan(lower[0]))

    def test_nan_H_sep_propagates_to_nan_branches(self):
        kinetic_coeffs = (1.0, 0.0, 0.0)
        upper, lower = SymbolicSeparatrixHelper._dE_sep_branches(
            np.array([0.0, 1.0]),
            kinetic_coeffs=kinetic_coeffs,
            potential=lambda x: np.zeros_like(x),
            H_sep=np.array([4.0, np.nan]),
        )
        np.testing.assert_allclose(upper[0], 2.0)
        np.testing.assert_allclose(lower[0], -2.0)
        self.assertTrue(np.isnan(upper[1]))
        self.assertTrue(np.isnan(lower[1]))

    def test_no_real_root_marks_branch_nan(self):
        """``K = dE**2 = -1`` has no real roots -> both branches NaN."""
        kinetic_coeffs = (1.0, 0.0, 0.0)
        upper, lower = SymbolicSeparatrixHelper._dE_sep_branches(
            np.array([0.0]),
            kinetic_coeffs=kinetic_coeffs,
            potential=lambda x: np.zeros_like(x),
            H_sep=np.array([-1.0]),
        )
        self.assertTrue(np.isnan(upper[0]))
        self.assertTrue(np.isnan(lower[0]))


class TestGetSeparatrixAsymmetricDriftExact(unittest.TestCase):
    """
    End-to-end check that ``DriftExact`` with ``alpha_0 = 0`` and
    non-trivial ``higher_order_alpha`` produces an asymmetric
    separatrix.

    Reproduces a scenario reported by a user where, with the old
    ``np.stack([dE_sep, -dE_sep])`` mirror, particles inside the real
    bucket appeared to cross the drawn lower-branch separatrix because
    ``K(dE)`` carried a non-zero ``dE**3`` coefficient driven by
    ``alpha_1``. After the fix the lower-branch ``|dE|`` is materially
    larger than the upper-branch ``|dE|`` at the stable phase.
    """

    def test_alpha_0_zero_gives_asymmetric_branches(self):
        from blond import (
            Beam,
            BiGaussian,
            MagneticCyclePerTurn,
            Ring,
            SingleHarmonicRFStation,
            momentum_compaction_factor,
            proton,
        )
        from blond.physics.drifts import DriftExact

        ring = Ring(26658.883)
        rf_station = SingleHarmonicRFStation(
            harmonic=35640,
            voltage=6e6,
            phi_rf=0.0,
        )
        energy_cycle = MagneticCyclePerTurn.init_from_linspace(
            values=np.linspace(450e9, 450e9, 2),
            reference_particle=proton,
        )
        drift = DriftExact(
            orbit_length=26658.883,
            momentum_compaction_factor=momentum_compaction_factor(
                transition_gamma=55.759505,
            ),
            higher_order_alpha=np.array([-3.2163e-4, -3.2163e-4]),
        )
        # Set alpha_0 to zero -- below transition only via -1/gamma**2,
        # so eta is tiny and the higher-order alpha contributions to
        # K(dE) become a significant fraction of the dE^2 term.
        drift.momentum_compaction_factor *= 0
        beam = Beam(intensity=1e9, particle_type=proton)
        ring.add_elements((drift, rf_station))
        sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
        sim.prepare_beam(
            beam=beam,
            preparation_routine=BiGaussian(
                sigma_dt=1e-10,
                sigma_dE=1e8,
                reinsertion=False,
                seed=1,
                n_macroparticles=10,
            ),
        )

        helper = SymbolicSeparatrixHelper.from_simulation(simulation=sim)

        # Sanity check: K(dE) has a non-zero dE^3 coefficient -- this is
        # the root cause of the asymmetry that the old mirror missed.
        kinetic_coeffs, _ = helper._substitute_symbols(beam=beam)
        degree = len(kinetic_coeffs) - 1
        self.assertGreaterEqual(degree, 3)
        c3 = kinetic_coeffs[degree - 3]
        self.assertNotEqual(c3, 0.0)

        omega = float(rf_station.omega_rf_design)
        t_rf = 2.0 * np.pi / omega
        # Sample inside the bucket containing the stable phase at
        # dt = t_rf (below-transition stable phase for phi_rf = 0).
        dt = np.linspace(0.6 * t_rf, 1.4 * t_rf, 401)
        sep = helper.get_separatrix(beam=beam, dt=dt)
        upper, lower = sep[0], sep[1]

        # Strip NaN regions outside the bucket.
        finite = np.isfinite(upper) & np.isfinite(lower)
        self.assertTrue(finite.any(), "separatrix is entirely NaN")
        max_upper = np.nanmax(upper[finite])
        max_abs_lower = np.nanmax(np.abs(lower[finite]))

        # The lower branch is materially larger in magnitude than the
        # upper branch (~63% asymmetry for these parameters); the old
        # mirror code would force |lower| == |upper| identically.
        self.assertGreater(max_abs_lower, 1.2 * max_upper)
        # And the two branches are NOT pointwise mirror images.
        self.assertGreater(
            np.nanmax(np.abs(upper[finite] + lower[finite])), 1e7
        )


class TestSymbolicSeparatrixHelperFromSimulation(unittest.TestCase):
    """Cover `SymbolicSeparatrixHelper.from_simulation`."""

    def test_raises_when_no_symbolic_hamiltonian_elements(self):
        simulation = Mock()
        simulation.ring.elements.get_elements.return_value = []

        with self.assertRaisesRegex(
            ValueError,
            "No elements with `HasSymbolicHamiltonian` found.",
        ):
            SymbolicSeparatrixHelper.from_simulation(simulation=simulation)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
