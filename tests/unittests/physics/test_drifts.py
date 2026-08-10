import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import sympy
from scipy.constants import c
from scipy.constants import speed_of_light as c0

from blond import Simulation, momentum_compaction_factor
from blond.core.backends.backend import Numpy64Bit, backend
from blond.core.base import DynamicParameter
from blond.core.beam.base import BeamBaseClass
from blond.core.beam.beams import ProbeBeam
from blond.core.beam.particle_types import lead_82
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.physics.drifts import DriftBaseClass, DriftExact, DriftSimple
from blond.testing.backend_testing import multi_backend_testcase


class DriftBaseClassHelper(DriftBaseClass):
    def track_reference(self, reference: ReferenceCoordinates, **kwargs):
        pass

    def eta_0(self, gamma: float) -> backend.float:
        pass

    def _track(self, beam: BeamBaseClass) -> None:
        pass


class TestDriftBaseClass(unittest.TestCase):
    def setUp(self):
        self.drift_base_class = DriftBaseClassHelper(
            orbit_length=123, section_index=0
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_on_init_simulation(self):
        simulation = Mock(Simulation)
        self.drift_base_class.on_init_simulation(simulation=simulation)

    def test_on_run_simulation(self):
        simulation = Mock(Simulation)
        self.drift_base_class.on_run_simulation(
            simulation=simulation,
            n_turns=11,
            beam=Mock(BeamBaseClass),
        )

    def test_orbit_length(self):
        self.assertEqual(123, self.drift_base_class.orbit_length)

    def test_radiation_integrals(self):
        self.assertIsNone(self.drift_base_class.radiation_integrals)

        radiation_integrals = np.array(
            [
                0.646747216157,
                0.0005936549319,
                5.6814536525e-08,
                5.92870407301e-09,
                1.71368060083e-11,
            ]
        )
        drift_base_class = DriftBaseClassHelper(
            orbit_length=123,
            section_index=0,
            radiation_integrals=radiation_integrals,
        )

        np.testing.assert_equal(
            drift_base_class.radiation_integrals,
            radiation_integrals,
        )


class TestDriftSimple(unittest.TestCase):
    def setUp(self):
        self.gamma = 2.5
        self.drift_simple = DriftSimple.headless(
            momentum_compaction_factor=20.0,  # highly relativistic
            orbit_length=0.25 * 25,
            section_index=0,
        )

    def test_setters2(self):
        drift_simple = DriftSimple(
            momentum_compaction_factor=20.0,  # highly relativistic
            orbit_length=0.25 * 25,
            section_index=0,
        )
        drift_simple.momentum_compaction_factor = 1.0
        drift_simple.momentum_compaction_factor = -1.0

    def test_array_setup(self):
        self.drift_simple = DriftSimple.headless(
            momentum_compaction_factor=momentum_compaction_factor(
                np.array([20.0])
            ),  # highly relativistic
            orbit_length=0.25 * 25,
            section_index=0,
            # array input is scheduled, so a live turn_counter is required
            turn_counter=DynamicParameter(value_init=0),
        )

        beam = Mock(BeamBaseClass)
        beam.reference = Mock()
        beam.common_array_size = 1
        beam.reference.time = 0.0
        beam.reference.gamma = 1.0
        beam.reference.velocity = 0.5
        beam.reference.beta = 0.1
        beam.reference.total_energy = 1.0
        beam.write_partial_dt.return_value = backend.ones(
            10, dtype=backend.float
        )
        beam.read_partial_dE.return_value = backend.zeros(
            10, dtype=backend.float
        )
        self.drift_simple.track(beam=beam)

    def test_error_throwing_on_unscheduled(self):
        from types import SimpleNamespace

        simulation = Mock(Simulation)
        simulation.turn_counter = SimpleNamespace(value=0)
        self.drift_simple = DriftSimple(
            section_index=1, orbit_length=0
        )  # will raise Exception because of missing transition gamma
        with self.assertRaises(ValueError):
            self.drift_simple.on_init_simulation(simulation=simulation)

    def test___init__(self):
        np.testing.assert_array_equal(
            self.drift_simple.momentum_compaction_factor, 20.0
        )
        self.assertEqual(self.drift_simple.orbit_length, 0.25 * 25)

    def test_transition_gamma(self):
        np.testing.assert_array_equal(
            self.drift_simple.momentum_compaction_factor, 20.0
        )

    def test_alpha_0(self):
        np.testing.assert_array_equal(
            self.drift_simple.alpha_0,
            self.drift_simple.momentum_compaction_factor,
        )

    def test_eta_0(self):
        # eta_0 = alpha_0 - 1 / gamma^2
        rel_eta = self.drift_simple.alpha_0 - 1 / self.gamma**2

        np.testing.assert_array_equal(
            self.drift_simple.eta_0(gamma=self.gamma), (rel_eta)
        )

    def test_on_init_simulation(self):
        from types import SimpleNamespace

        from blond.core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        simulation.turn_counter = SimpleNamespace(value=0)
        simulation.ring.circumference = 10
        self.drift_simple.on_init_simulation(simulation=simulation)

    def test_track(self):
        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

        beam = Mock(BeamBaseClass)
        beam.reference = Mock(ReferenceCoordinates)
        beam.common_array_size = 1
        beam.reference.time = float(0)
        beam.reference.beta = 0.5
        beam.reference.velocity = float(beam.reference.beta * c0)
        beam.reference.gamma = float(np.sqrt(1 - 0.25))  # beta**2
        beam.reference.total_energy = float(938)
        beam.dE = backend.linspace(
            -1e6, 1e6, 10, dtype=backend.float
        )  # delta E in eV
        beam.dt = backend.linspace(
            -1e-6, 1e-6, 10, dtype=backend.float
        )  # delta t in s
        beam.write_partial_dt.return_value = beam.dt
        beam.read_partial_dE.return_value = beam.dE
        self.drift_simple.momentum_compaction_factor = (
            momentum_compaction_factor(transition_gamma=20.0)
        )  # highly relativistic

        self.drift_simple.track(beam=beam)
        np.testing.assert_allclose(
            copy_to_cpu(beam.dt),
            [
                0.00023563017947381346,
                0.0001832679173685216,
                0.0001309056552632297,
                7.854339315793783e-05,
                2.6181131052645944e-05,
                -2.6181131052645917e-05,
                -7.85433931579378e-05,
                -0.0001309056552632297,
                -0.0001832679173685216,
                -0.00023563017947381346,
            ],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            copy_to_cpu(beam.dE),
            np.linspace(-1e6, 1e6, 10),
        )
        self.assertEqual(
            beam.reference.beta,
            0.5,  # unchanged
        )
        self.assertEqual(
            beam.reference.time,
            self.drift_simple.orbit_length
            / (0.5 * c0),  # drifted by length of drift
        )

    def test_init(self):
        DriftSimple(
            orbit_length=1.0, section_index=0, momentum_compaction_factor=2.5
        )

    @multi_backend_testcase("Numpy64Bit")
    @pytest.mark.backend_mutation
    def test_compare_track_ham(self):
        from blond.core.beam.particle_types import proton

        drift = DriftSimple.headless(
            momentum_compaction_factor=1e-3,
            orbit_length=10.0,
            section_index=0,
        )
        dE_values = np.linspace(-1e6, 1e6, 11)
        beam = ProbeBeam(
            dE=dE_values,
            particle_type=proton,
            reference_total_energy=1e9,
        )
        dt_before = beam.dt.copy_as_numpy()

        # Predicted dt change: dH/d(dE) evaluated at each particle's dE.
        dE_s, beta_s, gamma_s, E_s = sympy.symbols(
            "dE beta gamma E", real=True
        )
        dH_ddE = sympy.lambdify(
            (dE_s, beta_s, gamma_s, E_s),
            sympy.diff(drift.get_hamilton_symbolic(), dE_s),
            modules="numpy",
        )
        predicted = dH_ddE(
            dE_values,
            beam.reference.beta,
            beam.reference.gamma,
            beam.reference.total_energy,
        )

        drift.track(beam=beam)
        actual = beam.dt.copy_as_numpy() - dt_before

        np.testing.assert_allclose(actual, predicted, rtol=1e-12)

    def test_get_hamilton_symbolic_replace_symbols_false_keeps_alpha_0(self):
        """With ``replace_symbols=False`` the momentum-compaction factor
        must stay the free symbol ``alpha_0`` instead of being baked in
        as a float, and resubstituting its numeric value must reproduce
        the ``replace_symbols=True`` Hamiltonian.
        """
        drift = DriftSimple.headless(
            momentum_compaction_factor=1e-3,
            orbit_length=10.0,
            section_index=0,
        )
        alpha_0_s = sympy.Symbol("alpha_0", real=True)

        ham_sym = drift.get_hamilton_symbolic(replace_symbols=False)
        self.assertIn("alpha_0", {s.name for s in ham_sym.free_symbols})

        ham_num = drift.get_hamilton_symbolic(replace_symbols=True)
        resubstituted = ham_sym.subs(alpha_0_s, float(drift.alpha_0))
        self.assertEqual(sympy.simplify(resubstituted - ham_num), 0)


class TestDriftExact(unittest.TestCase):
    def setUp(self):
        self.gamma = 2.5
        # params from
        # https://proceedings.jacow.org/e08/papers/thpc044.pdf
        self.drift_exact = DriftExact(
            orbit_length=63.13,
            section_index=0,
            momentum_compaction_factor=0.0001278,
            higher_order_alpha=np.array([1.49]),
        )

    def test_track(self):
        beam = Mock(BeamBaseClass)
        beam.reference = Mock(ReferenceCoordinates)
        beam.common_array_size = 1
        beam.reference.time = float(0)
        beam.reference.beta = 0.5
        beam.reference.velocity = float(beam.reference.beta * c0)
        beam.reference.gamma = float(np.sqrt(1 - 0.25))
        beam.reference.total_energy = float(938)

        beam.dE = backend.linspace(
            -1e6, 1e6, 10, dtype=backend.float
        )  # delta E in eV
        beam.dt = backend.linspace(
            -1e-6, 1e-6, 10, dtype=backend.float
        )  # delta t in s
        beam.write_partial_dt.return_value = beam.dt
        beam.read_partial_dE.return_value = beam.dE
        self.drift_exact._simulation = Mock(Simulation)
        self.drift_exact._simulation.turn_counter = DynamicParameter(1)

        self.drift_exact.schedule(
            "higher_order_alpha",
            np.array(
                [[1.49, 23], [1.49, 24]],
            ),
        )
        self.drift_exact.track(beam=beam)

    def test_track_empty_beam_skips_drift(self):
        from blond.core.simulation.simulation import Simulation

        beam = Mock(BeamBaseClass)
        beam.reference = Mock(ReferenceCoordinates)
        beam.common_array_size = 0
        beam.reference.time = float(0)
        beam.reference.beta = 0.5
        beam.reference.velocity = float(0.5 * c0)
        beam.reference.gamma = float(np.sqrt(1 - 0.25))
        beam.reference.total_energy = float(938)
        self.drift_exact._simulation = Mock(Simulation)
        self.drift_exact._simulation.turn_counter = DynamicParameter(1)
        self.drift_exact.schedule(
            "higher_order_alpha",
            np.array([[1.49, 23], [1.49, 24]]),
        )
        self.drift_exact.track(beam=beam)

    @pytest.mark.backend_mutation
    def test_track_vs_blond2(self):
        backend.change_backend(Numpy64Bit)

        def drift_blond2(dE, T, energy, beta, alpha_0, alpha_1, alpha_2):
            invbetasq = 1 / (beta * beta)
            invenesq = 1 / (energy * energy)

            beam_delta = (
                np.sqrt(
                    1.0 + invbetasq * (dE * dE * invenesq + 2.0 * dE / energy)
                )
                - 1.0
            )

            dt = T * (
                (
                    1.0
                    + alpha_0 * beam_delta
                    + alpha_1 * (beam_delta * beam_delta)
                    + alpha_2 * (beam_delta * beam_delta * beam_delta)
                )
                * (1.0 + dE / energy)
                / (1.0 + beam_delta)
                - 1.0
            )
            return dt

        beam = ProbeBeam(
            dE=np.linspace(-10, 10, 41),
            particle_type=lead_82,
            reference_total_energy=1e12,
        )

        drift = DriftExact.headless(
            orbit_length=10,
            section_index=0,
            momentum_compaction_factor=10,
            higher_order_alpha=[20, 30],
        )

        blond2_expected = drift_blond2(
            dE=beam.dE.copy_as_numpy(),
            T=drift.orbit_length / (beam.reference.beta * c),
            energy=beam.reference.total_energy,
            beta=beam.reference.beta,
            alpha_0=drift.alpha_0,
            alpha_1=drift.higher_order_alpha[0],
            alpha_2=drift.higher_order_alpha[1],
        )
        drift.track(beam=beam)

        np.testing.assert_allclose(blond2_expected, beam.dt.copy_as_numpy())

    @multi_backend_testcase("Numpy64Bit")
    @pytest.mark.backend_mutationn
    def test_compare_track_ham(self):
        """For ``higher_order_alpha`` lengths 1, 2, 3 (i.e. α_1, α_1..α_2,
        α_1..α_3 — α_0 is set separately by ``momentum_compaction_factor``),
        the tracker's dt change must equal ``dH/d(dE)`` from
        ``get_hamilton_symbolic``.

        ``DriftExact`` symbolically truncates ``H`` at order ``n_alpha + 2``
        in ``dE``. In principle the residual would shrink as
        ``(dE/E)**(n_alpha + 1)``, but in practice the dE**2 coefficient
        ``c1*beta**2 - c2`` suffers catastrophic cancellation near
        ``beta = 1`` and keeps only ~9 significant digits. That floor —
        not the truncation tail — sets the tolerance for every ``n_alpha``.
        """
        from blond.core.beam.particle_types import proton

        dE_s, beta_s, E_s = sympy.symbols("dE beta E", real=True)

        for higher_order_alpha in (
            np.array([1.0]),
            np.array([1.0, 0.5]),
            np.array([1.0, 0.5, 0.25]),
        ):
            with self.subTest(n_alpha=len(higher_order_alpha)):
                drift = DriftExact.headless(
                    orbit_length=10000.0,
                    section_index=0,
                    momentum_compaction_factor=1e-3,
                    higher_order_alpha=higher_order_alpha,
                )
                dE_values = np.linspace(-1e5, 1e5, 11)
                beam = ProbeBeam(
                    dE=dE_values,
                    particle_type=proton,
                    reference_total_energy=1e10,
                )
                dt_before = beam.dt.copy_as_numpy()

                dH_ddE = sympy.lambdify(
                    (dE_s, beta_s, E_s),
                    sympy.diff(drift.get_hamilton_symbolic(), dE_s),
                    modules="numpy",
                )
                predicted = dH_ddE(
                    dE_values,
                    beam.reference.beta,
                    beam.reference.total_energy,
                )

                drift.track(beam=beam)
                actual = beam.dt.copy_as_numpy() - dt_before

                np.testing.assert_allclose(actual, predicted, rtol=1e-7)

    def test_get_hamilton_symbolic_replace_symbols_false_preserves_higher_alpha(
        self,
    ):
        """
        With ``replace_symbols=False`` the analytical Hamiltonian must
        keep one ``dE``-polynomial term per configured higher-order
        alpha. Regression: an earlier implementation hard-coded
        ``higher = ()`` in symbolic mode, collapsing the truncation back
        to ``dE**2`` and silently dropping every ``alpha_k`` (``k >= 1``).
        """
        dE_s = sympy.Symbol("dE", real=True)

        for n_alpha in (0, 1, 2, 3):
            higher_order_alpha = (
                np.zeros(n_alpha) if n_alpha > 0 else np.array([])
            )
            drift = DriftExact.headless(
                orbit_length=10000.0,
                section_index=0,
                momentum_compaction_factor=1e-3,
                higher_order_alpha=higher_order_alpha,
            )
            with self.subTest(n_alpha=n_alpha):
                ham = drift.get_hamilton_symbolic(replace_symbols=False)
                # Polynomial degree in dE must reflect every configured
                # alpha: 2 base + n_alpha higher-order terms.
                self.assertEqual(
                    sympy.Poly(ham, dE_s).degree(),
                    n_alpha + 2,
                )
                # Each alpha_k symbol (k = 1..n_alpha) must actually
                # appear in the expression.
                free_names = {s.name for s in ham.free_symbols}
                for k in range(1, n_alpha + 1):
                    self.assertIn(f"alpha_{k}", free_names)


class TestDriftSpecial(unittest.TestCase):
    @unittest.skip
    def test_on_init_simulation(self):
        # TODO: implement test for `on_init_simulation`
        self.drift_special.on_init_simulation(simulation=None)

    @unittest.skip
    def test_track(self):
        # TODO: implement test for `track`
        self.drift_special.track(beam=None)


class TestDriftXSuite(unittest.TestCase):
    @unittest.skip
    def test_on_init_simulation(self):
        # TODO: implement test for `on_init_simulation`
        self.drift_x_suite.on_init_simulation(simulation=None)

    @unittest.skip
    def test_track(self):
        # TODO: implement test for `track`
        self.drift_x_suite.track(beam=None)


class TestDriftSubstepped(unittest.TestCase):
    """Sub-stepped energy adaptation within the drift region (DriftSubstepped)."""

    def setUp(self):
        from blond import mu_plus

        self.particle = mu_plus
        self.orbit_length = 5990.0
        self.alpha_0 = 10.395e-4
        self.E0 = 4.0e9
        self.harmonic = 25900
        self.n_turns = 5
        self.t_rev0 = self.orbit_length / float(
            ReferenceCoordinates(0.0, self.E0, self.particle).velocity
        )
        self.t_rf = self.t_rev0 / self.harmonic
        self.ramp_rate = 20e6 / self.t_rev0  # eV/s, ~20 MeV per turn

    def _cycle_stub(self):
        """Magnetic cycle stub: total energy ramps linearly with reference time."""
        E0, rate = self.E0, self.ramp_rate

        def get_target_total_energy(
            *, turn_i, section_i, reference_time, particle_type
        ):
            return E0 + rate * reference_time

        return SimpleNamespace(get_target_total_energy=get_target_total_energy)

    def _accumulated_reference_time(self, n_substeps):
        from blond.physics.drifts import DriftSubstepped

        drift = DriftSubstepped(
            orbit_length=self.orbit_length,
            n_substeps=n_substeps,
            momentum_compaction_factor=self.alpha_0,
        )
        drift.configure(
            turn_counter=SimpleNamespace(value=0),
            magnetic_cycle=self._cycle_stub(),
        )
        reference = ReferenceCoordinates(0.0, self.E0, self.particle)
        for _ in range(self.n_turns):
            drift.track_reference(reference)
        return reference.time

    def test_init_rejects_nonpositive_n_substeps(self):
        """n_substeps below 1 is rejected with a speaking ValueError."""
        from blond.physics.drifts import DriftSubstepped

        with self.assertRaisesRegex(
            ValueError, r"n_substeps must be >= 1, got 0"
        ):
            DriftSubstepped(
                orbit_length=self.orbit_length,
                n_substeps=0,
                momentum_compaction_factor=self.alpha_0,
            )

    def test_on_init_simulation_rejects_non_time_cycle(self):
        """A magnetic cycle that is not by-time is rejected with TypeError."""
        from blond.physics.drifts import DriftSubstepped

        simulation = Mock(Simulation)
        simulation.turn_counter = SimpleNamespace(value=0)
        simulation.ring.circumference = self.orbit_length
        # A by-turn-style stand-in: not a MagneticCycleByTime, so the
        # element cannot re-sample the energy as a function of time.
        simulation.magnetic_cycle = SimpleNamespace()
        drift = DriftSubstepped(
            orbit_length=self.orbit_length,
            n_substeps=4,
            momentum_compaction_factor=self.alpha_0,
        )
        with self.assertRaisesRegex(
            TypeError,
            r"DriftSubstepped requires a MagneticCycleByTime, "
            r"got SimpleNamespace",
        ):
            drift.on_init_simulation(simulation=simulation)

    def test_track_applies_schedule_at_live_turn(self):
        """The schedule branch in _track retunes alpha_0 per live turn."""
        from blond.core.beam.beams import ProbeBeam
        from blond.physics.drifts import DriftSubstepped

        alpha_per_turn = np.array([self.alpha_0, 3.0 * self.alpha_0])
        turn_counter = SimpleNamespace(value=1)
        flat_cycle = SimpleNamespace(
            get_target_total_energy=(
                lambda *, turn_i, section_i, reference_time, particle_type: (
                    self.E0
                )
            )
        )
        drift = DriftSubstepped(
            orbit_length=self.orbit_length,
            n_substeps=1,
        )
        drift.configure(turn_counter=turn_counter, magnetic_cycle=flat_cycle)
        drift.schedule(
            attribute="momentum_compaction_factor", value=alpha_per_turn
        )
        # scheduling immediately applies the turn-0 value ...
        self.assertEqual(drift.momentum_compaction_factor, alpha_per_turn[0])

        beam = ProbeBeam(
            dE=np.zeros(3),
            particle_type=self.particle,
            reference_total_energy=self.E0,
        )
        gamma_in = beam.reference.gamma
        drift.track(beam=beam)
        # ... while tracking re-applies it at the live turn index ...
        self.assertEqual(drift.momentum_compaction_factor, alpha_per_turn[1])
        # ... and the drift's slippage used the turn-1 value
        np.testing.assert_allclose(
            drift._last_eta_0,
            alpha_per_turn[1] - 1.0 / gamma_in**2,
            rtol=1e-12,
        )

    def test_reference_time_converges_with_substeps(self):
        """Reference time converges to the fine integral as n_substeps grows."""
        fine = self._accumulated_reference_time(8192)
        err_1 = abs(self._accumulated_reference_time(1) - fine) / self.t_rf
        err_256 = abs(self._accumulated_reference_time(256) - fine) / self.t_rf
        # single-beta-per-turn carries a significant arrival-time error ...
        self.assertGreater(err_1, 0.1)
        # ... that sub-stepping removes
        self.assertLess(err_256, 0.01)
        self.assertGreater(err_1, 20 * err_256)

    def test_single_substep_reproduces_plain_drift_time(self):
        """n_substeps=1 advances the clock exactly like the plain single-beta drift."""
        from blond.physics.drifts import DriftSimple, DriftSubstepped

        sub = DriftSubstepped(
            orbit_length=self.orbit_length,
            n_substeps=1,
            momentum_compaction_factor=self.alpha_0,
        )
        sub.configure(
            turn_counter=SimpleNamespace(value=0),
            magnetic_cycle=self._cycle_stub(),
        )
        plain = DriftSimple.headless(
            momentum_compaction_factor=self.alpha_0,
            orbit_length=self.orbit_length,
        )
        dt_sub = sub.track_reference(
            ReferenceCoordinates(0.0, self.E0, self.particle)
        )
        dt_plain = plain.track_reference(
            ReferenceCoordinates(0.0, self.E0, self.particle)
        )
        # one segment at the entering energy == the plain single-beta drift
        self.assertAlmostEqual(dt_sub / dt_plain, 1.0, places=12)

    def test_reference_reframing_preserves_absolute_energy(self):
        """Re-sampling the energy reframes dE, keeping absolute energy constant."""
        from blond.core.beam.beams import ProbeBeam
        from blond.physics.drifts import DriftSubstepped

        dE0 = np.linspace(-1e6, 1e6, 11)
        beam = ProbeBeam(
            dE=dE0.copy(),
            particle_type=self.particle,
            reference_total_energy=self.E0,
        )
        e_abs_before = beam.reference.total_energy + beam.dE.copy_as_numpy()

        drift = DriftSubstepped(
            orbit_length=self.orbit_length,
            n_substeps=4,
            momentum_compaction_factor=self.alpha_0,
        )
        drift.configure(
            turn_counter=SimpleNamespace(value=0),
            magnetic_cycle=self._cycle_stub(),
        )
        drift.track(beam=beam)

        # the reference actually ramped (so the test is non-trivial) ...
        self.assertGreater(beam.reference.total_energy, self.E0)
        # ... yet each particle's absolute energy E_ref + dE is unchanged
        e_abs_after = beam.reference.total_energy + beam.dE.copy_as_numpy()
        np.testing.assert_allclose(e_abs_after, e_abs_before, rtol=1e-12)

    def test_no_ramp_reproduces_plain_drift_beam(self):
        """With a flat energy program, the beam map equals the plain drift."""
        from blond.core.beam.beams import ProbeBeam
        from blond.physics.drifts import DriftSimple, DriftSubstepped

        dE0 = np.linspace(-1e6, 1e6, 11)
        flat_cycle = SimpleNamespace(
            get_target_total_energy=(
                lambda *, turn_i, section_i, reference_time, particle_type: (
                    self.E0
                )
            )
        )

        beam_sub = ProbeBeam(
            dE=dE0.copy(),
            particle_type=self.particle,
            reference_total_energy=self.E0,
        )
        dt_sub_before = beam_sub.dt.copy_as_numpy()
        sub = DriftSubstepped(
            orbit_length=self.orbit_length,
            n_substeps=1,
            momentum_compaction_factor=self.alpha_0,
        )
        sub.configure(
            turn_counter=SimpleNamespace(value=0), magnetic_cycle=flat_cycle
        )
        sub.track(beam=beam_sub)
        dt_sub_change = beam_sub.dt.copy_as_numpy() - dt_sub_before

        beam_plain = ProbeBeam(
            dE=dE0.copy(),
            particle_type=self.particle,
            reference_total_energy=self.E0,
        )
        dt_plain_before = beam_plain.dt.copy_as_numpy()
        plain = DriftSimple.headless(
            momentum_compaction_factor=self.alpha_0,
            orbit_length=self.orbit_length,
        )
        plain.track(beam=beam_plain)
        dt_plain_change = beam_plain.dt.copy_as_numpy() - dt_plain_before

        np.testing.assert_allclose(dt_sub_change, dt_plain_change, rtol=1e-12)
        # a flat program leaves dE untouched
        np.testing.assert_allclose(
            beam_sub.dE.copy_as_numpy(), dE0, rtol=0, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
