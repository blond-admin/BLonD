import cmath
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.constants import speed_of_light as c0

from blond import Simulation
from blond.core.backends.backend import Numpy32Bit, Numpy64Bit, backend
from blond.core.beam.base import BeamBaseClass
from blond.physics.drifts import DriftBaseClass, DriftSimple


class DriftBaseClassHelper(DriftBaseClass):
    def eta_0(self, gamma: float) -> backend.float:
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
            turn_i_init=1,
            beam=Mock(BeamBaseClass),
        )

    def test_orbit_length(self):
        self.assertEqual(123, self.drift_base_class.orbit_length)


class TestDriftSimple(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        backend.change_backend(Numpy64Bit)

    def setUp(self):
        self.gamma = 2.5
        self.drift_simple = DriftSimple.headless(
            transition_gamma=20.0,  # highly relativistic
            orbit_length=0.25 * 25,
            section_index=0,
        )

    def test_setters1(self):
        drift_simple = DriftSimple(
            transition_gamma=20.0,  # highly relativistic
            orbit_length=0.25 * 25,
            section_index=0,
        )
        drift_simple.transition_gamma = 1.0
        drift_simple.transition_gamma = 1.0j

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
            transition_gamma=np.array([20.0]),  # highly relativistic
            orbit_length=0.25 * 25,
            section_index=0,
        )

        beam = Mock(BeamBaseClass)
        beam.reference_time = 0.0
        beam.reference_gamma = 1.0
        beam.reference_velocity = 0.5
        beam.reference_beta = 0.1
        beam.reference_total_energy = 1.0
        beam.write_partial_dt.return_value = np.ones(10)
        beam.read_partial_dE.return_value = np.zeros(10)
        self.drift_simple.track(beam=beam)

    def test_error_throwing_on_unscheduled(self):
        simulation = Mock(Simulation)
        self.drift_simple = DriftSimple(
            section_index=1, orbit_length=0
        )  # will raise Exception because of missing transition gamma
        with self.assertRaises(ValueError):
            self.drift_simple.on_init_simulation(simulation=simulation)

    def test___init__(self):
        np.testing.assert_array_equal(self.drift_simple.transition_gamma, 20.0)
        self.assertEqual(self.drift_simple.orbit_length, 0.25 * 25)

    def test_transition_gamma(self):
        np.testing.assert_array_equal(self.drift_simple.transition_gamma, 20.0)

    def test_alpha_0(self):
        np.testing.assert_array_equal(
            self.drift_simple.alpha_0,
            1 / self.drift_simple.transition_gamma**2,
        )

    def test_momentum_compaction_factor(self):
        np.testing.assert_array_equal(
            self.drift_simple.momentum_compaction_factor,
            1 / self.drift_simple.transition_gamma**2,
        )

    def test_eta_0(self):
        # eta_0 = alpha_0 - 1 / gamma^2
        rel_eta = self.drift_simple.alpha_0 - 1 / self.gamma**2

        np.testing.assert_array_equal(
            self.drift_simple.eta_0(gamma=self.gamma), backend.float(rel_eta)
        )

    def test_invalidate_cache(self):
        self.drift_simple.invalidate_cache()

    def test_on_init_simulation(self):
        from blond.core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        simulation.ring.circumference = 10
        self.drift_simple.on_init_simulation(simulation=simulation)

    def test_track(self):
        beam = Mock(BeamBaseClass)
        beam.reference_time = backend.float(0)
        beam.reference_beta = backend.float(0.5)
        beam.reference_velocity = backend.float(beam.reference_beta * c0)
        beam.reference_gamma = backend.float(np.sqrt(1 - 0.25))  # beta**2
        beam.reference_total_energy = backend.float(938)
        beam.dE = np.linspace(
            -1e6, 1e6, 10, dtype=backend.float
        )  # delta E in eV
        beam.dt = np.linspace(
            -1e-6, 1e-6, 10, dtype=backend.float
        )  # delta t in s
        beam.write_partial_dt.return_value = beam.dt
        beam.read_partial_dE.return_value = beam.dE

        self.drift_simple.track(beam=beam)
        np.testing.assert_allclose(
            beam.dt,
            [
                0.0002356301947884534,
                0.0001832679292799082,
                0.00013090566377136297,
                7.854339826281781e-05,
                2.61811327542726e-05,
                -2.6181132754272573e-05,
                -7.854339826281778e-05,
                -0.00013090566377136297,
                -0.0001832679292799082,
                -0.0002356301947884534,
            ],
        )
        np.testing.assert_allclose(
            beam.dE,
            np.linspace(-1e6, 1e6, 10),
        )
        self.assertEqual(
            beam.reference_beta,
            0.5,  # unchanged
        )
        self.assertEqual(
            beam.reference_time,
            self.drift_simple.orbit_length
            / (0.5 * c0),  # drifted by length of drift
        )

    def test_setters_negative_compaction(self):
        self.drift_simple.momentum_compaction_factor = -2.5
        self.assertEqual(self.drift_simple.momentum_compaction_factor, -2.5)
        self.assertEqual(
            self.drift_simple.transition_gamma, 1 / cmath.sqrt(-2.5)
        )

    def test_setters_complex_transition(self):
        self.drift_simple.transition_gamma = 1 / cmath.sqrt(-2.5)
        self.assertEqual(self.drift_simple.momentum_compaction_factor, -2.5)
        self.assertEqual(
            self.drift_simple.transition_gamma, 1 / cmath.sqrt(-2.5)
        )

    def test_setters_real_transition(self):
        self.drift_simple.transition_gamma = 1 / cmath.sqrt(2.5)
        self.assertEqual(self.drift_simple.momentum_compaction_factor, 2.5)
        self.assertEqual(
            self.drift_simple.transition_gamma, 1 / cmath.sqrt(2.5)
        )

    def test_init(self):
        DriftSimple(orbit_length=1.0, section_index=0, transition_gamma=2.5j)

        DriftSimple(
            orbit_length=1.0, section_index=0, momentum_compaction_factor=2.5
        )
        with self.assertRaises(ValueError):
            DriftSimple(
                orbit_length=1.0,
                section_index=0,
                momentum_compaction_factor=2.5,
                transition_gamma=2.5j,
            )

    @classmethod
    def tearDownClass(cls):
        backend.change_backend(Numpy32Bit)


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


if __name__ == "__main__":
    unittest.main()
