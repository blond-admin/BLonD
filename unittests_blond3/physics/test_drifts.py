import unittest
from unittest.mock import Mock, patch

import numpy as np

from blond3 import Simulation
from blond3._core.backends.backend import backend, Numpy64Bit, Numpy32Bit
from blond3._core.beam.base import BeamBaseClass
from blond3.physics.drifts import DriftBaseClass, DriftSimple
from scipy.constants import speed_of_light as c0


class TestDriftBaseClass(unittest.TestCase):
    def setUp(self):
        self.drift_base_class = DriftBaseClass(
            effective_length=123, section_index=0
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_on_init_simulation(self):
        simulation = Mock(Simulation)
        self.drift_base_class.on_init_simulation(simulation=simulation)

    def test_on_run_simulation(self):
        simulation = Mock(Simulation)
        self.drift_base_class.on_run_simulation(
            simulation=simulation, n_turns=11, turn_i_init=1,
            beam=Mock(BeamBaseClass),
        )

    def test_effective_length(self):
        self.assertEqual(123, self.drift_base_class.effective_length)


class TestDriftSimple(unittest.TestCase):
    def setUp(self):
        backend.change_backend(Numpy64Bit)
        self.gamma = 2.5
        self.drift_simple = DriftSimple.headless(
            transition_gamma=20.0,  # highly relativistic
            effective_length=0.25*25,
            section_index=0,
        )

    def test___init__(self):
        np.testing.assert_array_equal(self.drift_simple.transition_gamma, 20.0)
        self.assertEqual(self.drift_simple.effective_length, 0.25*25)

    def test_transition_gamma(self):
        np.testing.assert_array_equal(self.drift_simple.transition_gamma, 20.0)

    def test_alpha_0(self):
        np.testing.assert_array_equal(
            self.drift_simple.alpha_0, 1 / self.drift_simple.transition_gamma**2
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
        from blond3._core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        simulation.ring.effective_circumference = 10
        self.drift_simple.on_init_simulation(simulation=simulation)

    def test_track(self):

        beam = Mock(BeamBaseClass)
        beam.reference_time = backend.float(0)
        beam.reference_beta = backend.float(0.5)
        beam.reference_velocity = backend.float(beam.reference_beta * c0)
        beam.reference_gamma = backend.float(np.sqrt(1 - 0.25))  # beta**2
        beam.reference_total_energy = backend.float(938)
        beam.dE = np.linspace(-1e6, 1e6, 10, dtype=backend.float)  # delta E in eV
        beam.dt = np.linspace(-1e-6, 1e-6, 10, dtype=backend.float)  # delta t in s
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
            self.drift_simple.effective_length / (0.5 * c0),  # drifted by length of drift
        )

    def tearDown(self):
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
