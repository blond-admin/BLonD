import unittest
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.constants import c
from scipy.constants import speed_of_light as c0

from blond import Cupy32Bit, Numpy32Bit, Simulation, momentum_compaction_factor
from blond.core.backends.backend import Numpy64Bit, backend
from blond.core.base import DynamicParameter
from blond.core.beam.base import BeamBaseClass
from blond.core.beam.beams import ProbeBeam
from blond.core.beam.particle_types import lead_82
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.physics.drifts import DriftBaseClass, DriftExact, DriftSimple


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
        simulation = Mock(Simulation)
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

    def test_invalidate_cache(self):
        self.drift_simple.invalidate_cache()

    def test_on_init_simulation(self):
        from blond.core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        simulation.ring.circumference = 10
        self.drift_simple.on_init_simulation(simulation=simulation)

    def test_track(self):
        beam = Mock(BeamBaseClass)
        beam.reference = Mock(ReferenceCoordinates)
        beam.common_array_size = 1
        beam.reference.time = float(0)
        beam.reference.beta = float(0.5)
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
            rtol=1e-5 if backend.float == np.float32 else 1e-12,
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
        beam.reference.beta = float(0.5)
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
        self.drift_exact._simulation.turn_i = DynamicParameter(1)

        self.drift_exact.schedule(
            "higher_order_alpha",
            np.array(
                [[1.49, 23], [1.49, 24]],
            ),
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
