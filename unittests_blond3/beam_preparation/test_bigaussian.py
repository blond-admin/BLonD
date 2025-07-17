import unittest

from blond3 import BiGaussian
from blond3.beam_preparation.bigaussian import _get_dE_from_dt_core, _get_dE_from_dt
from blond3.testing.simulation import ExampleSimulation01


class TestFunctions(unittest.TestCase):
    def test__get_dE_from_dt_core(self):
        dE_amplitude = _get_dE_from_dt_core(
            beta=0.99,
            dt_amplitude=0.1,
            energy=1e9,
            eta0=0.1,
            harmonic=1,
            omega_rf=400e3,
            particle_charge=1,
            phi_rf=2,
            phi_s=2,
            voltage=1e3,
        )

        self.assertEqual(dE_amplitude, 336852503.57639045)  # pinned to some
        # random values.
        # would be better if this would be pinned to a theoretic expected value

    @unittest.skip("Implement phi_s in RF first")  # TODO
    def test_get_dE_from_dt(self):
        dE_amplitude = _get_dE_from_dt(
            simulation=ExampleSimulation01().simulation, dt_amplitude=0.1
        )
        self.assertEqual(dE_amplitude, 0)


class TestBiGaussian(unittest.TestCase):
    def setUp(self):
        self.bi_gaussian = BiGaussian(
            n_macroparticles=10,
            sigma_dt=50e-9,
            sigma_dE=None,
            reinsertion=False,
            seed=0,
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip("Implement phi_s in RF first")  # TODO
    def test_on_prepare_beam(self):
        self.bi_gaussian.prepare_beam(simulation=ExampleSimulation01().simulation)


if __name__ == "__main__":
    unittest.main()
