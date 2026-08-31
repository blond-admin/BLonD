import unittest

import numpy as np

from blond import BiGaussian, SingleHarmonicRFStation
from blond.beam_preparation.bigaussian import (
    _get_dE_from_dt,
    _get_dE_from_dt_core,
)
from blond.testing.simulation import ExampleSimulation01


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
    def test___init__(self):
        bi_gaussian = BiGaussian(
            n_macroparticles=10,
            sigma_dt=50e-9,
            sigma_dE=None,
            reinsertion=False,
            seed=0,
        )

    def test_prepare_beam(self):
        simulation_ = ExampleSimulation01()
        bi_gaussian = BiGaussian(
            n_macroparticles=1e4,
            sigma_dt=50e-9,
            sigma_dE=None,
            reinsertion=False,
            seed=0,
        )
        bi_gaussian.prepare_beam(
            simulation=simulation_.simulation, beam=simulation_.beam1
        )
        self.assertAlmostEqual(
            float(np.std(simulation_.beam1.read_partial_dt())), 50e-9
        )

    def test_reinsertion_gives_up_instead_of_spinning_forever(self):
        """A bucket no sample can land in must raise, not hang.

        ``reinsertion=True`` re-draws the particles that fell outside the
        separatrix until none is left.  When the separatrix admits (almost)
        nothing -- e.g. a design RF phase that degenerates the single-RF
        bucket, or a sigma far wider than the bucket -- no draw ever
        succeeds and the loop cannot terminate.  Spinning forever is the
        worst possible failure mode: it is indistinguishable from a slow
        run and burns CPU silently (one occurrence cost ~3.3 CPU-hours
        before it was noticed).  It has to give up and say why.
        """
        simulation_ = ExampleSimulation01()
        station = simulation_.simulation.ring.elements.get_element(
            SingleHarmonicRFStation, recursive=False
        )
        # pi / 2 degenerates the bucket this matcher searches.
        station.phi_rf_design = np.pi / 2.0
        bi_gaussian = BiGaussian(
            n_macroparticles=64,
            sigma_dt=50e-9,
            sigma_dE=None,
            reinsertion=True,
            seed=0,
        )
        with self.assertRaises(ValueError) as raised:
            bi_gaussian.prepare_beam(
                simulation=simulation_.simulation, beam=simulation_.beam1
            )
        message = str(raised.exception)
        # The diagnostic must be actionable: how many are still outside,
        # and what to change.
        self.assertIn("reinsertion", message)
        self.assertIn("separatrix", message)
        self.assertIn("phi_rf_design", message)

    def test_prepare_beam2(self):
        simulation_ = ExampleSimulation01()
        bi_gaussian = BiGaussian(
            n_macroparticles=1e4,
            sigma_dt=50e-9,
            sigma_dE=60e9,
            reinsertion=False,
            seed=0,
        )
        bi_gaussian.prepare_beam(
            simulation=simulation_.simulation, beam=simulation_.beam1
        )
        self.assertAlmostEqual(
            float(np.std(simulation_.beam1.read_partial_dt()) / 50e-9),
            1,
            places=1,  # low precision because of few
            # particles in this testcase
        )
        self.assertAlmostEqual(
            float(np.std(simulation_.beam1.read_partial_dE()) / 60e9),
            1,
            places=1,  # low precision because of
            # few particles in this testcase
        )


if __name__ == "__main__":
    unittest.main()
