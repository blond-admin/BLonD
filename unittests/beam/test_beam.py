import unittest

import numpy as np

from blond.beam.beam import Beam, Proton, MuMinus, MuPlus, Particle
from blond.input_parameters.ring import Ring


class TestBeam(unittest.TestCase):
    def setUp(self):
        number_of_turns = 20

        # Define general parameters
        ring = Ring(
            ring_length=26658.883,
            alpha_0=1. / 55.759505 ** 2,
            synchronous_data=np.linspace(450e9, 460.005e9, number_of_turns + 1),
            particle=Proton(),
            n_turns=number_of_turns,
        )

        # Define beam and distribution
        self.beam = Beam(ring=ring, n_macroparticles=1001, intensity=1e9)

    def test___init__(self):
        pass # calls __init__ in  self.setUp

    def test_n_macroparticles_not_alive(self):
        self.beam.id[1] = 0
        self.assertEqual(self.beam.n_macroparticles_not_alive, 1)

    def test_eliminate_lost_particles(self):
        self.beam.id[1] = 0
        self.beam.eliminate_lost_particles()
        self.assertEqual(self.beam.n_macroparticles_eliminated, 1)
        self.assertEqual(self.beam.n_macroparticles_not_alive, 0)


    @unittest.skip
    def test_gather(self):
        # TODO: implement test for `gather`
        self.beam.gather(all_gather=None)

    @unittest.skip
    def test_gather_losses(self):
        # TODO: implement test for `gather_losses`
        self.beam.gather_losses(all_gather=None)

    @unittest.skip
    def test_gather_statistics(self):
        # TODO: implement test for `gather_statistics`
        self.beam.gather_statistics(all_gather=None)


    def test_losses_below_energy(self):
        self.beam.dE = np.arange(len(self.beam.dE))
        de_min = self.beam.dE[len(self.beam.dE)//2]
        self.assertTrue(np.all(self.beam.id > 0))
        self.beam.losses_below_energy(dE_min=de_min)

        self.assertTrue(np.sum(self.beam.id == 0), len(self.beam.id) // 2)


    @unittest.skip
    def test_particle_decay(self):
        # TODO: implement test for `particle_decay`
        self.beam.particle_decay(time=None)

    @unittest.skip
    def test_split(self):
        # TODO: implement test for `split`
        self.beam.split(random=None, fast=None)


class TestMuMinus(unittest.TestCase):
    def setUp(self):
        self.mu_minus = MuMinus()
    def test___init__(self):
        pass # calls __init__ in  self.setUp


class TestMuPlus(unittest.TestCase):
    def setUp(self):
        self.mu_plus = MuPlus()

    def test___init__(self):
        pass # calls __init__ in  self.setUp


class TestParticle(unittest.TestCase):
    def setUp(self):
        self.particle = Particle(user_mass=1.0, user_charge=1.0,
                                 user_decay_rate=1.0)

    def test___init__(self):
        pass # calls __init__ in  self.setUp

if __name__ == '__main__':
    unittest.main()