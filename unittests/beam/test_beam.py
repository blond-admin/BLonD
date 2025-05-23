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
            Particle=Proton(),
            n_turns=number_of_turns,
        )

        # Define beam and distribution
        self.beam = Beam(Ring=ring, n_macroparticles=1001, intensity=1e9)

    def test_n_macroparticles_not_alive(self):
        self.beam.id[1] = 0
        self.assertEqual(self.beam.n_macroparticles_not_alive, 1)

    def test_n_macroparticles_eliminated(self):
        self.beam.id[1] = 0
        self.beam.eliminate_lost_particles()
        self.assertEqual(self.beam.n_macroparticles_eliminated, 1)
        self.assertEqual(self.beam.n_macroparticles_not_alive, 0)


if __name__ == '__main__':
    unittest.main()

    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.beam = Beam(Ring=None, n_macroparticles=None, intensity=None, dE=None, dt=None)
    @unittest.skip
    def test___init__(self):
        pass # calls __init__ in  self.setUp

    @unittest.skip
    def test__sumsq_dE(self):
        # TODO: implement test for `_sumsq_dE`
        self.beam._sumsq_dE()

    @unittest.skip
    def test__sumsq_dE(self):
        # TODO: implement test for `_sumsq_dE`
        self.beam._sumsq_dE(val=None)

    @unittest.skip
    def test__sumsq_dt(self):
        # TODO: implement test for `_sumsq_dt`
        self.beam._sumsq_dt()

    @unittest.skip
    def test__sumsq_dt(self):
        # TODO: implement test for `_sumsq_dt`
        self.beam._sumsq_dt(val=None)

    @unittest.skip
    def test_eliminate_lost_particles(self):
        # TODO: implement test for `eliminate_lost_particles`
        self.beam.eliminate_lost_particles()

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

    @unittest.skip
    def test_is_splitted(self):
        # TODO: implement test for `is_splitted`
        self.beam.is_splitted()

    @unittest.skip
    def test_is_splitted(self):
        # TODO: implement test for `is_splitted`
        self.beam.is_splitted(val=None)

    @unittest.skip
    def test_losses_below_energy(self):
        # TODO: implement test for `losses_below_energy`
        self.beam.losses_below_energy(dE_min=None)

    @unittest.skip
    def test_n_total_macroparticles(self):
        # TODO: implement test for `n_total_macroparticles`
        self.beam.n_total_macroparticles()

    @unittest.skip
    def test_n_total_macroparticles(self):
        # TODO: implement test for `n_total_macroparticles`
        self.beam.n_total_macroparticles(val=None)

    @unittest.skip
    def test_n_total_macroparticles_lost(self):
        # TODO: implement test for `n_total_macroparticles_lost`
        self.beam.n_total_macroparticles_lost()

    @unittest.skip
    def test_n_total_macroparticles_lost(self):
        # TODO: implement test for `n_total_macroparticles_lost`
        self.beam.n_total_macroparticles_lost(val=None)

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
