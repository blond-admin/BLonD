import unittest

import numpy as np

from blond.beam.beam import Beam, Proton
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
