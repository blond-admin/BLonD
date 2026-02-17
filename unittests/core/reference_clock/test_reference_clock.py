import unittest

import numpy as np
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from scipy.constants import speed_of_light as c0

from blond import uranium_29


class TestReferenceCoordinates(unittest.TestCase):
    def setUp(self):
        # https://fr.wikipedia.org/wiki/Facteur_de_Lorentz

        gamma = 1.25  # from wikipedia

        beta = 0.6  # from wikipedia

        # gamma m c ** 2, but mass is  already in eV

        self.reference_coorinates = ReferenceCoordinates(
            time=1.2,
            total_energy=gamma * uranium_29.mass,
            particle_type=uranium_29,
        )

    def test___init__(self):
        # self.reference_coorinates.__init__()
        pass  # done implicitly by setUp

    def test_particle_type(self):
        self.assertEqual(self.reference_coorinates.particle_type, uranium_29)

    def test_total_energy_setter(self):
        self.assertEqual(
            self.reference_coorinates.total_energy, 1.25 * uranium_29.mass
        )

    def test_total_energy_getter(self):
        self.reference_coorinates.total_energy = 1.5 * uranium_29.mass
        self.assertEqual(
            self.reference_coorinates.total_energy, 1.5 * uranium_29.mass
        )

    def test_gamma(self):
        self.assertAlmostEqual(1.25, self.reference_coorinates.gamma)

    def test_beta(self):
        self.assertAlmostEqual(0.6, self.reference_coorinates.beta)

    def test_velocity(self):
        np.testing.assert_allclose(
            0.6 * c0, self.reference_coorinates.velocity
        )
