import unittest

from scipy.constants import speed_of_light as c0

from blond import uranium_29
from blond.core.reference_clock.reference_clock_numba import (
    beta,
    gamma,
    velocity,
)


class TestCallables(unittest.TestCase):
    def setUp(self):
        mass = uranium_29.mass  # in eV
        # https://fr.wikipedia.org/wiki/Facteur_de_Lorentz
        gamma = 1.25  # from wikipedia
        beta = 0.6  # from wikipedia
        # gamma m c ** 2, but mass is  already in eV
        self.total_energy = gamma * mass
        self.mass_inv = 1 / mass

    def test_gamma(self):
        gamma_expected = 1.25
        self.assertAlmostEqual(
            gamma_expected,
            gamma(self.total_energy, self.mass_inv),
        )

    def test_beta(self):
        beta_expected = 0.6
        self.assertAlmostEqual(
            beta_expected,
            beta(self.total_energy, self.mass_inv),
        )

    def test_velocity(self):
        velocity_expected = 0.6 * c0
        self.assertAlmostEqual(
            velocity_expected,
            velocity(self.total_energy, self.mass_inv),
        )


if __name__ == "__main__":
    unittest.main()
