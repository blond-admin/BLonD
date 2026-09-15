import unittest

import numpy as np
from scipy.constants import (
    c,
    e,
    electron_mass,
    epsilon_0,
    hbar,
    m_e,
    m_p,
    proton_mass,
)

from blond import proton
from blond.core.base import BeamPhysicsRelevant
from blond.core.beam.particle_types import ParticleType, electron, mu_plus


class TestParticleType(unittest.TestCase):
    def setUp(self):
        self.mass = 1e-10
        self.user_decay_rate = 1e-5
        self.particle_type = ParticleType(
            mass=self.mass,
            charge=+1,
            user_decay_rate=1e-5,
        )

    def test_inputs(self):
        self.assertEqual(self.particle_type.charge, 1)
        self.assertEqual(self.particle_type.mass, self.mass)
        self.assertEqual(
            self.particle_type.user_decay_rate, self.user_decay_rate
        )
        self.assertEqual(self.particle_type.mass_inv, 1 / self.mass)

        expected_classical_radius = (
            0.25 / (np.pi * epsilon_0) * e**2 * 1**2 / (self.mass * e)
        )
        self.assertEqual(
            self.particle_type.classical_particle_radius,
            expected_classical_radius,
        )
        self.assertEqual(
            self.particle_type.sands_radiation_constant,
            4 * np.pi / 3 * expected_classical_radius / self.mass**3,
        )
        self.assertEqual(
            self.particle_type.quantum_radiation_constant,
            55.0 / (32.0 * np.sqrt(3.0)) * hbar * c / (self.mass * e),
        )

    def test_particle_library(self):
        # Electron
        self.assertTrue(electron.mass == electron_mass * c**2 / e)
        self.assertTrue(electron.mass == m_e * c**2 / e)
        self.assertTrue(electron.charge == -1)
        self.assertAlmostEqual(
            electron.classical_particle_radius, 2.8179403205e-15, places=9
        )

        # Proton
        self.assertTrue(proton.mass == proton_mass * c**2 / e)
        self.assertTrue(proton.mass == m_p * c**2 / e)
        self.assertTrue(proton.charge == 1)

        # Muon
        self.assertAlmostEqual(
            mu_plus.mass,
            105658375.5,  # eV
            places=1,
        )
        self.assertTrue(mu_plus.charge == 1)

    def test__eq__(self):
        element = BeamPhysicsRelevant
        with self.assertRaisesRegex(
            expected_exception=TypeError,
            expected_regex=f"Cannot compare {type(self.particle_type)} "
            f"to {type(element)}",
        ):
            self.particle_type.__eq__(element)

        self.assertFalse(self.particle_type.__eq__(proton))
        self.assertFalse(self.particle_type.__eq__(electron))
        self.assertTrue(self.particle_type.__eq__(self.particle_type))
