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
from blond.core.beam.particle_types import (
    ParticleType,
    electron,
    mu_minus,
    mu_plus,
)


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


class TestDecayToggle(unittest.TestCase):
    """The decay is metadata until switched on; the toggle is a copy."""

    def test_inactive_by_default_and_rate_is_then_zero(self):
        particle = ParticleType(mass=1e8, charge=1, user_decay_rate=3.0)
        self.assertFalse(particle.decay_active)
        self.assertEqual(particle.user_decay_rate, 3.0)
        self.assertEqual(particle.decay_rate, 0.0)

    def test_active_applies_the_user_rate(self):
        particle = ParticleType(
            mass=1e8, charge=1, user_decay_rate=3.0, decay_active=True
        )
        self.assertTrue(particle.decay_active)
        self.assertEqual(particle.decay_rate, 3.0)

    def test_with_decay_active_returns_a_distinct_equal_but_for_flag(self):
        inactive = ParticleType(mass=1e8, charge=1, user_decay_rate=3.0)
        active = inactive.with_decay_active(True)
        self.assertIsNot(active, inactive)
        self.assertFalse(inactive.decay_active)
        self.assertTrue(active.decay_active)
        self.assertEqual(active.mass, inactive.mass)
        self.assertEqual(active.charge, inactive.charge)
        self.assertEqual(active.user_decay_rate, inactive.user_decay_rate)
        # Different physics, so not equal and not the same hash key.
        self.assertNotEqual(active, inactive)
        self.assertNotEqual(hash(active), hash(inactive))
        # Round trip.
        self.assertEqual(active.with_decay_active(False), inactive)
        self.assertEqual(hash(active.with_decay_active(False)), hash(inactive))

    def test_muons_carry_the_rate_but_are_inactive(self):
        """The shipped ``mu_plus`` must not change any existing result."""
        self.assertFalse(mu_plus.decay_active)
        self.assertFalse(mu_minus.decay_active)
        self.assertAlmostEqual(
            mu_plus.user_decay_rate, 1 / 2.1969811e-6, delta=1e-3
        )
        self.assertEqual(mu_plus.decay_rate, 0.0)
        self.assertEqual(
            mu_plus.with_decay_active().decay_rate, mu_plus.user_decay_rate
        )

    def test_repr_states_the_toggle(self):
        self.assertIn("inactive", repr(mu_plus))
        self.assertIn("(active)", repr(mu_plus.with_decay_active()))
