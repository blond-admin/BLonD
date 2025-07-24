import unittest
import numpy as np
from scipy.constants import c
from _core.beam.particle_types import ParticleType
from acc_math.analytic.synchrotron_radiation_maths import (
    calculate_energy_loss_per_turn,
    calculate_partition_numbers,
    calculate_damping_times_in_second,
    calculate_damping_times_in_turn,
    calculate_natural_horizontal_emittance,
    calculate_natural_energy_spread,
    calculate_natural_bunch_length,
)
from blond3 import electron


class TestSynchrotronRadiationMaths_float_inputs(unittest.TestCase):
    def setUp(self):
        # Example of the FCC-ee high-energy booster at injection
        self.particle_type = ParticleType(mass=1, charge=-1)
        self.beam_energy = 1
        self.synchrotron_radiation_integrals = np.array(
            [
                1,
                2 * np.pi / self.particle_type.quantum_radiation_constant,
                np.pi / (self.particle_type.quantum_radiation_constant) ** 2,
                0,
                2 * np.pi / (self.particle_type.quantum_radiation_constant) ** 2,
            ]
        )
        self.revolution_frequency = 1
        self.energy_lost_per_turn = 1
        self.angular_synchrotron_frequency = 1
        self.momentum_compaction_factor = 1 / c

    def test_calculate_partition_numbers(self):
        jx, jy, jz = calculate_partition_numbers(
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals
        )
        jx_1 = calculate_partition_numbers(
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            which_plane="horizontal",
        )
        jz_1 = calculate_partition_numbers(
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            which_plane="longitudinal",
        )
        self.assertEqual(1.0, jy)
        self.assertEqual(1.0, jx, jx_1)
        self.assertEqual(1.0, jz, jz_1)

    def test_calculate_damping_times_in_turn(self):
        damping_times_in_turn = calculate_damping_times_in_turn(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
        )
        taux_1 = calculate_damping_times_in_turn(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
            which_plane="horizontal",
        )
        tauz_1 = calculate_damping_times_in_turn(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
            which_plane="horizontal",
        )
        self.assertEqual(2, damping_times_in_turn[1])
        self.assertEqual(2, damping_times_in_turn[0], taux_1)
        self.assertEqual(1, damping_times_in_turn[2], tauz_1)

    def test_calculate_damping_times_in_seconds(self):
        damping_times_in_second = calculate_damping_times_in_second(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
            revolution_frequency=self.revolution_frequency,
        )
        taux_1 = calculate_damping_times_in_second(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
            revolution_frequency=self.revolution_frequency,
            which_plane="horizontal",
        )
        tauz_1 = calculate_damping_times_in_second(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
            revolution_frequency=self.revolution_frequency,
            which_plane="horizontal",
        )
        self.assertEqual(2, damping_times_in_second[1])
        self.assertEqual(2, damping_times_in_second[0], taux_1)
        self.assertEqual(1, damping_times_in_second[2], tauz_1)

    def test_calculate_energy_loss_per_turn(self):
        self.assertEqual(
            1.0,
            calculate_energy_loss_per_turn(
                energy=self.beam_energy,
                synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                particle_type=self.particle_type,
            ),
        )

    def test_calculate_natural_horizontal_emittance(self):
        self.assertEqual(
            1.0,
            calculate_natural_horizontal_emittance(
                energy=self.beam_energy,
                synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                particle_type=self.particle_type,
            ),
        )

    def test_calculate_natural_energy_spread(self):
        self.assertEqual(
            0.5,
            calculate_natural_energy_spread(
                energy=self.beam_energy,
                synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                particle_type=self.particle_type,
            ),
        )

    def test_calculate_natural_bunch_length(self):
        self.assertEqual(
            0.5,
            calculate_natural_bunch_length(
                energy=self.beam_energy,
                synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                angular_synchrotron_frequency=self.angular_synchrotron_frequency,
                momentum_compaction_factor=self.momentum_compaction_factor,
                particle_type=self.particle_type,
            ),
        )


class TestSynchrotronRadiationMaths_array_inputs(unittest.TestCase):
    def setUp(self):
        # Example of the FCC-ee high-energy booster at injection
        self.particle_type = ParticleType(mass=1, charge=-1)
        self.beam_energy = np.array([1.0, 1.0, 1.0, 0.0, 1.0])
        self.synchrotron_radiation_integrals = np.array(
            [
                1,
                2 * np.pi / self.particle_type.quantum_radiation_constant,
                np.pi / (self.particle_type.quantum_radiation_constant) ** 2,
                0,
                2 * np.pi / (self.particle_type.quantum_radiation_constant) ** 2,
            ]
        )
        self.revolution_frequency = 1
        self.energy_lost_per_turn = 1
        self.angular_synchrotron_frequency = 1
        self.momentum_compaction_factor = 1 / c

    def test_calculate_partition_numbers(self):
        jx, jy, jz = calculate_partition_numbers(
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals
        )
        jx_1 = calculate_partition_numbers(
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            which_plane="horizontal",
        )
        jz_1 = calculate_partition_numbers(
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            which_plane="longitudinal",
        )
        self.assertEqual(1.0, jy)
        self.assertEqual(1.0, jx, jx_1)
        self.assertEqual(1.0, jz, jz_1)

    def test_calculate_damping_times_in_turn(self):
        damping_times_in_turn = calculate_damping_times_in_turn(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
        )
        taux_1 = calculate_damping_times_in_turn(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
            which_plane="horizontal",
        )
        tauz_1 = calculate_damping_times_in_turn(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
            which_plane="horizontal",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.array([2.0, 2.0, 2.0, 0.0, 2.0]), damping_times_in_turn[1]
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.array([2.0, 2.0, 2.0, 0.0, 2.0]), damping_times_in_turn[0], taux_1
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.array([1.0, 1.0, 1.0, 0.0, 1.0]), damping_times_in_turn[2], tauz_1
            )
        )

    def test_calculate_damping_times_in_seconds(self):
        damping_times_in_second = calculate_damping_times_in_second(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
            revolution_frequency=self.revolution_frequency,
        )
        taux_1 = calculate_damping_times_in_second(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
            revolution_frequency=self.revolution_frequency,
            which_plane="horizontal",
        )
        tauz_1 = calculate_damping_times_in_second(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
            revolution_frequency=self.revolution_frequency,
            which_plane="horizontal",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.array([2.0, 2.0, 2.0, 0.0, 2.0]), damping_times_in_second[1]
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.array([2.0, 2.0, 2.0, 0.0, 2.0]), damping_times_in_second[0], taux_1
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.array([1.0, 1.0, 1.0, 0.0, 1.0]), damping_times_in_second[2], tauz_1
            )
        )

    def test_calculate_energy_loss_per_turn(self):
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.array([1.0, 1.0, 1.0, 0.0, 1.0]),
                calculate_energy_loss_per_turn(
                    energy=self.beam_energy,
                    synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                    particle_type=self.particle_type,
                ),
            )
        )

    def test_calculate_natural_horizontal_emittance(self):
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.array([1.0, 1.0, 1.0, 0.0, 1.0]),
                calculate_natural_horizontal_emittance(
                    energy=self.beam_energy,
                    synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                    particle_type=self.particle_type,
                ),
            )
        )

    def test_calculate_natural_energy_spread(self):
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.array([0.5, 0.5, 0.5, 0.0, 0.5]),
                calculate_natural_energy_spread(
                    energy=self.beam_energy,
                    synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                    particle_type=self.particle_type,
                ),
            )
        )

    def test_calculate_natural_bunch_length(self):
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.array([0.5, 0.5, 0.5, 0.0, 0.5]),
                calculate_natural_bunch_length(
                    energy=self.beam_energy,
                    synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                    angular_synchrotron_frequency=self.angular_synchrotron_frequency,
                    momentum_compaction_factor=self.momentum_compaction_factor,
                    particle_type=self.particle_type,
                ),
            )
        )
