import unittest

import numpy as np
from scipy.constants import c

from blond import backend
from blond._core.beam.particle_types import ParticleType
from blond.acc_math.analytic.synchrotron_radiation.synchrotron_radiation_maths import (
    calculate_damping_times_in_seconds,
    calculate_damping_times_in_turns,
    calculate_energy_loss_per_turn,
    calculate_horizontal_damping_partition_number,
    calculate_horizontal_damping_time_in_seconds,
    calculate_horizontal_damping_time_in_turns,
    calculate_longitudinal_damping_partition_number,
    calculate_longitudinal_damping_time_in_seconds,
    calculate_longitudinal_damping_time_in_turns,
    calculate_natural_bunch_length,
    calculate_natural_energy_spread,
    calculate_natural_horizontal_emittance,
    calculate_partition_numbers,
)


class TestSynchrotronRadiationMaths_float_inputs(unittest.TestCase):
    def setUp(self):
        self.particle_type = ParticleType(mass=1, charge=-1)
        self.beam_energy = 1
        self.synchrotron_radiation_integrals = np.array(
            [
                1,
                2 * np.pi / self.particle_type.sands_radiation_constant,
                np.pi
                / (
                    self.particle_type.quantum_radiation_constant
                    * self.particle_type.sands_radiation_constant
                ),
                0,
                2
                * np.pi
                / (
                    self.particle_type.sands_radiation_constant
                    * self.particle_type.quantum_radiation_constant
                ),
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
        jx_1 = calculate_horizontal_damping_partition_number(
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
        )
        jz_1 = calculate_longitudinal_damping_partition_number(
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
        )
        self.assertEqual(1.0, jy)
        self.assertEqual(jx, jx_1, msg="Expected value = 1")
        self.assertEqual(jz, jz_1, msg="Expected value = 2")

    def test_calculate_damping_times_in_turn(self):
        damping_times_in_turn = calculate_damping_times_in_turns(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
        )
        tau_x = calculate_horizontal_damping_time_in_turns(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
        )
        tau_z = calculate_longitudinal_damping_time_in_turns(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
        )
        self.assertEqual(2, damping_times_in_turn[1], msg="Expected value = 2")
        self.assertEqual(
            damping_times_in_turn[0], tau_x, msg="Expected value = 2"
        )
        self.assertEqual(
            damping_times_in_turn[2], tau_z, msg="Expected value = 1"
        )

    def test_calculate_damping_times_in_seconds(self):
        damping_times_in_second = calculate_damping_times_in_seconds(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
            revolution_frequency=self.revolution_frequency,
        )
        tau_x = calculate_horizontal_damping_time_in_seconds(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
            revolution_frequency=self.revolution_frequency,
        )
        tau_z = calculate_longitudinal_damping_time_in_seconds(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
            revolution_frequency=self.revolution_frequency,
        )
        self.assertEqual(
            2, damping_times_in_second[1], msg="Expected value = 2"
        )
        self.assertEqual(
            damping_times_in_second[0], tau_x, msg="Expected value = 2"
        )
        self.assertEqual(
            damping_times_in_second[2], tau_z, msg="Expected value = 1"
        )

    def test_calculate_energy_loss_per_turn(self):
        self.assertAlmostEqual(
            1.0,
            calculate_energy_loss_per_turn(
                energy=self.beam_energy,
                synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                particle_type=self.particle_type,
            ),
            places=6 if backend.float == np.float32 else 12,
        )

    def test_calculate_natural_horizontal_emittance(self):
        self.assertAlmostEqual(
            1.0,
            calculate_natural_horizontal_emittance(
                energy=self.beam_energy,
                synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                particle_type=self.particle_type,
            ),
            places=6 if backend.float == np.float32 else 12,
        )

    def test_calculate_natural_energy_spread(self):
        self.assertAlmostEqual(
            0.5,
            calculate_natural_energy_spread(
                energy=self.beam_energy,
                synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                particle_type=self.particle_type,
            ),
            places=6 if backend.float == np.float32 else 12,
        )

    def test_calculate_natural_bunch_length(self):
        self.assertAlmostEqual(
            0.5,
            calculate_natural_bunch_length(
                energy=self.beam_energy,
                synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                angular_synchrotron_frequency=self.angular_synchrotron_frequency,
                momentum_compaction_factor=self.momentum_compaction_factor,
                particle_type=self.particle_type,
            ),
            places=6 if backend.float == np.float32 else 12,
        )


class TestSynchrotronRadiationMaths_array_inputs(unittest.TestCase):
    def setUp(self):
        # Example of the FCC-ee high-energy booster at injection
        self.particle_type = ParticleType(mass=1, charge=-1)
        self.beam_energy = np.array([1.0, 1.0, 1.0, 0.0, 1.0])
        self.synchrotron_radiation_integrals = np.array(
            [
                1,
                2 * np.pi / self.particle_type.sands_radiation_constant,
                np.pi
                / (
                    self.particle_type.quantum_radiation_constant
                    * self.particle_type.sands_radiation_constant
                ),
                0,
                2
                * np.pi
                / (
                    self.particle_type.sands_radiation_constant
                    * self.particle_type.quantum_radiation_constant
                ),
            ]
        )
        self.revolution_frequency = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        self.energy_lost_per_turn = 1
        self.angular_synchrotron_frequency = 1
        self.momentum_compaction_factor = 1 / c

    def test_calculate_partition_numbers(self):
        jx, jy, jz = calculate_partition_numbers(
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals
        )
        jx_1 = calculate_horizontal_damping_partition_number(
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
        )
        jz_1 = calculate_longitudinal_damping_partition_number(
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
        )
        self.assertEqual(1.0, jy)
        self.assertEqual(jx, jx_1, msg="Expected value = 1")
        self.assertEqual(jz, jz_1, msg="Expected value = 2")

    def test_calculate_damping_times_in_turn(self):
        damping_times_in_turn = calculate_damping_times_in_turns(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
        )
        tau_x = calculate_horizontal_damping_time_in_turns(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
        )
        tau_z = calculate_longitudinal_damping_time_in_turns(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
        )
        np.testing.assert_array_almost_equal(
            np.array([2.0, 2.0, 2.0, 0.0, 2.0]),
            damping_times_in_turn[1],
            decimal=6 if backend.float == np.float32 else 12,
        )
        np.testing.assert_array_almost_equal(
            damping_times_in_turn[0],
            tau_x,
            decimal=6 if backend.float == np.float32 else 12,
        )
        np.testing.assert_array_almost_equal(
            damping_times_in_turn[2],
            tau_z,
            decimal=6 if backend.float == np.float32 else 12,
        )

    def test_calculate_damping_times_in_seconds(self):
        # todo: test assert identical lengths for revolution frequency and
        # energy arrays
        damping_times_in_second = calculate_damping_times_in_seconds(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
            revolution_frequency=self.revolution_frequency,
        )
        tau_x = calculate_horizontal_damping_time_in_seconds(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
            revolution_frequency=self.revolution_frequency,
        )
        tau_z = calculate_longitudinal_damping_time_in_seconds(
            energy=self.beam_energy,
            synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
            energy_loss_per_turn=self.energy_lost_per_turn,
            revolution_frequency=self.revolution_frequency,
        )
        np.testing.assert_array_almost_equal(
            np.array([2.0, 2.0, 2.0, 0.0, 2.0]),
            damping_times_in_second[1],
            decimal=6 if backend.float == np.float32 else 12,
        )
        np.testing.assert_array_almost_equal(
            damping_times_in_second[0],
            tau_x,
            decimal=6 if backend.float == np.float32 else 12,
        )
        np.testing.assert_array_almost_equal(
            damping_times_in_second[2],
            tau_z,
            decimal=6 if backend.float == np.float32 else 12,
        )

    def test_calculate_energy_loss_per_turn(self):
        np.testing.assert_array_almost_equal(
            np.array([1.0, 1.0, 1.0, 0.0, 1.0]),
            calculate_energy_loss_per_turn(
                energy=self.beam_energy,
                synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                particle_type=self.particle_type,
            ),
            decimal=6 if backend.float == np.float32 else 12,
        )

    def test_calculate_natural_horizontal_emittance(self):
        np.testing.assert_array_almost_equal(
            np.array([1.0, 1.0, 1.0, 0.0, 1.0]),
            calculate_natural_horizontal_emittance(
                energy=self.beam_energy,
                synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                particle_type=self.particle_type,
            ),
            decimal=6 if backend.float == np.float32 else 12,
        )

    def test_calculate_natural_energy_spread(self):
        np.testing.assert_array_almost_equal(
            np.array([0.5, 0.5, 0.5, 0.0, 0.5]),
            calculate_natural_energy_spread(
                energy=self.beam_energy,
                synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                particle_type=self.particle_type,
            ),
            decimal=6 if backend.float == np.float32 else 12,
        )

    def test_calculate_natural_bunch_length(self):
        np.testing.assert_array_almost_equal(
            np.array([0.5, 0.5, 0.5, 0.0, 0.5]),
            calculate_natural_bunch_length(
                energy=self.beam_energy,
                synchrotron_radiation_integrals=self.synchrotron_radiation_integrals,
                angular_synchrotron_frequency=self.angular_synchrotron_frequency,
                momentum_compaction_factor=self.momentum_compaction_factor,
                particle_type=self.particle_type,
            ),
            decimal=6 if backend.float == np.float32 else 12,
        )
