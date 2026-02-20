import unittest

import numpy as np
from scipy.constants import c

from blond import backend, electron
from blond.acc_math.analytic.synchrotron_radiation.synchrotron_radiation_maths import *
from blond.core.beam.particle_types import ParticleType
from blond.generals.exceptions import UnevenArraySizes


class TestSynchrotronRadiationMaths_float_inputs(unittest.TestCase):
    def setUp(self):
        # FCC-ee high-energy booster at injection energy
        self.particle_type = electron
        self.beam_energy = 20e9
        self.radiation_integrals = np.array(
            [
                0.646747216157,
                0.000593654931851,
                5.6814536525e-08,
                5.92870407301e-09,
                1.698280783e-11,
            ]
        )
        self.revolution_frequency = 3300
        self.angular_synchrotron_frequency = 1
        self.momentum_compaction_factor = 7.120435962 * 1e-6
        self.decimals = 6 if backend.float == np.float32 else 12

    def test_calculate_partition_numbers(self):
        jx, jy, jz = calculate_partition_numbers(
            radiation_integrals=self.radiation_integrals
        )
        jx_1 = calculate_horizontal_damping_partition_number(
            radiation_integrals=self.radiation_integrals,
        )
        jz_1 = calculate_longitudinal_damping_partition_number(
            radiation_integrals=self.radiation_integrals,
        )
        self.assertEqual(1.0, jy)
        self.assertEqual(jx, jx_1, msg="Expected value ~= 1")
        self.assertEqual(jz, jz_1, msg="Expected value ~= 2")

    def test_calculate_damping_times_in_turn(self):
        damping_times_in_turn = calculate_damping_times_in_turns(
            energy=self.beam_energy,
            radiation_integrals=self.radiation_integrals,
            particle_type=self.particle_type,
        )
        tau_x = calculate_horizontal_damping_time_in_turns(
            energy=self.beam_energy,
            radiation_integrals=self.radiation_integrals,
            particle_type=self.particle_type,
        )
        tau_z = calculate_longitudinal_damping_time_in_turns(
            energy=self.beam_energy,
            radiation_integrals=self.radiation_integrals,
            particle_type=self.particle_type,
        )
        self.assertEqual(
            29910.62041820081,
            damping_times_in_turn[1],
            msg="Expected value = 29911",
        )
        self.assertEqual(
            damping_times_in_turn[0], tau_x, msg="Expected value = 29911"
        )
        self.assertEqual(
            damping_times_in_turn[2], tau_z, msg="Expected value = 14955"
        )

    def test_calculate_damping_times_in_seconds(self):
        damping_times_in_second = calculate_damping_times_in_seconds(
            energy=self.beam_energy,
            radiation_integrals=self.radiation_integrals,
            particle_type=self.particle_type,
            revolution_frequency=self.revolution_frequency,
        )
        tau_x = calculate_horizontal_damping_time_in_seconds(
            energy=self.beam_energy,
            radiation_integrals=self.radiation_integrals,
            particle_type=self.particle_type,
            revolution_frequency=self.revolution_frequency,
        )
        tau_z = calculate_longitudinal_damping_time_in_seconds(
            energy=self.beam_energy,
            radiation_integrals=self.radiation_integrals,
            particle_type=self.particle_type,
            revolution_frequency=self.revolution_frequency,
        )
        self.assertAlmostEqual(
            9.06382436915176,
            damping_times_in_second[1],
            msg="Expected value = 9.1s",
            places=self.decimals,
        )
        self.assertEqual(
            damping_times_in_second[0], tau_x, msg="Expected value = 9.1s"
        )
        self.assertEqual(
            damping_times_in_second[2], tau_z, msg="Expected value = 4.5s"
        )
        self.decimals = 6 if backend.float == np.float32 else 12

    def test_calculate_energy_loss_per_turn(self):
        self.assertAlmostEqual(
            1337317.6296824566,
            calculate_energy_loss_per_turn(
                energy=self.beam_energy,
                radiation_integrals=self.radiation_integrals,
                particle_type=self.particle_type,
            ),
            places=self.decimals,
        )

    def test_calculate_natural_horizontal_emittance(self):
        self.assertAlmostEqual(
            1.6792612747193685e-11,
            calculate_natural_horizontal_emittance(
                energy=self.beam_energy,
                radiation_integrals=self.radiation_integrals,
                particle_type=self.particle_type,
            ),
            places=self.decimals,
        )

    def test_calculate_natural_energy_spread(self):
        self.assertAlmostEqual(
            0.00016759685785477585,
            calculate_natural_energy_spread(
                energy=self.beam_energy,
                radiation_integrals=self.radiation_integrals,
                particle_type=self.particle_type,
            ),
            places=self.decimals,
        )

    def test_calculate_natural_bunch_length(self):
        self.assertAlmostEqual(
            0.35776113525601044,
            calculate_natural_bunch_length(
                energy=self.beam_energy,
                radiation_integrals=self.radiation_integrals,
                angular_synchrotron_frequency=self.angular_synchrotron_frequency,
                momentum_compaction_factor=self.momentum_compaction_factor,
                particle_type=self.particle_type,
            ),
            places=self.decimals,
        )


class TestSynchrotronRadiationMaths_array_inputs(unittest.TestCase):
    def setUp(self):
        # Example of the FCC-ee high-energy booster at injection
        self.particle_type = ParticleType(mass=1, charge=-1)
        self.beam_energy = np.array([1.0, 1.0, 1.0, 10, 1.0])
        self.radiation_integrals = np.array(
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
        self.decimals = 6 if backend.float == np.float32 else 12

    def test_calculate_partition_numbers(self):
        jx, jy, jz = calculate_partition_numbers(
            radiation_integrals=self.radiation_integrals
        )
        jx_1 = calculate_horizontal_damping_partition_number(
            radiation_integrals=self.radiation_integrals,
        )
        jz_1 = calculate_longitudinal_damping_partition_number(
            radiation_integrals=self.radiation_integrals,
        )
        self.assertEqual(1.0, jy)
        self.assertEqual(jx, jx_1, msg="Expected value = 1")
        self.assertEqual(jz, jz_1, msg="Expected value = 2")

    def test_calculate_damping_times_in_turn(self):
        damping_times_in_turn = calculate_damping_times_in_turns(
            energy=self.beam_energy,
            radiation_integrals=self.radiation_integrals,
            particle_type=self.particle_type,
        )
        tau_x = calculate_horizontal_damping_time_in_turns(
            energy=self.beam_energy,
            radiation_integrals=self.radiation_integrals,
            particle_type=self.particle_type,
        )
        tau_z = calculate_longitudinal_damping_time_in_turns(
            energy=self.beam_energy,
            radiation_integrals=self.radiation_integrals,
            particle_type=self.particle_type,
        )
        np.testing.assert_array_almost_equal(
            np.array([2.0, 2.0, 2.0, 0.002, 2.0]),
            damping_times_in_turn[1],
            decimal=self.decimals,
        )
        np.testing.assert_array_almost_equal(
            damping_times_in_turn[0],
            tau_x,
            decimal=self.decimals,
        )
        np.testing.assert_array_almost_equal(
            damping_times_in_turn[2],
            tau_z,
            decimal=self.decimals,
        )

    def test_calculate_damping_times_in_seconds(self):
        with self.assertRaises(UnevenArraySizes):
            calculate_damping_times_in_seconds(
                energy=self.beam_energy,
                radiation_integrals=self.radiation_integrals,
                particle_type=self.particle_type,
                revolution_frequency=self.revolution_frequency[0:2],
            )
        damping_times_in_second = calculate_damping_times_in_seconds(
            energy=self.beam_energy,
            radiation_integrals=self.radiation_integrals,
            particle_type=self.particle_type,
            revolution_frequency=self.revolution_frequency,
        )
        tau_x = calculate_horizontal_damping_time_in_seconds(
            energy=self.beam_energy,
            radiation_integrals=self.radiation_integrals,
            particle_type=self.particle_type,
            revolution_frequency=self.revolution_frequency,
        )
        tau_z = calculate_longitudinal_damping_time_in_seconds(
            energy=self.beam_energy,
            radiation_integrals=self.radiation_integrals,
            particle_type=self.particle_type,
            revolution_frequency=self.revolution_frequency,
        )
        np.testing.assert_array_almost_equal(
            np.array([2.0, 2.0, 2.0, 0.002, 2.0]),
            damping_times_in_second[1],
            decimal=self.decimals,
        )
        np.testing.assert_array_almost_equal(
            damping_times_in_second[0],
            tau_x,
            decimal=self.decimals,
        )
        np.testing.assert_array_almost_equal(
            damping_times_in_second[2],
            tau_z,
            decimal=self.decimals,
        )

    def test_calculate_energy_loss_per_turn(self):
        np.testing.assert_array_almost_equal(
            np.array([1.0, 1.0, 1.0, 10000, 1.0]),
            calculate_energy_loss_per_turn(
                energy=self.beam_energy,
                radiation_integrals=self.radiation_integrals,
                particle_type=self.particle_type,
            ),
            decimal=self.decimals,
        )

    def test_calculate_natural_horizontal_emittance(self):
        np.testing.assert_array_almost_equal(
            np.array([1.0, 1.0, 1.0, 100.0, 1.0]),
            calculate_natural_horizontal_emittance(
                energy=self.beam_energy,
                radiation_integrals=self.radiation_integrals,
                particle_type=self.particle_type,
            ),
            decimal=self.decimals,
        )

    def test_calculate_natural_energy_spread(self):
        np.testing.assert_array_almost_equal(
            np.array([0.5, 0.5, 0.5, 5.0, 0.5]),
            calculate_natural_energy_spread(
                energy=self.beam_energy,
                radiation_integrals=self.radiation_integrals,
                particle_type=self.particle_type,
            ),
            decimal=self.decimals,
        )

    def test_calculate_natural_bunch_length(self):
        np.testing.assert_array_almost_equal(
            np.array([0.5, 0.5, 0.5, 5.0, 0.5]),
            calculate_natural_bunch_length(
                energy=self.beam_energy,
                radiation_integrals=self.radiation_integrals,
                angular_synchrotron_frequency=self.angular_synchrotron_frequency,
                momentum_compaction_factor=self.momentum_compaction_factor,
                particle_type=self.particle_type,
            ),
            decimal=self.decimals,
        )
