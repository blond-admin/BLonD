import unittest

import numpy as np

from blond import positron
from blond.specifics.fccee.generate_rings import (
    generate_fccee_booster_basic_simulation,
)


class TestFCCColliderBasicSimulation(unittest.TestCase):
    def setUp(self):
        self.simZ = generate_fccee_booster_basic_simulation()
        self.simW = generate_fccee_booster_basic_simulation(
            operation_mode="WW"
        )
        self.simZH = generate_fccee_booster_basic_simulation(
            operation_mode="ZH"
        )
        self.simttbar = generate_fccee_booster_basic_simulation(
            operation_mode="ttbar"
        )
        self.radiation_integrals = np.array(
            [
                0.646747216157,
                0.0005936549319,
                5.6814536525e-08,
                5.92870407301e-09,
                1.698280783e-11,
            ]
        )

    def test_inputs(self):
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="Operation mode not recognised.",
        ):
            generate_fccee_booster_basic_simulation(
                operation_mode="not an expected operation mode"
            )

    def test_simulation_Z(self):
        self.assertEqual(self.simZ.ring.circumference, 90.65874532 * 1e3)
        np.testing.assert_array_equal(
            self.simZ.ring.radiation_integrals,
            self.radiation_integrals,
        )
        self.assertEqual(self.simZ.magnetic_cycle.reference_particle, positron)
        self.assertEqual(
            self.simZ.magnetic_cycle.get_target_total_energy(
                turn_i=0, section_i=0, reference_time=0, particle_type=positron
            ),
            20e9,
        )

    def test_simulation_W(self):
        self.assertEqual(self.simW.ring.circumference, 90.65874532 * 1e3)
        np.testing.assert_array_equal(
            self.simW.ring.radiation_integrals,
            self.radiation_integrals,
        )
        self.assertEqual(self.simW.magnetic_cycle.reference_particle, positron)
        self.assertEqual(
            self.simW.magnetic_cycle.get_target_total_energy(
                turn_i=0, section_i=0, reference_time=0, particle_type=positron
            ),
            20e9,
        )

    def test_simulation_ZH(self):
        self.assertEqual(self.simZH.ring.circumference, 90.65874532 * 1e3)
        np.testing.assert_array_equal(
            self.simW.ring.radiation_integrals,
            self.radiation_integrals,
        )
        self.assertEqual(
            self.simZH.magnetic_cycle.reference_particle, positron
        )
        self.assertEqual(
            self.simZH.magnetic_cycle.get_target_total_energy(
                turn_i=0, section_i=0, reference_time=0, particle_type=positron
            ),
            20e9,
        )

    def test_simulation_ttbar(self):
        self.assertEqual(self.simttbar.ring.circumference, 90.65874532 * 1e3)
        np.testing.assert_array_equal(
            self.simW.ring.radiation_integrals,
            self.radiation_integrals,
        )
        self.assertEqual(
            self.simttbar.magnetic_cycle.reference_particle, positron
        )
        self.assertEqual(
            self.simttbar.magnetic_cycle.get_target_total_energy(
                turn_i=0, section_i=0, reference_time=0, particle_type=positron
            ),
            20e9,
        )
