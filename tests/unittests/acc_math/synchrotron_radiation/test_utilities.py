import unittest

import numpy as np

from blond import backend, electron
from blond.acc_math.analytic.synchrotron_radiation.utilities import (
    calculate_isomagnetic_radiation_integrals,
    gather_longitudinal_synchrotron_radiation_parameters,
)


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

        self.circumference = 90.65874532 * 1e3
        self.momentum_compaction_factor = 7.120435962 * 1e-6
        self.bending_radius = 14428.78745218723
        self.decimals = 6 if backend.float == np.float32 else 12

    def test_gather_longitudinal_synchrotron_radiation_parameters(self):
        energy_loss, tau_z, sigmaE = (
            gather_longitudinal_synchrotron_radiation_parameters(
                energy=self.beam_energy,
                radiation_integrals=self.radiation_integrals,
                particle_type=self.particle_type,
            )
        )

        self.assertAlmostEqual(
            1337317.6296824566,
            energy_loss,
            msg="Expected value = 1.3 MeV per turn",
            places=self.decimals,
        )
        self.assertAlmostEqual(
            14955.235531740671,
            tau_z,
            msg="Expected value = 14955 turns i.e. 4.5s",
            places=self.decimals,
        )
        self.assertAlmostEqual(
            0.00016759685785477585,
            sigmaE,
            msg="Expected value = 1.6e-4",
            places=self.decimals,
        )

    def test_calculate_isomagnetic_radiation_integrals(self):
        np.testing.assert_array_almost_equal(
            np.array(
                [
                    0.6455297904463271,
                    0.0004354617689116441,
                    3.018006678347967e-08,
                    3.100677639434604e-09,
                    0,
                ]
            ),
            calculate_isomagnetic_radiation_integrals(
                circumference=self.circumference,
                bending_radius=self.bending_radius,
                momentum_compaction_factor=self.momentum_compaction_factor,
            ),
            decimal=self.decimals,
        )
