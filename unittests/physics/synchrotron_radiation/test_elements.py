import unittest
from unittest.mock import Mock

import numpy as np
from scipy.constants import speed_of_light as c0

from blond import (
    SynchrotronRadiationBaseClass,
    SynchrotronRadiationDrift,
    SynchrotronRadiationSection,
    electron,
)
from blond.acc_math.analytic.synchrotron_radiation.utilities import (
    gather_longitudinal_synchrotron_radiation_parameters,
)
from blond.core.backends.backend import backend
from blond.core.beam.base import BeamBaseClass


class TestRFStationBaseClass(unittest.TestCase):
    def setUp(self) -> None:
        radiation_integrals = np.array(
            [
                0.646747216157,
                0.0005936549319,
                5.6814536525e-08,
                5.92870407301e-09,
                1.71368060083e-11,
            ]
        )
        self.SRB = SynchrotronRadiationBaseClass(share_of_synchrotron_radiation_integrals = 0.1 * radiation_integrals)
        self.SRD = SynchrotronRadiationDrift(share_of_synchrotron_radiation_integrals
                                             = 0.1 * radiation_integrals )
        self.SRS = SynchrotronRadiationSection(share_of_synchrotron_radiation_integrals
                                             = 0.1 * radiation_integrals )

        self.beam = Mock(BeamBaseClass)
        self.beam.particle_type = electron
        self.beam.reference_time = 0
        self.beam.reference_beta = 0.99
        self.beam.reference_velocity = self.beam.reference_beta * c0
        self.beam.reference_gamma = np.sqrt(1 - 0.99**2)  # beta**2
        self.beam.reference_total_energy = 20e9
        self.beam.dE = np.linspace(
            -1e6, 1e6, 10, dtype=backend.float
        )  # delta E
        # in eV
        self.beam.dt = np.linspace(
            -1e-6, 1e-6, 10, dtype=backend.float
        )  # delta t
        # in s
        self.beam.n_macroparticles_partial.return_value = 10
        self.beam.read_partial_dE.return_value = self.beam.dE

        self.decimal = 6 if backend.float == np.float32 else 12

        self.U0, self.tau_z, self.sigma0 = gather_longitudinal_synchrotron_radiation_parameters(
                particle_type=self.beam.particle_type,
                energy=self.beam.reference_total_energy,
                synchrotron_radiation_integrals=radiation_integrals,
            )

        self.seed = 500

    def test_calculate_kick(self):
        np.random.seed(seed=self.seed)
        energy_kick_from_base_class = self.SRB._calculate_kick(beam =
                                                               self.beam,
                                                               seed =
                                                               self.seed)
        self.assertAlmostEqual(self.SRB._energy_lost_due_to_synchrotron_radiation,
                         np.float64(133731.76297928384),
                         places = self.decimal)
        self.assertAlmostEqual(self.SRB._damping_time,
                               np.float64(149552.35530506275),
                         places = self.decimal)
        self.assertAlmostEqual(self.SRB._natural_energy_spread,
                         np.float64(0.0001675968578478592),
                         places = self.decimal)

        #TODO test random generation
        # expected_energy_kick = (-2.0 / (self.tau_z * 10) *
        #                                 self.beam.read_partial_dE(
        # ) - 2.0 * self.sigma0 /
        #  np.sqrt((self.tau_z * 10))
        #                         * self.beam.reference_total_energy *
        # np.random.normal(size=self.beam.n_macroparticles_partial()))
        # np.testing.assert_almost_equal(energy_kick_from_base_class,
        #                                expected_energy_kick,
        #                                decimal = self.decimal,
        #                                )

    def test_update_beam_energy(self):
        previous_energy = self.beam.read_partial_dE()

        self.SRB._update_beam_energy(beam = self.beam)

        new_energy = self.beam.read_partial_dE()
