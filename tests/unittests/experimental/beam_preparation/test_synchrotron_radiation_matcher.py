import unittest

import numpy as np

from blond import (
    Beam,
    DriftSimple,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    positron,
)
from blond.experimental.beam_preparation.synchrotron_radiation_matcher import (
    SynchrotronRadiationMatcher,
    sawtooth_factor,
)
from blond.physics.synchrotron_radiation.synchrotron_radiation_master import (
    SynchrotronRadiationMaster,
)


class TestFunctions(unittest.TestCase):
    def test_sawtooth_factor(self):
        # n_sections = 1, sr+drift
        self.assertEqual(sawtooth_factor(1, "sr+drift"), 0.0)
        # n_sections = 2, sr+drift
        self.assertEqual(sawtooth_factor(2, "sr+drift"), 0.25)
        # n_sections = 1, drift+sr
        self.assertEqual(sawtooth_factor(1, "drift+sr"), 1.0)
        # invalid order
        with self.assertRaisesRegex(ValueError, "The order should either be"):
            sawtooth_factor(1, "invalid")


class TestSynchrotronRadiationMatcher(unittest.TestCase):
    def setUp(self):
        self.radiation_integrals = np.array(
            [
                0.646747216157,
                0.0005936549319,
                5.6814536525e-08,
                5.92870407301e-09,
                1.71368060083e-11,
            ]
        )
        self.circumference = 90.65874532 * 1e3
        self.momentum_compaction_factor = (
            self.radiation_integrals[0] / self.circumference
        )
        self.reference_energy = 20e9

        self.ring = Ring(
            self.circumference,
            radiation_integrals=self.radiation_integrals,
        )

        self.rf = SingleHarmonicRFStation()
        self.rf.harmonic = 242400
        self.rf.voltage = 50.1e6
        self.rf.phi_rf_design = 0

        self.drift = DriftSimple(
            orbit_length=self.circumference,
            momentum_compaction_factor=self.momentum_compaction_factor,
        )

        self.ring.add_element(self.rf)
        self.ring.add_element(self.drift)

        self.sr_master = SynchrotronRadiationMaster()
        self.sr_master.prepare_ring_for_synchrotron_radiation_tracking(
            ring=self.ring
        )

        self.energy = self.reference_energy
        self.cycle = MagneticCyclePerTurn(
            value_init=self.energy,
            values_after_turn=np.array([self.energy, self.energy]),
            reference_particle=positron,
            in_unit="total energy",
        )
        self.beam = Beam(intensity=2.725e10, particle_type=positron)
        self.simulation = Simulation(ring=self.ring, magnetic_cycle=self.cycle)

    def test___init__(self):
        matcher = SynchrotronRadiationMatcher(
            synchrotron_radiation_master=self.sr_master,
            n_macroparticles=10,
            seed=42,
        )
        self.assertEqual(matcher._n_macroparticles_local, 10)
        self.assertEqual(matcher._seed, 42)
        self.assertIsInstance(matcher._sr_master, SynchrotronRadiationMaster)

    def test_prepare_beam_invalid_layout(self):
        ring_invalid = Ring(self.circumference)

        one_turn_execution_order = (
            DriftSimple(
                momentum_compaction_factor=self.momentum_compaction_factor,
                orbit_length=0.3 * self.circumference,
                section_index=0,
            ),
            SingleHarmonicRFStation(
                harmonic=self.rf.harmonic,
                phi_rf=self.rf.phi_rf_design,
                voltage=self.rf.voltage,
                section_index=0,
            ),
            DriftSimple(
                momentum_compaction_factor=self.momentum_compaction_factor,
                orbit_length=0.7 * self.circumference,
                section_index=1,
            ),
            SingleHarmonicRFStation(
                harmonic=self.rf.harmonic,
                phi_rf=self.rf.phi_rf_design,
                voltage=self.rf.voltage,
                section_index=1,
            ),
        )
        ring_invalid.add_elements(one_turn_execution_order, reorder=False)
        sim_invalid = Simulation(ring=ring_invalid, magnetic_cycle=self.cycle)

        matcher = SynchrotronRadiationMatcher(
            synchrotron_radiation_master=self.sr_master,
            n_macroparticles=10,
        )
        with self.assertRaisesRegex(ValueError, "presently only implemented"):
            matcher.prepare_beam(simulation=sim_invalid, beam=self.beam)

    def test_prepare_beam(self):
        matcher = SynchrotronRadiationMatcher(
            synchrotron_radiation_master=self.sr_master,
            n_macroparticles=1e3,
            seed=42,
        )
        matcher.prepare_beam(simulation=self.simulation, beam=self.beam)

        self.assertEqual(len(self.beam.read_partial_dt()), int(1e3))
        self.assertEqual(len(self.beam.read_partial_dE()), int(1e3))

        # Check generated distribution stats against theoretical expectations
        all_base_params = matcher.get_all_base_params(
            self.simulation, self.beam
        )
        cov = matcher.compute_covariance_matrix(all_base_params)
        beta_cs = cov[0, 0]
        gamma_cs = cov[1, 1]

        expected_std_dE = (
            all_base_params["sigma_dE"] * all_base_params["energy"]
        )
        expected_std_dt = expected_std_dE * np.sqrt(beta_cs / gamma_cs)

        np.testing.assert_allclose(
            float(np.std(self.beam.read_partial_dt())),
            expected_std_dt,
            rtol=0.1,
        )
        np.testing.assert_allclose(
            float(np.std(self.beam.read_partial_dE())),
            expected_std_dE,
            rtol=0.1,
        )

    def test_compute_covariance_matrix(self):
        matcher = SynchrotronRadiationMatcher(
            synchrotron_radiation_master=self.sr_master,
            n_macroparticles=10,
        )
        params = {
            "energy": 1e9,
            "charge": 1,
            "rf_voltage": 1e6,
            "energy_loss_per_turn": 1e3,
            "sigma_dE": 1e-3,
            "beta": 1.0,
            "eta_0": 1e-3,
            "t_rev": 1e-5,
            "t_rf": 1e-6,
            "omega_rf": 2 * np.pi * 1e6,
            "phi_s": np.pi,  # Use pi for stable phase above transition to avoid invalid values
        }
        cov = matcher.compute_covariance_matrix(params)

        self.assertEqual(cov.shape, (2, 2))
        self.assertAlmostEqual(cov[0, 1], cov[1, 0])  # symmetric

        # Expected values computed analytically for the Courant-Snyder parameters
        expected_beta_cs = 1.2615662610100802e-15
        expected_gamma_cs = 792665459521222.2
        expected_alpha_cs = -0.003963337582522509

        np.testing.assert_allclose(
            cov,
            np.array(
                [
                    [expected_beta_cs, -expected_alpha_cs],
                    [-expected_alpha_cs, expected_gamma_cs],
                ]
            ),
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
