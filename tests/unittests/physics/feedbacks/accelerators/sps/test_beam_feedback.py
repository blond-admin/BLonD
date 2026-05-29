import unittest

import numpy as np

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    MultiHarmonicRFStation,
    Ring,
    Simulation,
    StaticProfile,
    backend,
    proton,
)
from blond.core.backends.backend import Numpy64Bit
from blond.physics.feedbacks.accelerators.sps import (
    SPSBeamControl,
)

circumference = 2 * np.pi * 1100.009  # [m]
momentum = 25.92e9
intensity = 1.6e11
n_turns = 2_000
h = 4620
gamma_t = 17.95
alpha = 1 / gamma_t / gamma_t

n_macroparticles = 100_000
tau_bunch = 1.2e-9
injection_offset_phase = 20
reference = -20

voltage_200 = 4.4788e6
voltage_800 = 0.1 * voltage_200
phase_200 = 0.0
phase_800 = np.pi

k_phi_n = 8810.900275164733
k_phi_nm1 = 70.59968917093718
k_eps_n = 278.1481404221182
k_z_n = 0.8881084849451452
k_a_n = 70.59968917093718
k_b_n = 0.0
action_delay = 0


class TestSPSBeamFeedback(unittest.TestCase):
    def create_scenario(
        self,
        k_phi_n,
        k_phi_nm1,
        k_eps_n,
        k_z_n,
        k_a_n,
        k_b_n,
        phi_sync,
        pl_gain,
    ):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")

        energy = np.sqrt(momentum**2 + proton.mass**2)
        rel_gamma = energy / proton.mass
        rel_beta = np.sqrt(1 - 1 / rel_gamma**2)

        beam = Beam(
            intensity,
            proton,
        )

        cycle = ConstantMagneticCycle(proton, momentum, in_unit="momentum")

        lattice = DriftSimple(
            orbit_length=circumference, momentum_compaction_factor=alpha
        )

        cavity = MultiHarmonicRFStation(
            voltage=np.array([voltage_200, voltage_800]),
            phi_rf=np.array([phase_200, phase_800]),
            harmonic=np.array([h, 4 * h]),
            n_harmonics=2,
            main_harmonic_idx=0,
        )

        f_rf = cavity.calc_main_harmonic_omega_rf_design(
            rel_beta, lattice.orbit_length
        ) / (2 * np.pi)
        f_rev = f_rf / h
        t_rf = 1 / f_rf
        t_rev = 1 / f_rev

        self.profile = StaticProfile(
            cut_left=-1.5 * t_rf,
            cut_right=2.5 * t_rf,
            n_bins=4 * 2**6,
        )

        bigaussian = BiGaussian(
            n_macroparticles, sigma_dt=tau_bunch / 4, seed=1234
        )
        self.beam_control = SPSBeamControl(
            profile=self.profile,
            k_phi_n=k_phi_n,
            k_phi_nm1=k_phi_nm1,
            k_eps_n=k_eps_n,
            k_z_n=k_z_n,
            k_a_n=k_a_n,
            k_b_n=k_b_n,
            phi_sync=phi_sync,
            pl_gain=pl_gain,
            action_delay=action_delay,
        )

        cavity.attach_beam_feedback(self.beam_control)

        ring = Ring(
            circumference,
        )

        ring.add_elements(
            [self.profile, cavity, self.beam_control, lattice],
        )

        simulation = Simulation(
            ring,
            cycle,
        )

        simulation.prepare_beam(beam, bigaussian)

        beam._dt.array_local += injection_offset_phase * t_rf / 360

        self.profile.track(beam)

        simulation.finalize(
            (beam,),
            n_turns,
        )
        self.beam_control.reference = reference * np.pi / 180

        self.beam_control.track(beam)

    def test_sps_beam_control_with_single_values(self):
        self.create_scenario(
            k_phi_n=k_phi_n,
            k_phi_nm1=k_phi_nm1,
            k_eps_n=k_eps_n,
            k_z_n=k_z_n,
            k_a_n=k_a_n,
            k_b_n=k_b_n,
            phi_sync=reference * np.pi / 180,
            pl_gain=1.0,
        )

        # Checks the correction calculation of the recursion parameters for the synchronization loop
        np.testing.assert_array_equal(
            self.beam_control.k_phi_n, k_phi_n * np.ones(n_turns + 1)
        )

        np.testing.assert_array_equal(
            self.beam_control.k_phi_nm1, k_phi_nm1 * np.ones(n_turns + 1)
        )

        np.testing.assert_array_equal(
            self.beam_control.k_eps_n, k_eps_n * np.ones(n_turns + 1)
        )

        np.testing.assert_array_equal(
            self.beam_control.k_z_n, k_z_n * np.ones(n_turns + 1)
        )

        np.testing.assert_array_equal(
            self.beam_control.k_a_n, k_a_n * np.ones(n_turns + 1)
        )

        np.testing.assert_array_equal(
            self.beam_control.k_b_n, k_b_n * np.ones(n_turns + 1)
        )

        # Beam-phase loop output
        self.assertAlmostEqual(
            self.beam_control.domega_dphi, -3073.547859760765
        )

        # Synchro loop output
        self.assertAlmostEqual(
            self.beam_control.domega_sync, -97.09201717330986
        )

        # Frequency loop output
        self.assertAlmostEqual(self.beam_control.domega_freq, -0.0)

        # Total correction from beam control
        self.assertAlmostEqual(self.beam_control.domega_rf, 0.0)

        np.testing.assert_almost_equal(
            self.beam_control.domega_rf_corr,
            [-3170.639876934075, 0.0],
            decimal=10,
        )

    def test_sps_beam_control_with_arrays(self):
        self.create_scenario(
            k_phi_n=k_phi_n * np.ones(n_turns + 1),
            k_phi_nm1=k_phi_nm1 * np.ones(n_turns + 1),
            k_eps_n=k_eps_n * np.ones(n_turns + 1),
            k_z_n=k_z_n * np.ones(n_turns + 1),
            k_a_n=k_a_n * np.ones(n_turns + 1),
            k_b_n=k_b_n * np.ones(n_turns + 1),
            phi_sync=reference * np.pi / 180 * np.ones(n_turns + 1),
            pl_gain=1.0 * np.ones(n_turns + 1),
        )

        # Checks the correction calculation of the recursion parameters for the synchronization loop
        np.testing.assert_array_equal(
            self.beam_control.k_phi_n, k_phi_n * np.ones(n_turns + 1)
        )

        np.testing.assert_array_equal(
            self.beam_control.k_phi_nm1, k_phi_nm1 * np.ones(n_turns + 1)
        )

        np.testing.assert_array_equal(
            self.beam_control.k_eps_n, k_eps_n * np.ones(n_turns + 1)
        )

        np.testing.assert_array_equal(
            self.beam_control.k_z_n, k_z_n * np.ones(n_turns + 1)
        )

        np.testing.assert_array_equal(
            self.beam_control.k_a_n, k_a_n * np.ones(n_turns + 1)
        )

        np.testing.assert_array_equal(
            self.beam_control.k_b_n, k_b_n * np.ones(n_turns + 1)
        )

        # Beam-phase loop output
        self.assertAlmostEqual(
            self.beam_control.domega_dphi, -3073.547859760765
        )

        # Synchro loop output
        self.assertAlmostEqual(
            self.beam_control.domega_sync, -97.09201717330986
        )

        # Frequency loop output
        self.assertAlmostEqual(self.beam_control.domega_freq, -0.0)

        # Total correction from beam control
        self.assertAlmostEqual(self.beam_control.domega_rf, 0.0)

        np.testing.assert_almost_equal(
            self.beam_control.domega_rf_corr,
            [-3170.639876934075, 0.0],
            decimal=10,
        )

    def test_sps_beam_control_incorrect_arrays(self):
        with self.assertRaises(ValueError):
            self.create_scenario(
                k_phi_n=k_phi_n * np.ones(n_turns),
                k_phi_nm1=k_phi_nm1 * np.ones(n_turns + 1),
                k_eps_n=k_eps_n * np.ones(n_turns + 1),
                k_z_n=k_z_n * np.ones(n_turns + 1),
                k_a_n=k_a_n * np.ones(n_turns + 1),
                k_b_n=k_b_n * np.ones(n_turns + 1),
                phi_sync=reference * np.pi / 180 * np.ones(n_turns + 1),
                pl_gain=1.0 * np.ones(n_turns + 1),
            )

        with self.assertRaises(ValueError):
            self.create_scenario(
                k_phi_n=k_phi_n * np.ones(n_turns + 1),
                k_phi_nm1=k_phi_nm1 * np.ones(n_turns),
                k_eps_n=k_eps_n * np.ones(n_turns + 1),
                k_z_n=k_z_n * np.ones(n_turns + 1),
                k_a_n=k_a_n * np.ones(n_turns + 1),
                k_b_n=k_b_n * np.ones(n_turns + 1),
                phi_sync=reference * np.pi / 180 * np.ones(n_turns + 1),
                pl_gain=1.0 * np.ones(n_turns + 1),
            )

        with self.assertRaises(ValueError):
            self.create_scenario(
                k_phi_n=k_phi_n * np.ones(n_turns + 1),
                k_phi_nm1=k_phi_nm1 * np.ones(n_turns + 1),
                k_eps_n=k_eps_n * np.ones(n_turns),
                k_z_n=k_z_n * np.ones(n_turns + 1),
                k_a_n=k_a_n * np.ones(n_turns + 1),
                k_b_n=k_b_n * np.ones(n_turns + 1),
                phi_sync=reference * np.pi / 180 * np.ones(n_turns + 1),
                pl_gain=1.0 * np.ones(n_turns + 1),
            )

        with self.assertRaises(ValueError):
            self.create_scenario(
                k_phi_n=k_phi_n * np.ones(n_turns + 1),
                k_phi_nm1=k_phi_nm1 * np.ones(n_turns + 1),
                k_eps_n=k_eps_n * np.ones(n_turns + 1),
                k_z_n=k_z_n * np.ones(n_turns),
                k_a_n=k_a_n * np.ones(n_turns + 1),
                k_b_n=k_b_n * np.ones(n_turns + 1),
                phi_sync=reference * np.pi / 180 * np.ones(n_turns + 1),
                pl_gain=1.0 * np.ones(n_turns + 1),
            )

        with self.assertRaises(ValueError):
            self.create_scenario(
                k_phi_n=k_phi_n * np.ones(n_turns + 1),
                k_phi_nm1=k_phi_nm1 * np.ones(n_turns + 1),
                k_eps_n=k_eps_n * np.ones(n_turns + 1),
                k_z_n=k_z_n * np.ones(n_turns + 1),
                k_a_n=k_a_n * np.ones(n_turns),
                k_b_n=k_b_n * np.ones(n_turns + 1),
                phi_sync=reference * np.pi / 180 * np.ones(n_turns + 1),
                pl_gain=1.0 * np.ones(n_turns + 1),
            )

        with self.assertRaises(ValueError):
            self.create_scenario(
                k_phi_n=k_phi_n * np.ones(n_turns + 1),
                k_phi_nm1=k_phi_nm1 * np.ones(n_turns + 1),
                k_eps_n=k_eps_n * np.ones(n_turns + 1),
                k_z_n=k_z_n * np.ones(n_turns + 1),
                k_a_n=k_a_n * np.ones(n_turns + 1),
                k_b_n=k_b_n * np.ones(n_turns),
                phi_sync=reference * np.pi / 180 * np.ones(n_turns + 1),
                pl_gain=1.0 * np.ones(n_turns + 1),
            )

        with self.assertRaises(ValueError):
            self.create_scenario(
                k_phi_n=k_phi_n * np.ones(n_turns + 1),
                k_phi_nm1=k_phi_nm1 * np.ones(n_turns + 1),
                k_eps_n=k_eps_n * np.ones(n_turns + 1),
                k_z_n=k_z_n * np.ones(n_turns + 1),
                k_a_n=k_a_n * np.ones(n_turns + 1),
                k_b_n=k_b_n * np.ones(n_turns + 1),
                phi_sync=reference * np.pi / 180 * np.ones(n_turns),
                pl_gain=1.0 * np.ones(n_turns + 1),
            )

        with self.assertRaises(ValueError):
            self.create_scenario(
                k_phi_n=k_phi_n * np.ones(n_turns + 1),
                k_phi_nm1=k_phi_nm1 * np.ones(n_turns + 1),
                k_eps_n=k_eps_n * np.ones(n_turns + 1),
                k_z_n=k_z_n * np.ones(n_turns + 1),
                k_a_n=k_a_n * np.ones(n_turns + 1),
                k_b_n=k_b_n * np.ones(n_turns + 1),
                phi_sync=reference * np.pi / 180 * np.ones(n_turns),
                pl_gain=1.0 * np.ones(n_turns),
            )
