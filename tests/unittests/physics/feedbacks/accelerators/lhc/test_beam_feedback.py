import unittest

import numpy as np

circumference = 26658.8832  # [m]
momentum = 450e9
intensity = 1.6e11
n_turns = 2_000
voltage = 5e6
h = 35640
gamma_t = 53.8
alpha = 1 / gamma_t / gamma_t

n_macroparticles = 100_000
tau_bunch = 1.2e-9
injection_offset_phase = 20
reference = -20


class TestLHCBeamFeedback(unittest.TestCase):
    def setUp(self):
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
        from blond.physics.feedbacks.accelerators.lhc import (
            LHCBeamControl,
        )

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
            voltage=np.array([voltage]),
            phi_rf=np.array([0.0]),
            harmonic=np.array([h]),
            n_harmonics=1,
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
        self.beam_control = LHCBeamControl(
            pl_gain=1 / (5 * t_rev) * 1,
            sl_gain=1 / (5 * t_rev) / 10,
            profile=self.profile,
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
        self.lhc_y_init = self.beam_control.lhc_y
        self.beam_control.reference = reference * np.pi / 180

        self.beam_control.track(beam)

    def test_lhc_beam_control_init(self):
        # Checks the correction calculation of the recursion parameters for the synchronization loop
        self.assertAlmostEqual(self.beam_control.lhc_t[0], 0.0177111)

        self.assertAlmostEqual(self.lhc_y_init, 0)

        self.assertAlmostEqual(self.beam_control.lhc_a[0], 2.64280271)

    def test_lhc_beam_phase_loop(self):
        # Checks the calculation done by the beam phase loop for the first turn
        self.assertAlmostEqual(
            self.beam_control.dphi * 180 / np.pi,
            injection_offset_phase,
            places=2,
        )

        self.assertAlmostEqual(
            self.beam_control.pl_gain * self.beam_control.dphi,
            785.0284369961593,
        )

    def test_lhc_synchronization_loop(self):
        # Checks the calculation done for the synchro loop for the first turn
        dphi_rf = self.beam_control.cavities[0].delta_phi_rf

        synch_corr = self.beam_control.sl_gain * (
            self.lhc_y_init
            + self.beam_control.lhc_a[0]
            * (dphi_rf + self.beam_control.reference)
        )

        self.assertAlmostEqual(synch_corr[0], -207.48175328)

        self.assertAlmostEqual(self.beam_control.lhc_y[0], 0.01015636)

    def test_correction_calculation(self):
        # Checks the correct calculation of the corrections for the next turn
        self.assertAlmostEqual(
            self.beam_control.domega_rf[0] / 2 / np.pi, -91.91940957997551
        )
