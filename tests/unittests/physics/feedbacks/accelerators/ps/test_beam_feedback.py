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
from blond.physics.feedbacks.accelerators.ps import (
    PSBeamControl,
)

circumference = 2 * np.pi * 100.0  # [m]
# momentum = 2.791277166873131e9
intensity = 1.6e11
n_turns = 2_000
h = 8
gamma_t = 6.248296
alpha = 1 / gamma_t / gamma_t
bending_radius = 70.79

n_macroparticles = 100_000
tau_bunch = 1.2e-9
injection_offset_phase = 20
reference = -20

voltage = 50.0e3

PL_gain = 0.01924
RL_gain = 155.05


class TestPSBeamFeedback(unittest.TestCase):
    def create_scenario(
        self,
        pl_gain,
        rl_gain,
        momentum,
        initialize_steady_state=True,
        prev_in_phase=0.0,
        prev_out_phase=0.0,
        prev_out_radial=0.0,
    ):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")

        energy = np.sqrt(momentum**2 + proton.mass**2)
        rel_gamma = energy / proton.mass
        rel_beta = np.sqrt(1 - 1 / rel_gamma**2)

        bl_transition = [rel_gamma < gamma_t] * n_turns

        if rel_gamma > gamma_t:
            phase = 0
        else:
            phase = np.pi

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
            phi_rf=np.array([phase]),
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

        self.beam_control = PSBeamControl(
            profile=self.profile,
            pl_gain=pl_gain / t_rev,
            rl_gain=rl_gain / t_rev / bending_radius,
            below_transition=np.array(bl_transition, dtype=bool),
            sample_de=50,
            gd_pl=5.704,
            gi_pl=1 - 8.66e-5,
            g_rl=0.993671,
            initialize_steady_state=initialize_steady_state,
            prev_in_phase=prev_in_phase,
            prev_out_phase=prev_out_phase,
            prev_out_radial=prev_out_radial,
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

    def test_ps_beam_control_below_transition(self):
        self.create_scenario(pl_gain=PL_gain, rl_gain=RL_gain, momentum=2.79e9)

        # Check beam-phase loop output
        self.assertAlmostEqual(
            self.beam_control.domega_dphi, -17054.02707404136, places=5
        )

        # Check radial loop output
        self.assertAlmostEqual(
            self.beam_control.domega_dr, 0.04106727491338032, places=5
        )

        # Check total correction
        self.assertAlmostEqual(
            self.beam_control.domega_rf, -17053.98600676645, places=5
        )

    def test_ps_beam_control_above_transition(self):
        self.create_scenario(
            pl_gain=PL_gain, rl_gain=RL_gain, momentum=25.92e9
        )

        # Check beam-phase loop output
        self.assertAlmostEqual(
            self.beam_control.domega_dphi, -17980.799776858436, places=5
        )

        # Check radial loop output
        self.assertAlmostEqual(
            self.beam_control.domega_dr, -0.025791985204956448, places=5
        )

        # Check total correction
        self.assertAlmostEqual(
            self.beam_control.domega_rf, -17980.82556884364, places=5
        )

    def test_ps_beam_control_init_not_steady_state(self):
        self.create_scenario(
            pl_gain=PL_gain,
            rl_gain=RL_gain,
            momentum=2.79e9,
            initialize_steady_state=False,
            prev_in_phase=30 * np.pi / 180,
            prev_out_phase=-30 * np.pi / 180,
            prev_out_radial=0.1,
        )

        # Check beam-phase loop output
        self.assertAlmostEqual(
            self.beam_control.domega_dphi, 13488.60850349915, places=5
        )

        # Check radial loop output
        self.assertAlmostEqual(
            self.beam_control.domega_dr, -98427.55953880296, places=4
        )

        # Check total correction
        self.assertAlmostEqual(
            self.beam_control.domega_rf, -84938.9510353038, places=4
        )
