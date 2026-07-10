import unittest

import numpy as np

from blond import (
    Beam,
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
intensity = 1.6e11
n_turns = 2_000
h = 8
gamma_t = 6.248296
alpha = 1 / gamma_t / gamma_t
bending_radius = 70.79

n_macroparticles = 100_000
tau_bunch = 20e-9
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

        self.simulation = Simulation(
            ring,
            cycle,
        )

        eta = ring.calc_average_eta_0(rel_gamma)
        bucket_height = np.sqrt(
            2 * voltage * rel_beta**2 * energy / (np.pi * h * np.abs(eta))
        )

        dt_scale = tau_bunch / 4
        de_scale = dt_scale / t_rf * bucket_height * 3

        self.beam = Beam.simple_gaussian(
            n_macroparticles=1_000_000,
            intensity=intensity,
            particle_type=proton,
            dt_scale=dt_scale,
            dE_scale=de_scale,
            dE_offset=0,
            dt_offset=t_rf / 2,
            seed=1234,
            reference_total_energy=energy,
        )

        self.beam._dt.array_local += injection_offset_phase * t_rf / 360

        self.profile.track(self.beam)

        self.simulation.finalize(
            (self.beam,),
            n_turns,
        )
        self.beam_control.reference = reference * np.pi / 180

        self.beam_control.track(self.beam)

    def test_ps_beam_control_below_transition(self):
        self.create_scenario(pl_gain=PL_gain, rl_gain=RL_gain, momentum=2.79e9)

        # Check beam-phase loop output
        self.assertAlmostEqual(
            self.beam_control.domega_dphi, -17335.4154043551646, places=5
        )

        # Check radial loop output
        self.assertAlmostEqual(
            self.beam_control.domega_dr, -0.01623603489296729, places=5
        )

        # Check total correction
        self.assertAlmostEqual(
            self.beam_control.delta_omega_rf, -17335.431640390056, places=5
        )

        # Check one-turn memory variables
        self.assertAlmostEqual(
            self.beam_control.prev_in_phase, 0.3492812266877152, places=5
        )
        self.assertAlmostEqual(
            self.beam_control.prev_out_phase, 1.9923001170267276, places=5
        )
        self.assertAlmostEqual(
            self.beam_control.prev_out_radial,
            1.029877479036536e-07,
            places=5,
        )

        # Track the next turn
        self.simulation.turn_i.value = 1
        self.beam_control.track(self.beam)

        # Check one-turn memory variables
        self.assertAlmostEqual(
            self.beam_control.prev_in_phase, 0.3466191124113115, places=5
        )
        self.assertAlmostEqual(
            self.beam_control.prev_out_phase, 1.992127583836593, places=5
        )
        self.assertAlmostEqual(
            self.beam_control.prev_out_radial, 2.0532368635082497e-07, places=5
        )

        # Check beam-phase loop output
        self.assertAlmostEqual(
            self.beam_control.domega_dphi, -17333.914157381147, places=5
        )

        # Check radial loop output
        self.assertAlmostEqual(
            self.beam_control.domega_dr, -0.032369311921096995, places=5
        )

        # Check total correction
        self.assertAlmostEqual(
            self.beam_control.delta_omega_rf, -17333.946526693067, places=5
        )

    def test_ps_beam_control_above_transition(self):
        self.create_scenario(
            pl_gain=PL_gain, rl_gain=-RL_gain, momentum=25.92e9
        )

        # Check beam-phase loop output
        self.assertAlmostEqual(
            self.beam_control.domega_dphi, -18276.739631042263, places=5
        )

        # Check radial loop output
        self.assertAlmostEqual(
            self.beam_control.domega_dr, 0.010196918671247831, places=5
        )

        # Check total correction
        self.assertAlmostEqual(
            self.beam_control.delta_omega_rf, -18276.72943412359, places=5
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
            self.beam_control.domega_dphi, 13207.22017318535, places=5
        )

        # Check radial loop output
        self.assertAlmostEqual(
            self.beam_control.domega_dr, -98427.61684211278, places=4
        )

        # Check total correction
        self.assertAlmostEqual(
            self.beam_control.delta_omega_rf, -85220.39666892742, places=4
        )

    def test_ps_beam_control_switch_harmonic_fail(self):
        self.create_scenario(pl_gain=PL_gain, rl_gain=RL_gain, momentum=2.79e9)

        with self.assertRaises(ValueError):
            self.beam_control.update_main_rf_stations(harmonic=7)
