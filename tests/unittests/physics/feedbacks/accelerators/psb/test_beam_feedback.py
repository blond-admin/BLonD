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
from blond.physics.feedbacks.accelerators.psb import (
    PSBBeamControl,
)

circumference = 2 * np.pi * 25  # [m]
momentum = 310891054.809
intensity = 1.6e11
n_turns = 2_000
h = 1
gamma_t = 4.076750841
alpha = 1 / gamma_t / gamma_t

n_macroparticles = 100_000
tau_bunch = 200e-9
injection_offset_phase = 20
reference = -20

voltage = 8e3


class TestPSBBeamFeedback(unittest.TestCase):
    def create_scenario(
        self,
        pl_gain=0.0,
        pl_schedule=None,
        rl_gain_a=0.0,
        rl_gain_b=0.0,
        period=0.00001,
        coefficients=None,
        n_tracks=16,
    ):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")

        energy = np.sqrt(momentum**2 + proton.mass**2)
        rel_gamma = energy / proton.mass
        rel_beta = np.sqrt(1 - 1 / rel_gamma**2)

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
            cut_left=0.0,
            cut_right=t_rev,
            n_bins=4 * 2**6,
        )

        self.beam_control = PSBBeamControl(
            profile=self.profile,
            pl_gain=pl_gain,
            rl_gain_a=rl_gain_a,
            rl_gain_b=rl_gain_b,
            period=period,
            coefficients=coefficients,
        )

        if pl_schedule is not None:
            self.beam_control.schedule(
                attribute="pl_gain", value=pl_gain * pl_schedule
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

        dt_scale = tau_bunch
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

        for i in range(n_tracks):
            self.simulation.turn_counter.value = i

            for element in ring.elements.elements:
                element.track(self.beam)

    def test_psb_beam_control_phase_loop(self):
        self.create_scenario(pl_gain=1 / 25e-6, period=10e-6)

        # Check memory of the beam-phase loop
        self.assertAlmostEqual(
            self.beam_control.dphi_sum, 0.32288702705805883, places=5
        )
        self.assertAlmostEqual(
            self.beam_control.dphi_av, 0.341706136003109, places=5
        )
        self.assertAlmostEqual(
            self.beam_control.dphi_av_prev, 0.341706136003109, places=5
        )

        # Check radial loop offsets
        self.assertAlmostEqual(
            self.beam_control.dR_over_R_prev, 1.0596857013162176e-06, places=5
        )
        self.assertAlmostEqual(
            self.beam_control.dR_over_R, 1.0596857013162176e-06, places=5
        )

        # Check calculated corrections
        self.assertAlmostEqual(
            self.beam_control.domega_PL, 13654.946393646453, places=5
        )
        self.assertAlmostEqual(self.beam_control.domega_RL, 0.0, places=5)
        self.assertAlmostEqual(
            self.beam_control.delta_omega_rf, -13654.946393646453, places=5
        )

    def test_psb_beam_control_phase_loop_with_schedule(self):
        pl_schedule = np.zeros(16)
        pl_schedule[10:] = 1
        self.create_scenario(
            pl_gain=1 / 25e-6, pl_schedule=pl_schedule, period=10e-6
        )

        # Check memory of the beam-phase loop
        self.assertAlmostEqual(
            self.beam_control.dphi_sum, 0.3221992010414767, places=5
        )
        self.assertAlmostEqual(
            self.beam_control.dphi_av, 0.341375890599374, places=5
        )
        self.assertAlmostEqual(
            self.beam_control.dphi_av_prev, 0.341375890599374, places=5
        )

        # Check radial loop offsets
        self.assertAlmostEqual(self.beam_control.dR_over_R_prev, 0.0, places=5)
        self.assertAlmostEqual(self.beam_control.dR_over_R, 0.0, places=5)

        # Check calculated corrections
        self.assertAlmostEqual(
            self.beam_control.domega_PL, 13697.498714931087, places=5
        )
        self.assertAlmostEqual(self.beam_control.domega_RL, 0.0, places=5)
        self.assertAlmostEqual(
            self.beam_control.delta_omega_rf, -13697.498714931087, places=5
        )

    def test_psb_beam_control_radial_loop(self):
        self.create_scenario(
            pl_gain=0.0, period=10e-6, rl_gain_a=1.0e7, rl_gain_b=1.0e11
        )

        # Check memory of the beam-phase loop
        self.assertAlmostEqual(
            self.beam_control.dphi_sum, 0.33481055779883225, places=5
        )
        self.assertAlmostEqual(
            self.beam_control.dphi_av, 0.341375890599374, places=5
        )
        self.assertAlmostEqual(
            self.beam_control.dphi_av_prev, 0.341375890599374, places=5
        )

        # Check radial loop offsets
        self.assertAlmostEqual(
            self.beam_control.dR_over_R_prev, 0.0, places=10
        )
        self.assertAlmostEqual(self.beam_control.dR_over_R, 0.0, places=10)

        # Check calculated corrections
        self.assertAlmostEqual(self.beam_control.domega_PL, 0.0, places=5)
        self.assertAlmostEqual(self.beam_control.domega_RL, 0.0, places=5)
        self.assertAlmostEqual(self.beam_control.delta_omega_rf, 0.0, places=5)

    def test_psb_beam_control_coefficients(self):
        self.create_scenario(
            pl_gain=1 / 25e-6,
            period=10e-6,
            rl_gain_a=1.0e7,
            rl_gain_b=1.0e11,
            coefficients=[
                0.999019,
                -0.999019,
                0.0,
                1.0,
                -0.998038,
                0.0,
            ],
        )

        # Check memory of the beam-phase loop
        self.assertAlmostEqual(
            self.beam_control.dphi_sum, 0.22530578692634615, places=5
        )
        self.assertAlmostEqual(
            self.beam_control.dphi_av, 0.341706136003109, places=5
        )
        self.assertAlmostEqual(
            self.beam_control.dphi_av_prev, 0.341706136003109, places=5
        )

        # Check radial loop offsets
        self.assertAlmostEqual(
            self.beam_control.dR_over_R_prev, 1.0596857013162176e-06, places=10
        )
        self.assertAlmostEqual(
            self.beam_control.dR_over_R, 1.0596857013162176e-06, places=10
        )

        # Check calculated corrections
        self.assertAlmostEqual(
            self.beam_control.domega_PL, 13654.946393646453, places=5
        )
        self.assertAlmostEqual(
            self.beam_control.domega_RL, 105979.16698863493, places=2
        )
        self.assertAlmostEqual(
            self.beam_control.delta_omega_rf, -119634.11338228139, places=2
        )

    def test_psb_beam_control_act_every_turn(self):
        self.create_scenario(
            pl_gain=1 / 25e-6,
            period=0.0,
            rl_gain_a=1.0e7,
            rl_gain_b=1.0e11,
            coefficients=[
                0.999019,
                -0.999019,
                0.0,
                1.0,
                -0.998038,
                0.0,
            ],
            n_tracks=2,
        )

        # Check memory of the beam-phase loop
        self.assertAlmostEqual(self.beam_control.dphi_sum, 0.0, places=5)
        self.assertAlmostEqual(
            self.beam_control.dphi_av, 0.3497491642743108, places=5
        )
        self.assertAlmostEqual(
            self.beam_control.dphi_av_prev, 0.3497491642743108, places=5
        )

        # Check radial loop offsets
        self.assertAlmostEqual(
            self.beam_control.dR_over_R_prev, 1.326094468588297e-07, places=10
        )
        self.assertAlmostEqual(
            self.beam_control.dR_over_R, 1.326094468588297e-07, places=10
        )

        # Check calculated corrections
        self.assertAlmostEqual(
            self.beam_control.domega_PL, 13976.256485308591, places=5
        )
        self.assertAlmostEqual(
            self.beam_control.domega_RL, 13262.270780351559, places=2
        )
        self.assertAlmostEqual(
            self.beam_control.delta_omega_rf, -27238.527265660152, places=2
        )
