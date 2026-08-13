import copy
import unittest

import numpy as np

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    proton,
)
from blond.physics.feedbacks.accelerators.lhc import (
    LHCCavityFeedback,
    LHCCavityFeedbackCommissioning,
)

circumference = 26658.8832  # [m]
momentum = 450e9
intensity = 1.6e11
rf_voltage = 5e6
rf_phase = 0.0
h = 35640
gamma_t = 53.8
alpha = 1 / gamma_t / gamma_t

energy = np.sqrt(momentum**2 + proton.mass**2)
rel_gamma = energy / proton.mass
rel_beta = np.sqrt(1 - 1 / rel_gamma**2)

n_macroparticles = 100_000
tau_bunch = 1.2e-9
number_of_bunches = 72
bunch_spacing = 10
bucket_shift = 1000

g_a = 6.79e-6
g_d = 10
g_o = 10
tau_a = 170e-6
tau_d = 400e-6
tau_o = 110e-6
tau_loop = 650e-9
tau_otfb = 1200e-9


class TestLHCCavityFeedback(unittest.TestCase):
    def create_scenario(
        self,
        commissioning: LHCCavityFeedbackCommissioning,
        disable_fine_grid: bool = False,
        n_turns: int = 20,
        q_l: float = 20_000,
        n_pretrack: int = 200,
    ):
        beam = Beam(
            intensity,
            proton,
        )

        cycle = ConstantMagneticCycle(proton, momentum, in_unit="momentum")

        lattice = DriftSimple(
            orbit_length=circumference, momentum_compaction_factor=alpha
        )

        cavity = SingleHarmonicRFStation(
            voltage=rf_voltage,
            phi_rf=rf_phase,
            harmonic=h,
        )

        f_rf = cavity.calc_main_harmonic_omega_rf_design(
            rel_beta, lattice.orbit_length
        ) / (2 * np.pi)
        t_rf = 1 / f_rf

        profile = StaticProfile(
            cut_left=(-5.5 + bucket_shift) * t_rf,
            cut_right=(6.5 + number_of_bunches * bunch_spacing + bucket_shift)
            * t_rf,
            n_bins=(10 * number_of_bunches + 12) * 2**5,
        )

        bigaussian = BiGaussian(
            n_macroparticles=n_macroparticles,
            sigma_dt=tau_bunch / 4,
            seed=1234,
        )

        self.cavity_feedback = LHCCavityFeedback(
            profile,
            tau_loop=tau_loop,
            tau_otfb=tau_otfb,
            rffb=commissioning,
            q_l=q_l,
            n_pretrack=n_pretrack,
        )
        self.cavity_feedback.disable_fine_grid = disable_fine_grid

        cavity.attach_cavity_feedback(self.cavity_feedback)

        ring = Ring(
            circumference,
        )

        ring.add_elements(
            [profile, cavity, lattice],
        )

        simulation = Simulation(
            ring,
            cycle,
        )

        simulation.prepare_beam(beam, bigaussian)

        beam_copy = copy.deepcopy(beam)

        for i in range(1, number_of_bunches):
            _dt = beam_copy.write_partial_dt()
            _dt += bunch_spacing * t_rf
            beam.add_beam(beam_copy)

        _dt = beam.write_partial_dt()

        profile.track(beam)

        simulation.finalize(
            (beam,),
            n_turns,
        )

    def test_pre_tracking(self):
        commissioning = LHCCavityFeedbackCommissioning(
            g_a=g_a,
            g_d=g_d,
            g_o=g_o,
            tau_a=tau_a,
            tau_d=tau_d,
            tau_o=tau_o,
            open_tuner=True,
            enable_klystron=False,
            clamping=False,
            saturation=False,
        )
        self.create_scenario(
            commissioning=commissioning,
        )

        # Note: that target voltage is different from rf_voltage setpoint due to regulation errors
        target_voltage = 619927.756800819
        self.assertAlmostEqual(
            np.mean(np.abs(self.cavity_feedback.buffers_coarse.v_ant.curr))
            / target_voltage,
            1,
        )

        target_power = 53414.80900014226
        self.assertAlmostEqual(
            np.mean(np.abs(self.cavity_feedback.generator_power()))
            / target_power,
            1,
        )

    def test_with_and_without_otfb(self):
        pass

    def test_tuner_loop(self):
        pass

    def test_klystron_model(self):
        pass


class TestLHCCavityFeedbackTransferFunction(unittest.TestCase):
    def create_scenario(self):
        pass

    def test_open_loop_response(self):
        pass

    def test_closed_loop_response(self):
        pass

    def test_one_turn_delay_feedback_reponse(self):
        pass
