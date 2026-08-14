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
bucket_shift = 0

g_a = 6.79e-6
g_d = 10
g_o = 10
tau_a = 170e-6
tau_d = 400e-6
tau_o = 110e-6
tau_loop = 650e-9
tau_otfb = 1200e-9


class TestLHCCavityFeedback(unittest.TestCase):
    @staticmethod
    def create_scenario(
        commissioning: LHCCavityFeedbackCommissioning,
        disable_fine_grid: bool = False,
        n_turns: int = 20,
        q_l: float = 20_000,
        n_pretrack: int = 200,
        detuning: float = 0.0,
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

        cavity_feedback = LHCCavityFeedback(
            profile,
            tau_loop=tau_loop,
            tau_otfb=tau_otfb,
            rffb=commissioning,
            q_l=q_l,
            n_pretrack=n_pretrack,
            f_c=detuning + 400.789e6,
        )
        cavity_feedback.disable_fine_grid = disable_fine_grid

        cavity.attach_cavity_feedback(cavity_feedback)

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

        simulation.finalize(
            (beam,),
            n_turns,
        )

        profile.track(beam)

        return simulation, beam

    def test_with_and_without_otfb(self):
        commissioning = LHCCavityFeedbackCommissioning(
            g_a=g_a,
            g_d=g_d,
            g_o=g_o,
            tau_a=tau_a,
            tau_d=tau_d,
            tau_o=tau_o,
            open_tuner=True,
            open_otfb=True,
            enable_klystron=False,
            clamping=False,
            saturation=False,
        )
        simulation_without_otfb, beam_without_otfb = self.create_scenario(
            commissioning=commissioning, n_pretrack=100, disable_fine_grid=True
        )

        commissioning = LHCCavityFeedbackCommissioning(
            g_a=g_a,
            g_d=g_d,
            g_o=g_o,
            tau_a=tau_a,
            tau_d=tau_d,
            tau_o=tau_o,
            open_tuner=True,
            open_otfb=False,
            enable_klystron=False,
            clamping=False,
            saturation=False,
        )
        simulation_with_otfb, beam_with_otfb = self.create_scenario(
            commissioning=commissioning, n_pretrack=100, disable_fine_grid=True
        )

        rf_station = simulation_without_otfb.ring.elements.get_element(
            SingleHarmonicRFStation
        )
        cavity_feedback_without_otfb = (
            rf_station.get_main_harmonic_cavity_feedback()
        )

        # Note: that target voltage is different from rf_voltage setpoint due to regulation errors
        target_voltage = 619927.7567935854
        self.assertAlmostEqual(
            np.mean(
                np.abs(cavity_feedback_without_otfb.buffers_coarse.v_ant.curr)
            )
            / target_voltage,
            1,
        )

        target_power = 53414.808997960674
        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback_without_otfb.generator_power()))
            / target_power,
            1,
        )

        rf_station = simulation_with_otfb.ring.elements.get_element(
            SingleHarmonicRFStation
        )
        cavity_feedback_with_otfb: LHCCavityFeedback = (
            rf_station.get_main_harmonic_cavity_feedback()
        )

        # Note: that target voltage is different from rf_voltage setpoint due to regulation errors
        target_voltage = 619927.756800819
        self.assertAlmostEqual(
            np.mean(
                np.abs(cavity_feedback_with_otfb.buffers_coarse.v_ant.curr)
            )
            / target_voltage,
            1,
        )

        target_power = 53414.80900014226
        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback_with_otfb.generator_power()))
            / target_power,
            1,
        )

        # Track one turn to check the one-turn delay of the OTFB
        simulation_without_otfb.run_simulation(
            beam_without_otfb, n_turns=1, show_progressbar=False, verbose=False
        )
        simulation_with_otfb.run_simulation(
            beam_with_otfb, n_turns=1, show_progressbar=False, verbose=False
        )

        rf_station = simulation_without_otfb.ring.elements.get_element(
            SingleHarmonicRFStation
        )
        cavity_feedback_without_otfb: LHCCavityFeedback = (
            rf_station.get_main_harmonic_cavity_feedback()
        )

        rf_station = simulation_with_otfb.ring.elements.get_element(
            SingleHarmonicRFStation
        )
        cavity_feedback_with_otfb: LHCCavityFeedback = (
            rf_station.get_main_harmonic_cavity_feedback()
        )

        self.assertAlmostEqual(
            np.max(cavity_feedback_without_otfb.generator_power())
            / np.max(cavity_feedback_with_otfb.generator_power()),
            1,
            places=7,
        )

        # Track another turn to check the one-turn delay of the OTFB
        cavity_feedback_without_otfb.track(beam_without_otfb)
        cavity_feedback_with_otfb.track(beam_with_otfb)

        target_power_without_otfb = 310167.0496694454
        target_power_with_otfb = 310020.65010378964

        self.assertAlmostEqual(
            np.max(cavity_feedback_without_otfb.generator_power())
            / target_power_without_otfb,
            1,
            places=7,
        )

        self.assertAlmostEqual(
            np.max(cavity_feedback_with_otfb.generator_power())
            / target_power_with_otfb,
            1,
            places=7,
        )

    def test_tuner_loop(self):
        commissioning = LHCCavityFeedbackCommissioning(
            g_a=g_a,
            g_d=g_d,
            g_o=g_o,
            tau_a=tau_a,
            tau_d=tau_d,
            tau_o=tau_o,
            open_tuner=False,
            mu=-10,
            open_otfb=False,
            enable_klystron=False,
            clamping=False,
            saturation=False,
        )
        simulation, beam = self.create_scenario(
            commissioning=commissioning, n_pretrack=100, disable_fine_grid=True
        )

        rf_station = simulation.ring.elements.get_element(
            SingleHarmonicRFStation
        )
        cavity_feedback: LHCCavityFeedback = (
            rf_station.get_main_harmonic_cavity_feedback()
        )
        cavity_feedback.rf_beam_current(beam)

        theoretical_detuning = cavity_feedback.half_detuning(
            imag_peak_beam_current=np.max(
                np.abs(cavity_feedback.buffers_coarse.i_beam.curr)
            ),
            r_over_q=cavity_feedback.r_over_q,
            rf_frequency=cavity_feedback.omega_rf / 2 / np.pi,
            voltage=np.mean(np.abs(cavity_feedback.buffers_coarse.v_ant.curr)),
        )

        q_l_optimum = cavity_feedback.optimum_Q_L(
            detuning=theoretical_detuning,
            rf_frequency=cavity_feedback.omega_rf / 2 / np.pi,
        )
        cavity_feedback.q_l = q_l_optimum

        theoretical_rf_power = cavity_feedback.half_detuning_power(
            peak_beam_current=np.max(
                np.abs(cavity_feedback.buffers_coarse.i_beam.curr)
            ),
            voltage=np.mean(np.abs(cavity_feedback.buffers_coarse.v_ant.curr)),
        )

        for i in range(100):
            cavity_feedback.track(beam)

        model_detuning = cavity_feedback.d_omega / 2 / np.pi
        model_rf_power = np.mean(cavity_feedback.generator_power())

        self.assertAlmostEqual(
            theoretical_detuning / model_detuning, 1.0136520554274586, places=5
        )

        self.assertAlmostEqual(
            theoretical_rf_power / float(model_rf_power),
            1.0229780380697582,
            places=5,
        )

    def test_klystron_model(self):
        commissioning = LHCCavityFeedbackCommissioning(
            g_a=g_a,
            g_d=g_d,
            g_o=g_o,
            tau_a=tau_a,
            tau_d=tau_d,
            tau_o=tau_o,
            open_tuner=True,
            open_otfb=False,
            enable_klystron=True,
            saturation=True,
            clamping=True,
        )
        simulation, beam = self.create_scenario(
            commissioning=commissioning,
            n_pretrack=100,
            disable_fine_grid=True,
            detuning=-11.2e3,
        )

        rf_station = simulation.ring.elements.get_element(
            SingleHarmonicRFStation
        )
        cavity_feedback: LHCCavityFeedback = (
            rf_station.get_main_harmonic_cavity_feedback()
        )
        cavity_feedback.track(beam)

        target_mean_voltage = 620023.0464245899
        target_max_voltage = 634889.7936566119

        target_mean_power = 116051.16450970304
        target_max_power = 130066.94003941363

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.buffers_coarse.v_ant.curr))
            / target_mean_voltage,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.buffers_coarse.v_ant.curr))
            / target_max_voltage,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.generator_power()))
            / target_mean_power,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.generator_power()))
            / target_max_power,
            1,
            places=5,
        )

        cavity_feedback.track(beam)

        target_mean_voltage = 619945.0477434999
        target_max_voltage = 621708.9643980105

        target_mean_power = 114983.06619079916
        target_max_power = 138233.44015627756

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.buffers_coarse.v_ant.curr))
            / target_mean_voltage,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.buffers_coarse.v_ant.curr))
            / target_max_voltage,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.generator_power()))
            / target_mean_power,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.generator_power()))
            / target_max_power,
            1,
            places=5,
        )

    def test_induced_voltage_calculation(self):
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
