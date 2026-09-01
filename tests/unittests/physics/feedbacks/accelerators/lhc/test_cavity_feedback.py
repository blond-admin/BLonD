import copy
import unittest

import numpy as np
import pytest

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
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.feedbacks.accelerators.lhc import (
    LHCCavityFeedback,
    LHCCavityFeedbackCommissioning,
)
from blond.physics.feedbacks.transfer_function_analysis import (
    estimate_transfer_function,
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
        commissioning: LHCCavityFeedbackCommissioning = None,
        disable_fine_grid: bool = False,
        n_turns: int = 20,
        q_l: float = 20_000,
        n_pretrack: int = 100,
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
            commissioning=commissioning,
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

    @pytest.mark.skip(reason="Unittest takes too long to run")
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

        commissioning = None
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
            commissioning=commissioning, n_pretrack=50, disable_fine_grid=True
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
            rf_frequency=rf_station.omega_rf / 2 / np.pi,
            voltage=np.mean(np.abs(cavity_feedback.buffers_coarse.v_ant.curr)),
        )

        q_l_optimum = cavity_feedback.optimum_Q_L(
            detuning=theoretical_detuning,
            rf_frequency=rf_station.omega_rf / 2 / np.pi,
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
            n_pretrack=50,
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
        target_max_power = 138231.87042979817

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

    def test_compare_coarse_and_fine_grids(self):
        simulation, beam = self.create_scenario(
            n_pretrack=50, disable_fine_grid=False
        )

        rf_station = simulation.ring.elements.get_element(
            SingleHarmonicRFStation
        )
        cavity_feedback: LHCCavityFeedback = (
            rf_station.get_main_harmonic_cavity_feedback()
        )
        cavity_feedback.track(beam)

        n_mov_avg = 2**5 * 10

        fine_ant_buffer = cavity_feedback.buffers_fine.v_ant / 8
        fine_ant_buffer = np.convolve(
            fine_ant_buffer, np.ones(n_mov_avg) / n_mov_avg, "valid"
        )

        fine_ant_buffer_interp = np.interp(
            cavity_feedback.rf_centers[1:72],
            copy_to_cpu(cavity_feedback.profile.hist_x)[n_mov_avg - 1 : :],
            fine_ant_buffer,
        )

        np.testing.assert_allclose(
            cavity_feedback.buffers_coarse.v_ant.curr.real[1:72],
            fine_ant_buffer_interp.real,
            rtol=5e-5,
        )

        np.testing.assert_allclose(
            cavity_feedback.buffers_coarse.v_ant.curr.imag[1:72],
            fine_ant_buffer_interp.imag,
            rtol=5e-3,
        )

    def test_optimum_ql_with_beam(self):
        optimum_q = LHCCavityFeedback.optimum_Q_L_beam(
            r_over_q=45, real_peak_beam_current=1.111, voltage=1e6
        )
        self.assertAlmostEqual(optimum_q, 20002.0002000)

    def test_linear_interp_scalar(self):
        x_arr = np.array([0, 1, 2, 3, 4, 5])
        y_arr = np.array([0, 1, 2, 3, 4, 5])

        t_val = -1
        val = LHCCavityFeedback._linear_interp_scalar(
            x=x_arr, y=y_arr, t=t_val
        )
        self.assertAlmostEqual(val, t_val)

        t_val = 6
        val = LHCCavityFeedback._linear_interp_scalar(
            x=x_arr, y=y_arr, t=t_val
        )
        self.assertAlmostEqual(val, t_val)

    def test_hardware_commissioning_without_excitation(self):
        f_rf = 400.789e6
        harmonic = 35640
        n_pretrack = 50
        commissioning = LHCCavityFeedbackCommissioning()

        profile = StaticProfile(cut_left=0, cut_right=2.5e-9, n_bins=4)

        cavity_feedback = LHCCavityFeedback(
            profile=profile, commissioning=commissioning, n_pretrack=n_pretrack
        )
        cavity_feedback.disable_fine_grid = True

        cavity_feedback.set_hardware_commissioning(
            omega_rf=2 * np.pi * f_rf, harmonic=harmonic
        )

        np.testing.assert_allclose(
            np.zeros(cavity_feedback.n_coarse, dtype=complex),
            cavity_feedback.buffers_coarse.v_ant.curr,
        )


class TestLHCCavityFeedbackTransferFunction(unittest.TestCase):
    @staticmethod
    def create_scenario(
        commissioning: LHCCavityFeedbackCommissioning,
        cut_data: int = 0,
        n_pretrack: int = 200,
    ):
        f_rf = 400.789e6
        harmonic = 35640

        profile = StaticProfile(cut_left=0, cut_right=2.5e-9, n_bins=4)

        cavity_feedback = LHCCavityFeedback(
            profile=profile, commissioning=commissioning, n_pretrack=n_pretrack
        )
        cavity_feedback.disable_fine_grid = True

        cavity_feedback.set_hardware_commissioning(
            omega_rf=2 * np.pi * f_rf, harmonic=harmonic
        )

        return estimate_transfer_function(
            input_signal=cavity_feedback.v_excitation_in,
            output_signal=cavity_feedback.v_excitation_out,
            t_s=cavity_feedback.T_s,
            data_cut=cut_data,
        )

    @pytest.mark.skip(reason="Unittest takes too long to run")
    def test_open_loop_response(self):
        cut_data = 3564 * 5
        r_over_q = 45
        q_l = 20_000
        domega = 0.0
        f_rf = 400.789e6
        omega_rf = 2 * np.pi * f_rf

        commissioning = LHCCavityFeedbackCommissioning(
            g_a=g_a,
            g_d=g_d,
            g_o=g_o,
            tau_a=tau_a,
            tau_d=tau_d,
            tau_o=tau_o,
            excitation=True,
            open_loop=True,
            open_otfb=True,
        )

        f_est, h_est = self.create_scenario(
            commissioning=commissioning, cut_data=cut_data, n_pretrack=200
        )

        h_a = lambda s: g_a * tau_a * s / (1 + tau_a * s)
        h_d = lambda s: g_a * g_d / (1 + tau_d * s)
        h_delay = lambda s: np.exp(-tau_loop * s)
        z_cav = lambda s: (
            r_over_q * q_l / (1 + 2 * q_l * (s - 1j * domega) / omega_rf)
        )

        h_open = lambda s: 2 * h_delay(s) * (h_a(s) + h_d(s)) * z_cav(s)

        h_actual = h_open(1j * 2 * np.pi * f_est)

        np.testing.assert_allclose(
            actual=h_est.real, desired=h_actual.real, atol=35
        )

        np.testing.assert_allclose(
            actual=h_est.imag, desired=h_actual.imag, atol=10
        )

    def test_closed_loop_response(self):
        cut_data = 3564 * 5
        r_over_q = 45
        q_l = 20_000
        domega = 0.0
        f_rf = 400.789e6
        omega_rf = 2 * np.pi * f_rf

        commissioning = LHCCavityFeedbackCommissioning(
            g_a=g_a,
            g_d=g_d,
            g_o=g_o,
            tau_a=tau_a,
            tau_d=tau_d,
            tau_o=tau_o,
            excitation=True,
            open_loop=False,
            open_otfb=True,
        )

        f_est, h_est = self.create_scenario(
            commissioning=commissioning, cut_data=cut_data, n_pretrack=100
        )

        h_a = lambda s: g_a * tau_a * s / (1 + tau_a * s)
        h_d = lambda s: g_a * g_d / (1 + tau_d * s)
        h_delay = lambda s: np.exp(-tau_loop * s)
        z_cav = lambda s: (
            r_over_q * q_l / (1 + 2 * q_l * (s - 1j * domega) / omega_rf)
        )

        h_open = lambda s: 2 * h_delay(s) * (h_a(s) + h_d(s)) * z_cav(s)
        h_closed = lambda s: h_open(s) / (1 + h_open(s))

        h_actual = h_closed(1j * 2 * np.pi * f_est)

        np.testing.assert_allclose(
            actual=h_est.real, desired=h_actual.real, atol=0.03
        )

        np.testing.assert_allclose(
            actual=h_est.imag, desired=h_actual.imag, atol=0.03
        )

    @pytest.mark.skip(reason="Unittest takes too long to run")
    def test_one_turn_delay_feedback_response(self):
        cut_data = 3564 * 5
        r_over_q = 45
        a_comb = 15 / 16
        q_l = 20_000
        domega = 0.0
        f_rf = 400.789e6
        omega_rf = 2 * np.pi * f_rf

        f_span = 750e3

        t_rev = 35640 / f_rf

        commissioning = LHCCavityFeedbackCommissioning(
            g_a=g_a,
            g_d=g_d,
            g_o=g_o,
            tau_a=tau_a,
            tau_d=tau_d,
            tau_o=tau_o,
            excitation=True,
            open_loop=False,
            open_otfb=False,
        )

        f_est, h_est = self.create_scenario(
            commissioning=commissioning, cut_data=cut_data, n_pretrack=200
        )

        h_est = h_est[(f_est > -f_span) & (f_est < f_span)]
        f_est = f_est[(f_est > -f_span) & (f_est < f_span)]

        h_a = lambda s: g_a * tau_a * s / (1 + tau_a * s)
        h_d = lambda s: g_a * g_d / (1 + tau_d * s)
        h_delay = lambda s: np.exp(-tau_loop * s)

        h_comb = lambda s: (
            g_o
            * (1 - a_comb)
            * np.exp(-t_rev * s)
            / (1 - a_comb * np.exp(-t_rev * s))
        )
        h_ac = lambda s: tau_o * s / (1 + tau_o * s)
        h_comp = lambda s: np.exp(tau_otfb * s)
        h_otfb = lambda s: h_ac(s) * h_comb(s) * h_ac(s) * h_comp(s)

        z_cav = lambda s: (
            r_over_q * q_l / (1 + 2 * q_l * (s - 1j * domega) / omega_rf)
        )

        h_open = lambda s: (
            2 * h_delay(s) * (h_a(s) * (h_otfb(s) + 1) + h_d(s)) * z_cav(s)
        )
        h_closed = lambda s: h_open(s) / (1 + h_open(s))

        h_actual = h_closed(1j * 2 * np.pi * f_est)

        np.testing.assert_allclose(
            actual=h_est.real, desired=h_actual.real, atol=0.6
        )

        np.testing.assert_allclose(
            actual=h_est.imag, desired=h_actual.imag, atol=0.5
        )

    def test_otfb_excitation_response_1(self):
        cut_data = 3564 * 5
        r_over_q = 45
        a_comb = 15 / 16
        q_l = 20_000
        domega = 0.0
        f_rf = 400.789e6
        omega_rf = 2 * np.pi * f_rf

        f_span = 750e3

        t_rev = 35640 / f_rf

        commissioning = LHCCavityFeedbackCommissioning(
            g_a=g_a,
            g_d=g_d,
            g_o=g_o,
            tau_a=tau_a,
            tau_d=tau_d,
            tau_o=tau_o,
            excitation_otfb_1=True,
            excitation_otfb_2=False,
            open_loop=False,
            open_otfb=False,
        )

        f_est, h_est = self.create_scenario(
            commissioning=commissioning, cut_data=cut_data, n_pretrack=100
        )

        h_est = h_est[(f_est > -f_span) & (f_est < f_span)]
        f_est = f_est[(f_est > -f_span) & (f_est < f_span)]

        h_a = lambda s: g_a * tau_a * s / (1 + tau_a * s)
        h_d = lambda s: g_a * g_d / (1 + tau_d * s)
        h_delay = lambda s: np.exp(-tau_loop * s)

        h_comb = lambda s: (
            g_o
            * (1 - a_comb)
            * np.exp(-t_rev * s)
            / (1 - a_comb * np.exp(-t_rev * s))
        )
        h_ac = lambda s: tau_o * s / (1 + tau_o * s)
        h_comp = lambda s: np.exp(tau_otfb * s)
        h_otfb = lambda s: h_ac(s) * h_comb(s) * h_ac(s) * h_comp(s)

        z_cav = lambda s: (
            r_over_q * q_l / (1 + 2 * q_l * (s - 1j * domega) / omega_rf)
        )

        h_open = lambda s: (
            2 * h_delay(s) * (h_a(s) * (h_otfb(s) + 1) + h_d(s)) * z_cav(s)
        )
        h_closed = lambda s: h_open(s) / (1 + h_open(s))

        h_actual = h_closed(1j * 2 * np.pi * f_est)

    def test_otfb_excitation_response_2(self):
        cut_data = 3564 * 5
        r_over_q = 45
        a_comb = 15 / 16
        q_l = 20_000
        domega = 0.0
        f_rf = 400.789e6
        omega_rf = 2 * np.pi * f_rf

        f_span = 750e3

        t_rev = 35640 / f_rf

        commissioning = LHCCavityFeedbackCommissioning(
            g_a=g_a,
            g_d=g_d,
            g_o=g_o,
            tau_a=tau_a,
            tau_d=tau_d,
            tau_o=tau_o,
            excitation_otfb_1=False,
            excitation_otfb_2=True,
            open_loop=False,
            open_otfb=False,
        )

        f_est, h_est = self.create_scenario(
            commissioning=commissioning, cut_data=cut_data, n_pretrack=100
        )

        h_est = h_est[(f_est > -f_span) & (f_est < f_span)]
        f_est = f_est[(f_est > -f_span) & (f_est < f_span)]

        h_a = lambda s: g_a * tau_a * s / (1 + tau_a * s)
        h_d = lambda s: g_a * g_d / (1 + tau_d * s)
        h_delay = lambda s: np.exp(-tau_loop * s)

        h_comb = lambda s: (
            g_o
            * (1 - a_comb)
            * np.exp(-t_rev * s)
            / (1 - a_comb * np.exp(-t_rev * s))
        )
        h_ac = lambda s: tau_o * s / (1 + tau_o * s)
        h_comp = lambda s: np.exp(tau_otfb * s)
        h_otfb = lambda s: h_ac(s) * h_comb(s) * h_ac(s) * h_comp(s)

        z_cav = lambda s: (
            r_over_q * q_l / (1 + 2 * q_l * (s - 1j * domega) / omega_rf)
        )

        h_open = lambda s: (
            2 * h_delay(s) * (h_a(s) * (h_otfb(s) + 1) + h_d(s)) * z_cav(s)
        )
        h_closed = lambda s: h_open(s) / (1 + h_open(s))

        h_actual = h_closed(1j * 2 * np.pi * f_est)
