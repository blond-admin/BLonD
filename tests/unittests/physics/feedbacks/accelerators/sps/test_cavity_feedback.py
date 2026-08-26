import copy
import unittest

import numpy as np
from scipy.signal import find_peaks

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    MultiHarmonicRFStation,
    Ring,
    Simulation,
    StaticProfile,
    proton,
)
from blond.physics.feedbacks.accelerators.sps import (
    SPSCavityFeedback,
    SPSCavityFeedbackCommissioning,
    SPSOneTurnFeedback,
)
from blond.physics.feedbacks.transfer_function_analysis import (
    estimate_transfer_function,
)

# Initialize the accelerator
circumference = 2 * np.pi * 1100.009  # [m]
momentum = 25.92e9
intensity = 2.6e11
n_turns = 200
h = 4620
gamma_t = 17.95
alpha = 1 / gamma_t / gamma_t

voltage_200 = 4.4788e6
voltage_800 = 0.1 * voltage_200
phase_200 = 0.0
phase_800 = np.pi

energy = np.sqrt(momentum**2 + proton.mass**2)
rel_gamma = energy / proton.mass
rel_beta = np.sqrt(1 - 1 / rel_gamma**2)

# The beam
number_of_bunches = 72  # Length of the batch [number of bunches]
n_macroparticles = 100_000  # Number of macroparticles per bunch [-]
tau_bunch = 3.0e-9  # Bunch length [s]
bunch_spacing = 5  # Bunch spacing [number of rf buckets]
injection_energy_error = 0  # Injection energy error [eV]
injection_phase_error = 0  # 40
bucket_shift = 0

# Beam control parameters
G_llrf = 20
G_tx = 1
G_ff = 1
a_comb = 63 / 64


class TestSPSOneTurnFeedback(unittest.TestCase):
    @staticmethod
    def create_scenario(
        commissioning: SPSCavityFeedbackCommissioning = None,
        n_sections: int = 3,
        v_part: float = 4 / 9,
        twc_tau: float = None,
        n_pretrack: int = 1000,
    ):
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

        cavity_feedback = SPSOneTurnFeedback(
            profile=profile,
            n_sections=n_sections,
            commissioning=commissioning,
            v_part=v_part,
            g_ff=G_ff,
            g_tx=G_tx,
            g_llrf=G_llrf,
            a_comb=a_comb,
            n_pretrack=n_pretrack,
        )

        if twc_tau is not None:
            cavity_feedback.TWC.tau = twc_tau

        cavity.attach_cavity_feedback(cavity_feedback, harmonic_index=0)

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

        simulation.finalize(
            (beam,),
            n_turns,
        )

        return simulation, beam

    def test_custom_setpoint(self):
        v_set_custom = np.zeros(2 * h, dtype=complex)
        one_turn = np.zeros(h, dtype=complex)
        one_turn[: h // 2] = np.linspace(0, 1, h // 2) * (1 + 1j * 0)
        one_turn[h // 2 :] = np.linspace(1, 0, h // 2) * (1 + 1j * 0)
        v_set_custom[:h] = one_turn
        v_set_custom[h:] = one_turn

        commissioning = SPSCavityFeedbackCommissioning(v_set=v_set_custom)

        simulation, beam = self.create_scenario(commissioning=commissioning)

        rf_station = simulation.ring.elements.get_element(
            MultiHarmonicRFStation
        )
        cavity_feedback: SPSOneTurnFeedback = (
            rf_station.get_main_harmonic_cavity_feedback()
        )

        np.testing.assert_array_equal(
            cavity_feedback.buffers_coarse.v_setpoint.curr, one_turn
        )

        np.testing.assert_allclose(
            cavity_feedback.buffers_coarse.v_ant.curr, one_turn, atol=0.02
        )

    def test_incorrect_custom_setpoint(self):
        v_len = h + 10
        v_set_custom = np.zeros(2 * v_len, dtype=complex)
        one_turn = np.zeros(v_len, dtype=complex)
        one_turn[: v_len // 2] = np.linspace(0, 1, v_len // 2) * (1 + 1j * 0)
        one_turn[v_len // 2 :] = np.linspace(1, 0, v_len // 2) * (1 + 1j * 0)
        v_set_custom[:v_len] = one_turn
        v_set_custom[v_len:] = one_turn

        commissioning = SPSCavityFeedbackCommissioning(v_set=v_set_custom)

        with self.assertRaises(RuntimeError):
            simulation, beam = self.create_scenario(
                commissioning=commissioning
            )

    def test_failure_in_init(self):
        # Check incorrect partitioning
        with self.assertRaises(ValueError):
            self.create_scenario(v_part=1.1)

        # Check incorrect number of sections
        with self.assertRaises(ValueError):
            self.create_scenario(n_sections=2)

    def test_incorrect_tws_tau(self):
        with self.assertRaises(ValueError):
            self.create_scenario(twc_tau=1e-9)

    def test_standard_commissioning(self):
        simulation, beam = self.create_scenario()

        rf_station = simulation.ring.elements.get_element(
            MultiHarmonicRFStation
        )
        cavity_feedback: SPSOneTurnFeedback = (
            rf_station.get_main_harmonic_cavity_feedback()
        )

        self.assertEqual(cavity_feedback.open_loop, 1)

        self.assertEqual(cavity_feedback.open_fb, 1)

        self.assertEqual(cavity_feedback.open_drive, 1)

        self.assertEqual(cavity_feedback.open_ff, 0)

        self.assertEqual(cavity_feedback.custom_setpoint, None)

        self.assertEqual(cavity_feedback.rot_iq, 1)

        self.assertEqual(cavity_feedback.excitation, 0)

    def test_pretrack(self):
        f_rf = 400.789e6 / 2
        harmonic = 4620
        v_set_custom = np.ones(2 * h, dtype=complex) * 1e6

        commissioning = SPSCavityFeedbackCommissioning(
            open_fb=True, v_set=v_set_custom
        )

        profile = StaticProfile(cut_left=0, cut_right=2.5e-9, n_bins=4)

        cavity_feedback = SPSOneTurnFeedback(
            profile=profile,
            n_sections=3,
            n_cavities=1,
            commissioning=commissioning,
            n_pretrack=0,
        )
        cavity_feedback.disable_fine_grid = True

        cavity_feedback.set_hardware_commissioning(
            omega_rf=2 * np.pi * f_rf, harmonic=harmonic
        )

        np.testing.assert_allclose(
            cavity_feedback.buffers_coarse.v_ant.curr, np.zeros(harmonic)
        )

        cavity_feedback = SPSOneTurnFeedback(
            profile=profile,
            n_sections=3,
            n_cavities=1,
            commissioning=commissioning,
            n_pretrack=1000,
        )
        cavity_feedback.disable_fine_grid = True

        cavity_feedback.set_hardware_commissioning(
            omega_rf=2 * np.pi * f_rf, harmonic=harmonic
        )

        np.testing.assert_allclose(
            np.abs(cavity_feedback.buffers_coarse.v_ant.curr),
            np.abs(cavity_feedback.buffers_coarse.v_setpoint.curr),
            rtol=0.05,
        )


class TestSPSCavityFeedback(unittest.TestCase):
    @staticmethod
    def create_scenario(
        commissioning: SPSCavityFeedbackCommissioning | list = None,
        post_ls2: bool = True,
        n_pretrack: int = 1000,
        v_part: float = None,
        a_comb: float | None = 63 / 64,
        g_llrf: float = 20,
    ):
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

        cavity_feedback = SPSCavityFeedback(
            profile=profile,
            commissioning=commissioning,
            g_ff=G_ff,
            g_tx=G_tx,
            g_llrf=g_llrf,
            a_comb=a_comb,
            post_LS2=post_ls2,
            n_pretrack=n_pretrack,
            v_part=v_part,
        )

        cavity.attach_cavity_feedback(cavity_feedback, harmonic_index=0)

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
        _dt += injection_phase_error * t_rf / 360

        simulation.finalize(
            (beam,),
            n_turns,
        )

        profile.track(beam)

        return simulation, beam

    @staticmethod
    def voltage_and_power_from_first_two_turns(
        cavity_feedback: SPSCavityFeedback, beam: Beam
    ):
        n_turns = 2
        mean_voltage_otfb_1 = np.zeros(n_turns + 1)
        mean_voltage_otfb_2 = np.zeros(n_turns + 1)
        max_voltage_otfb_1 = np.zeros(n_turns + 1)
        max_voltage_otfb_2 = np.zeros(n_turns + 1)

        mean_power_otfb_1 = np.zeros(n_turns + 1)
        mean_power_otfb_2 = np.zeros(n_turns + 1)
        max_power_otfb_1 = np.zeros(n_turns + 1)
        max_power_otfb_2 = np.zeros(n_turns + 1)

        for i in range(n_turns + 1):
            mean_voltage_otfb_1[i] = np.mean(
                np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr)
            )
            mean_voltage_otfb_2[i] = np.mean(
                np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr)
            )

            max_voltage_otfb_1[i] = np.max(
                np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr)
            )
            max_voltage_otfb_2[i] = np.max(
                np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr)
            )

            mean_power_otfb_1[i] = np.mean(
                np.abs(cavity_feedback.OTFB_1.calc_power()[-h:])
            )
            mean_power_otfb_2[i] = np.mean(
                np.abs(cavity_feedback.OTFB_2.calc_power()[-h:])
            )

            max_power_otfb_1[i] = np.max(
                np.abs(cavity_feedback.OTFB_1.calc_power()[-h:])
            )
            max_power_otfb_2[i] = np.max(
                np.abs(cavity_feedback.OTFB_2.calc_power()[-h:])
            )

            cavity_feedback.track(beam)

        output = (
            mean_voltage_otfb_1,
            mean_voltage_otfb_2,
            max_voltage_otfb_1,
            max_voltage_otfb_2,
            mean_power_otfb_1,
            mean_power_otfb_2,
            max_power_otfb_1,
            max_power_otfb_2,
        )

        return output

    def test_one_turn_delay_feedback(self):
        commissioning = None
        simulation, beam = self.create_scenario(commissioning=commissioning)

        rf_station = simulation.ring.elements.get_element(
            MultiHarmonicRFStation
        )
        cavity_feedback: SPSCavityFeedback = (
            rf_station.get_main_harmonic_cavity_feedback()
        )

        output = self.voltage_and_power_from_first_two_turns(
            cavity_feedback, beam
        )

        self.assertEqual(cavity_feedback.OTFB_1.open_ff, 0)
        self.assertEqual(cavity_feedback.OTFB_2.open_ff, 0)

        target_mean_voltage_3sec = [
            669693.8922412858,
            719594.6231249172,
            700716.0597870443,
        ]
        target_mean_voltage_4sec = [
            887723.7656973798,
            991858.371108505,
            957526.8680339582,
        ]
        np.testing.assert_allclose(
            output[0],
            target_mean_voltage_3sec,
        )
        np.testing.assert_allclose(
            output[1],
            target_mean_voltage_4sec,
        )

        target_max_voltage_3sec = [
            669693.8922502627,
            1353966.801056147,
            1134006.369873736,
        ]
        target_max_voltage_4sec = [
            887723.7657088347,
            2288145.950835744,
            1885204.821227101,
        ]
        np.testing.assert_allclose(
            output[2],
            target_max_voltage_3sec,
        )
        np.testing.assert_allclose(
            output[3],
            target_max_voltage_4sec,
        )

        target_mean_power_3sec = [
            118552.70780873713,
            118634.04286945327,
            118835.5114545722,
        ]
        target_mean_power_4sec = [
            123909.11559327216,
            124112.44113981347,
            123737.62042847557,
        ]
        np.testing.assert_allclose(
            output[4],
            target_mean_power_3sec,
        )
        np.testing.assert_allclose(
            output[5],
            target_mean_power_4sec,
        )

        target_max_power_3sec = [
            118552.70791481061,
            125515.57560204196,
            156460.55283835143,
        ]
        target_max_power_4sec = [
            123909.11569149903,
            135697.80857785046,
            184763.7947763473,
        ]
        np.testing.assert_allclose(
            output[6],
            target_max_power_3sec,
        )
        np.testing.assert_allclose(
            output[7],
            target_max_power_4sec,
        )

    def test_feedforward(self):
        commissioning = SPSCavityFeedbackCommissioning(
            open_ff=False,
        )
        simulation, beam = self.create_scenario(commissioning=commissioning)

        rf_station = simulation.ring.elements.get_element(
            MultiHarmonicRFStation
        )
        cavity_feedback: SPSCavityFeedback = (
            rf_station.get_main_harmonic_cavity_feedback()
        )

        output = self.voltage_and_power_from_first_two_turns(
            cavity_feedback, beam
        )

        self.assertEqual(cavity_feedback.OTFB_1.open_ff, 1)
        self.assertEqual(cavity_feedback.OTFB_2.open_ff, 1)

        target_mean_voltage_3sec = [
            669693.8922412858,
            719750.2763709549,
            670134.4944808393,
        ]
        target_mean_voltage_4sec = [
            887723.7656973798,
            992215.3445669926,
            883823.0669048417,
        ]
        np.testing.assert_allclose(
            output[0],
            target_mean_voltage_3sec,
        )
        np.testing.assert_allclose(
            output[1],
            target_mean_voltage_4sec,
        )

        target_max_voltage_3sec = [
            669693.8922502627,
            1353966.801056147,
            744371.5534235353,
        ]
        target_max_voltage_4sec = [
            887723.7657088347,
            2288145.950835744,
            1049911.8808480394,
        ]
        np.testing.assert_allclose(
            output[2],
            target_max_voltage_3sec,
        )
        np.testing.assert_allclose(
            output[3],
            target_max_voltage_4sec,
        )

        target_mean_power_3sec = [
            118552.70780873713,
            119333.04968393837,
            147777.61765445358,
        ]
        target_mean_power_4sec = [
            123909.11559327216,
            125489.86623813033,
            173491.12236045886,
        ]
        np.testing.assert_allclose(
            output[4],
            target_mean_power_3sec,
        )
        np.testing.assert_allclose(
            output[5],
            target_mean_power_4sec,
        )

        target_max_power_3sec = [
            118552.70791481061,
            263030.6139606892,
            756477.2050868174,
        ]
        target_max_power_4sec = [
            123909.11569149903,
            347876.7127743392,
            1181683.3857683493,
        ]
        np.testing.assert_allclose(
            output[6],
            target_max_power_3sec,
        )
        np.testing.assert_allclose(
            output[7],
            target_max_power_4sec,
        )

    def test_pre_ls2_settings(self):
        commissioning = SPSCavityFeedbackCommissioning(
            open_ff=True,
        )
        simulation, beam = self.create_scenario(
            commissioning=commissioning, post_ls2=False
        )

        rf_station = simulation.ring.elements.get_element(
            MultiHarmonicRFStation
        )
        cavity_feedback: SPSCavityFeedback = (
            rf_station.get_main_harmonic_cavity_feedback()
        )

        output = self.voltage_and_power_from_first_two_turns(
            cavity_feedback, beam
        )

        target_mean_voltage_4sec = [
            986359.7396637555,
            1087845.4111294912,
            1054147.9821517656,
        ]
        target_mean_voltage_5sec = [
            1237766.6594053463,
            1401106.087261154,
            1345022.3575921939,
        ]
        np.testing.assert_allclose(
            output[0],
            target_mean_voltage_4sec,
        )
        np.testing.assert_allclose(
            output[1],
            target_mean_voltage_5sec,
        )

        target_max_voltage_4sec = [
            986359.7396764832,
            2352372.3501146217,
            1952416.74259459,
        ]
        target_max_voltage_5sec = [
            1237766.6594214134,
            3456614.0311293746,
            2806388.1290586162,
        ]
        np.testing.assert_allclose(
            output[2],
            target_max_voltage_4sec,
        )
        np.testing.assert_allclose(
            output[3],
            target_max_voltage_5sec,
        )

        target_mean_power_4sec = [
            152974.21678181743,
            153184.438964153,
            152422.63681253025,
        ]
        target_mean_power_5sec = [
            146456.5969673544,
            146818.49131173396,
            148077.8242585613,
        ]
        np.testing.assert_allclose(
            output[4],
            target_mean_power_4sec,
        )
        np.testing.assert_allclose(
            output[5],
            target_mean_power_5sec,
        )

        target_max_power_4sec = [
            152974.21690113,
            165080.96144978618,
            211365.16790033312,
        ]
        target_max_power_5sec = [
            146456.59708978832,
            168548.10856555318,
            256557.62510228768,
        ]
        np.testing.assert_allclose(
            output[6],
            target_max_power_4sec,
        )
        np.testing.assert_allclose(
            output[7],
            target_max_power_5sec,
        )

    def test_mixed_feedback_settings(self):
        commissioning_1 = SPSCavityFeedbackCommissioning(
            open_ff=True,
        )
        commissioning_2 = SPSCavityFeedbackCommissioning(
            open_ff=False,
        )
        simulation, beam = self.create_scenario(
            commissioning=[commissioning_1, commissioning_2]
        )

        rf_station = simulation.ring.elements.get_element(
            MultiHarmonicRFStation
        )
        cavity_feedback: SPSCavityFeedback = (
            rf_station.get_main_harmonic_cavity_feedback()
        )

        output = self.voltage_and_power_from_first_two_turns(
            cavity_feedback, beam
        )

        self.assertEqual(cavity_feedback.OTFB_1.open_ff, 0)
        self.assertEqual(cavity_feedback.OTFB_2.open_ff, 1)

        target_mean_voltage_3sec = [
            669693.8922412858,
            719594.6231249172,
            700716.0597870443,
        ]
        target_mean_voltage_4sec = [
            887723.7656973798,
            992215.3445669926,
            883823.0669048417,
        ]
        np.testing.assert_allclose(
            output[0],
            target_mean_voltage_3sec,
        )
        np.testing.assert_allclose(
            output[1],
            target_mean_voltage_4sec,
        )

        target_max_voltage_3sec = [
            669693.8922502627,
            1353966.801056147,
            1134006.369873736,
        ]
        target_max_voltage_4sec = [
            887723.7657088347,
            2288145.950835744,
            1049911.8808480394,
        ]
        np.testing.assert_allclose(
            output[2],
            target_max_voltage_3sec,
        )
        np.testing.assert_allclose(
            output[3],
            target_max_voltage_4sec,
        )

        target_mean_power_3sec = [
            118552.70780873713,
            118634.04286945327,
            118835.5114545722,
        ]
        target_mean_power_4sec = [
            123909.11559327216,
            125489.86623813033,
            173491.12236045886,
        ]
        np.testing.assert_allclose(
            output[4],
            target_mean_power_3sec,
        )
        np.testing.assert_allclose(
            output[5],
            target_mean_power_4sec,
        )

        target_max_power_3sec = [
            118552.70791481061,
            125515.57560204196,
            156460.55283835143,
        ]
        target_max_power_4sec = [
            123909.11569149903,
            347876.7127743392,
            1181683.3857683493,
        ]
        np.testing.assert_allclose(
            output[6],
            target_max_power_3sec,
        )
        np.testing.assert_allclose(
            output[7],
            target_max_power_4sec,
        )

    def test_nonexplicit_comb_filter_coefficient_and_partitioning(self):
        simulation, beam = self.create_scenario(
            post_ls2=True, v_part=0.5, a_comb=None
        )

        rf_station = simulation.ring.elements.get_element(
            MultiHarmonicRFStation
        )
        cavity_feedback: SPSCavityFeedback = (
            rf_station.get_main_harmonic_cavity_feedback()
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            * cavity_feedback.OTFB_1.n_cavities
            / 0.5
            / voltage_200,
            0.9968353014814768,
            places=4,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            * cavity_feedback.OTFB_2.n_cavities
            / (1 - 0.5)
            / voltage_200,
            0.9910285854440698,
            places=4,
        )

        simulation, beam = self.create_scenario(
            post_ls2=False, v_part=0.5, a_comb=None, g_llrf=10
        )

        rf_station = simulation.ring.elements.get_element(
            MultiHarmonicRFStation
        )
        cavity_feedback: SPSCavityFeedback = (
            rf_station.get_main_harmonic_cavity_feedback()
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            * cavity_feedback.OTFB_1.n_cavities
            / 0.5
            / voltage_200,
            0.9834997877347231,
            places=4,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            * cavity_feedback.OTFB_2.n_cavities
            / (1 - 0.5)
            / voltage_200,
            0.9906089027461353,
            places=4,
        )

    def test_incorrect_turns(self):
        with self.assertRaises(RuntimeError):
            self.create_scenario(n_pretrack=-1)

    def test_incorrect_partitioning(self):
        with self.assertRaises(RuntimeError):
            self.create_scenario(v_part=-0.2)


class TestSPSCavityFeedbackTransferFunction(unittest.TestCase):
    def get_simulated_transfer_function(
        self,
        commissioning: SPSCavityFeedbackCommissioning,
        cut_data: int = 0,
        n_pretrack: int = 1000,
    ):
        f_rf = 400.789e6 / 2
        harmonic = 4620

        profile = StaticProfile(cut_left=0, cut_right=2.5e-9, n_bins=4)

        self.cavity_feedback = SPSOneTurnFeedback(
            profile=profile,
            n_sections=3,
            n_cavities=1,
            commissioning=commissioning,
            n_pretrack=n_pretrack,
        )
        self.cavity_feedback.disable_fine_grid = True

        self.cavity_feedback.set_hardware_commissioning(
            omega_rf=2 * np.pi * f_rf, harmonic=harmonic
        )

        return estimate_transfer_function(
            input_signal=self.cavity_feedback.v_excitation_in,
            output_signal=self.cavity_feedback.v_excitation_out,
            t_s=self.cavity_feedback.T_s,
            data_cut=cut_data,
        )

    @staticmethod
    def get_analytic_transfer_function(cavity_feedback: SPSOneTurnFeedback):
        r_g = cavity_feedback.TWC.R_gen
        tau = cavity_feedback.TWC.tau
        f_c = cavity_feedback.TWC.omega_r / 2 / np.pi
        f_rf = cavity_feedback.omega_rf / 2 / np.pi
        g_otfb = cavity_feedback.G_tx * cavity_feedback.G_llrf
        a_comb = cavity_feedback.a_comb
        n_clock = cavity_feedback.n_coarse
        f_clock = n_clock * f_rf / h

        z_gen = lambda f: (
            r_g * (np.sinc(tau * (f - f_c)) + np.sinc(tau * (f + f_c)))
        )
        h_comb = lambda z: (
            g_otfb
            * (1 - a_comb)
            / (1 - a_comb * z ** (-n_clock))
            * z ** (-n_clock)
        )
        h_cav = lambda s: 1 / (1 + tau * s / 2)

        h_open = lambda f: (
            z_gen(f)
            * h_comb(np.exp(2j * np.pi * (f - f_rf) / f_clock))
            * h_cav(2j * np.pi * (f - f_c))
            / r_g
        )

        return h_open

    def test_open_loop_response(self):
        v_set_custom = np.ones(2 * h, dtype=complex) * 1e6 * 0.0

        commissioning = SPSCavityFeedbackCommissioning(
            excitation=True, open_ff=True, open_loop=True, v_set=v_set_custom
        )
        f_est, h_est = self.get_simulated_transfer_function(
            commissioning=commissioning, cut_data=h * 5, n_pretrack=4000
        )

        h_open = self.get_analytic_transfer_function(
            cavity_feedback=self.cavity_feedback
        )

        f_c = self.cavity_feedback.TWC.omega_r / 2 / np.pi
        f_rf = self.cavity_feedback.omega_rf / 2 / np.pi
        df = f_c - f_rf

        h_expected = (
            h_open(f_est + f_rf)
            * np.max(np.abs(h_est))
            / np.max(np.abs(h_open(f_est + f_rf)))
        )

        mask = (-2.1e6 < (f_est - df)) & ((f_est - df) < 2.1e6)

        h_est = h_est[mask]
        h_expected = h_expected[mask]
        f_est = f_est[mask]

        peaks_est, prop_est = find_peaks(np.abs(h_est), distance=800)
        peaks_exp, prop_exp = find_peaks(np.abs(h_expected), distance=800)

        self.assertEqual(len(peaks_exp), len(peaks_est))

        np.testing.assert_allclose(peaks_exp, peaks_est, atol=4)
