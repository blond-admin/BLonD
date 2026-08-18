import copy
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
    proton,
)
from blond.physics.feedbacks.accelerators.sps import (
    SPSCavityFeedback,
    SPSCavityFeedbackCommissioning,
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
    def test_custom_setpoint(self):
        # TODO: implement
        pass

    def test_failure_in_init(self):
        # TODO: implement
        pass

    def test_incorrect_tws_tau(self):
        # TODO: implement
        pass

    def test_standard_commissioning(self):
        # TODO: implement
        pass


class TestSPSCavityFeedback(unittest.TestCase):
    @staticmethod
    def create_scenario(
        commissioning: SPSCavityFeedbackCommissioning = None,
        post_ls2: bool = True,
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
            g_llrf=G_llrf,
            a_comb=a_comb,
            post_LS2=post_ls2,
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

    def test_one_turn_delay_feedback(self):
        commissioning = None
        simulation, beam = self.create_scenario(commissioning=commissioning)

        rf_station = simulation.ring.elements.get_element(
            MultiHarmonicRFStation
        )
        cavity_feedback: SPSCavityFeedback = (
            rf_station.get_main_harmonic_cavity_feedback()
        )

        self.assertEqual(cavity_feedback.OTFB_1.open_ff, 0)
        self.assertEqual(cavity_feedback.OTFB_2.open_ff, 0)

        target_mean_voltage_3sec = 669693.8922412858
        target_mean_voltage_4sec = 887723.7656973798

        target_max_voltage_3sec = 669693.8922502627
        target_max_voltage_4sec = 887723.7657088347

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            / target_mean_voltage_3sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            / target_mean_voltage_4sec,
            1,
            places=5,
        )

        # Mean is equal to max since the voltage and power should
        # be flat before the arrival of the beam
        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            / target_mean_voltage_3sec,
            1,
            places=5,
        )

        # Mean is equal to max since the voltage and power should
        # be flat before the arrival of the beam
        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            / target_mean_voltage_4sec,
            1,
            places=5,
        )

        target_mean_power_3sec = 118552.70780873713
        target_mean_power_4sec = 123909.11559327216

        target_max_power_3sec = 118552.70791481061
        target_max_power_4sec = 123909.11569149903

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.calc_power()))
            / target_mean_power_3sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.calc_power()))
            / target_mean_power_4sec,
            1,
            places=5,
        )

        # Mean is equal to max since the voltage and power should
        # be flat before the arrival of the beam
        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_1.calc_power()))
            / target_mean_power_3sec,
            1,
            places=5,
        )

        # Mean is equal to max since the voltage and power should
        # be flat before the arrival of the beam
        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_2.calc_power()))
            / target_mean_power_4sec,
            1,
            places=5,
        )

        ##################
        ##### Turn 1 #####
        ##################

        cavity_feedback.track(beam=beam)

        target_mean_voltage_3sec = 719594.6231249172
        target_mean_voltage_4sec = 991858.371108505

        target_max_voltage_3sec = 1353966.801056147
        target_max_voltage_4sec = 2288145.950835744

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            / target_mean_voltage_3sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            / target_mean_voltage_4sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            / target_max_voltage_3sec,
            1,
            places=5,
        )
        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            / target_max_voltage_4sec,
            1,
            places=5,
        )

        target_mean_power_3sec = 118634.04286945327
        target_mean_power_4sec = 124112.44113981347

        target_max_power_3sec = 125515.57560204196
        target_max_power_4sec = 135697.80857785046

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.calc_power()[-h:]))
            / target_mean_power_3sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.calc_power()[-h:]))
            / target_mean_power_4sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_1.calc_power()[-h:]))
            / target_max_power_3sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_2.calc_power()[-h:]))
            / target_max_power_4sec,
            1,
            places=5,
        )

        ##################
        ##### Turn 2 #####
        ##################

        cavity_feedback.track(beam=beam)

        target_mean_voltage_3sec = 700716.0597870443
        target_mean_voltage_4sec = 957526.8680339582

        target_max_voltage_3sec = 1134006.369873736
        target_max_voltage_4sec = 1885204.821227101

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            / target_mean_voltage_3sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            / target_mean_voltage_4sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            / target_max_voltage_3sec,
            1,
            places=5,
        )
        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            / target_max_voltage_4sec,
            1,
            places=5,
        )

        target_mean_power_3sec = 118835.5114545722
        target_mean_power_4sec = 123737.62042847557

        target_max_power_3sec = 156460.55283835143
        target_max_power_4sec = 184763.7947763473

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.calc_power()[-h:]))
            / target_mean_power_3sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.calc_power()[-h:]))
            / target_mean_power_4sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_1.calc_power()[-h:]))
            / target_max_power_3sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_2.calc_power()[-h:]))
            / target_max_power_4sec,
            1,
            places=5,
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

        self.assertEqual(cavity_feedback.OTFB_1.open_ff, 1)
        self.assertEqual(cavity_feedback.OTFB_2.open_ff, 1)

        target_mean_voltage_3sec = 669693.8922412858
        target_mean_voltage_4sec = 887723.7656973798

        target_max_voltage_3sec = 669693.8922502627
        target_max_voltage_4sec = 887723.7657088347

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            / target_mean_voltage_3sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            / target_mean_voltage_4sec,
            1,
            places=5,
        )

        # Mean is equal to max since the voltage and power should
        # be flat before the arrival of the beam
        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            / target_mean_voltage_3sec,
            1,
            places=5,
        )

        # Mean is equal to max since the voltage and power should
        # be flat before the arrival of the beam
        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            / target_mean_voltage_4sec,
            1,
            places=5,
        )

        target_mean_power_3sec = 118552.70780873713
        target_mean_power_4sec = 123909.11559327216

        target_max_power_3sec = 118552.70791481061
        target_max_power_4sec = 123909.11569149903

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.calc_power()))
            / target_mean_power_3sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.calc_power()))
            / target_mean_power_4sec,
            1,
            places=5,
        )

        # Mean is equal to max since the voltage and power should
        # be flat before the arrival of the beam
        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_1.calc_power()))
            / target_mean_power_3sec,
            1,
            places=5,
        )

        # Mean is equal to max since the voltage and power should
        # be flat before the arrival of the beam
        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_2.calc_power()))
            / target_mean_power_4sec,
            1,
            places=5,
        )

        ##################
        ##### Turn 1 #####
        ##################

        cavity_feedback.track(beam=beam)

        target_mean_voltage_3sec = 719750.2763709549
        target_mean_voltage_4sec = 992215.3445669926

        target_max_voltage_3sec = 1353966.801056147
        target_max_voltage_4sec = 2288145.950835744

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            / target_mean_voltage_3sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            / target_mean_voltage_4sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            / target_max_voltage_3sec,
            1,
            places=5,
        )
        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            / target_max_voltage_4sec,
            1,
            places=5,
        )

        target_mean_power_3sec = 119333.04968393837
        target_mean_power_4sec = 125489.86623813033

        target_max_power_3sec = 263030.6139606892
        target_max_power_4sec = 347876.7127743392

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.calc_power()[-h:]))
            / target_mean_power_3sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.calc_power()[-h:]))
            / target_mean_power_4sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_1.calc_power()[-h:]))
            / target_max_power_3sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_2.calc_power()[-h:]))
            / target_max_power_4sec,
            1,
            places=5,
        )

        ##################
        ##### Turn 2 #####
        ##################

        cavity_feedback.track(beam=beam)

        target_mean_voltage_3sec = 670134.4944808393
        target_mean_voltage_4sec = 883823.0669048417

        target_max_voltage_3sec = 744371.5534235353
        target_max_voltage_4sec = 1049911.8808480394

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            / target_mean_voltage_3sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            / target_mean_voltage_4sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            / target_max_voltage_3sec,
            1,
            places=5,
        )
        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            / target_max_voltage_4sec,
            1,
            places=5,
        )

        target_mean_power_3sec = 147777.61765445358
        target_mean_power_4sec = 173491.12236045886

        target_max_power_3sec = 756477.2050868174
        target_max_power_4sec = 1181683.3857683493

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.calc_power()[-h:]))
            / target_mean_power_3sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.calc_power()[-h:]))
            / target_mean_power_4sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_1.calc_power()[-h:]))
            / target_max_power_3sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_2.calc_power()[-h:]))
            / target_max_power_4sec,
            1,
            places=5,
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

        target_mean_voltage_4sec = 986359.7396637555
        target_mean_voltage_5sec = 1237766.6594053463

        target_max_voltage_4sec = 986359.7396764832
        target_max_voltage_5sec = 1237766.6594214134

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            / target_mean_voltage_4sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            / target_mean_voltage_5sec,
            1,
            places=5,
        )

        # Mean is equal to max since the voltage and power should
        # be flat before the arrival of the beam
        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            / target_mean_voltage_4sec,
            1,
            places=5,
        )

        # Mean is equal to max since the voltage and power should
        # be flat before the arrival of the beam
        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            / target_mean_voltage_5sec,
            1,
            places=5,
        )

        target_mean_power_4sec = 152974.21678181743
        target_mean_power_5sec = 146456.5969673544

        target_max_power_4sec = 152974.21690113
        target_max_power_5sec = 146456.59708978832

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.calc_power()))
            / target_mean_power_4sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.calc_power()))
            / target_mean_power_5sec,
            1,
            places=5,
        )

        # Mean is equal to max since the voltage and power should
        # be flat before the arrival of the beam
        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_1.calc_power()))
            / target_mean_power_4sec,
            1,
            places=5,
        )

        # Mean is equal to max since the voltage and power should
        # be flat before the arrival of the beam
        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_2.calc_power()))
            / target_mean_power_5sec,
            1,
            places=5,
        )

        ##################
        ##### Turn 1 #####
        ##################

        cavity_feedback.track(beam=beam)

        target_mean_voltage_4sec = 1087845.4111294912
        target_mean_voltage_5sec = 1401106.087261154

        target_max_voltage_4sec = 2352372.3501146217
        target_max_voltage_5sec = 3456614.0311293746

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            / target_mean_voltage_4sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            / target_mean_voltage_5sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            / target_max_voltage_4sec,
            1,
            places=5,
        )
        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            / target_max_voltage_5sec,
            1,
            places=5,
        )

        target_mean_power_4sec = 153184.438964153
        target_mean_power_5sec = 146818.49131173396

        target_max_power_4sec = 165080.96144978618
        target_max_power_5sec = 168548.10856555318

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.calc_power()[-h:]))
            / target_mean_power_4sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.calc_power()[-h:]))
            / target_mean_power_5sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_1.calc_power()[-h:]))
            / target_max_power_4sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_2.calc_power()[-h:]))
            / target_max_power_5sec,
            1,
            places=5,
        )

        ##################
        ##### Turn 2 #####
        ##################

        cavity_feedback.track(beam=beam)

        target_mean_voltage_4sec = 1054147.9821517656
        target_mean_voltage_5sec = 1345022.3575921939

        target_max_voltage_4sec = 1952416.74259459
        target_max_voltage_5sec = 2806388.1290586162

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            / target_mean_voltage_4sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            / target_mean_voltage_5sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr))
            / target_max_voltage_4sec,
            1,
            places=5,
        )
        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr))
            / target_max_voltage_5sec,
            1,
            places=5,
        )

        target_mean_power_4sec = 152422.63681253025
        target_mean_power_5sec = 148077.8242585613

        target_max_power_4sec = 211365.16790033312
        target_max_power_5sec = 256557.62510228768

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_1.calc_power()[-h:]))
            / target_mean_power_4sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.mean(np.abs(cavity_feedback.OTFB_2.calc_power()[-h:]))
            / target_mean_power_5sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_1.calc_power()[-h:]))
            / target_max_power_4sec,
            1,
            places=5,
        )

        self.assertAlmostEqual(
            np.max(np.abs(cavity_feedback.OTFB_2.calc_power()[-h:]))
            / target_max_power_5sec,
            1,
            places=5,
        )

    def test_cpp_convolution(self):
        commissioning = SPSCavityFeedbackCommissioning(
            open_ff=False, cpp_conv=True
        )
        simulation, beam = self.create_scenario(
            commissioning=commissioning, post_ls2=True
        )

        rf_station = simulation.ring.elements.get_element(
            MultiHarmonicRFStation
        )
        cavity_feedback: SPSCavityFeedback = (
            rf_station.get_main_harmonic_cavity_feedback()
        )

        cavity_feedback.track(beam=beam)
        cavity_feedback.track(beam=beam)

        mean_voltage_3sec_cpp = np.mean(
            np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr)
        )
        max_voltage_3sec_cpp = np.max(
            np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr)
        )

        mean_voltage_4sec_cpp = np.mean(
            np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr)
        )
        max_voltage_4sec_cpp = np.max(
            np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr)
        )

        mean_power_3sec_cpp = np.mean(
            np.abs(cavity_feedback.OTFB_1.calc_power()[-h:])
        )
        max_power_3sec_cpp = np.max(
            np.abs(cavity_feedback.OTFB_1.calc_power()[-h:])
        )

        mean_power_4sec_cpp = np.mean(
            np.abs(cavity_feedback.OTFB_2.calc_power()[-h:])
        )
        max_power_4sec_cpp = np.max(
            np.abs(cavity_feedback.OTFB_2.calc_power()[-h:])
        )

        commissioning = SPSCavityFeedbackCommissioning(
            open_ff=False, cpp_conv=False
        )
        simulation, beam = self.create_scenario(
            commissioning=commissioning, post_ls2=True
        )

        rf_station = simulation.ring.elements.get_element(
            MultiHarmonicRFStation
        )
        cavity_feedback: SPSCavityFeedback = (
            rf_station.get_main_harmonic_cavity_feedback()
        )

        cavity_feedback.track(beam=beam)
        cavity_feedback.track(beam=beam)

        mean_voltage_3sec_py = np.mean(
            np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr)
        )
        max_voltage_3sec_py = np.max(
            np.abs(cavity_feedback.OTFB_1.buffers_coarse.v_ant.curr)
        )

        mean_voltage_4sec_py = np.mean(
            np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr)
        )
        max_voltage_4sec_py = np.max(
            np.abs(cavity_feedback.OTFB_2.buffers_coarse.v_ant.curr)
        )

        mean_power_3sec_py = np.mean(
            np.abs(cavity_feedback.OTFB_1.calc_power()[-h:])
        )
        max_power_3sec_py = np.max(
            np.abs(cavity_feedback.OTFB_1.calc_power()[-h:])
        )

        mean_power_4sec_py = np.mean(
            np.abs(cavity_feedback.OTFB_2.calc_power()[-h:])
        )
        max_power_4sec_py = np.max(
            np.abs(cavity_feedback.OTFB_2.calc_power()[-h:])
        )

        # Check voltages
        self.assertAlmostEqual(
            mean_voltage_3sec_cpp, mean_voltage_3sec_py, places=5
        )

        self.assertAlmostEqual(
            max_voltage_3sec_cpp, max_voltage_3sec_py, places=5
        )

        self.assertAlmostEqual(
            mean_voltage_4sec_cpp, mean_voltage_4sec_py, places=5
        )

        self.assertAlmostEqual(
            max_voltage_4sec_cpp, max_voltage_4sec_py, places=5
        )

        # Check RF power
        self.assertAlmostEqual(
            mean_power_3sec_cpp, mean_power_3sec_py, places=5
        )

        self.assertAlmostEqual(max_power_3sec_cpp, max_power_3sec_py, places=5)

        self.assertAlmostEqual(
            mean_power_4sec_cpp, mean_power_4sec_py, places=5
        )

        self.assertAlmostEqual(max_power_4sec_cpp, max_power_4sec_py, places=5)

    def test_two_different_commissionings(self):
        # TODO: implement
        pass

    def test_debugging(self):
        # TODO: implement
        pass


class TestSPSCavityFeedbackTransferFunction(unittest.TestCase):
    def create_scenario(self):
        # TODO: implement
        pass

    def test_open_loop_response(self):
        # TODO: implement
        pass

    def test_closed_loop_response(self):
        # TODO: implement
        pass

    def test_one_turn_delay_feedback_reponse(self):
        # TODO: implement
        pass
