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
n_macroparticles = 1_000_000  # Number of macroparticles per bunch [-]
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

        if commissioning is None:
            commissioning = SPSCavityFeedbackCommissioning(
                open_loop=False, open_ff=False, debug=False
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
        commissioning = SPSCavityFeedbackCommissioning(
            open_ff=True,
        )
        simulation, beam = self.create_scenario(commissioning=commissioning)

        rf_station = simulation.ring.elements.get_element(
            MultiHarmonicRFStation
        )
        cavity_feedback: SPSCavityFeedback = (
            rf_station.get_main_harmonic_cavity_feedback()
        )

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

        target_mean_voltage_3sec = 719879.2193976191
        target_mean_voltage_4sec = 992377.4497708735

        target_max_voltage_3sec = 1357783.0747418394
        target_max_voltage_4sec = 2294980.95775722

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

        target_mean_power_3sec = 118633.90084294468
        target_mean_power_4sec = 124112.49981515526

        target_max_power_3sec = 125500.47766321833
        target_max_power_4sec = 135697.80286706766

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

        target_mean_voltage_3sec = 700907.8794683248
        target_mean_voltage_4sec = 957895.2296272368

        target_max_voltage_3sec = 1136882.8438123984
        target_max_voltage_4sec = 1890516.6957680818

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

        target_mean_power_3sec = 118833.55065205484
        target_mean_power_4sec = 123735.66400154989

        target_max_power_3sec = 156602.43733882008
        target_max_power_4sec = 185052.7158483995

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

        target_mean_voltage_3sec = 720035.4455132243
        target_mean_voltage_4sec = 992736.2671321571

        target_max_voltage_3sec = 1357783.0747418392
        target_max_voltage_4sec = 2294980.9577572206

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

        target_mean_power_3sec = 119337.69544427162
        target_mean_power_4sec = 125500.61683927145

        target_max_power_3sec = 263959.57960349275
        target_max_power_4sec = 349473.06627607805

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

        target_mean_voltage_3sec = 670127.2437983346
        target_mean_voltage_4sec = 883802.2505492448

        target_max_voltage_3sec = 744536.4172064766
        target_max_voltage_4sec = 1050367.0253132517

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

        target_mean_power_3sec = 147970.4698244471
        target_mean_power_4sec = 173847.1718177561

        target_max_power_3sec = 760564.1896078447
        target_max_power_4sec = 1188856.9519884584

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

        target_mean_voltage_4sec = 1088362.0268588583
        target_mean_voltage_5sec = 1401933.1885392386

        target_max_voltage_4sec = 2359186.7095714654
        target_max_voltage_5sec = 3467498.487802676

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

        target_mean_power_4sec = 153184.39271708208
        target_mean_power_5sec = 146818.7618668209

        target_max_power_4sec = 165076.26405158968
        target_max_power_5sec = 168591.50535078964

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

        target_mean_voltage_4sec = 1054513.0622475357
        target_mean_voltage_5sec = 1345602.7211031278

        target_max_voltage_4sec = 1957690.0938218157
        target_max_voltage_5sec = 2814792.525540921

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

        target_mean_power_4sec = 152417.90311351998
        target_mean_power_5sec = 148083.69118486813

        target_max_power_4sec = 211620.68634508818
        target_max_power_5sec = 257114.3703686788

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
