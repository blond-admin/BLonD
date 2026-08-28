import unittest
from unittest.mock import Mock

import numpy as np

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    MultiHarmonicRFStation,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    proton,
)
from blond.physics.feedbacks.buffers import OneTurnBufferBase
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedback

circumference = 26658.8832  # [m]
momentum = 450e9
intensity = 1.6e11
n_turns = 1
voltage = 5e6
h = 35640
gamma_t = 53.8
alpha = 1 / gamma_t / gamma_t

n_macroparticles = 100_000
tau_bunch = 1.2e-9


class DummyFeedback(IQCavityFeedback[OneTurnBufferBase, OneTurnBufferBase]):
    buffer_cls_coarse = OneTurnBufferBase
    buffer_cls_fine = OneTurnBufferBase

    def __init__(
        self,
        profile: StaticProfile,
        n_cavities: int = 4,
        harmonic_index: int = 0,
        n_periods_coarse=1,
    ):
        super().__init__(
            profile=profile,
            n_cavities=n_cavities,
            n_periods_coarse=n_periods_coarse,
            harmonic_index=harmonic_index,
        )

    def update_fb_variables(self):
        pass

    def circuit_track(self, no_beam: bool = False) -> None:
        pass


class TestIQCavityFeedback(unittest.TestCase):
    def test_n_periods_coarse_as_float(self):
        profile = StaticProfile(cut_left=0.0, cut_right=2.5e-9, n_bins=64)

        self.assertWarns(
            UserWarning, DummyFeedback, profile, n_periods_coarse=0.5
        )

    def test_n_periods_coarse_below_one(self):
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

        cavity = SingleHarmonicRFStation(
            voltage=voltage,
            phi_rf=0.0,
            harmonic=h,
        )

        f_rf = cavity.calc_main_harmonic_omega_rf_design(
            rel_beta, lattice.orbit_length
        ) / (2 * np.pi)
        f_rev = f_rf / h
        t_rf = 1 / f_rf
        t_rev = 1 / f_rev

        profile = StaticProfile(
            cut_left=-1.5 * t_rf,
            cut_right=2.5 * t_rf,
            n_bins=4 * 2**6,
        )

        cavity_feedback = DummyFeedback(profile=profile, n_periods_coarse=0.5)

        cavity.attach_cavity_feedback(cavity_feedback)

        bigaussian = BiGaussian(
            n_macroparticles, sigma_dt=tau_bunch / 4, seed=1234
        )

        ring = Ring(
            circumference,
        )
        ring_elements = [profile, cavity, lattice]
        ring.add_elements(
            ring_elements,
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

        expected_rf_centers = (
            np.arange(cavity_feedback.n_coarse) * cavity_feedback.T_s
            + 0.5 * t_rf * cavity_feedback.n_periods_coarse
        )

        np.testing.assert_allclose(
            expected_rf_centers,
            cavity_feedback.rf_centers,
        )

    @staticmethod
    def create_rf_parameter_tests(single_harmonic: bool = False):
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

        if single_harmonic:
            cavity = SingleHarmonicRFStation(
                voltage=voltage,
                phi_rf=0.0,
                harmonic=h,
            )
        else:
            cavity = MultiHarmonicRFStation(
                n_harmonics=2,
                main_harmonic_idx=0,
                voltage=np.array([voltage, voltage * 0.2]),
                phi_rf=np.array([0.0, np.pi]),
                harmonic=np.array([h, 4 * h]),
            )

        f_rf = cavity.calc_main_harmonic_omega_rf_design(
            rel_beta, lattice.orbit_length
        ) / (2 * np.pi)
        f_rev = f_rf / h
        t_rf = 1 / f_rf
        t_rev = 1 / f_rev

        profile = StaticProfile(
            cut_left=-1.5 * t_rf,
            cut_right=2.5 * t_rf,
            n_bins=4 * 2**6,
        )

        cavity_feedback = DummyFeedback(profile=profile, n_periods_coarse=0.5)

        cavity.attach_cavity_feedback(cavity_feedback, harmonic_index=0)

        bigaussian = BiGaussian(
            n_macroparticles, sigma_dt=tau_bunch / 4, seed=1234
        )

        ring = Ring(
            circumference,
        )
        ring_elements = [profile, cavity, lattice]
        ring.add_elements(
            ring_elements,
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

        return cavity_feedback, cavity

    def test_get_design_rf_parameters_multi_harmonic(self):
        cavity_feedback, cavity = self.create_rf_parameter_tests()

        harm, omega_rf_d, phi_rf_d = (
            cavity_feedback.get_harmonic_and_omega_rf_phi_rf_design()
        )

        self.assertEqual(harm, cavity.harmonic[cavity_feedback.harmonic_index])
        self.assertEqual(
            omega_rf_d, cavity.omega_rf_design[cavity_feedback.harmonic_index]
        )
        self.assertEqual(
            phi_rf_d, cavity.phi_rf_design[cavity_feedback.harmonic_index]
        )

    def test_get_design_rf_parameters_single_harmonic(self):
        cavity_feedback, cavity = self.create_rf_parameter_tests(
            single_harmonic=True
        )

        harm, omega_rf_d, phi_rf_d = (
            cavity_feedback.get_harmonic_and_omega_rf_phi_rf_design()
        )

        self.assertEqual(harm, cavity.harmonic)
        self.assertEqual(omega_rf_d, cavity.omega_rf_design)
        self.assertEqual(phi_rf_d, cavity.phi_rf_design)

    def test_get_rf_parameters_incorrect_rf_station(self):
        cavity_feedback, cavity = self.create_rf_parameter_tests(
            single_harmonic=True
        )
        wrong_rf_station = Mock(spec=DriftSimple)

        cavity_feedback._parent_rf_station = wrong_rf_station

        with self.assertRaises(TypeError):
            _ = cavity_feedback.get_harmonic_and_omega_rf_phi_rf()

        with self.assertRaises(TypeError):
            _ = cavity_feedback.get_harmonic_and_omega_rf_phi_rf_design()
