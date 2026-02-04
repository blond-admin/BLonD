import unittest
from pathlib import Path

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
from blond.experimental.physics.feedbacks.accelerators.lhc.beam_feedback import (
    LHCBeamControl,
)
from blond.handle_results.helpers import callers_relative_path


class TestSingleBunchInjectionWithPhaseLoop(unittest.TestCase):
    blond2_data = np.load(
        Path(
            callers_relative_path(
                "../resources/lhc_beam_control_40.0deg.npz",
                stacklevel=1,
            )
        )
    )

    @classmethod
    def setUpClass(cls):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")

        circumference = 26658.8832  # [m]
        momentum = 450e9
        intensity = 1.6e11
        n_turns = 2_000
        voltage = 5e6
        h = 35640
        gamma_t = 53.8
        alpha = 1 / gamma_t / gamma_t

        injection_offset_phase = 40

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

        cavity = MultiHarmonicRFStation(
            voltage=np.array([voltage]),
            phi_rf=np.array([0.0]),
            harmonic=np.array([h]),
            n_harmonics=1,
            main_harmonic_idx=0,
        )

        f_rf = cavity.get_main_harmonic_omega_rf_design(
            rel_beta, lattice.orbit_length
        ) / (2 * np.pi)
        f_rev = f_rf / h
        t_rf = 1 / f_rf
        t_rev = 1 / f_rev

        profile = StaticProfile(
            cut_left=-5.5 / f_rf,
            cut_right=(6.5 + 10) / f_rf,
            n_bins=2**5 * (12 + 10),
        )

        bigaussian = BiGaussian(1_000_000, sigma_dt=1.2e-9 / 4, seed=1234)

        beam_control = LHCBeamControl(
            profile,
            pl_gain=1 / (5 * t_rev) * 1,
            sl_gain=1 / (5 * t_rev) / 10 * 1,
        )

        cavity.attach_beam_feedback(beam_control)

        ring = Ring(
            circumference,
        )

        ring.add_elements(
            [profile, beam_control, cavity, lattice],
        )

        simulation = Simulation(
            ring,
            cycle,
        )

        simulation.prepare_beam(beam, bigaussian)

        beam._dt.array_local += injection_offset_phase * t_rf / 360

        profile.track(beam)

        cls.pl_error = np.zeros(n_turns)
        cls.delta_phi_rf = np.zeros(n_turns)
        cls.omega_rf = np.zeros(n_turns)
        cls.phi_rf = np.zeros(n_turns)

        simulation.finalize(
            (beam,),
            n_turns,
        )

        for i in range(n_turns):
            simulation.turn_i.value = i

            cls.omega_rf[i] = cavity.omega_rf_actual[0]
            cls.phi_rf[i] = cavity.phi_rf_actual[0]

            for element in ring.elements.elements:
                element.track(beam)

            cls.pl_error[i] = beam_control.dphi * 180 / np.pi
            cls.delta_phi_rf[i] = cavity.delta_phi_rf[0] * 180 / np.pi

    def test_phase_loop_error(self):
        np.testing.assert_allclose(
            self.pl_error,
            self.blond2_data["beam_loop_error"],
            atol=1e-1,
            err_msg="Error in phase loop error signal",
        )

    def test_synchronization_loop_error(self):
        np.testing.assert_allclose(
            self.delta_phi_rf,
            self.blond2_data["synchro_loop_error"],
            atol=1e-1,
            err_msg="Error in synchronization loop error signal",
        )

    def test_rf_frequency_swing(self):
        np.testing.assert_allclose(
            self.omega_rf,
            self.blond2_data["omega_rf"],
            atol=1e-2,
            err_msg="Error in rf frequency swing",
        )

    def test_rf_phase_swing(self):
        np.testing.assert_allclose(
            self.phi_rf,
            self.blond2_data["phi_rf"],
            atol=1e-2,
            err_msg="Error in rf phase swing",
        )
