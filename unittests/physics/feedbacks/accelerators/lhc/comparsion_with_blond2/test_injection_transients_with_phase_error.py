import unittest
from pathlib import Path

import numpy as np

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    MultiHarmonicRfStation,
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
from blond.experimental.physics.feedbacks.accelerators.lhc.cavity_feedback import (
    LHCCavityLoop,
    LHCCavityLoopCommissioning,
)
from blond.handle_results.helpers import callers_relative_path


class TestInjectionWithPhaseError(unittest.TestCase):
    blond2_data = np.load(
        Path(
            callers_relative_path(
                "../resources/lhc_convergence_to_steadystate_40.0deg.npz",
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
        n_bunches = 36
        intensity = 1.6e11 * n_bunches
        n_turns = 500
        voltage = 5e6
        h = 35640
        gamma_t = 53.606713
        delta_f = -3480

        bunch_lengths = 1.2e-9

        bucket_shift = 10_000
        injection_phase_error = 40

        energy = np.sqrt(momentum**2 + proton.mass**2)
        rel_gamma = energy / proton.mass
        rel_beta = np.sqrt(1 - 1 / rel_gamma**2)

        beam = Beam(
            intensity,
            proton,
        )

        cycle = ConstantMagneticCycle(proton, momentum, in_unit="momentum")

        lattice = DriftSimple(
            orbit_length=circumference,
            momentum_compaction_factor=1.0 / gamma_t / gamma_t,
        )

        cavity = MultiHarmonicRfStation(
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
            cut_left=(bucket_shift - 5.5) / f_rf,
            cut_right=(bucket_shift + 6.5 + n_bunches * 10) / f_rf,
            n_bins=2**5 * (12 + n_bunches * 10),
        )

        # LHC cavity feedback
        commissioning = LHCCavityLoopCommissioning(
            G_a=6.79e-6,
            G_d=10,
            G_o=10,
            tau_a=170e-6,
            tau_d=400e-6,
            tau_o=110e-6,
        )
        cavity_control = LHCCavityLoop(
            profile=profile,
            tau_otfb=1.2e-6,
            f_c=f_rf + delta_f,
            RFFB=commissioning,
            n_pretrack=200,
        )
        cavity.attach_cavity_feedback(cavity_control)

        # LHC beam feedback
        beam_control = LHCBeamControl(
            profile,
            pl_gain=1 / (5 * t_rev) * 1,
            sl_gain=1 / (5 * t_rev) / 10 * 1,
            current_thres=0.5,
        )
        cavity.attach_beam_feedback(beam_control)

        bigaussian = BiGaussian(100_000, sigma_dt=bunch_lengths / 4, seed=1234)

        ring = Ring(
            circumference,
        )

        ring.add_elements(
            [profile, cavity, beam_control, lattice],
        )

        simulation = Simulation(
            ring,
            cycle,
        )

        simulation.prepare_beam(beam, bigaussian)

        _dt_tmp = beam._dt
        _dE_tmp = beam._dE
        _flags_tmp = beam._flags
        _ids_tmp = beam._ids

        for i in range(1, n_bunches):
            beam._dt = np.append(beam._dt, _dt_tmp + 10 * t_rf * i)
            beam._dE = np.append(beam._dE, _dE_tmp)
            beam._flags = np.append(beam._flags, _flags_tmp)
            beam._ids = np.append(beam._ids, _ids_tmp)

        beam._dt += bucket_shift * t_rf + injection_phase_error / 360 * t_rf

        simulation.finalize(
            (beam,),
            n_turns,
        )

        cls.v_ant = np.zeros((n_turns, h // 10), dtype=complex)
        cls.i_beam = np.zeros((n_turns, h // 10), dtype=complex)
        cls.rf_power = np.zeros((n_turns, h // 10), dtype=complex)
        cls.rf_beam_current_phase = np.zeros((n_turns, n_bunches))
        cls.beam_loop_phase = np.zeros(n_turns)

        for i in range(n_turns):
            simulation.turn_i.value = i

            for element in ring.elements.elements:
                element.track(beam)

            cls.v_ant[i, :] = cavity_control.V_ANT_COARSE[-h // 10 :]
            cls.i_beam[i, :] = cavity_control.I_BEAM_COARSE[-h // 10 :]
            cls.rf_power[i, :] = cavity_control.generator_power()[-h // 10 :]
            cls.beam_loop_phase[i] = beam_control.phi_beam * 180 / np.pi
            cls.rf_beam_current_phase[i, :] = -np.angle(
                cavity_control.I_BEAM_COARSE[
                    cavity_control.n_coarse
                    + bucket_shift // 10 : cavity_control.n_coarse
                    + bucket_shift // 10
                    + n_bunches
                ]
            )

        cls.rf_beam_current_phase = np.mean(
            np.unwrap(cls.rf_beam_current_phase) * 180 / np.pi, axis=1
        )
        cls.rf_beam_current_phase = (
            cls.rf_beam_current_phase
            - cls.rf_beam_current_phase[0]
            + injection_phase_error
        )
        cls.beam_loop_phase = (
            cls.beam_loop_phase
            - cls.beam_loop_phase[0]
            + injection_phase_error
        )

    def test_beam_phase_loop(self):
        np.testing.assert_allclose(
            self.beam_loop_phase + 10,
            self.blond2_data["beam_loop_phase"] + 10,
            rtol=4e-5,
            err_msg="Error in phase loop error signal",
        )

    def test_rf_beam_current(self):
        np.testing.assert_allclose(
            self.rf_beam_current_phase + 10,
            self.blond2_data["rf_beam_current_phase"] + 10,
            rtol=4e-5,
            err_msg="Error in turn-by-turn phase of rf beam current",
        )

        np.testing.assert_allclose(
            np.abs(self.i_beam),
            np.abs(self.blond2_data["rf_beam_current"]),
            rtol=1e-5,
            err_msg="Error in absolute value of rf beam current",
        )

        np.testing.assert_allclose(
            np.angle(self.i_beam, deg=True),
            np.angle(self.blond2_data["rf_beam_current"], deg=True),
            rtol=2e-3,
            err_msg="Error in phase value of rf beam current",
        )

    def test_rf_voltage_transient(self):
        np.testing.assert_allclose(
            np.abs(self.v_ant),
            np.abs(self.blond2_data["rf_voltage"]),
            rtol=9e-6,
            err_msg="Error in absolute value of rf voltage",
        )

        np.testing.assert_allclose(
            np.angle(self.v_ant, deg=True) + 10,
            np.angle(self.blond2_data["rf_voltage"], deg=True) + 10,
            rtol=4e-5,
            err_msg="Error in phase value of rf voltage",
        )

    def test_rf_power_transient(self):
        np.testing.assert_allclose(
            np.abs(self.rf_power),
            np.abs(self.blond2_data["rf_power"]),
            rtol=2e-3,
            err_msg="Error in absolute value of rf power",
        )

        np.testing.assert_allclose(
            np.angle(self.rf_power, deg=True),
            np.angle(self.blond2_data["rf_power"], deg=True),
            atol=1e-9,
            err_msg="Error in phase value of rf power",
        )
