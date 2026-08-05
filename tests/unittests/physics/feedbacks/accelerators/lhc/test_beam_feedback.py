import unittest
from unittest.mock import Mock

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
    backend,
    proton,
)
from blond.core.backends.backend import Numpy64Bit
from blond.experimental.physics.feedbacks.base import (
    LocalFeedback,
)
from blond.physics.feedbacks.accelerators.lhc import (
    LHCBeamControl,
)

circumference = 26658.8832  # [m]
momentum = 450e9
intensity = 1.6e11
n_turns = 2_000
voltage = 5e6
h = 35640
gamma_t = 53.8
alpha = 1 / gamma_t / gamma_t

n_macroparticles = 100_000
tau_bunch = 1.2e-9
injection_offset_phase = 20
reference = -20


class TestLHCBeamFeedback(unittest.TestCase):
    def create_scenario(
        self,
        open_synchro: bool = False,
        time_offset: float | None = None,
        delay: int = 0,
        mock_cavity_feedback: bool = False,
        current_thres: float | None = None,
        mock_phase_noise=None,
    ):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")

        energy = np.sqrt(momentum**2 + proton.mass**2)
        rel_gamma = energy / proton.mass
        rel_beta = np.sqrt(1 - 1 / rel_gamma**2)

        self.beam = Beam(
            intensity,
            proton,
        )

        cycle = ConstantMagneticCycle(proton, momentum, in_unit="momentum")

        lattice = DriftSimple(
            orbit_length=circumference, momentum_compaction_factor=alpha
        )

        if mock_cavity_feedback:
            cavity_feedback = Mock(spec=LocalFeedback)
            n_coarse = 3564
            cavity_feedback.n_coarse = n_coarse
            _i_coarse = np.zeros(n_coarse, dtype=complex)
            _i_coarse[0] = 1.5 + 0 * 1j
            cavity_feedback.I_BEAM_COARSE = _i_coarse
            _v_ant = np.zeros(n_coarse, dtype=complex)
            _v_ant[:] = voltage * np.exp(1j * 10 / 180 * np.pi)
            cavity_feedback.V_ANT_COARSE = _v_ant
        else:
            cavity_feedback = None

        if mock_phase_noise:
            phase_noise = Mock()
            noise_val = 10 / 180 * np.pi
            phase_noise.dphi = noise_val * np.ones(n_turns, dtype=float)
            phase_noise.dphi[1:] = -10 / 180 * np.pi
        else:
            phase_noise = None

        cavity = SingleHarmonicRFStation(
            voltage=voltage,
            phi_rf=0.0,
            harmonic=h,
            cavity_feedback=cavity_feedback,
        )

        f_rf = cavity.calc_main_harmonic_omega_rf_design(
            rel_beta, lattice.orbit_length
        ) / (2 * np.pi)
        f_rev = f_rf / h
        t_rf = 1 / f_rf
        t_rev = 1 / f_rev

        self.profile = StaticProfile(
            cut_left=-1.5 * t_rf,
            cut_right=2.5 * t_rf,
            n_bins=4 * 2**6,
        )

        bigaussian = BiGaussian(
            n_macroparticles, sigma_dt=tau_bunch / 4, seed=1234
        )
        self.beam_control = LHCBeamControl(
            pl_gain=1 / (5 * t_rev) * 1,
            sl_gain=1 / (5 * t_rev) / 10 * int(not open_synchro),
            profile=self.profile,
            time_offset=time_offset,
            delay=delay,
            current_thres=current_thres,
            phase_noise=phase_noise,
        )

        cavity.attach_beam_feedback(self.beam_control)

        ring = Ring(
            circumference,
        )
        ring_elements = [self.profile, cavity, self.beam_control, lattice]
        ring.add_elements(
            ring_elements,
        )

        self.simulation = Simulation(
            ring,
            cycle,
        )

        self.simulation.prepare_beam(self.beam, bigaussian)

        self.beam._dt.array_local += injection_offset_phase * t_rf / 360

        self.profile.track(self.beam)

        self.simulation.finalize(
            (self.beam,),
            n_turns,
        )
        self.lhc_y_init = self.beam_control.lhc_y
        self.beam_control.reference = reference * np.pi / 180

        self.beam_control.track(self.beam)

    def create_double_scenario(
        self,
        open_synchro: bool = False,
        time_offset: float | None = None,
        delay: int = 0,
        mock_cavity_feedback: bool = False,
        one_mock_only: bool = False,
        current_thres: float | None = None,
        mock_phase_noise=None,
    ):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")

        energy = np.sqrt(momentum**2 + proton.mass**2)
        rel_gamma = energy / proton.mass
        rel_beta = np.sqrt(1 - 1 / rel_gamma**2)

        self.beam = Beam(
            intensity,
            proton,
        )

        cycle = ConstantMagneticCycle(proton, momentum, in_unit="momentum")

        circ_ratio = 0.0001
        lattice1 = DriftSimple(
            orbit_length=circumference * circ_ratio,
            momentum_compaction_factor=alpha,
            section_index=0,
        )
        lattice2 = DriftSimple(
            orbit_length=circumference * (1 - circ_ratio),
            momentum_compaction_factor=alpha,
            section_index=1,
        )

        if mock_cavity_feedback:
            cavity_feedback = Mock(spec=LocalFeedback)
            n_coarse = 3564
            cavity_feedback.n_coarse = n_coarse
            _i_coarse = np.zeros(n_coarse, dtype=complex)
            _i_coarse[0] = 1.5 + 0 * 1j
            cavity_feedback.I_BEAM_COARSE = _i_coarse
            _v_ant = np.zeros(n_coarse, dtype=complex)
            _v_ant[:] = voltage * np.exp(1j * 10 / 180 * np.pi)
            cavity_feedback.V_ANT_COARSE = _v_ant
        else:
            cavity_feedback = None

        if mock_phase_noise:
            phase_noise = Mock()
            noise_val = 10 / 180 * np.pi
            phase_noise.dphi = noise_val * np.ones(n_turns, dtype=float)
            phase_noise.dphi[1:] = -10 / 180 * np.pi
        else:
            phase_noise = None

        cavity1 = SingleHarmonicRFStation(
            voltage=voltage / 2,
            phi_rf=0.0,
            harmonic=h,
            cavity_feedback=cavity_feedback,
            section_index=0,
        )
        cavity2 = SingleHarmonicRFStation(
            voltage=voltage / 2,
            phi_rf=0.0,
            harmonic=h,
            cavity_feedback=cavity_feedback if not one_mock_only else None,
            section_index=1,
        )

        f_rf = cavity1.calc_main_harmonic_omega_rf_design(
            rel_beta, lattice1.orbit_length + lattice2.orbit_length
        ) / (2 * np.pi)
        f_rev = f_rf / h
        t_rf = 1 / f_rf
        t_rev = 1 / f_rev

        self.profile = StaticProfile(
            cut_left=-1.5 * t_rf,
            cut_right=2.5 * t_rf,
            n_bins=4 * 2**6,
        )

        bigaussian = BiGaussian(
            n_macroparticles, sigma_dt=tau_bunch / 4, seed=1234
        )
        self.beam_control = LHCBeamControl(
            pl_gain=1 / (5 * t_rev) * 1,
            sl_gain=1 / (5 * t_rev) / 10 * int(not open_synchro),
            profile=self.profile,
            time_offset=time_offset,
            delay=delay,
            current_thres=current_thres,
            phase_noise=phase_noise,
        )

        cavity1.attach_beam_feedback(self.beam_control)
        cavity2.attach_beam_feedback(self.beam_control)

        ring = Ring(
            circumference,
        )

        ring_elements = [
            self.profile,
            cavity1,
            lattice1,
            cavity2,
            self.beam_control,
            lattice2,
        ]

        ring.add_elements(
            ring_elements,
        )

        self.simulation = Simulation(
            ring,
            cycle,
        )

        self.simulation.prepare_beam(self.beam, bigaussian)

        self.beam._dt.array_local += injection_offset_phase * t_rf / 360

        self.profile.track(self.beam)

        self.simulation.finalize(
            (self.beam,),
            n_turns,
        )
        self.lhc_y_init = self.beam_control.lhc_y
        self.beam_control.reference = reference * np.pi / 180

        self.beam_control.track(self.beam)

    def test_lhc_beam_control_synchro_closed(self):
        self.create_scenario(open_synchro=False)

        # Checks the correction calculation of the recursion parameters for the synchronization loop
        self.assertAlmostEqual(self.beam_control.lhc_t, 0.0177111)

        self.assertAlmostEqual(self.lhc_y_init, 0)

        self.assertAlmostEqual(self.beam_control.lhc_a, 2.64280271)

        # Checks the calculation done by the beam phase loop for the first turn
        self.assertAlmostEqual(
            self.beam_control.dphi * 180 / np.pi,
            injection_offset_phase,
            places=2,
        )

        self.assertAlmostEqual(
            self.beam_control.pl_gain * self.beam_control.dphi,
            785.0284369961593,
        )

        # Checks the calculation done for the synchro loop for the first turn
        dphi_rf = self.beam_control.cavities[0].delta_phi_rf

        synch_corr = self.beam_control.sl_gain * (
            self.lhc_y_init
            + self.beam_control.lhc_a * (dphi_rf + self.beam_control.reference)
        )

        self.assertAlmostEqual(synch_corr, -207.48175328)

        self.assertAlmostEqual(self.beam_control.lhc_y, 0.01015636)

        # Checks the correct calculation of the corrections for the next turn
        self.assertAlmostEqual(
            self.beam_control.delta_omega_rf / 2 / np.pi, -91.91940957997551
        )

        self.assertAlmostEqual(
            self.beam_control.cavities[0].delta_omega_rf / 2 / np.pi,
            -91.91940957997551,
        )

    def test_lhc_beam_control_synchro_open(self):
        self.create_scenario(open_synchro=True)

        # Checks the correction calculation of the recursion parameters for the synchronization loop
        self.assertAlmostEqual(self.beam_control.lhc_t, 0.0)

        self.assertAlmostEqual(self.lhc_y_init, 0)

        self.assertAlmostEqual(self.beam_control.lhc_a, 0.0)

        # Checks the calculation done by the beam phase loop for the first turn
        self.assertAlmostEqual(
            self.beam_control.dphi * 180 / np.pi,
            injection_offset_phase,
            places=2,
        )

        self.assertAlmostEqual(
            self.beam_control.pl_gain * self.beam_control.dphi,
            785.0284369961593,
        )

        # Checks the calculation done for the synchro loop for the first turn
        dphi_rf = self.beam_control.cavities[0].delta_phi_rf

        synch_corr = self.beam_control.sl_gain * (
            self.lhc_y_init
            + self.beam_control.lhc_a * (dphi_rf + self.beam_control.reference)
        )

        self.assertAlmostEqual(synch_corr, 0.0)

        self.assertAlmostEqual(self.beam_control.lhc_y, 0.0)

        # Checks the correct calculation of the corrections for the next turn
        self.assertAlmostEqual(
            self.beam_control.delta_omega_rf / 2 / np.pi, -124.94115621564328
        )

        self.assertAlmostEqual(
            self.beam_control.cavities[0].delta_omega_rf / 2 / np.pi,
            -124.94115621564328,
        )

    def test_lhc_beam_control_time_offset(self):
        self.create_scenario(time_offset=0.0)
        # Checks the calculation done by the beam phase loop for the first turn
        self.assertAlmostEqual(
            self.beam_control.dphi * 180 / np.pi,
            injection_offset_phase,
            places=2,
        )

    def test_lhc_beam_control_delay(self):
        self.create_scenario(
            delay=1,
        )

        self.assertAlmostEqual(
            self.beam_control.cavities[0].delta_omega_rf, 0.0
        )

    def test_lhc_beam_control_current_threshold(self):
        with self.assertRaises(RuntimeError):
            self.create_scenario(mock_cavity_feedback=True)

    def test_lhc_beam_control_cavity_sum_phase(self):
        self.create_scenario(mock_cavity_feedback=True, current_thres=0.5)

        self.assertAlmostEqual(
            self.beam_control.dphi * 180 / np.pi,
            injection_offset_phase + 10,
            places=2,
        )

    @pytest.mark.skip
    def test_lhc_beam_control_warning_with_two_cavity_controllers(self):
        with self.assertWarns(Warning):
            self.create_double_scenario(
                mock_cavity_feedback=True,
                current_thres=0.5,
                one_mock_only=True,
            )

    def test_lhc_beam_control_mock_rf_noise(self):
        self.create_scenario(
            mock_phase_noise=True,
        )

        self.assertAlmostEqual(
            self.beam_control.dphi * 180 / np.pi,
            injection_offset_phase + 10,
            places=2,
        )

        # Track the next turn
        self.simulation.turn_i.value = 1
        self.beam_control.track(self.beam)

        self.assertAlmostEqual(
            self.beam_control.dphi * 180 / np.pi,
            injection_offset_phase - 10,
            places=2,
        )

    @pytest.mark.skip
    def test_lhc_beam_control_double_rf_station(self):
        self.create_double_scenario(
            mock_cavity_feedback=True,
            current_thres=0.5,
        )

        self.assertAlmostEqual(
            self.beam_control.dphi * 180 / np.pi,
            injection_offset_phase + 10,
            places=2,
        )
