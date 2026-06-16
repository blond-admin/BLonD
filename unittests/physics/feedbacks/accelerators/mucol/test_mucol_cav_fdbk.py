"""Unit tests for the muon collider cavity feedback timing class."""

import unittest
import warnings
from unittest.mock import Mock, PropertyMock, patch

import numpy as np

from blond import (
    Beam,
    Resonators,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    mu_plus,
)
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass
from blond.physics.feedbacks.helpers import rf_beam_current
from blond.physics.impedances.solvers import (
    SingleTurnResonatorConvolutionSolver,
)

# Package-relative imports: the dirs above ``mucol`` have no __init__.py, so
# these test helpers are not importable by an absolute path under pytest.
from .stubs import StubBeam
from .support import lab_frame_voltage


class TestCavityFeedback(unittest.TestCase):
    """Tests for IQCavityFeedbackTimingClass step-size sanity checks."""

    def setUp(self):
        """Build a cavity feedback instance with RCS1 4-station parameters."""
        # RCS1 4 stations
        self.prof = Mock(StaticProfile)
        self.prof.hist_x = np.linspace(
            5.791514370530446e-10, 1.7351942079901463e-09, num=1024
        )
        self.prof.hist_y = np.zeros(1024)
        self.prof.cut_left = self.prof.hist_x[0]

        self.R_over_Q = 518
        self.Q_L = 1287601.7251526634
        self.n_cavities = 42.217908605563096
        self.generator_current = 0.0233441090290177 + 0.04958176818202371j
        self.initial_voltage = 30e6
        self.n_rf_periods_per_coarse_grid = 1  # TODO: check for 2 and 0.5
        self.delta_omega = -6717.47508329349

        self.cav_fdbk = IQCavityFeedbackTimingClass(
            profile=self.prof,
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            generator_current=self.generator_current,
            n_cavities=self.n_cavities,
            initial_voltage=self.initial_voltage,
            n_rf_periods_per_coarse_grid=self.n_rf_periods_per_coarse_grid,
            delta_omega=self.delta_omega,
        )

    def test_circuit_track_applies_delta_omega_phase_shift(self):
        """
        Check circuit_track() applies the delta_omega phase shift.

        circuit_track() feeds `relative_detuning = delta_omega / omega_input`
        into cavity_response(), which advances the antenna voltage each
        coarse-grid step by a complex factor containing `1j * relative_detuning
        * omega_times_T_s == 1j * delta_omega * delta_t`.

        This test drives circuit_track() with a hand-built, constant-step
        rf_centers grid and zero generator/beam current (no_beam=True), so
        the antenna voltage on the coarse grid should evolve purely by that
        per-step complex multiplier. We recompute the expected trajectory
        from the multiplier directly and compare it element-wise to what
        circuit_track() actually produced.
        """
        omega_input = 2 * np.pi * 1e9
        n_steps = 50
        dt = 1.2e-9

        # Build a constant-step rf_centers grid covering a single segment.
        self.cav_fdbk.rf_centers = np.arange(1, n_steps + 1) * dt
        self.cav_fdbk.rf_centers_lengths = np.array([n_steps])
        self.cav_fdbk.residual_time_last_rf_centers_calculation = 0.0
        self.cav_fdbk.last_rf_centers_entry = None

        # Zero out generator/beam current contributions so only the
        # `(1 - 0.5*omega*dt/Q_L + 1j*relative_detuning*omega*dt)` term
        # governs the antenna voltage evolution.
        self.cav_fdbk.generator_current_constant = 0.0 + 0.0j
        self.cav_fdbk.generator_current_coarse_grid = np.zeros(
            n_steps, dtype=complex
        )
        self.cav_fdbk.last_val_generator_current = 0.0 + 0.0j
        self.cav_fdbk.last_val_beam_current = 0.0 + 0.0j

        v0 = self.initial_voltage + 0.0j
        self.cav_fdbk.last_val_ant_voltage = v0
        self.cav_fdbk.antenna_voltage_coarse_grid = np.zeros(
            n_steps, dtype=complex
        )

        self.cav_fdbk.circuit_track(
            omega_input=omega_input,
            no_beam=True,
            start_index=0,
            end_index=n_steps,
        )

        v = self.cav_fdbk.antenna_voltage_coarse_grid

        omega_times_T_s = omega_input * dt
        expected_multiplier = (
            1 - 0.5 * omega_times_T_s / self.Q_L + 1j * self.delta_omega * dt
        )
        expected = v0 * expected_multiplier ** np.arange(1, n_steps + 1)

        np.testing.assert_allclose(v, expected, rtol=1e-12)

    def _patched_carrier_props(self, omega_carrier, sampling_time_coarse):
        """
        Patch the carrier properties that need a missing cavity object.

        Parameters
        ----------
        omega_carrier
            Carrier angular frequency to patch in.
        sampling_time_coarse
            Coarse-grid sampling time to patch in.

        Returns
        -------
        patch_omega
            Context manager patching ``omega_carrier``.
        patch_dt
            Context manager patching ``sampling_time_coarse``.
        """
        return (
            patch.object(
                IQCavityFeedbackTimingClass,
                "omega_carrier",
                new_callable=PropertyMock,
                return_value=omega_carrier,
            ),
            patch.object(
                IQCavityFeedbackTimingClass,
                "sampling_time_coarse",
                new_callable=PropertyMock,
                return_value=sampling_time_coarse,
            ),
        )

    def test_step_size_check_warns_for_large_decay_per_step(self):
        """Warn when the per-step decay is between the soft and hard limits."""
        # 0.5 * omega * dt / Q_L should be between the soft (0.1) and hard
        # (2.0) thresholds: large enough to warn, small enough not to raise
        omega_carrier = 2 * np.pi * 1e9
        sampling_time_coarse = 1e-9
        self.cav_fdbk.Q_L = 10.0
        self.cav_fdbk.delta_omega = 0.0  # avoid triggering the other warning

        patch_omega, patch_dt = self._patched_carrier_props(
            omega_carrier, sampling_time_coarse
        )
        with patch_omega, patch_dt, self.assertWarns(UserWarning) as cm:
            self.cav_fdbk._check_step_sizes()
        self.assertIn("decay_per_step", str(cm.warning))

    def test_step_size_check_warns_for_large_detuning_phase_per_step(self):
        """Warn when the per-step detuning phase exceeds the soft limit."""
        # delta_omega * dt should clearly exceed the 0.1 threshold
        omega_carrier = 2 * np.pi * 1e9
        sampling_time_coarse = 1e-6
        self.cav_fdbk.Q_L = 1e12  # avoid triggering the decay warning
        self.cav_fdbk.delta_omega = 1e6

        patch_omega, patch_dt = self._patched_carrier_props(
            omega_carrier, sampling_time_coarse
        )
        with patch_omega, patch_dt, self.assertWarns(UserWarning) as cm:
            self.cav_fdbk._check_step_sizes()
        self.assertIn("detuning_phase_per_step", str(cm.warning))

    def test_step_size_check_no_warning_for_small_step_parameters(self):
        """Do not warn when both per-step parameters are well below the limit."""
        # both 0.5 * omega * dt / Q_L and delta_omega * dt are well below 0.1
        omega_carrier = 2 * np.pi * 1e9
        sampling_time_coarse = 1e-9
        self.cav_fdbk.Q_L = 1e12
        self.cav_fdbk.delta_omega = 1.0

        patch_omega, patch_dt = self._patched_carrier_props(
            omega_carrier, sampling_time_coarse
        )
        with (
            patch_omega,
            patch_dt,
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            self.cav_fdbk._check_step_sizes()
        self.assertEqual(caught, [])

    def test_cavity_response_warns_for_large_beam_kick(self):
        """Warn when the relative beam kick is between the soft and hard limits."""
        # relative_kick should be between the soft (0.1) and hard (1.0)
        # thresholds: large enough to warn, small enough not to raise
        omega_times_T_s = 1.0
        self.cav_fdbk.rf_centers = np.array([1e-9, 2e-9])
        self.cav_fdbk.rf_centers_lengths = np.array([2])
        self.cav_fdbk.antenna_voltage_coarse_grid = np.array(
            [1.0 + 0.0j, 0.0j]
        )
        self.cav_fdbk.generator_current_coarse_grid = np.zeros(
            2, dtype=complex
        )
        self.cav_fdbk.beam_current_forward_coarse_grid = np.array(
            [0.0 + 0.0j, 1e-3 + 0.0j]
        )

        with self.assertWarns(UserWarning) as cm:
            self.cav_fdbk.cavity_response(
                omega_times_T_s=omega_times_T_s,
                coarse_grid_index_to_update=1,
                relative_detuning=0.0,
                no_beam=False,
            )
        self.assertIn("relative_kick", str(cm.warning))

    def test_cavity_response_no_warning_for_small_beam_kick(self):
        """Do not warn when the relative beam kick is negligibly small."""
        # beam_current * 0.5 * R_over_Q * omega_times_T_s is a tiny fraction
        # of the previous antenna voltage
        omega_times_T_s = 1e-9
        self.cav_fdbk.rf_centers = np.array([1e-9, 2e-9])
        self.cav_fdbk.rf_centers_lengths = np.array([2])
        self.cav_fdbk.antenna_voltage_coarse_grid = np.array(
            [self.initial_voltage + 0.0j, 0.0j]
        )
        self.cav_fdbk.generator_current_coarse_grid = np.zeros(
            2, dtype=complex
        )
        self.cav_fdbk.beam_current_forward_coarse_grid = np.array(
            [0.0 + 0.0j, 1.0 + 0.0j]
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.cav_fdbk.cavity_response(
                omega_times_T_s=omega_times_T_s,
                coarse_grid_index_to_update=1,
                relative_detuning=0.0,
                no_beam=False,
            )
        self.assertEqual(caught, [])

    def test_step_size_check_raises_for_unphysical_decay_per_step(self):
        """Raise when the per-step decay exceeds the hard limit."""
        # 0.5 * omega * dt / Q_L > 2.0 makes the Euler decay factor negative
        omega_carrier = 2 * np.pi * 1e9
        sampling_time_coarse = 1e-3
        self.cav_fdbk.Q_L = 1.0
        self.cav_fdbk.delta_omega = 0.0  # avoid the other (hard) error first

        patch_omega, patch_dt = self._patched_carrier_props(
            omega_carrier, sampling_time_coarse
        )
        with patch_omega, patch_dt, self.assertRaises(ValueError) as cm:
            self.cav_fdbk._check_step_sizes()
        self.assertIn("decay_per_step", str(cm.exception))

    def test_step_size_check_raises_for_unphysical_detuning_phase_per_step(
        self,
    ):
        """Raise when the per-step detuning phase exceeds the hard limit."""
        # delta_omega * dt > 2.0 makes the per-step rotation exceed one
        # step's worth of phase -- the discretization can no longer track
        # the cavity phase
        omega_carrier = 2 * np.pi * 1e9
        sampling_time_coarse = 1e-6
        self.cav_fdbk.Q_L = 1e12  # avoid the decay error/warning
        self.cav_fdbk.delta_omega = 1e7

        patch_omega, patch_dt = self._patched_carrier_props(
            omega_carrier, sampling_time_coarse
        )
        with patch_omega, patch_dt, self.assertRaises(ValueError) as cm:
            self.cav_fdbk._check_step_sizes()
        self.assertIn("detuning_phase_per_step", str(cm.exception))

    def test_step_size_check_fires_on_run_simulation(self):
        """
        An unphysical detuning aborts the run-start initialisation.

        End-to-end companion to the patched-property step-size tests above:
        here ``_check_step_sizes`` runs inside ``on_run_simulation`` with the
        carrier frequency resolved through a real RF station. Only
        ``delta_omega`` is relevant, so the beam and simulation are stubbed --
        no beam preparation or tracking is needed.
        """
        t_rf = 1.0e-9
        omega_rf = 2 * np.pi / t_rf
        profile = StaticProfile.from_rad(np.pi * 1.5, np.pi * 4.5, 1024, t_rf)
        feedback = IQCavityFeedbackTimingClass(
            profile=profile,
            R_over_Q=518.0,
            Q_L=1.29e4,
            generator_current=0.0,
            n_cavities=1,
            initial_voltage=0.0,
            n_rf_periods_per_coarse_grid=1,
            # detuning_phase_per_step = delta_omega * sampling_time_coarse
            # ~ 1e12 * 1e-9 ~ 1000, far beyond the hard limit of 2.0.
            delta_omega=1e12,
        )
        # Voltage/harmonic are placeholders; only omega_rf enters the check.
        rf = SingleHarmonicRFStation(
            voltage=30e6,
            phi_rf=0.0,
            harmonic=25900,
            cavity_feedback=feedback,
            profile=profile,
        )
        # Normally set by the station's own initialisation at run start.
        rf.omega_rf_design = omega_rf

        # on_run_simulation only needs the ring's reference-altering elements
        # (to locate the parent station) and a deepcopy-able beam reference.
        stub_simulation = Mock()
        stub_simulation.ring.elements.get_elements.return_value = (rf,)

        with self.assertRaises(ValueError) as cm:
            feedback.on_run_simulation(
                simulation=stub_simulation,
                beam=StubBeam(2.7e12),
                n_turns=1,
            )
        self.assertIn("detuning_phase_per_step", str(cm.exception))

    def test_cavity_response_raises_for_unphysical_beam_kick(self):
        """Raise when the beam kick exceeds the previous antenna voltage."""
        # beam-induced kick exceeds the previous antenna voltage itself,
        # i.e. the Euler step would flip the sign of the antenna voltage
        omega_times_T_s = 1.0
        self.cav_fdbk.rf_centers = np.array([1e-9, 2e-9])
        self.cav_fdbk.rf_centers_lengths = np.array([2])
        self.cav_fdbk.antenna_voltage_coarse_grid = np.array(
            [1.0 + 0.0j, 0.0j]
        )
        self.cav_fdbk.generator_current_coarse_grid = np.zeros(
            2, dtype=complex
        )
        self.cav_fdbk.beam_current_forward_coarse_grid = np.array(
            [0.0 + 0.0j, 1e8 + 0.0j]
        )

        with self.assertRaises(ValueError) as cm:
            self.cav_fdbk.cavity_response(
                omega_times_T_s=omega_times_T_s,
                coarse_grid_index_to_update=1,
                relative_detuning=0.0,
                no_beam=False,
            )
        self.assertIn("relative_kick", str(cm.exception))


class TestFineGridResonatorBenchmark(unittest.TestCase):
    """
    Benchmark FB against resonator induced voltage single turn.

    Benchmark the single-turn (fine-grid) cavity beam-loading response of
    IQCavityFeedbackTimingClass against an independent resonator induced
    voltage model, on a real Gaussian-plus-noise beam profile.

    The fine-grid antenna voltage (generator current zeroed) is the purely
    beam-induced voltage. Demodulated at omega_rf, it is remodulated to the
    lab frame and compared to the induced voltage of a matching Resonators
    source (R_s = R_over_Q * Q_L, Q = Q_L, f_r = f_rf + delta_omega/2pi)
    convolved with the same profile.
    """

    R_over_Q = 518.0
    Q_L = 1287601.7251526634
    f_rf = 1.3e9
    intensity = 2.7e12
    n_macroparticles = int(1e6)
    n_bins = 2**12

    def _build_beam_and_profile(self, seed=0):
        rng = np.random.default_rng(seed)
        t_rf = 1.0 / self.f_rf
        profile = StaticProfile.from_rad(
            0.5 * np.pi, 3.5 * np.pi, self.n_bins, t_rf
        )
        t_center = t_rf
        sigma_t = 0.06 * t_rf
        n_noise = self.n_macroparticles // 10
        n_gauss = self.n_macroparticles - n_noise
        dt = np.concatenate(
            [
                rng.normal(t_center, sigma_t, n_gauss),
                rng.uniform(
                    t_center - 4 * sigma_t, t_center + 4 * sigma_t, n_noise
                ),
            ]
        )
        beam = Beam(
            intensity=self.intensity,
            particle_type=mu_plus,
            is_counter_rotating=False,
        )
        beam.setup_beam(
            dt=dt, dE=np.zeros_like(dt), mpi_mode="root-distributes"
        )
        profile.track(beam=beam)
        return beam, profile

    def _cavity_lab_voltage(self, beam, profile, delta_omega):
        omega_rf = 2.0 * np.pi * self.f_rf
        charges_fine = rf_beam_current(
            beam=beam,
            profile=profile,
            omega_c=omega_rf,
            T_rev=1.0 / self.f_rf,
            use_lowpass_filter=False,
            external_reference=False,
        )
        cav = IQCavityFeedbackTimingClass(
            profile=profile,
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            generator_current=0.0 + 0.0j,
            n_cavities=1,
            initial_voltage=0.0,
            delta_omega=delta_omega,
        )
        cav.beam_current_fine_grid = charges_fine / profile.hist_step
        cav.generator_current_fine_grid = np.zeros(self.n_bins, dtype=complex)
        cav.cavity_response_fine(
            initial_voltage_fine_grid=0.0,
            initial_voltage_gradient_fine_grid=0.0,
            initial_generator_current_fine_grid=0.0,
            samples_per_rf_fine_grid=omega_rf * profile.hist_step,
            relative_detuning=delta_omega / omega_rf,
        )
        return lab_frame_voltage(
            cav.antenna_voltage_fine_grid,
            omega_rf,
            profile.hist_x,
            use_real=True,
        )

    def _resonator_induced_voltage(self, beam, profile, delta_omega):
        res = Resonators(
            shunt_impedances=self.R_over_Q * self.Q_L,
            quality_factors=self.Q_L,
            center_frequencies=self.f_rf + delta_omega / (2.0 * np.pi),
        )
        wf = WakeField(
            sources=(res,),
            solver=SingleTurnResonatorConvolutionSolver(),
            profile=profile,
        )
        wf.solver.on_wakefield_init_simulation(Mock(), wf)
        return np.asarray(wf.solver.calc_induced_voltage(beam=beam))

    def _assert_matches(self, delta_omega):
        beam, profile = self._build_beam_and_profile()
        v_cav = self._cavity_lab_voltage(beam, profile, delta_omega)
        v_res = self._resonator_induced_voltage(beam, profile, delta_omega)

        # Best-fit amplitude scale (should be ~1 with the sign convention
        # already folded into v_cav).
        scale = np.dot(v_res, v_cav) / np.dot(v_cav, v_cav)
        nrmse = (
            np.sqrt(np.mean((v_res - scale * v_cav) ** 2))
            / np.abs(v_res).max()
        )
        corr = np.corrcoef(v_cav, v_res)[0, 1]

        self.assertGreater(
            corr, 0.999, f"shape mismatch (corr={corr}) for {delta_omega=}"
        )
        self.assertAlmostEqual(
            scale,
            1.0,
            delta=0.05,
            msg=f"amplitude scale off ({scale}) for {delta_omega=}",
        )
        self.assertLess(
            nrmse,
            1e-2,
            f"waveform mismatch (nrmse={nrmse}) for {delta_omega=}",
        )

    def test_fine_grid_matches_resonator_on_resonance(self):
        """On resonance, fine-grid response matches the resonator model."""
        self._assert_matches(delta_omega=0.0)

    def test_fine_grid_matches_resonator_positive_detuning(self):
        """With positive detuning, the phase shift matches a detuned resonator."""
        self._assert_matches(delta_omega=5e6)

    def test_fine_grid_matches_resonator_negative_detuning(self):
        """With negative detuning, the phase shift matches a detuned resonator."""
        self._assert_matches(delta_omega=-2e7)


if __name__ == "__main__":
    unittest.main()
